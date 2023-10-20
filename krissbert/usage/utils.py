# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Dict
import os
import sys
import time
import logging
import json
import gzip
import pandas as pd
from dataclasses import dataclass, field
import ujson

import torch
from torch import Tensor as T
from transformers import PreTrainedTokenizer
from bigbio.dataloader import BigBioConfigHelpers
from tqdm.auto import tqdm

tqdm.pandas()

sys.path.append("../..")
from bigbio_utils import (
    dataset_to_df,
    DATASET_NAMES,
    CUIS_TO_EXCLUDE,
    CUIS_TO_REMAP,
    resolve_abbreviation,
    dataset_to_documents,
    get_left_context,
    get_right_context,
)


logger = logging.getLogger()


@dataclass
class Mention:
    cui: str
    start: int
    end: int
    text: str
    types: str


@dataclass
class ContextualMention:
    mention: str
    cuis: List[str]
    ctx_l: str
    ctx_r: str

    def to_tensor(self, tokenizer: PreTrainedTokenizer, max_length: int) -> T:
        ctx_l_ids = tokenizer.encode(
            text=self.ctx_l,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )
        ctx_r_ids = tokenizer.encode(
            text=self.ctx_r,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )
        mention_ids = tokenizer.encode(
            text=self.mention,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )

        # Concatenate context and mention to the max length.
        token_ids = (
            tokenizer.convert_tokens_to_ids(["<ENT>"])
            + mention_ids
            + tokenizer.convert_tokens_to_ids(["</ENT>"])
        )
        max_ctx_len = max_length - len(token_ids) - 2  # Exclude [CLS] and [SEP]
        max_ctx_l_len = max_ctx_len // 2
        max_ctx_r_len = max_ctx_len - max_ctx_l_len
        if len(ctx_l_ids) < max_ctx_l_len and len(ctx_r_ids) < max_ctx_r_len:
            token_ids = ctx_l_ids + token_ids + ctx_r_ids
        elif len(ctx_l_ids) >= max_ctx_l_len and len(ctx_r_ids) >= max_ctx_r_len:
            token_ids = (
                ctx_l_ids[-max_ctx_l_len:] + token_ids + ctx_r_ids[:max_ctx_r_len]
            )
        elif len(ctx_l_ids) >= max_ctx_l_len:
            ctx_l_len = max_ctx_len - len(ctx_r_ids)
            token_ids = ctx_l_ids[-ctx_l_len:] + token_ids + ctx_r_ids
        else:
            ctx_r_len = max_ctx_len - len(ctx_l_ids)
            token_ids = ctx_l_ids + token_ids + ctx_r_ids[:ctx_r_len]

        token_ids = [tokenizer.cls_token_id] + token_ids

        # The above snippet doesn't guarantee the max length limit.
        token_ids = token_ids[: max_length - 1] + [tokenizer.sep_token_id]

        if len(token_ids) < max_length:
            token_ids = token_ids + [tokenizer.pad_token_id] * (
                max_length - len(token_ids)
            )

        return torch.tensor(token_ids)


@dataclass
class DetailedContextualMention(ContextualMention):
    document_id: str
    types: str
    offsets: List[List[int]]
    # start: int
    # end: int
    mention_id: str
    # split: str

    def to_dict(self):
        return {
            "text": self.mention,
            "document_id": self.document_id,
            "type": self.types,
            "offsets": self.offsets,
            # "start": self.start,
            # "end": self.end,
            "db_ids": self.cuis,
        }


@dataclass
class Document:
    id: str = None
    title: str = None
    abstract: str = None
    mentions: List[Mention] = field(default_factory=list)

    def concatenate_text(self) -> str:
        return " ".join([self.title, self.abstract])

    @classmethod
    def from_PubTator(cls, path: str, split_path_prefix: str) -> Dict[str, List]:
        docs = []
        with gzip.open(path, "rb") as f:
            for b in f.read().decode().strip().split("\n\n"):
                d = cls()
                s = ""
                for i, ln in enumerate(b.split("\n")):
                    if i == 0:
                        id, type, text = ln.strip().split("|", 2)
                        assert type == "t"
                        d.id, d.title = id, text
                    elif i == 1:
                        id, type, text = ln.strip().split("|", 2)
                        assert type == "a"
                        assert d.id == id
                        d.abstract = text
                        s = d.concatenate_text()
                    else:
                        items = ln.strip().split("\t")
                        assert d.id == items[0]
                        cui = items[5].split("UMLS:")[-1]
                        assert len(cui) == 8, breakpoint()
                        m = Mention(
                            cui=cui,
                            start=int(items[1]),
                            end=int(items[2]),
                            text=items[3],
                            types=items[4].split(","),
                        )
                        assert m.text == s[m.start : m.end]
                        d.mentions.append(m)
                docs.append(d)
        dataset = split_dataset(docs, split_path_prefix)
        print_dataset_stats(dataset)
        return dataset

    def to_contextual_mentions(self, max_length: int = 64) -> List[ContextualMention]:
        text = self.concatenate_text()
        mentions = []
        for m in self.mentions:
            assert m.text == text[m.start : m.end]
            # Context
            ctx_l, ctx_r = (
                text[: m.start].strip().split(),
                text[m.end :].strip().split(),
            )
            ctx_l, ctx_r = " ".join(ctx_l[-max_length:]), " ".join(ctx_r[:max_length])
            cm = ContextualMention(
                mention=m.text,
                cuis=[m.cui],
                ctx_l=ctx_l,
                ctx_r=ctx_r,
            )
            mentions.append(cm)
        return mentions


# @dataclass
# class BigBioDocument:
#     dataset_name: str
#     pmid: str
#     passages: List[str]
#     mentions: List[Mention] = field(default_factory=list)

#     def concatenate_text(self) -> str:
#         return ' '.join(self.passages)

#     def


class BigBioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        splits: List[str],
        resolve_abbreviations: bool = True,
        abbreviations_dict_path="../../data/abbreviations.json",
    ):
        self.dataset_name = dataset_name
        conhelps = BigBioConfigHelpers()
        self.data = conhelps.for_config_name(f"{dataset_name}_bigbio_kb").load_dataset()
        self.splits = splits
        self.name_to_cuis = {}

        exclude = CUIS_TO_EXCLUDE[dataset_name]
        remap = CUIS_TO_REMAP[dataset_name]
        df = dataset_to_df(
            self.data, cuis_to_exclude=exclude, entity_remapping_dict=remap
        )
        df["start"] = df["offsets"].map(lambda x: x[0][0])
        df["end"] = df["offsets"].map(lambda x: x[-1][-1])
        df = df[df.split.isin(splits)]
        print(dataset_name, splits, df.split.unique())
        self.df = df

        self.documents = dataset_to_documents(self.data)

        if resolve_abbreviations:
            if abbreviations_dict_path is not None:
                self.abbreviations = ujson.load(open(abbreviations_dict_path))
                self.df.text = self.df[["document_id", "text"]].apply(
                    lambda x: resolve_abbreviation(
                        document_id=x[0],
                        text=x[1],
                        abbreviations_dict=self.abbreviations,
                    ),
                    axis=1,
                )
                print("Resolved abbreviations")
            else:
                print(
                    "No abbreviations dictionary found.  Setting resolve_abbreviations to False"
                )
        # self.abbreviations_dict =

        self._post_init()

    def _df_to_contextual_mentions(self, max_length: int = 64):
        self.df["ctx_l"] = self.df[["document_id", "start"]].progress_apply(
            lambda x: get_left_context(self.documents[x[0]], x[1], strip=True), axis=1
        )
        self.df["ctx_r"] = self.df[["document_id", "end"]].progress_apply(
            lambda x: get_right_context(self.documents[x[0]], x[1], strip=True), axis=1
        )

        self.df = self.df.rename(
            {"db_ids": "cuis", "type": "types", "text": "mention"}, axis=1
        ).drop(["split", "start", "end"], axis=1)

        return [
            DetailedContextualMention(**x) for x in self.df.to_dict(orient="records")
        ]

    # def _doc_to_contextual_mentions(self, doc, max_length: int = 64):
    #     pmid = doc["document_id"]
    #     text = " ".join([x for p in doc["passages"] for x in p["text"]])
    #     mentions = []
    #     for e in doc["entities"]:
    #         if len(e["normalized"]) == 0:
    #             continue
    #         mention = " ".join(e["text"])
    #         # db_name = e["normalized"][0]["db_name"]
    #         # db_id = e["normalized"][0]["db_id"]
    #         e_type = e["type"]
    #         curies = "|".join(
    #             [x["db_name"] + ":" + x["db_id"] for x in e["normalized"]]
    #         )
    #         start = e["offsets"][0][0]
    #         end = e["offsets"][-1][-1]

    #         # Quality control check for data
    #         for t, offset in zip(e["text"], e["offsets"]):
    #             st = offset[0]
    #             ed = offset[1]
    #             if t != text[st:ed]:
    #                 logger.info(f"pmid: {pmid}")
    #                 logger.info(f"entity: {e}")
    #                 logger.info(f"entity text: {t}")
    #                 logger.info(f"text in article: {text[st:ed]}")
    #             # assert t == text[st:ed]

    #         mentions.append([mention, e_type, curies, start, end])

    #     columns = ["mention", "type", "curie", "start", "end"]
    #     df = pd.DataFrame(mentions, columns=columns)
    #     dedup = (
    #         df.groupby(["start", "end"])
    #         .agg(
    #             {
    #                 "mention": "first",
    #                 "type": lambda x: "|".join(x),
    #                 "curie": lambda x: list(
    #                     set([c for concat in x for c in concat.split("|")])
    #                 ),
    #             }
    #         )
    #         .reset_index()
    #     )
    #     dedup["pmid"] = pmid
    #     dedup["ctx_l"] = dedup.start.map(
    #         lambda x: " ".join(text[:x].strip().split()[-max_length:])
    #     )
    #     dedup["ctx_r"] = dedup.end.map(
    #         lambda x: " ".join(text[x:].strip().split()[:max_length])
    #     )

    #     dedup = dedup.rename({"curie": "cuis", "type": "types"}, axis=1)

    #     return [DetailedContextualMention(**x) for x in dedup.to_dict(orient="records")]

    def to_contextual_mentions(self):
        """
        Turn dataset into set of contextual mentions
        """
        # contextual_mentions = []

        # for split in self.splits:
        #     logger.info(f"Split: {split}")
        #     for doc in tqdm(self.data[split]):
        #         contextual_mentions.extend(self._doc_to_contextual_mentions(doc))

        # self.mentions = contextual_mentions

        self.mentions = self._df_to_contextual_mentions()

    def _post_init(self):
        self.to_contextual_mentions()

        for m in self.mentions:
            if m.mention not in self.name_to_cuis:
                self.name_to_cuis[m.mention] = set()
            self.name_to_cuis[m.mention].update(m.cuis)

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, idx: int) -> DetailedContextualMention:
        return self.mentions[idx]


def split_dataset(docs: List, split_path_prefix: str) -> Dict[str, List]:
    split_kv = {"train": "trng", "dev": "dev", "test": "test"}
    id_to_split = {}
    dataset = {}
    for k, v in split_kv.items():
        dataset[k] = []
        path = split_path_prefix + v + ".txt"
        for i in open(path, encoding="utf-8").read().strip().split("\n"):
            assert i not in id_to_split, breakpoint()
            id_to_split[i] = k
    for doc in docs:
        split = id_to_split[doc.id]
        dataset[split].append(doc)
    return dataset


def print_dataset_stats(dataset: Dict[str, List[Document]]) -> None:
    all_docs = []
    for v in dataset.values():
        all_docs.extend(v)
    for split, docs in {"all": all_docs, **dataset}.items():
        logger.info(f"***** {split} *****")
        logger.info(f"Documents: {len(docs)}")
        logger.info(f"Mentions: {sum(len(d.mentions) for d in docs)}")
        cuis = set()
        for d in docs:
            for m in d.mentions:
                cuis.add(m.cui)
        logger.info(f"Mentioned concepts: {len(cuis)}")


class MedMentionsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, split: str) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.docs = Document.from_PubTator(
            path=os.path.join(self.dataset_path, "corpus_pubtator.txt.gz"),
            split_path_prefix=os.path.join(self.dataset_path, "corpus_pubtator_pmids_"),
        )[split]
        self.mentions = []
        self.name_to_cuis = {}
        self._post_init()

    def _post_init(self):
        for d in tqdm(self.docs):
            self.mentions.extend(d.to_contextual_mentions())
        for m in self.mentions:
            if m.mention not in self.name_to_cuis:
                self.name_to_cuis[m.mention] = set()
            self.name_to_cuis[m.mention].update(m.cuis)

    def __getitem__(self, index: int) -> ContextualMention:
        return self.mentions[index]

    def __len__(self) -> int:
        return len(self.mentions)


class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str) -> None:
        super().__init__()
        self.file = dataset_path
        self.data = []
        self.load_data()

    def load_data(self) -> None:
        with open(self.file, encoding="utf-8") as f:
            logger.info("Reading file %s" % self.file)
            for ln in f:
                if ln.strip():
                    self.data.append(json.loads(ln))
        logger.info("Loaded data size: {}".format(len(self.data)))

    def __getitem__(self, index: int) -> ContextualMention:
        d = self.data[index]
        return ContextualMention(
            ctx_l=d["context_left"],
            ctx_r=d["context_right"],
            mention=d["mention"],
            cuis=d["cuis"],
        )

    def __len__(self) -> int:
        return len(self.data)


def generate_vectors(
    encoder: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    max_length: int,
    is_prototype: bool = False,
):
    n = len(dataset)
    total = 0
    results = []
    start_time = time.time()
    logger.info("Start encoding...")
    for i, batch_start in enumerate(range(0, n, batch_size)):
        batch = [
            dataset[i] for i in range(batch_start, min(n, batch_start + batch_size))
        ]
        batch_token_tensors = [m.to_tensor(tokenizer, max_length) for m in batch]

        ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
        seg_batch = torch.zeros_like(ids_batch)
        attn_mask = ids_batch != tokenizer.pad_token_id

        with torch.inference_mode():
            out = encoder(
                input_ids=ids_batch, token_type_ids=seg_batch, attention_mask=attn_mask
            )
            out = out[0][:, 0, :]
        out = out.cpu()

        num_mentions = out.size(0)
        total += num_mentions

        if is_prototype:
            meta_batch = [{"cuis": m.cuis} for m in batch]
            assert len(meta_batch) == num_mentions
            results.extend(
                [(meta_batch[i], out[i].view(-1).numpy()) for i in range(num_mentions)]
            )
        else:
            results.extend(out.cpu().split(1, dim=0))

        if (i + 1) % 10 == 0:
            eta = (n - total) * (time.time() - start_time) / 60 / total
            logger.info(f"Batch={i + 1}, Encoded mentions={total}, ETA={eta:.1f}m")

    assert len(results) == n
    logger.info(f"Total encoded mentions={n}")
    if not is_prototype:
        results = torch.cat(results, dim=0)

    return results


if __name__ == "__main__":
    # data = BigBioDataset("medmentions_full", splits=["train"])
    # print(len(data))
    # # for i, d in enumerate(data):
    # #     if i < 10:
    # #         print(len())

    # data = BigBioDataset("medmentions_full", splits=["train", "validation"])
    # print(len(data))

    MedMentionsDataset("MedMentions/full/data/", "train")
