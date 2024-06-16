from typing import List, Dict
import time
import logging
import json
from dataclasses import dataclass
import ujson

import torch
from torch import Tensor as T
from transformers import PreTrainedTokenizer
from bioel.utils.bigbio_utils import (
    load_bigbio_dataset,
    dataset_to_df,
    resolve_abbreviation,
    dataset_to_documents,
    get_left_context,
    get_right_context,
    add_deabbreviations,
)
from bioel.utils.dataset_consts import (
    DATASET_NAMES,
    CUIS_TO_EXCLUDE,
    CUIS_TO_REMAP,
)
from tqdm.auto import tqdm

tqdm.pandas()


logger = logging.getLogger()


@dataclass
class Mention:
    cui: str
    start: int
    end: int
    text: str
    types: str
    deabbreviated_text: str


@dataclass
class ContextualMention:
    mention: str  # text
    cuis: List[str]
    ctx_l: str
    ctx_r: str
    deabbreviated_text: str

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
    split: str

    def to_dict(self):
        return {
            "document_id": self.document_id,
            "offsets": self.offsets,
            "text": self.mention,
            "type": self.types,
            # "start": self.start,
            # "end": self.end,
            "db_ids": self.cuis,
            "split": self.split,
            "deabbreviated_mention": self.deabbreviated_text,
            "mention_id": self.mention_id,
        }


class BigBioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        splits: List[str],
        abbreviations_path=None,
    ):
        self.dataset_name = dataset_name
        self.data = load_bigbio_dataset(dataset_name)
        self.splits = splits
        self.abbreviations_path = abbreviations_path
        self.name_to_cuis = {}

        exclude = CUIS_TO_EXCLUDE[dataset_name]
        remap = CUIS_TO_REMAP[dataset_name]

        if self.abbreviations_path:
            self.data = add_deabbreviations(
                dataset=self.data, path_to_abbrev=self.abbreviations_path
            )
            print("Resolved abbreviations")
        else:
            print("No abbreviations dictionary found.")

        df = dataset_to_df(
            self.data, cuis_to_exclude=exclude, entity_remapping_dict=remap
        )
        df["start"] = df["offsets"].map(lambda x: x[0][0])
        df["end"] = df["offsets"].map(lambda x: x[-1][-1])
        df = df[df.split.isin(splits)]
        print(dataset_name, splits, df.split.unique())

        self.df = df

        self.documents = dataset_to_documents(self.data)

        self._post_init()

    def _df_to_contextual_mentions(self, max_length: int = 64):
        self.df["ctx_l"] = self.df[["document_id", "start"]].progress_apply(
            lambda x: get_left_context(
                self.documents[x.iloc[0]], x.iloc[1], strip=True
            ),
            axis=1,
        )
        self.df["ctx_r"] = self.df[["document_id", "end"]].progress_apply(
            lambda x: get_right_context(
                self.documents[x.iloc[0]], x.iloc[1], strip=True
            ),
            axis=1,
        )

        self.df = self.df.rename(
            {"db_ids": "cuis", "type": "types", "text": "mention"}, axis=1
        ).drop(["start", "end"], axis=1)

        return [
            DetailedContextualMention(**x) for x in self.df.to_dict(orient="records")
        ]

    def to_contextual_mentions(self):
        """
        Turn dataset into set of contextual mentions
        """
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
    # Tests
    data = BigBioDataset("medmentions_full", splits=["train"])
    print(len(data))

    data = BigBioDataset("medmentions_full", splits=["train", "validation"])
    print(len(data))
