import os
import ujson
import warnings
from collections import defaultdict
import pandas as pd

# from bigbio.dataloader import BigBioConfigHelpers
import json
import joblib
import numpy as np
import datetime

from tqdm.auto import tqdm

tqdm.pandas()

from datasets import load_dataset
from bioel.utils.bigbio_utils import (
    dataset_to_documents,
    dataset_to_df,
    resolve_abbreviation,
    load_bigbio_dataset,
)
from bioel.utils.dataset_consts import (
    CUIS_TO_REMAP,
    CUIS_TO_EXCLUDE,
    VALIDATION_DOCUMENT_IDS,
)
from bioel.ontology import BiomedicalOntology
from sklearn.feature_extraction.text import TfidfVectorizer

from bioel.ontology import BiomedicalOntology
from typing import List


def cuis_to_aliases(ontology: BiomedicalOntology, save_dir: str, dataset_name: str):
    """
    Creates the .txt file that maps the Cuis to the aliases and canonical name for each entities
    from the provided ontology.

    Parameters
    ----------
    ontology_dir: str, required.
        The path where the ontology is stored.
    save_dir : str, required.
        The path where the mapping file "{dataset_name}_aliases.txt" will be stored
    dataset_name : str, required.
        The name of the dataset for which the mapping will be created.
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    import re

    file_path = os.path.join(save_dir, f"{dataset_name}_aliases.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        for cui, entity in ontology.entities.items():
            file.write(f"{cui}||{entity.name}\n")
            if len(entity.aliases) != 0:
                if isinstance(entity.aliases, list):
                    for word in entity.aliases:
                        if word is not None:
                            # print(word)
                            word = word.strip()
                            file.write(f"{cui}||{word}\n")
                else:
                    # words = entity.aliases.split('; , ')
                    words = re.split("[;|]", entity.aliases)
                    for word in words:
                        word = word.strip()
                        file.write(f"{cui}||{word}\n")
            if entity.equivalant_cuis:
                for eqcui in entity.equivalant_cuis:
                    if eqcui != entity.cui:
                        file.write(f"{eqcui}||{entity.name}\n")
                        if entity.aliases:
                            words = re.split("[;|]", entity.aliases)
                            for word in words:
                                word = word.strip()
                                file.write(f"{eqcui}||{word}\n")

    return file_path


def create_tfidf_ann_index(out_path: str, input_path_kb: str):
    """
    Build tfidf vectorizer and ann index.

    Parameters
    ----------
    out_path: str, required.
        The path where the various model pieces will be saved.
    input_path_kb : KnowledgeBase, required
        The kb items to generate the index and vectors for.

    """
    tfidf_vectorizer_path = f"{out_path}/tfidf_vectorizer.joblib"
    concept_aliases = []
    os.makedirs(out_path, exist_ok=True)

    # Open the text file for reading
    with open(input_path_kb, "r") as file:
        # Read each line from the file
        for line in file:
            # Split the line at "||" and extract the word after the separator
            parts = line.strip().split("||")
            if len(parts) > 1:
                alias = parts[1]
                # Append the alias to the list
                concept_aliases.append(alias)

    # concept_aliases = list(kb.alias_to_cuis.keys())

    # NOTE: here we are creating the tf-idf vectorizer with float32 type, but we can serialize the
    # resulting vectors using float16, meaning they take up half the memory on disk. Unfortunately
    # we can't use the float16 format to actually run the vectorizer, because of this bug in sparse
    # matrix representations in scipy: https://github.com/scipy/scipy/issues/7408
    print(f"Fitting tfidf vectorizer on {len(concept_aliases)} aliases")
    tfidf_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 3), min_df=10, dtype=np.float32
    )
    start_time = datetime.datetime.now()
    concept_alias_tfidfs = tfidf_vectorizer.fit_transform(concept_aliases)
    print(f"Saving tfidf vectorizer to {tfidf_vectorizer_path}")
    joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print(f"Fitting and saving vectorizer took {total_time.total_seconds()} seconds")


def put_mention_in_context(
    doc_id,
    doc,
    mention,
    offsets,
    max_len,
    start_delimiter="START ",
    end_delimiter=" END",
    resolve_abbrevs=False,
    abbrev_dict=None,
):
    """
    Put a mention in context of surrouding text
    """
    # Resolve abbreviaions if desired
    if resolve_abbrevs and abbrev_dict is not None:
        mention = resolve_abbreviation(doc_id, mention, abbrev_dict)

    tagged_mention = start_delimiter + mention + end_delimiter
    start = offsets[0][0]
    end = offsets[-1][-1]
    before_context = doc[:start]
    after_context = doc[end:]
    before_split_context = before_context.split(" ")
    after_split_context = after_context.split(" ")
    len_before = len(before_split_context)
    len_after = len(after_split_context)

    max_ctx_l_len = max_len // 2
    max_ctx_r_len = max_len - max_ctx_l_len

    if len_before < max_ctx_l_len and len_after < max_ctx_r_len:
        mention_in_context = before_context + tagged_mention + after_context

    elif len_before >= max_ctx_l_len and len_after >= max_ctx_r_len:
        mention_in_context = (
            " ".join(before_split_context[-max_ctx_l_len:])
            + " "
            + tagged_mention
            + " "
            + " ".join(after_split_context[:max_ctx_r_len])
        )

    elif len_before >= max_ctx_l_len:
        ctx_l_len = max_len - len_after
        mention_in_context = (
            " ".join(before_split_context[-ctx_l_len:])
            + " "
            + tagged_mention
            + after_context
        )
    else:
        ctx_r_len = max_len - len_before
        mention_in_context = (
            before_context
            + tagged_mention
            + " "
            + " ".join(after_split_context[:ctx_r_len])
        )

    return mention_in_context


def contextualize_mentions(
    doc_dict,
    deduplicated,
    max_len=128,
    resolve_abbrevs=False,
    abbrev_dict=None,
    start_delimiter="START ",
    end_delimiter=" END",
):
    deduplicated["contextualized_mention"] = deduplicated[
        ["document_id", "mention", "offsets"]
    ].progress_apply(
        lambda x: put_mention_in_context(
            x[0],
            doc_dict[x[0]],
            x[1],
            x[2],
            max_len=max_len,
            resolve_abbrevs=resolve_abbrevs,
            abbrev_dict=abbrev_dict,
        ),
        axis=1,
    )

    return deduplicated


def create_training_files(
    save_dir: str,
    dataset_name: str,
    document_dict: dict,
    deduplicated,
    abbreviations_dict: dict,
    cui2alias: dict,
    resolve_abbrevs: bool = True,
):
    # Get contextualized mentions
    # print("deduplicated here", deduplicated)
    test_docs = deduplicated[deduplicated["split"] == "validation"]
    print(test_docs)

    df = contextualize_mentions(
        document_dict,
        deduplicated,
        resolve_abbrevs=resolve_abbrevs,
        abbrev_dict=abbreviations_dict,
    )

    # df["entity_aliases"] = df["db_ids"].map(
    #     lambda x: list(set([z for y in x for z in cui2alias[y]]))
    # )
    # df["most_similar_alias"] = df[["mention", "entity_aliases"]].apply(
    #     lambda x: get_most_similar_alias(x[0], x[1], vectorizer)
    # )

    # Get closest synonym for each mention
    tfidf_vectorizer = f"file_dumps/{dataset_name}/tfidf_vectorizer.joblib"
    vectorizer = joblib.load(tfidf_vectorizer)

    # df["entity_aliases"] = df["db_ids"].map(
    #     lambda x: list(set([z for y in x for z in cui2alias[y]]))
    # )
    def get_aliases(db_ids):
        aliases = []
        for db_id in db_ids:
            if db_id in cui2alias:
                aliases.extend(cui2alias[db_id])
        return list(set(aliases))

    df["entity_aliases"] = df["db_ids"].map(get_aliases)

    # print(df[df.entity_aliases.map(lambda x: len(x)==0)])
    print("Getting most similar alias")
    df["most_similar_alias"] = df[["mention", "entity_aliases"]].progress_apply(
        lambda x: get_most_similar_alias(
            mention=x[0], cui_alias_list=x[1], vectorizer=vectorizer
        ),
        axis=1,
    )

    # Make json string in correct format for decoder
    df["source_json"] = df["contextualized_mention"].map(lambda x: ujson.dumps([x]))

    df["target_json"] = df[["mention", "most_similar_alias"]].progress_apply(
        lambda x: ujson.dumps([f"{x[0]} is", x[1] if x[1] is not None else "unknown"]),
        axis=1,
    )

    # Store/check results
    # print(df.head(5))
    # print("Saving Pickle)")
    # df.to_pickle(os.path.join(save_dir, "processed_mentions.pickle"))

    # Write data files to pickle
    for split in df.split.unique():
        subset = df.query("split == @split")
        split_name = split
        if split in ["valid", "validation"]:
            split_name = "dev"

        # Write files
        with open(os.path.join(save_dir, f"{split_name}label.txt"), "w") as f:
            entity_link_list = ["|".join(x) for x in subset.db_ids.tolist()]
            f.write("\n".join(entity_link_list))

        with open(os.path.join(save_dir, f"{split_name}.source"), "w") as f:
            f.write("\n".join(subset.source_json.tolist()))

        with open(os.path.join(save_dir, f"{split_name}.target"), "w") as f:
            f.write("\n".join(subset.target_json.tolist()))


def get_most_similar_alias(mention, cui_alias_list, vectorizer):
    """
    Get most similar CUI alias to current mention using TF-IDF.
    Returns None if cui_alias_list is empty.
    """
    if not cui_alias_list:
        return None
    most_similar_idx = cal_similarity_tfidf(cui_alias_list, mention, vectorizer)[0]
    return cui_alias_list[most_similar_idx]


def cal_similarity_tfidf(a: list, b: str, vectorizer):
    features_a = vectorizer.transform(a)
    features_b = vectorizer.transform([b])
    features_T = features_a.T
    sim = features_b.dot(features_T).todense()
    return sim[0].argmax(), np.max(np.array(sim)[0])


def create_target_kb_dict(file_path, dataset_name):
    with open(file_path + f"{dataset_name}_aliases.txt", "r") as f:
        lines = f.read().split("\n")

    # Construct dict mapping each CURIE to a list of aliases
    umls_dict = {}
    for line in tqdm(lines):
        if len(line.split("||")) != 2:
            print(line)
            continue
        cui, name = line.split("||")
        if cui in umls_dict:
            umls_dict[cui].add(name.lower())
        else:
            umls_dict[cui] = set([name.lower()])

    processed_dict = {key: list(val) for key, val in umls_dict.items()}
    return processed_dict


def data_preprocess(
    dataset_name: str,
    save_dir: str,
    ontology: BiomedicalOntology,
    path_to_abbrev: str,
    resolve_abbrevs=False,
):

    if not resolve_abbrevs:
        save_dir = os.path.join(save_dir, "no_abbr_res/")
    else:
        save_dir = os.path.join(save_dir, "abbr_res/")

    save_dir = os.path.join(save_dir, f"{dataset_name}/")

    mappings_dir = cuis_to_aliases(ontology, save_dir, dataset_name)
    create_tfidf_ann_index(f"file_dumps/{dataset_name}", mappings_dir)

    # read data
    # dataset = load_dataset(f"bigbio/{dataset_name}", name=f"{dataset_name}_bigbio_kb")
    dataset = load_bigbio_dataset(f"{dataset_name}")

    target_kb_dict = create_target_kb_dict(save_dir, dataset_name)
    # create target_kb.json
    with open(os.path.join(save_dir, "target_kb.json"), "w") as target_kb:
        str_target = ujson.dumps(target_kb_dict, indent=2)
        target_kb.write(str_target)

    if resolve_abbrevs:
        with open(path_to_abbrev) as json_file:
            abbreviations_dict = ujson.load(json_file)
    else:
        abbreviations_dict = None

    entity_remapping_dict = CUIS_TO_REMAP[dataset_name]
    entities_to_exclude = CUIS_TO_EXCLUDE[dataset_name]

    if dataset_name in VALIDATION_DOCUMENT_IDS:
        validation_pmids = VALIDATION_DOCUMENT_IDS[dataset_name]
    else:
        validation_pmids = None

    deduplicated = dataset_to_df(
        dataset,
        entity_remapping_dict=entity_remapping_dict,
        cuis_to_exclude=entities_to_exclude,
        val_split_ids=validation_pmids,
    )
    deduplicated["mention"] = deduplicated["text"]

    for split in deduplicated.split.unique():
        print(split)
    split_docs = dataset_to_documents(dataset)

    # create source file
    create_training_files(
        save_dir,
        dataset_name=dataset_name,
        document_dict=split_docs,
        deduplicated=deduplicated,
        abbreviations_dict=abbreviations_dict,
        cui2alias=target_kb_dict,
        resolve_abbrevs=resolve_abbrevs,
    )

    return save_dir
