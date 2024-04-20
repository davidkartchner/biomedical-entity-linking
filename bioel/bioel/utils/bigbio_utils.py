import warnings
import os
import ujson
from collections import defaultdict
import pandas as pd
from datasets import load_dataset

from bioel.utils.dataset_consts import *

from datasets import load_dataset


def load_bigbio_dataset(dataset_name, abbrev=False, path_to_abbrev=None):
    """
    Load BigBio dataset and include abbreviations if specified.
    """
    # Load the dataset
    if dataset_name in {"medmentions_st21pv", "medmentions_full"}:
        dataset = load_dataset(
            f"bigbio/medmentions",
            name=f"{dataset_name}_bigbio_kb",
            trust_remote_code=True,
        )
    else:
        dataset = load_dataset(
            f"bigbio/{dataset_name}",
            name=f"{dataset_name}_bigbio_kb",
            trust_remote_code=True,
        )

    # If abbreviations are required, load the JSON file and update the dataset
    if abbrev:
        # Load the abbreviations.json file
        with open(path_to_abbrev, "r") as f:
            abbreviations = ujson.load(f)

        # Update the 'entities' in the dataset with abbreviations
        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                lambda example: {
                    "entities": [
                        {
                            **entity,
                            "abbreviation": match_abbreviation(
                                entity, example["document_id"], abbreviations
                            ),
                        }
                        for entity in example["entities"]
                    ]
                },
                batched=False,
            )

    return dataset


def cache_deduplicated_dataset(deduplicated_df):
    """
    Cache deduplicated dataset file for faster loading
    """
    raise NotImplementedError


def check_if_cached(dataset_name):
    """
    Check if dataset has been preprocessed and cached
    """
    raise NotImplementedError


def load_cached_dataset(dataset, splits_to_include: list = None):
    """
    Load cached deduplciated dataset as dataframe
    """
    raise NotImplementedError


def dataset_to_documents(dataset):
    """
    Return dictionary of documents in BigBio dataset
    """
    docs = {}

    for split in dataset.keys():
        for doc in dataset[split]:
            doc_id = pmid = doc["document_id"]
            doc_text = "\n".join([" ".join(x["text"]) for x in doc["passages"]])
            docs[doc_id] = doc_text
    return docs


def get_dataset_documents(dataset_name):
    """
    Get the documents from a BigBio dataset
    """
    dataset = load_bigbio_dataset(dataset_name)
    return dataset_to_documents(dataset)


def make_mention_id(document_id, running_mention_count):
    """
    Make a unique ID for each mention
    """
    running_mention_count[document_id] += 1
    return f"{document_id}.{running_mention_count[document_id]}"


# def get_dataset_df(dataset_name, abbrev_dict: dict = None):
#     data = load_bigbio_dataset(dataset_name)

#     data_df = dataset_to_df(
#         data,
#         entity_remapping_dict=CUIS_TO_REMAP[dataset_name],
#         cuis_to_exclude=CUIS_TO_EXCLUDE[dataset_name],
#         val_split_ids=VALIDATION_DOCUMENT_IDS[dataset_name],
#     )

#     if abbrev_dict is not None:
#         data_df["deabbreviated_text"] = data_df[["document_id", "text"]].apply(
#             lambda x: resolve_abbreviation(
#                 document_id=x[0],
#                 text=x[1],
#                 abbreviations_dict=abbrev_dict,
#             ),
#             axis=1,
#         )

#     return data_df


# Function to match text with abbreviations for each entity
def match_abbreviation(entity, doc_id, abbrev_dict):
    doc_abbrevs = abbrev_dict.get(str(doc_id), {})
    for text in entity["text"]:
        for abbr, full_form in doc_abbrevs.items():
            if full_form == text:
                return abbr
    return None


def dataset_to_df(
    dataset,
    splits_to_include: list = None,
    entity_remapping_dict: dict = None,
    cuis_to_exclude: list = None,
    val_split_ids: list = None,
    # abbreviations_dict: dict = None,
):
    """
    Convert BigBio dataset to pandas DataFrame

    Params:
    ------------------
        dataset: BigBio Dataset
            Dataset to load from BigBio

        splits_to_include: list of str
            List of splits to include in mo
    """
    columns = [
        # 'context', # string
        "document_id",  # string
        "mention_id",  # string
        "text",  # string
        "type",  # list
        "offsets",  # list of lists
        # "db_name",
        "db_ids",  # list
        "split",  # string
        # "abbreviation_resolved", # bool
        "abbreviation",  # string
    ]
    all_lines = []

    if splits_to_include is None:
        splits_to_include = dataset.keys()

    for split in splits_to_include:
        if split not in dataset.keys():
            warnings.warn(f"Split '{split}' not in dataset.  Omitting.")
        for doc in dataset[split]:
            pmid = doc["document_id"]
            for e in doc["entities"]:
                if len(e["normalized"]) == 0:
                    continue
                text = " ".join(e["text"])
                # abbreviation_resolved = False
                offsets = ";".join(
                    [",".join([str(y) for y in x]) for x in e["offsets"]]
                )
                # db_name = e["normalized"][0]["db_name"]
                db_ids = [
                    x["db_name"] + ":" + x["db_id"].strip() for x in e["normalized"]
                ]

                # Get the abbreviation if it exists, else set to None or an empty string
                abbreviation = e.get("abbreviation", None)

                # Remap entity IDs when identifier has changed in database
                if entity_remapping_dict is not None:
                    db_ids = [
                        entity_remapping_dict[x] if x in entity_remapping_dict else x
                        for x in db_ids
                    ]

                # Remove any IDs not included in database
                # Remove mentions with no entity link in database
                if cuis_to_exclude is not None:
                    new_db_ids = [x for x in db_ids if x not in cuis_to_exclude]
                else:
                    new_db_ids = db_ids
                if len(new_db_ids) == 0:
                    continue

                # Add mention + metadata to list of mentions
                all_lines.append(
                    [
                        pmid,
                        e["id"],
                        text,
                        e["type"],
                        # e['offsets'],
                        offsets,
                        # db_name,
                        new_db_ids,
                        split,
                        # abbreviation_resolved,
                        abbreviation,
                    ]
                )

    df = pd.DataFrame(all_lines, columns=columns)

    deduplicated = (
        df.groupby(["document_id", "offsets"])
        .agg(
            {
                "text": "first",
                "type": lambda x: list(set([a for a in x])),
                "db_ids": lambda db_ids: list(set([y for x in db_ids for y in x])),
                "split": "first",
                "abbreviation": "first",
            }
        )
        .reset_index()
    )

    deduplicated["offsets"] = deduplicated["offsets"].map(
        lambda x: [[int(z) for z in y.split(",")] for y in x.split(";")]
    )

    # Order mentions in consistent way (i.e. )
    deduplicated["first_offset"] = deduplicated["offsets"].map(lambda x: x[0][0])
    deduplicated["last_offset"] = deduplicated["offsets"].map(lambda x: x[-1][-1])
    deduplicated = deduplicated.sort_values(
        by=[
            "document_id",
            "first_offset",
            "last_offset",
        ]
    )
    deduplicated = deduplicated.drop(["first_offset", "last_offset"], axis=1)

    # Make a unique mention ID for each mention
    mention_counts = defaultdict(int)
    deduplicated["mention_id"] = deduplicated["document_id"].map(
        lambda x: make_mention_id(x, running_mention_count=mention_counts)
    )

    # Split off validation set if not given
    if val_split_ids is not None:
        print(type(val_split_ids[0]), type(deduplicated["document_id"][0]))
        deduplicated.loc[deduplicated["document_id"].isin(val_split_ids), "split"] = (
            "validation"
        )

    return deduplicated


def get_left_context(doc, start, max_length=64, strip=False):
    """
    Get all text in document that comes after a mentions up to $max_len words
    """
    if strip:
        return " ".join(doc[:start].strip().split()[-max_length:])
    else:
        return " ".join(doc[:start].split()[-max_length:])


def get_right_context(doc, end, max_length=64, strip=False):
    """
    Get all text in document that comes after a mentions up to $max_len words
    """
    if strip:
        return " ".join(doc[end:].strip().split()[:max_length])
    else:
        return " ".join(doc[end:].split()[:max_length])


def resolve_abbreviation(document_id, text, abbreviations_dict):
    """
    Return un-abbreviated form of entity name if it was found in abbreviations_dict, else return original text

    Inputs:
    -------------------------------
        document_id: str
            ID of document where mention was found

        text: str
            Text of mention

        abbreviations_dict: dict
            Dict of form {document_id:{text: unabbreviated_text}} containing abbreviations detected in each document
    """
    if text in abbreviations_dict[document_id]:
        return abbreviations_dict[document_id][text]
    else:
        return text


def load_dataset_df(name, abbrev=False, path_to_abbrev=None):
    """
    Load bigbio dataset and turn into pandas dataframe
    """
    data = load_bigbio_dataset(
        dataset_name=name, abbrev=abbrev, path_to_abbrev=path_to_abbrev
    )

    if name in CUIS_TO_EXCLUDE:
        exclude = CUIS_TO_EXCLUDE[name]
    else:
        exclude = None

    if name in CUIS_TO_REMAP:
        remap = CUIS_TO_REMAP[name]
    else:
        remap = None

    if name in VALIDATION_DOCUMENT_IDS:
        validation_pmids = VALIDATION_DOCUMENT_IDS[name]
    else:
        validation_pmids = None

    df = dataset_to_df(
        data,
        entity_remapping_dict=remap,
        cuis_to_exclude=exclude,
        val_split_ids=validation_pmids,
    )

    return df


def metamap_text_to_candidates(metamap_output):
    """
    Create mapping from text to list of candidates output by metamap

    Inputs:
    -------------------------------
        filepath: string or path-like
            Path to file containing output of MetaMap

    Returns:
    -------------------------------
        text2candidates: dict
            Dict of form {mention_text: [cand1, cand2, ...]} mapping each text string to candidates
            generated by metamap.  If no candidates were generated for a given key, value will be
            an empty string.
    """
    text2candidates = defaultdict(list)

    for row in metamap_output[
        ["text", "mapping_cui_list", "candidate_cui_list"]
    ].values:
        text = row[0]
        candidates = eval(row[1]) + eval(row[2])

        # TODO: Need to account for correct database
        raise NotImplementedError("Need to map UMLS values to correct DB")
        candidates = ["UMLS:" + x for x in candidates]

        text2candidates[text] = candidates
    return text2candidates


if __name__ == "__main__":
    dataset_name = "bc5cdr"
    dataset = load_bigbio_dataset(dataset_name)
    print(dataset_to_documents(dataset))
