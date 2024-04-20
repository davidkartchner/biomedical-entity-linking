import numpy as np
import pandas as pd
import seaborn as sns
import sys
import time
import itertools
import pickle

from tqdm.auto import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt

from bigbio_utils import (
    dataset_to_df,
    DATASET_NAMES,
    CUIS_TO_EXCLUDE,
    CUIS_TO_REMAP,
    resolve_abbreviation,
    dataset_to_documents,
    load_bigbio_dataset,
)
from dataset_consts import dataset_to_pretty_name, model_to_pretty_name, model_to_color

sns.set()
sns.set_style("whitegrid")

models = [
    "sapbert",
    "metamap",
    "krissbert",
    "scispacy",
    "medlinker",
    "arboel_biencoder",
    "arboel_crossencoder",
    "biobart",
    "biogenel",
]

to_add = ["bern2", "bootleg"]

pd.set_option("display.max_columns", 80)


def metamap_to_candidates(metamap_output, mappings_first=False):
    if mappings_first:
        cols = ["text", "mapping_cui_list", "candidate_cui_list"]
    else:
        cols = [
            "text",
            "candidate_cui_list",
            "mapping_cui_list",
        ]

    # Create mapping from text to candidates for MedMentions
    text2candidates = defaultdict(list)

    for row in metamap_output[cols].values:
        text = row[0]
        candidates = eval(row[1])
        for c in eval(row[2]):
            candidates
        # TODO: Need to account for correct database for
        candidates = [
            ["UMLS:" + x if ":" not in x else x.replace("ncbigene", "NCBIGene")]
            for x in candidates
        ]
        text2candidates[text] = candidates

    return text2candidates


def deduplicate_candidates(candidates, return_counts=False):
    deduplicated = []
    counts = defaultdict(int)
    for c in candidates:
        if type(c) == list:
            deduped_subset = [x for x in c if x not in counts]
            if len(deduped_subset) > 0:
                deduplicated.append(deduped_subset)

            for x in c:
                counts[x] += 1

        else:
            counts[c] += 1
            if c not in deduplicated:
                deduplicated.append(c)

    if return_counts:
        return deduplicated, counts

    return deduplicated


def output_list_to_candidates(output_list, k=20, sort_by_counts=False):
    """
    Turn model output into candidate list
    """
    output2candidates = defaultdict(list)
    for mention in output_list:
        doc_id = mention["document_id"]
        offsets = mention["offsets"]
        joined_offsets = ";".join([",".join([str(y) for y in x]) for x in offsets])
        # candidates = [y for x in mention['candidates'] for y in x]
        candidates = mention["candidates"]
        deduplicated_candidates, counts = deduplicate_candidates(
            candidates, return_counts=True
        )

        # Syntax to prioritize candidates that come up more frequently in retrieved results
        if sort_by_counts:
            top_cuis = sorted(
                [(k, v) for k, v in counts.items()], key=lambda x: x[1], reverse=True
            )
            grouped_candidates = []
            for k, g in itertools.groupby(top_cuis, key=lambda x: x[1]):
                grouped_candidates.append(list(g))

            output2candidates[(doc_id, joined_offsets)] = grouped_candidates
        else:
            output2candidates[(doc_id, joined_offsets)] = deduplicated_candidates[:k]

    return output2candidates


def add_candidates_to_df(df, candidate_dict, new_col_name, eval_col="text"):
    """
    Add candidates produced by a model to results dataframe for evaluation

    Parameters:
    ----------------------------
        df: pandas.DataFrame
            Dataframe of bigbio mention produced by bigbio_utils.dataset_to_df

        candidate_dict: dict
            Mapping of (document_id, joined_offsets) -> candidates

        new_col_name: str
            Name of new column

        eval_col: Column to evaluate on
    """
    df[new_col_name] = df[["document_id", "joined_offsets"]].apply(
        lambda x: candidate_dict[(x[0], x[1])], axis=1
    )


def list_flatten(nested_list):
    used = set([])
    flattened = []
    for x in nested_list:
        for y in x:
            if y not in used:
                flattened.append(y)
                used.add(y)

    return flattened


def min_hit_index(gold_cuis, candidates, eval_mode):
    """
    Find index of first hit in candidates
    """
    if eval_mode == "basic":
        flat_candidates = list_flatten(candidates)
        for i, c in enumerate(flat_candidates):
            if c in gold_cuis:
                return i
    elif eval_mode == "strict":
        for i, c in enumerate(candidates):
            if all(x in gold_cuis for x in c):
                return i

    elif eval_mode == "relaxed":
        for i, c in enumerate(candidates):
            if any(x in gold_cuis for x in c):
                return i

    else:
        raise ValueError(f"eval_mode {eval_mode} not supported")

    return 1000000


def recall_at_k(
    df,
    candidate_col,
    gold_col="db_ids",
    max_k: int = 10,
    filter_null=False,
    eval_mode="basic",
):
    """
    Compute recall@k for all values of k < max_k
    """
    # Filter rows wilt null values after CUI remapping
    if filter_null:
        before_row_count = df.shape[0]
        df = df[df[gold_col].map(lambda x: len(x) > 0)]
        after_row_count = df.shape[0]
        print(f"Dropped {before_row_count - after_row_count} rows with null db_ids")

    df[f"{candidate_col}_min_hit_index"] = df[[gold_col, candidate_col]].apply(
        lambda x: min_hit_index(x[0], x[1], eval_mode=eval_mode), axis=1
    )

    recall_at_k_dict = {}
    for k in range(1, max_k + 1):
        recall_at_k_dict[k] = (df[f"{candidate_col}_min_hit_index"] < k).mean()

    return recall_at_k_dict


def plot_recall_at_k(
    recall_dict, max_k=10, legend_key=None, ax=None, color=None, alpha=1
):
    if ax is not None:
        ax.plot(
            recall_dict.keys(),
            recall_dict.values(),
            marker="o",
            label=legend_key,
            color=color,
            alpha=alpha,
        )
    else:
        plt.plot(
            recall_dict.keys(),
            recall_dict.values(),
            marker="o",
            label=legend_key,
            color=color,
            alpha=alpha,
        )
    # if legend_key is not None:
    #     plt.legend()


def correct_medlinker_candidates(medlinker_output):
    for x in medlinker_output:
        new_candidates = []
        for y in x["candidates"]:
            single_candidate = []
            for c in y:
                if c is None:
                    continue
                if not c.startswith("MESH") and not c.startswith("OMIM"):
                    c = c.replace("ESH", "MESH")
                single_candidate.append(c)
            new_candidates.append(single_candidate)
        x["candidates"] = new_candidates
    return medlinker_output
