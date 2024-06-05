import numpy as np
import seaborn as sns
import ujson
import sys
import time
import itertools
import pickle
import math
import os

from tqdm.auto import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt

from bioel.utils.bigbio_utils import (
    load_bigbio_dataset,
    add_deabbreviations,
    dataset_to_documents,
    dataset_to_df,
    load_dataset_df,
    resolve_abbreviation,
    CUIS_TO_REMAP,
    CUIS_TO_EXCLUDE,
    DATASET_NAMES,
    VALIDATION_DOCUMENT_IDS,
)
from bioel.utils.dataset_consts import (
    dataset_to_pretty_name,
    model_to_pretty_name,
    model_to_color,
)


def metamap_to_candidates(metamap_output, mappings_first=False):
    """
    Converts MetaMap output into a dictionary mapping text mentions to candidate CUIs.

    Parameters
    ----------
    metamap_output : pandas.DataFrame
        A DataFrame containing MetaMap output with columns for text mentions and their corresponding mapping and candidate CUIs.
    mappings_first : bool, optional
        A flag indicating the order of columns in the DataFrame.
        If True, the columns are ["text", "mapping_cui_list", "candidate_cui_list"].
        If False, the columns are ["text", "candidate_cui_list", "mapping_cui_list"] (default is False).

    Returns
    -------
    dict
        A dictionary where keys are text mentions and values are lists of candidate CUIs.
    """
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
    """
    Deduplicates a list of candidate CUIs and optionally returns their counts.

    Parameters
    ----------
    candidates : list
        A list of candidate CUIs. Each element can be either a string (CUI) or a list of strings (CUIs).
    return_counts : bool, optional
        A flag indicating whether to return the counts of each CUI. Default is False.

    Returns
    -------
    list
        A list of deduplicated CUIs.
    dict (optional)
        A dictionary with CUIs as keys and their counts as values. Only returned if `return_counts` is True.
    """

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
    # Check if no candidates
    if candidates == [[]]:
        return 1000000

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


class Evaluate:
    def __init__(
        self, dataset_names, model_names, path_to_result, abbreviations_path=None
    ):
        """
        Parameters
        ----------
        dataset_names : list of str
            A list of dataset names to be used in the evaluation.
        model_names : list of str
            A list of model names to be evaluated.
        path_to_result : nested dict
            The path to the directory where the result files for each model and dataset are stored.
            Ex : path_to_result = {
                "bc5cdr": {
                    "arboEL": "/path/to/output_eval.json",
                    "krissbert": "/path/to/output_eval.json",
                },
                "ncbi_disease": {
                    "arboEL": "/path/to/output_eval.json",
                    "krissbert": "/path/to/output_eval.json",
                },
                "nlmchem" : {
                    "arboEL": "/path/to/output_eval.json",
                    "krissbert": "/path/to/output_eval.json",
                },
            }
        abbreviations_path : str
            The path to the JSON file containing abbreviations to be used for processing the datasets.
        """
        self.dataset_names = dataset_names
        self.model_names = model_names
        self.path_to_result = path_to_result
        self.abbreviations_path = abbreviations_path
        self.full_results = {}
        self.datasets = {}
        self.error_analysis_dfs = {}
        self.recall_all_eval_strategies = {}

    def load_results(self):
        """
        Loads the evaluation results for each model and dataset.

        Iterates over each dataset and model, loading the corresponding evaluation results from JSON files.
        Raises a FileNotFoundError if the expected result file for any model and dataset combination does not exist.
        """
        for name in self.dataset_names:
            self.datasets[name] = {}
            for model in self.model_names:
                file_path = self.path_to_result[name][model]
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Results file for model {model} on dataset {name} not found. Path to this file doesn't exist: {file_path}"
                    )
                self.datasets[name][model] = ujson.load(open(file_path))

    def process_datasets(self):
        """
        Processes each dataset by loading and preparing it for evaluation.

        For each dataset, this method:
        - Loads the dataset using `load_bigbio_dataset`.
        - Optionallly applies de-abbreviations to the dataset.
        - Converts the dataset to a DataFrame, applying entity remappings and exclusions.
        - Maps model results to the DataFrame and prepares it for evaluation.
        """
        for name in self.dataset_names:
            data = load_bigbio_dataset(name)
            exclude = CUIS_TO_EXCLUDE[name]
            remap = CUIS_TO_REMAP[name]
            if self.abbreviations_path:
                data_with_abbrev = add_deabbreviations(
                    dataset=data, path_to_abbrev=self.abbreviations_path
                )
                df = dataset_to_df(
                    data_with_abbrev,
                    entity_remapping_dict=remap,
                    cuis_to_exclude=exclude,
                )

            else:
                df = dataset_to_df(
                    data, entity_remapping_dict=remap, cuis_to_exclude=exclude
                )

            df["joined_offsets"] = df.offsets.map(
                lambda offsets: ";".join(
                    [",".join([str(y) for y in x]) for x in offsets]
                )
            )

            for model, results in self.datasets[name].items():
                print(f"dataset : {name}, model : {model}")
                for x in results:
                    x["candidates"] = [y for y in x["candidates"]]

                dict = {
                    x["mention_id"]: {
                        "db_ids": x["db_ids"],
                        "candidates": x["candidates"],
                    }
                    for x in results
                    if "abbr" not in x["mention_id"]
                }
                df[f"{model}"] = df["mention_id"].map(
                    lambda x: dict[x]["candidates"] if x in dict else [[] * 20]
                )

                resolve_abbr_dict = {
                    x["mention_id"].rstrip(".abbr_resolved"): {
                        "db_ids": x["db_ids"],
                        "candidates": x["candidates"],
                    }
                    for x in results
                    if "abbr" in x["mention_id"]
                }
                df[f"{model}_resolve_abbrev"] = df["mention_id"].map(
                    lambda x: (
                        resolve_abbr_dict[x]["candidates"]
                        if x in resolve_abbr_dict
                        else [[] * 20]
                    )
                )

            df = df[df["split"] == "test"].reset_index(drop=True)
            self.full_results[name] = df

    def evaluate(self, eval_strategies=["basic", "relaxed", "strict"]):
        """
        Evaluates the performance of models on each dataset using different evaluation strategies.

        For each evaluation strategy, this method:
        - Computes recall metrics for each model and dataset.
        - Handles cases where resolved abbreviations are used.
        - Stores detailed results for error analysis and recall metrics for different evaluation strategies.

        Parameters
        ----------
        eval_strategies : list of str, optional
            A list of evaluation strategies to be used (default is ["basic", "relaxed", "strict"]).

        """
        use_resolved_abbrevs = True if self.abbreviations_path else False
        for eval_strategy in tqdm(eval_strategies):
            all_recall = {}
            print(f"Eval Strategy: {eval_strategy}")

            dfs_one_eval_strategy = {}
            for name in self.dataset_names:
                print(name)
                df = self.full_results[name]
                recall_dict = {}
                for model in self.model_names:
                    if model in df.columns:
                        if model == "scispacy":
                            recall_dict[model] = recall_at_k(
                                df, model, eval_mode=eval_strategy, gold_col="db_ids"
                            )
                        else:
                            recall_dict[model] = recall_at_k(
                                df, model, eval_mode=eval_strategy
                            )

                        if model + "_resolve_abbrev" in df.columns:
                            cand_col = model + "_resolve_abbrev"
                            if model == "scispacy":
                                deabbrev_recall = recall_at_k(
                                    df,
                                    cand_col,
                                    eval_mode=eval_strategy,
                                    gold_col="db_ids",
                                )
                            else:
                                deabbrev_recall = recall_at_k(
                                    df, cand_col, eval_mode=eval_strategy
                                )
                            if use_resolved_abbrevs:
                                recall_dict[model] = deabbrev_recall

                all_recall[name] = recall_dict

                hit_index_cols = sorted([x for x in df.columns if "min_hit_index" in x])
                hit_index_df = df[
                    [
                        "document_id",
                        "text",
                        "deabbreviated_text",
                        "type",
                        "db_ids",
                        "mention_id",
                        "joined_offsets",
                    ]
                    + hit_index_cols
                ]
                hit_index_df = hit_index_df.rename(
                    {x: x.replace("_min_hit_index", "") for x in hit_index_df.columns},
                    axis=1,
                )

                dfs_one_eval_strategy[name] = hit_index_df

            self.error_analysis_dfs[eval_strategy] = dfs_one_eval_strategy
            self.recall_all_eval_strategies[eval_strategy] = all_recall

    def plot_results(self, eval_strategies=["basic", "relaxed", "strict"]):
        """
        Plots recall@k for each model and dataset using different evaluation strategies.

        This method:
        - Dynamically creates a grid of subplots based on the number of datasets.
        - Plots recall@k curves for each model on each dataset.
        - Uses distinct colors for each model for better visualization.
        - Handles different grid sizes and ensures proper labeling and legends.

        Parameters
        ----------
        eval_strategies : list of str, optional
            A list of evaluation strategies to be used (default is ['basic', 'relaxed', 'strict']).

        """
        # Define a color palette
        palette = sns.color_palette(
            "husl", len(self.model_names)
        )  # Using husl palette for distinct colors
        model_to_color = {model: palette[i] for i, model in enumerate(self.model_names)}

        for eval_strategy in tqdm(eval_strategies):
            print(f"{eval_strategy=}")
            all_recall = self.recall_all_eval_strategies[eval_strategy]

            # Determine the grid size based on the number of datasets
            num_datasets = len(self.dataset_names)
            num_cols = min(
                3, num_datasets
            )  # Limit the number of columns to 3 for better visualization
            num_rows = math.ceil(num_datasets / num_cols)

            fig, axs = plt.subplots(
                num_rows,
                num_cols,
                sharex=True,
                sharey=True,
                figsize=(num_cols * 4, num_rows * 4),
            )

            # Flatten the axs array if it's not 1D
            if num_rows == 1 and num_cols == 1:
                axs = [axs]
            elif num_rows == 1 or num_cols == 1:
                axs = axs.flatten()
            else:
                axs = axs.ravel()

            for ax in axs:
                if num_cols > 1:
                    ax.set_xlabel("k")
                if num_rows > 1:
                    ax.set_ylabel("recall@k")

            for idx, name in enumerate(self.dataset_names):
                recall_dict = all_recall[name]
                df = self.full_results[name]

                ax = axs[idx]
                ax.title.set_text(dataset_to_pretty_name.get(name, name))

                for model in self.model_names:
                    if model in recall_dict:
                        plot_recall_at_k(
                            recall_dict[model],
                            # legend_key=model_to_pretty_name.get(model, model),
                            legend_key=model,
                            ax=ax,
                            color=model_to_color[model],
                            alpha=0.7,
                        )

                print(f"{dataset_to_pretty_name.get(name, name)=}")

            # Hide any unused subplots
            for ax in axs[num_datasets:]:
                ax.axis("off")

            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=len(self.model_names),
                bbox_to_anchor=(0.5, -0.05),
            )

            fig.suptitle(
                f"Recall@K for all models using {eval_strategy} evaluation strategy",
                fontsize=16,
            )
            plt.show()
