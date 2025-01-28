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
import pandas as pd
from scipy.stats import chi2_contingency

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
        flat_candidates = list_flatten(
            candidates
        )  # Flatten list of lists to find the correct hit index (and not consider several cuis for one test)
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


def precision_at_k_DK(df, hit_column, k):
    """
    Calculates Precision@k for a given DataFrame and value of k. (David's version)
    ---------
    df : pandas.DataFrame
        DataFrame containing the results of the model.
    hit_column : str
        Name of the column containing the hit index.
    k : int
        Value of k.
    """
    hits_within_k = df[hit_column] <= (k - 1)
    precision = hits_within_k.sum() / len(df)
    return precision


def precision_at_k(df, hit_column, k):
    """
    Calculates Precision@k for a given DataFrame and value of k.
    ---------
    df : pandas.DataFrame
        DataFrame containing the results of the model.
    hit_column : str
        Name of the column containing the hit index.
    k : int
        Value of k.
    """
    precision_sum = 0
    for hit_index in df[hit_column]:
        if hit_index < k:
            precision_sum += 1 / k
    precision_at_k = precision_sum / len(df)
    return precision_at_k


def average_precision_at_k(hit_index, k):
    """
    Calculates Average Precision@k for a single query (mention).
    --------
    hit_index : int
        Index of the first hit.
    k : int
    """
    if hit_index <= (k - 1):  # Relevant item is within top k
        return 1 / (
            hit_index + 1
        )  # Precision at the position where the relevant item is found
    else:
        return 0


def mean_average_precision_at_k(df, hit_column, k):
    """
    Calculates MAP@k for the entire DataFrame.
    --------
    df : pandas.DataFrame
        DataFrame containing the results of the model.
    hit_column : str
        Name of the column containing the hit index.
    k : int
    """
    average_precisions = df[hit_column].apply(lambda x: average_precision_at_k(x, k))
    map_k = average_precisions.mean()  # Average over all queries
    return map_k


class Evaluate:
    def __init__(
        self,
        dataset_names,
        model_names,
        path_to_result,
        eval_strategies=["basic", "relaxed", "strict"],
        abbreviations_path=None,
        **kwargs,
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
        eval_strategies : list of str, optional
            A list of evaluation strategies to be used (default is ["basic", "relaxed", "strict"]).
        """
        self.dataset_names = dataset_names
        self.model_names = model_names
        self.path_to_result = path_to_result
        self.abbreviations_path = abbreviations_path
        self.eval_strategies = eval_strategies
        self.data = {}  # df for each dataset
        self.full_results = defaultdict(dict)
        self.datasets = {}
        self.error_analysis_dfs = {}  # Results with hit_index for error analysis
        self.recall_all_eval_strategies = (
            {}
        )  # Recall@k for all eval strategies, datasets, and models
        self.detailed_results_analysis = (
            {}
        )  # Detailed results for failure stage, statistical significance (p_values, chi2), accuracy per type, recall@k per type, MAP@k
        self.max_k = kwargs.get("max_k", 10)
        self.split = kwargs.get("split", "test")

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
                if file_path and os.path.exists(file_path):
                    self.datasets[name][model] = ujson.load(open(file_path))
                else:
                    print(
                        f"Skipping model {model} for dataset {name} because the file is missing or the path is invalid."
                    )

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

            df = df[df["split"] == self.split].reset_index(drop=True)
            self.data[name] = df

    def evaluate(self, eval_strategies=None):
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
        eval_strategies = eval_strategies if eval_strategies else self.eval_strategies
        self.full_results = defaultdict(dict)  # Initialize as defaultdict
        use_resolved_abbrevs = True if self.abbreviations_path else False
        for eval_strategy in tqdm(eval_strategies):
            all_recall = {}
            print(f"Eval Strategy: {eval_strategy}")

            dfs_one_eval_strategy = {}
            for name in self.dataset_names:
                print(name)
                self.full_results[eval_strategy][name] = self.data[name].copy()
                df = self.full_results[eval_strategy][name]
                recall_dict = {}
                for model in self.model_names:
                    if model in df.columns:
                        if model == "scispacy":
                            recall_dict[model] = recall_at_k(
                                df,
                                model,
                                eval_mode=eval_strategy,
                                gold_col="db_ids",
                                max_k=self.max_k,
                            )
                        else:
                            recall_dict[model] = recall_at_k(
                                df,
                                model,
                                eval_mode=eval_strategy,
                                max_k=self.max_k,
                            )

                        if model + "_resolve_abbrev" in df.columns:
                            cand_col = model + "_resolve_abbrev"
                            if model == "scispacy":
                                deabbrev_recall = recall_at_k(
                                    df,
                                    cand_col,
                                    eval_mode=eval_strategy,
                                    gold_col="db_ids",
                                    max_k=self.max_k,
                                )
                            else:
                                deabbrev_recall = recall_at_k(
                                    df,
                                    cand_col,
                                    eval_mode=eval_strategy,
                                    max_k=self.max_k,
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

    def plot_results(self, eval_strategies=None):
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
        eval_strategies = eval_strategies if eval_strategies else self.eval_strategies
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
                df = self.full_results[eval_strategy][name]

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
                            max_k=self.max_k,
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

    def detailed_results(self, k=None, eval_strategies=None):
        """
        Returns detailed results for further analysis : accuracy per type, recall@k per type, failure stage, statistical significance (p_values, chi2)
        The results are stored in the attribute detailed_results_analysis and is a nested dictionary with the following structure:
        self.detailed_results_analysis[eval_strat][dataset_name][model_name][feature]
        - eval_strat : evaluation strategy (e.g. basic, relaxed, strict)
        - dataset_name : name of the dataset (e.g. bc5cdr, ncbi_disease, nlmchem)
        - model_name : name of the model (e.g. arboEL, krissbert)
        - feature : feature of interest (count_miss_CG_per_type, count_miss_NED_per_type, count_success_per_type, count_hit_k_per_type, mentions_count_per_type, accuracy_per_type, recall_k_per_type, failure_stage_CG, failure_stage_NED, contingency_table_CG, contingency_table_NED, ChiSquare_test_NED, ChiSquare_test_CG)
        """
        k = k if k else self.max_k
        eval_strategies = eval_strategies if eval_strategies else self.eval_strategies
        resolve_abbrev = True if self.abbreviations_path else False
        for eval_strat in eval_strategies:
            self.detailed_results_analysis[eval_strat] = {}
            for dataset_name in self.dataset_names:
                self.detailed_results_analysis[eval_strat][dataset_name] = {}

                for model_name in self.model_names:
                    if (
                        model_name
                        not in self.error_analysis_dfs[eval_strat][
                            dataset_name
                        ].columns  # Check if the model has been evaluated on the dataset
                    ):
                        print(
                            f"Skipping model {model_name} for dataset {dataset_name} because the column doesn't exist."
                        )
                        continue

                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ] = {}

                    df = self.error_analysis_dfs[eval_strat][dataset_name]
                    assert (
                        "type" in df.columns
                    ), "'type' column is missing in the DataFrame for a comparison per type"
                    # Convert the 'type' column to strings
                    df["type"] = df["type"].apply(str)

                    name = model_name
                    if resolve_abbrev:
                        name += "_resolve_abbrev"
                    print(f"Processing {name}")

                    # Number of mentions where failure happened at CG step per type
                    count_miss_CG_per_type = (
                        df[df[name] == 1000000].groupby("type").size()
                    )
                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ][
                        "count_miss_CG_per_type"
                    ] = (
                        count_miss_CG_per_type.to_frame().reset_index()
                    )  # pandas series to dataframe + reset index (not aligned otherwise)

                    # Number of mentions where failure happened at NED step per type
                    count_miss_NED_per_type = (
                        df[(df[name] != 1000000) & (df[name] != 0)]
                        .groupby("type")
                        .size()
                    )
                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ][
                        "count_miss_NED_per_type"
                    ] = (
                        count_miss_NED_per_type.to_frame().reset_index()
                    )  # pandas series to dataframe + reset index (not aligned otherwise)

                    # Number of correctly linked mentions for each type
                    count_success_per_type = df[df[name] == 0].groupby("type").size()
                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ][
                        "count_success_per_type"
                    ] = count_success_per_type.to_frame().reset_index()

                    # Number of mentions with hit_index < k (recall@k) per type
                    count_hit_k_per_type = df[df[name] < k].groupby("type").size()
                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ][
                        "count_hit_k_per_type"
                    ] = count_hit_k_per_type.to_frame().reset_index()

                    # Total number of mentions for each type
                    mentions_count_per_type = df["type"].value_counts()
                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ][
                        "mentions_count_per_type"
                    ] = mentions_count_per_type.to_frame().reset_index()

                    # Accuracy for each type
                    accuracy_per_type = (
                        count_success_per_type / mentions_count_per_type
                    ).fillna(0)
                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ]["accuracy_per_type"] = accuracy_per_type.to_frame().reset_index()

                    # recall@k for each type
                    recall_k_per_type = (
                        count_hit_k_per_type / mentions_count_per_type
                    ).fillna(0)
                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ]["recall_k_per_type"] = recall_k_per_type.to_frame().reset_index()

                    # Failure from CG stage
                    failure_stage_CG = count_miss_CG_per_type.sum() / (
                        count_miss_CG_per_type.sum() + count_miss_NED_per_type.sum()
                    )
                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ]["failure_stage_CG"] = failure_stage_CG

                    # Failure from NED stage
                    failure_stage_NED = count_miss_NED_per_type.sum() / (
                        count_miss_CG_per_type.sum() + count_miss_NED_per_type.sum()
                    )
                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ]["failure_stage_NED"] = failure_stage_NED

                    # print(f"Number of mentions where the CG step failed for model {model_name} on dataset {dataset_name} with eval strategy {eval_strat} : {count_miss_CG}")
                    # print(f"Number of mentions where the NED step failed for model {model_name} on dataset {dataset_name} with eval strategy {eval_strat} : {count_miss_NED}")
                    # print(f"Number of of correctly linked mentions for model {model_name} on dataset {dataset_name} with eval strategy {eval_strat} : {count_success}")
                    # print(f"Number of mentions with hit_index < k for model {model_name} on dataset {dataset_name} with eval strategy {eval_strat} : {count_hit_k}")
                    # print(f"Accuracy per type for model {model_name} on dataset {name} with eval strategy {eval_strat} : {accuracy_per_type}")
                    # print(f"Recall@k per type for model {model_name} on dataset {name} with eval strategy {eval_strat} : {recall_k_per_type}")
                    # print(" '%' of mentions for which linking failed in CG for model {model_name} on dataset {dataset_name} with eval strategy {eval_strat} : ", failure_stage_CG)
                    # print(" '%' of mentions for which linking failed in NED for model {model_name} on dataset {dataset_name} with eval strategy {eval_strat} : ", failure_stage_NED)

                    # Create a new column candidate generation 'CG' to indicate if the correct cui is among the candidates (1) or not (0)
                    df["CG"] = df[name].apply(lambda x: 1 if x != 1000000 else 0)
                    # Create a new column Named Entity Disambiguation 'NED' to indicate if the prediction was correct (1) or incorrect (0)
                    df["NED"] = df[name].apply(lambda x: 1 if x == 0 else 0)
                    # Explode the list so that an entity is counted for each type it belongs to
                    df_exploded = df.explode("type")

                    # Create contingency tables
                    contingency_table_CG = pd.crosstab(
                        df_exploded["type"], df_exploded["CG"]
                    )
                    contingency_table_NED = pd.crosstab(
                        df_exploded["type"], df_exploded["NED"]
                    )
                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ]["contingency_table_CG"] = contingency_table_CG
                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ]["contingency_table_NED"] = contingency_table_NED

                    # Perform the Chi-square test
                    chi2_NED, p_value_NED, dof_NED, expected_NED = chi2_contingency(
                        contingency_table_NED,
                    )
                    chi2_CG, p_value_CG, dof_CG, expected_CG = chi2_contingency(
                        contingency_table_CG,
                    )
                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ]["ChiSquare_test_NED"] = chi2_contingency(contingency_table_NED)
                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ]["ChiSquare_test_CG"] = chi2_contingency(contingency_table_CG)

                    # # # Print results
                    # # print("Results for success in CG step:")
                    # # print(f"Degree of freedom (dof) = number of different classes : {dof_CG}")
                    # # # observed vs expected frequencies under the assumption of independence
                    # # # If observed < expected, then there is a statistical difference for the class
                    # # print(f"Expected frequencies table :{expected_CG}")
                    # print(f"Chi-square statistic for model {model_name} on dataset {dataset_name} with eval strategy {eval_strat} : {chi2_CG}")
                    # print(f"P-value for model {model_name} on dataset {dataset_name} with eval strategy {eval_strat} : {p_value_CG}")

                    # print('-'*50)
                    # # print("Results for succes in NED step:")
                    # # print(f"Degree of freedom (dof) = number of different classes : {dof_NED}")
                    # # print(f"Expected frequencies table :{expected_NED}") # Expected values for class=0 (failure) and class=1 (success) for the different categories (rows)
                    # print(f"Chi-square statistic for model {model_name} on dataset {dataset_name} with eval strategy {eval_strat} : {chi2_NED}")
                    # print(f"P-value for model {model_name} on dataset {dataset_name} with eval strategy {eval_strat} : {p_value_NED}")

                    precision_k_DK = {}  # David's version of precision@k
                    precision_k = {}
                    map_k = {}

                    # filtered_df = df[df["sapbert_resolve_abbrev"] != 1000000]  # Filter out mentions whose correct answer were not in the candidates
                    for i in range(1, k + 1):
                        precision_k_DK[k] = precision_at_k_DK(
                            df, "sapbert_resolve_abbrev", k
                        )
                        precision_k[i] = precision_at_k(df, name, i)
                        map_k[i] = mean_average_precision_at_k(df, name, i)
                        # precision_k_DK[k] = precision_at_k_DK(filtered_df, "sapbert_resolve_abbrev", k)
                        # precision_k[i] = precision_at_k(filtered_df, name, i)
                        # map_k[i] = mean_average_precision_at_k(filtered_df, name, i)

                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ]["precision@k"] = precision_k
                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ]["map@k"] = map_k
                    self.detailed_results_analysis[eval_strat][dataset_name][
                        model_name
                    ]["precision@k_DK"] = precision_k_DK
