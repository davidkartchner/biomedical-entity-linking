# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys

sys.path.append("../..")
sys.path.append("blink/biencoder")
import json
import math
import time
import torch
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
from tqdm import tqdm
import pickle
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components  # CC2
from special_partition.special_partition import (
    cluster_linking_partition,
)  # DD4 Implements MST + inference procedure

from collections import defaultdict
import blink.biencoder.data_process_mult as data_process
import blink.candidate_ranking.utils as utils
from blink.common.params import BlinkParser
from blink.biencoder.biencoder import BiEncoderRanker

from IPython import embed


def get_query_nn(
    knn, embeds, index, q_embed, searchK=None, gold_idxs=None, type_idx_mapping=None
):
    """
    Retrieves the top k neighbours of the query embedding (We know it's for mentions because of knn)

    Parameters
    ----------
    knn : int
        the number of nearest-neighbours to return
    embeds : ndarray of dim (number_mentions, H: embedding_size)
        matrix of embeddings
    index : faiss
        faiss index of the embeddings
    q_embed : ndarray of dim (1, H)
        2-D array containing the query embedding
    searchK: int
        optional parameter, the exact number of nearest-neighbours to retrieve and score
        (Used for approximate search)
    gold_idxs : array
        optional parameter, list of golden cui indexes (= the correct mentions for the query).
    type_idx_mapping : array
        optional parameter, list mapping type-specific indexes to the indexes of the full dictionary

    Returns
    -------
    nn_idxs : array
        indices of the nearest neighbour, sorted in descending order of scores
    scores : array
        similarity scores of the nearest neighbors, sorted in descending order
    """
    # To accomodate the approximate-nature of the knn procedure, retrieve more samples and then filter down
    k = (
        searchK if searchK is not None else max(16, 2 * knn)
    )  # 'is not None' not necessary

    # Find k nearest neighbours of the query
    _, nn_idxs = index.search(q_embed, k)
    nn_idxs = nn_idxs.astype(
        np.int64
    ).flatten()  # 2D array of dim = (1, k) --> 1D array of dim = (k,)
    if type_idx_mapping is not None:
        nn_idxs = type_idx_mapping[nn_idxs]
    nn_embeds = torch.tensor(embeds[nn_idxs]).cuda()

    # Compute query-candidate similarity scores
    scores = torch.flatten(torch.mm(torch.tensor(q_embed).cuda(), nn_embeds.T)).cpu()

    # Sort the candidates by descending order of scores
    nn_idxs, scores = zip(*sorted(zip(nn_idxs, scores), key=lambda x: -x[1]))  # CC1 DD1

    if gold_idxs is not None:
        # Calculate the knn index at which the gold cui is found (-1 if not found)
        # DD2 Only find one even though several can exist
        for topk, i in enumerate(nn_idxs):
            if i in gold_idxs:
                break
            topk = -1
        # Return only the top k neighbours, and the recall index
        return (
            np.array(nn_idxs[:knn], dtype=np.int64),
            np.array(scores[:knn]),
            topk,
        )  # DD3

    # Return only the top k neighbours
    return np.array(nn_idxs[:knn], dtype=np.int64), np.array(scores[:knn])


def partition_graph(
    graph, n_entities, directed, return_clusters=False, exclude=set(), threshold=None
):
    """
    Convert the initial graph constructed in train_biencoder_mst.evaluate(.) into the final graph resulting from the inference procedure

    Parameters
    ----------
    graph : dict
        object containing rows, cols, data, and shape of the entity-mention joint graph
    n_entities : int
        number of entities in the dictionary
    directed : bool
        whether the graph construction should be directed or undirected
    return_clusters : bool
        flag to indicate if clusters need to be returned from the partition
    exclude : set
        vertices (both rows and columns) whose edges should be dropped in the graph
    threshold : float
        similarity value below which edges should be dropped


    Returns
    -------
    partitioned_graph : coo_matrix
        partitioned graph with each mention connected to only one entity
    clusters : dict
        (optional) contains arrays of connected component indices of the graph
    """
    rows, cols, data, shape = (
        graph["rows"],
        graph["cols"],
        graph["data"],
        graph["shape"],
    )

    rows, cols, data = cluster_linking_partition(  # DD4
        rows, cols, data, n_entities, directed, exclude=exclude, threshold=threshold
    )
    # Construct the partitioned graph
    partitioned_graph = coo_matrix((data, (rows, cols)), shape=shape)

    if return_clusters:
        # Get an array of the graph with each index marked with the component label that it is connected to
        _, cc_labels = connected_components(  # CC2
            csgraph=partitioned_graph, directed=directed, return_labels=True
        )
        # Store clusters of indices marked with labels with at least 2 connected components
        unique_cc_labels, cc_sizes = np.unique(cc_labels, return_counts=True)  # CC3
        filtered_labels = unique_cc_labels[cc_sizes >= 2]
        clusters = defaultdict(list)
        for i, cc_label in enumerate(cc_labels):
            if cc_label in filtered_labels:
                clusters[cc_label].append(i)
        return partitioned_graph, clusters

    return partitioned_graph


def analyzeClusters(clusters, dictionary, queries, knn, n_train_mentions=0):
    """
    Evaluates the accuracy of entity-mention clustering

    Parameters
    ----------
    clusters : dict
        contains arrays of connected component indices of a graph
    dictionary : ndarray
        entity dictionary to evaluate
    queries : ndarray
        mention queries to evaluate
    knn : int
        the number of nearest-neighbour mention candidates considered

    Returns
    -------
    results : dict
        Contains n_entities, n_mentions, knn_mentions, accuracy, failure[], success[]
    """
    n_entities = len(dictionary)
    n_mentions = len(queries)

    results = {
        "n_entities": n_entities,
        "n_mentions": n_mentions,
        "knn_mentions": knn,
        "accuracy": 0,
        "failure": [],
        "success": [],
    }
    (
        _debug_n_mens_evaluated,
        _debug_clusters_wo_entities,
        _debug_clusters_w_mult_entities,
    ) = (0, 0, 0)

    print("Analyzing clusters...")
    for cluster in clusters.values():  # CC4 Scans all the values in clusters
        "Evaluate entity"
        # DD5 The lowest value in the cluster should always be the entity
        pred_entity_idx = cluster[0]
        # Track the graph index of the entity in the cluster
        pred_entity_idxs = [pred_entity_idx]
        if pred_entity_idx >= n_entities:
            # If the first element is a mention, then the cluster does not have an entity
            _debug_clusters_wo_entities += 1
            continue  # DD6 skip to the next cluster (for cluster in clusters.values)
        pred_entity = dictionary[pred_entity_idx]
        pred_entity_cuis = [
            str(pred_entity["cui"])
        ]  # Retrieve the cui of the entity indice
        _debug_tracked_mult_entities = False
        for i in range(1, len(cluster)):
            men_idx = cluster[i] - n_entities
            if men_idx < 0:
                # If elements after the first are entities, then the cluster has multiple entities
                if not _debug_tracked_mult_entities:
                    _debug_clusters_w_mult_entities += 1
                    _debug_tracked_mult_entities = True
                # Track the graph indices of each entity in the cluster
                pred_entity_idxs.append(cluster[i])
                # Predict based on all entities in the cluster
                pred_entity_cuis += list(
                    set([dictionary[cluster[i]]["cui"]]) - set(pred_entity_cuis)
                )
                continue
            men_idx -= n_train_mentions  # what's n_train_mentions ?
            if men_idx < 0:
                # Query is from train set
                continue
            " For each mentions in the cluster, we compare its mention_gold_cui to the predicted_cui"
            _debug_n_mens_evaluated += 1
            men_query = queries[men_idx]
            men_golden_cuis = list(
                map(str, men_query["label_cuis"])
            )  # One mention can can be associated with several CUIs in certain contexts
            report_obj = {
                "mention_id": men_query["mention_id"],
                "mention_name": men_query["mention_name"],
                "mention_gold_cui": "|".join(men_golden_cuis),  # CC5
                "mention_gold_cui_name": "|".join(
                    [
                        dictionary[i]["title"]
                        for i in men_query["label_idxs"][: men_query["n_labels"]]
                    ]
                ),
                "predicted_name": "|".join(
                    [d["title"] for d in [dictionary[i] for i in pred_entity_idxs]]
                ),
                "predicted_cui": "|".join(pred_entity_cuis),
            }
            # Correct prediction = if there is at least one common element between the predicted entity CUIs (pred_entity_cuis) and the gold standard entity CUIs (men_golden_cuis)
            if not set(pred_entity_cuis).isdisjoint(men_golden_cuis):
                results["accuracy"] += 1
                results["success"].append(report_obj)
            # Incorrect prediction
            else:
                results["failure"].append(report_obj)
    results["accuracy"] = (
        f"{results['accuracy'] / float(_debug_n_mens_evaluated) * 100} %"
    )
    print(f"Accuracy = {results['accuracy']}")
    # print('1st element of results["failure"] :', results["failure"][0])

    # Run sanity checks
    assert n_mentions == _debug_n_mens_evaluated
    assert _debug_clusters_wo_entities == 0
    assert _debug_clusters_w_mult_entities == 0

    return results


def filter_by_context_doc_id(mention_idxs, doc_id, doc_id_list, return_numpy=False):
    """
    Description
    -----------
    Filters and returns only mention indices that belong to a specific document identified by the doc_id.
    Ensures that the analysis are constrained within the context of that particular document.

    Parameters
    ----------
    - mention_idxs : ndarray(int) of dim = (number of mentions)
    Represents the indices of mentions
    - doc_id : int
    Indice of the target document
    - doc_id_list : ndarray(int) of dim = (number of mentions)
    Array of integers, where each element is a document ID associated with the corresponding mention in mention_idxs.
    The length of doc_id_list should match the total number of mentions referenced in mention_idxs.
    - return_numpy : bool
    A flag indicating whether to return the filtered list of mention indices as a NumPy array.
    If True, the function returns a NumPy array; otherwise, it returns a list
    -------
    Outputs:
    - mask : ndarray(bool) of dim = (number of mentions)
    Mask indicating where each mention's document ID (from "doc_id_list") matches the target "doc_id"
    - mention_idxs :
    Only contains mention indices that belong to the target document (=doc_id).
    """
    mask = [doc_id_list[i] == doc_id for i in mention_idxs]
    if isinstance(mention_idxs, list):
        mention_idxs = np.array(mention_idxs)
    mention_idxs = mention_idxs[mask]
    if not return_numpy:
        mention_idxs = list(mention_idxs)
    return mention_idxs, mask


def read_data(split, params, logger):
    """
    Description
    -----------
    Loads dataset samples from a specified path
    Optionally filters out samples without labels
    Checks if the dataset supports multiple labels per sample
    (has_mult_labels : bool)

    Parameters
    ----------
    split : str
        Indicates the portion of the dataset to load ("train", "test", "valid"), used by utils.read_dataset to determine which data to read.
    params : dict(str)
        Contains configuration options
    logger :
        An object used for logging messages about the process, such as the number of samples read.
    """
    samples = utils.read_dataset(split, params["data_path"])
    # Check if dataset has multiple ground-truth labels
    has_mult_labels = "labels" in samples[0].keys()
    if params["filter_unlabeled"]:
        # Filter samples without gold entities
        samples = list(
            filter(
                lambda sample: (
                    (len(sample["labels"]) > 0)
                    if has_mult_labels
                    else (sample["label"] is not None)
                ),
                samples,
            )
        )
    logger.info(f"Read {len(samples)} {split} samples.")
    return samples, has_mult_labels


def main(params):
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(params["output_path"], "log-eval")

    pickle_src_path = params["pickle_src_path"]
    if pickle_src_path is None or not os.path.exists(pickle_src_path):
        pickle_src_path = output_path

    embed_data_path = params["embed_data_path"]
    if embed_data_path is None or not os.path.exists(embed_data_path):
        embed_data_path = output_path

    # Init model
    reranker = BiEncoderRanker(params)
    reranker.model.eval()
    tokenizer = reranker.tokenizer
    n_gpu = reranker.n_gpu

    knn = params["knn"]
    use_types = params["use_types"]
    within_doc = params["within_doc"]
    data_split = params["data_split"]  # Default = "test"

    # Load test data
    test_samples = None
    entity_dictionary_loaded = False
    test_dictionary_pkl_path = os.path.join(pickle_src_path, "test_dictionary.pickle")
    test_tensor_data_pkl_path = os.path.join(pickle_src_path, "test_tensor_data.pickle")
    test_mention_data_pkl_path = os.path.join(
        pickle_src_path, "test_mention_data.pickle"
    )
    if params["transductive"]:
        train_tensor_data_pkl_path = os.path.join(
            pickle_src_path, "train_tensor_data.pickle"
        )
        train_mention_data_pkl_path = os.path.join(
            pickle_src_path, "train_mention_data.pickle"
        )
    if os.path.isfile(test_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(test_dictionary_pkl_path, "rb") as read_handle:
            test_dictionary = pickle.load(read_handle)
        entity_dictionary_loaded = True
    if os.path.isfile(test_tensor_data_pkl_path) and os.path.isfile(
        test_mention_data_pkl_path
    ):
        print("Loading stored processed test data...")
        with open(test_tensor_data_pkl_path, "rb") as read_handle:
            test_tensor_data = pickle.load(read_handle)
        with open(test_mention_data_pkl_path, "rb") as read_handle:
            mention_data = pickle.load(read_handle)
    else:
        test_samples = utils.read_dataset(data_split, params["data_path"])
        if not entity_dictionary_loaded:
            with open(
                os.path.join(params["data_path"], "dictionary.pickle"), "rb"
            ) as read_handle:
                test_dictionary = pickle.load(read_handle)

        # Check if dataset has multiple ground-truth labels
        mult_labels = "labels" in test_samples[0].keys()
        if params["filter_unlabeled"]:
            # Filter samples without gold entities
            test_samples = list(
                filter(
                    lambda sample: (
                        (len(sample["labels"]) > 0)
                        if mult_labels
                        else (sample["label"] is not None)
                    ),
                    test_samples,
                )
            )
        logger.info("Read %d test samples." % len(test_samples))

        mention_data, test_dictionary, test_tensor_data = (
            data_process.process_mention_data(
                test_samples,
                test_dictionary,
                tokenizer,
                params["max_context_length"],
                params["max_cand_length"],
                multi_label_key="labels" if mult_labels else None,
                context_key=params["context_key"],
                silent=params["silent"],
                logger=logger,
                debug=params["debug"],
                knn=knn,
                dictionary_processed=entity_dictionary_loaded,
            )
        )
        print("Saving processed test data...")
        if not entity_dictionary_loaded:
            with open(test_dictionary_pkl_path, "wb") as write_handle:
                pickle.dump(
                    test_dictionary, write_handle, protocol=pickle.HIGHEST_PROTOCOL
                )
            entity_dictionary_loaded = True
        with open(test_tensor_data_pkl_path, "wb") as write_handle:
            pickle.dump(
                test_tensor_data, write_handle, protocol=pickle.HIGHEST_PROTOCOL
            )
        with open(test_mention_data_pkl_path, "wb") as write_handle:
            pickle.dump(mention_data, write_handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Store test dictionary token ids
    test_dict_vecs = torch.tensor(
        list(map(lambda x: x["ids"], test_dictionary)), dtype=torch.long
    )
    # Store test mention token ids
    test_men_vecs = test_tensor_data[:][0]

    n_entities = len(test_dict_vecs)
    n_mentions = len(test_tensor_data)

    if within_doc:
        if test_samples is None:
            test_samples, _ = read_data(data_split, params, logger)
        test_context_doc_ids = [s["context_doc_id"] for s in test_samples]

    if params["transductive"]:
        if os.path.isfile(train_tensor_data_pkl_path) and os.path.isfile(
            train_mention_data_pkl_path
        ):
            print("Loading stored processed train data...")
            with open(train_tensor_data_pkl_path, "rb") as read_handle:
                train_tensor_data = pickle.load(read_handle)
            with open(train_mention_data_pkl_path, "rb") as read_handle:
                train_mention_data = pickle.load(read_handle)
        else:
            train_samples = utils.read_dataset("train", params["data_path"])

            # Check if dataset has multiple ground-truth labels
            mult_labels = "labels" in train_samples[0].keys()
            logger.info("Read %d test samples." % len(test_samples))

            train_mention_data, _, train_tensor_data = (
                data_process.process_mention_data(
                    train_samples,
                    test_dictionary,
                    tokenizer,
                    params["max_context_length"],
                    params["max_cand_length"],
                    multi_label_key="labels" if mult_labels else None,
                    context_key=params["context_key"],
                    silent=params["silent"],
                    logger=logger,
                    debug=params["debug"],
                    knn=knn,
                    dictionary_processed=entity_dictionary_loaded,
                )
            )
            print("Saving processed train data...")
            with open(train_tensor_data_pkl_path, "wb") as write_handle:
                pickle.dump(
                    train_tensor_data, write_handle, protocol=pickle.HIGHEST_PROTOCOL
                )
            with open(train_mention_data_pkl_path, "wb") as write_handle:
                pickle.dump(
                    train_mention_data, write_handle, protocol=pickle.HIGHEST_PROTOCOL
                )

        # Store train mention token ids
        train_men_vecs = train_tensor_data[:][0]
        n_mentions += len(train_tensor_data)
        n_train_mentions = len(train_tensor_data)

    # Values of k to run the evaluation against
    knn_vals = [0] + [2**i for i in range(int(math.log(knn, 2)) + 1)]
    # Store the maximum evaluation k
    max_knn = knn_vals[-1]

    time_start = time.time()

    # Check if graphs are already built
    graph_path = os.path.join(output_path, "graphs.pickle")
    if not params["only_recall"] and os.path.isfile(graph_path):
        print("Loading stored joint graphs...")
        with open(graph_path, "rb") as read_handle:
            joint_graphs = pickle.load(read_handle)
    else:
        # Initialize graphs to store mention-mention and mention-entity similarity score edges;
        # Keyed on k, the number of nearest mentions retrieved
        joint_graphs = {}
        for k in knn_vals:
            joint_graphs[k] = {
                "rows": np.array([]),
                "cols": np.array([]),
                "data": np.array([]),
                "shape": (n_entities + n_mentions, n_entities + n_mentions),
            }

        # Check and load stored embedding data
        embed_data_path = os.path.join(embed_data_path, "embed_data.t7")
        embed_data = None
        if os.path.isfile(embed_data_path):
            embed_data = torch.load(embed_data_path)

        if use_types:
            if embed_data is not None:
                logger.info("Loading stored embeddings and computing indexes")
                dict_embeds = embed_data["dict_embeds"]
                if "dict_idxs_by_type" in embed_data:
                    dict_idxs_by_type = embed_data["dict_idxs_by_type"]
                else:
                    dict_idxs_by_type = data_process.get_idxs_by_type(test_dictionary)
                dict_indexes = data_process.get_index_from_embeds(
                    dict_embeds,
                    dict_idxs_by_type,
                    force_exact_search=params["force_exact_search"],
                    probe_mult_factor=params["probe_mult_factor"],
                )
                men_embeds = embed_data["men_embeds"]
                if "men_idxs_by_type" in embed_data:
                    men_idxs_by_type = embed_data["men_idxs_by_type"]
                else:
                    men_idxs_by_type = data_process.get_idxs_by_type(mention_data)
                men_indexes = data_process.get_index_from_embeds(
                    men_embeds,
                    men_idxs_by_type,
                    force_exact_search=params["force_exact_search"],
                    probe_mult_factor=params["probe_mult_factor"],
                )
            else:
                logger.info("Dictionary: Embedding and building index")
                dict_embeds, dict_indexes, dict_idxs_by_type = (
                    data_process.embed_and_index(
                        reranker,
                        test_dict_vecs,
                        encoder_type="candidate",
                        n_gpu=n_gpu,
                        corpus=test_dictionary,
                        force_exact_search=params["force_exact_search"],
                        batch_size=params["embed_batch_size"],
                        probe_mult_factor=params["probe_mult_factor"],
                    )
                )
                logger.info("Queries: Embedding and building index")
                vecs = test_men_vecs
                men_data = mention_data
                if params["transductive"]:
                    vecs = torch.cat((train_men_vecs, vecs), dim=0)
                    men_data = train_mention_data + mention_data
                men_embeds, men_indexes, men_idxs_by_type = (
                    data_process.embed_and_index(
                        reranker,
                        vecs,
                        encoder_type="context",
                        n_gpu=n_gpu,
                        corpus=men_data,
                        force_exact_search=params["force_exact_search"],
                        batch_size=params["embed_batch_size"],
                        probe_mult_factor=params["probe_mult_factor"],
                    )
                )
        else:
            if embed_data is not None:
                logger.info("Loading stored embeddings and computing indexes")
                dict_embeds = embed_data["dict_embeds"]
                dict_index = data_process.get_index_from_embeds(
                    dict_embeds,
                    force_exact_search=params["force_exact_search"],
                    probe_mult_factor=params["probe_mult_factor"],
                )
                men_embeds = embed_data["men_embeds"]
                men_index = data_process.get_index_from_embeds(
                    men_embeds,
                    force_exact_search=params["force_exact_search"],
                    probe_mult_factor=params["probe_mult_factor"],
                )
            else:
                logger.info("Dictionary: Embedding and building index")
                dict_embeds, dict_index = data_process.embed_and_index(
                    reranker,
                    test_dict_vecs,
                    "candidate",
                    n_gpu=n_gpu,
                    force_exact_search=params["force_exact_search"],
                    batch_size=params["embed_batch_size"],
                    probe_mult_factor=params["probe_mult_factor"],
                )
                logger.info("Queries: Embedding and building index")
                vecs = test_men_vecs
                if params["transductive"]:
                    vecs = torch.cat((train_men_vecs, vecs), dim=0)
                men_embeds, men_index = data_process.embed_and_index(
                    reranker,
                    vecs,
                    "context",
                    n_gpu=n_gpu,
                    force_exact_search=params["force_exact_search"],
                    batch_size=params["embed_batch_size"],
                    probe_mult_factor=params["probe_mult_factor"],
                )

        # Save computed embedding data if not loaded from disk
        if embed_data is None:
            embed_data = {}
            embed_data["dict_embeds"] = dict_embeds
            embed_data["men_embeds"] = men_embeds
            if use_types:
                embed_data["dict_idxs_by_type"] = dict_idxs_by_type
                embed_data["men_idxs_by_type"] = men_idxs_by_type
            # NOTE: Cannot pickle faiss index because it is a SwigPyObject
            torch.save(
                embed_data, embed_data_path, pickle_protocol=pickle.HIGHEST_PROTOCOL
            )

        recall_accuracy = {
            2**i: 0 for i in range(int(math.log(params["recall_k"], 2)) + 1)
        }
        recall_idxs = [0.0] * params["recall_k"]

        logger.info("Starting KNN search...")
        # Fetch recall_k (default 16) knn entities for all mentions
        # Fetch (k+1) NN mention candidates
        if not use_types:
            _men_embeds = men_embeds
            if params["transductive"]:
                _men_embeds = _men_embeds[n_train_mentions:]
            nn_ent_dists, nn_ent_idxs = dict_index.search(
                _men_embeds, params["recall_k"]
            )
            n_mens_to_fetch = len(_men_embeds) if within_doc else max_knn + 1
            nn_men_dists, nn_men_idxs = men_index.search(_men_embeds, n_mens_to_fetch)
        else:
            query_len = len(men_embeds) - (
                n_train_mentions if params["transductive"] else 0
            )
            nn_ent_idxs = np.zeros((query_len, params["recall_k"]))
            nn_ent_dists = np.zeros((query_len, params["recall_k"]), dtype="float64")
            nn_men_idxs = -1 * np.ones((query_len, query_len), dtype=int)
            nn_men_dists = -1 * np.ones((query_len, query_len), dtype="float64")
            for entity_type in men_indexes:
                men_embeds_by_type = (
                    men_embeds[
                        men_idxs_by_type[entity_type][
                            men_idxs_by_type[entity_type] >= n_train_mentions
                        ]
                    ]
                    if params["transductive"]
                    else men_embeds[men_idxs_by_type[entity_type]]
                )
                nn_ent_dists_by_type, nn_ent_idxs_by_type = dict_indexes[
                    entity_type
                ].search(men_embeds_by_type, params["recall_k"])
                nn_ent_idxs_by_type = np.array(
                    list(
                        map(
                            lambda x: dict_idxs_by_type[entity_type][x],
                            nn_ent_idxs_by_type,
                        )
                    )
                )
                n_mens_to_fetch = len(men_embeds_by_type) if within_doc else max_knn + 1
                nn_men_dists_by_type, nn_men_idxs_by_type = men_indexes[
                    entity_type
                ].search(
                    men_embeds_by_type, min(n_mens_to_fetch, len(men_embeds_by_type))
                )
                nn_men_idxs_by_type = np.array(
                    list(
                        map(
                            lambda x: men_idxs_by_type[entity_type][x],
                            nn_men_idxs_by_type,
                        )
                    )
                )
                i = -1
                for idx in men_idxs_by_type[entity_type]:
                    if params["transductive"]:
                        idx -= n_train_mentions
                    if idx < 0:
                        continue
                    i += 1
                    nn_ent_idxs[idx] = nn_ent_idxs_by_type[i]
                    nn_ent_dists[idx] = nn_ent_dists_by_type[i]
                    nn_men_idxs[idx][: len(nn_men_idxs_by_type[i])] = (
                        nn_men_idxs_by_type[i]
                    )
                    nn_men_dists[idx][: len(nn_men_dists_by_type[i])] = (
                        nn_men_dists_by_type[i]
                    )
        logger.info("Search finished")

        logger.info("Building graphs")
        # Find the most similar entity and k-nn mentions for each mention query
        for idx in range(len(nn_ent_idxs)):
            # Get nearest entity candidate
            dict_cand_idx = nn_ent_idxs[idx][0]
            dict_cand_score = nn_ent_dists[idx][0]
            # Compute recall metric
            gold_idxs = mention_data[idx]["label_idxs"][: mention_data[idx]["n_labels"]]
            recall_idx = np.argwhere(nn_ent_idxs[idx] == gold_idxs[0])
            if len(recall_idx) != 0:
                recall_idx = int(recall_idx)
                recall_idxs[recall_idx] += 1.0
                for recall_k in recall_accuracy:
                    if recall_idx < recall_k:
                        recall_accuracy[recall_k] += 1.0
            if not params["only_recall"]:
                filter_mask_neg1 = nn_men_idxs[idx] != -1
                men_cand_idxs = nn_men_idxs[idx][filter_mask_neg1]
                men_cand_scores = nn_men_dists[idx][filter_mask_neg1]

                if within_doc:
                    men_cand_idxs, wd_mask = filter_by_context_doc_id(
                        men_cand_idxs,
                        test_context_doc_ids[idx],
                        test_context_doc_ids,
                        return_numpy=True,
                    )
                    men_cand_scores = men_cand_scores[wd_mask]

                # Filter candidates to remove mention query and keep only the top k candidates
                filter_mask = men_cand_idxs != idx
                men_cand_idxs, men_cand_scores = (
                    men_cand_idxs[filter_mask][:max_knn],
                    men_cand_scores[filter_mask][:max_knn],
                )

                if params["transductive"]:
                    idx += n_train_mentions
                # Add edges to the graphs
                for k in joint_graphs:
                    joint_graph = joint_graphs[k]
                    # Add mention-entity edge
                    joint_graph["rows"] = np.append(
                        joint_graph["rows"], [n_entities + idx]
                    )  # Mentions added at an offset of maximum entities
                    joint_graph["cols"] = np.append(joint_graph["cols"], dict_cand_idx)
                    joint_graph["data"] = np.append(
                        joint_graph["data"], dict_cand_score
                    )
                    if k > 0:
                        # Add mention-mention edges
                        joint_graph["rows"] = np.append(
                            joint_graph["rows"],
                            [n_entities + idx] * len(men_cand_idxs[:k]),
                        )
                        joint_graph["cols"] = np.append(
                            joint_graph["cols"], n_entities + men_cand_idxs[:k]
                        )
                        joint_graph["data"] = np.append(
                            joint_graph["data"], men_cand_scores[:k]
                        )

        if params["transductive"]:
            # Add positive infinity mention-entity edges from training queries to labeled entities
            for idx, train_men in enumerate(train_mention_data):
                dict_cand_idx = train_men["label_idxs"][0]
                for k in joint_graphs:
                    joint_graph = joint_graphs[k]
                    joint_graph["rows"] = np.append(
                        joint_graph["rows"], [n_entities + idx]
                    )  # Mentions added at an offset of maximum entities
                    joint_graph["cols"] = np.append(joint_graph["cols"], dict_cand_idx)
                    joint_graph["data"] = np.append(joint_graph["data"], float("inf"))

        # Compute and print recall metric
        recall_idx_mode = np.argmax(recall_idxs)
        recall_idx_mode_prop = recall_idxs[recall_idx_mode] / np.sum(recall_idxs)
        logger.info(
            f"""
        Recall metrics (for {len(nn_ent_idxs)} queries):
        ---------------"""
        )
        logger.info(
            f"highest recall idx = {recall_idx_mode} ({recall_idxs[recall_idx_mode]}/{np.sum(recall_idxs)} = {recall_idx_mode_prop})"
        )
        for recall_k in recall_accuracy:
            recall_accuracy[recall_k] /= len(nn_ent_idxs)
            logger.info(f"recall@{recall_k} = {recall_accuracy[recall_k]}")

        if params["only_recall"]:
            exit()

        # Pickle the graphs
        print("Saving joint graphs...")
        with open(graph_path, "wb") as write_handle:
            pickle.dump(joint_graphs, write_handle, protocol=pickle.HIGHEST_PROTOCOL)

        if params["only_embed_and_build"]:
            logger.info(f"Saved embedding data at: {embed_data_path}")
            logger.info(f"Saved graphs at: {graph_path}")
            exit()

    graph_mode = params.get("graph_mode", None)

    result_overview = {
        "n_entities": n_entities,
        "n_mentions": n_mentions - (n_train_mentions if params["transductive"] else 0),
    }
    results = {}
    if graph_mode is None or graph_mode not in ["directed", "undirected"]:
        results["directed"] = []
        results["undirected"] = []
    else:
        results[graph_mode] = []

    knn_fetch_time = time.time() - time_start
    graph_processing_time = time.time()
    n_graphs_processed = 0.0

    for mode in results:
        print(f"\nEvaluation mode: {mode.upper()}")
        for k in joint_graphs:
            if k <= knn:
                print(f"\nGraph (k={k}):")
                # Partition graph based on cluster-linking constraints
                partitioned_graph, clusters = partition_graph(
                    joint_graphs[k],
                    n_entities,
                    mode == "directed",
                    return_clusters=True,
                )
                # Infer predictions from clusters
                result = analyzeClusters(
                    clusters,
                    test_dictionary,
                    mention_data,
                    k,
                    n_train_mentions if params["transductive"] else 0,
                )
                # Store result
                results[mode].append(result)
                n_graphs_processed += 1

    avg_graph_processing_time = (
        time.time() - graph_processing_time
    ) / n_graphs_processed
    avg_per_graph_time = (knn_fetch_time + avg_graph_processing_time) / 60

    execution_time = (time.time() - time_start) / 60
    # Store results
    output_file_name = os.path.join(
        output_path,
        f"eval_results_{__import__('calendar').timegm(__import__('time').gmtime())}",
    )

    try:
        for recall_k in recall_accuracy:
            result_overview[f"recall@{recall_k}"] = recall_accuracy[recall_k]
    except:
        logger.info("Recall data not available since graphs were loaded from disk")

    for mode in results:
        mode_results = results[mode]
        result_overview[mode] = {}
        for r in mode_results:
            k = r["knn_mentions"]
            result_overview[mode][f"accuracy@knn{k}"] = r["accuracy"]
            logger.info(f"{mode} accuracy@knn{k} = {r['accuracy']}")
            output_file = f"{output_file_name}-{mode}-{k}.json"
            with open(output_file, "w") as f:
                json.dump(r, f, indent=2)
                print(f"\nPredictions ({mode}) @knn{k} saved at: {output_file}")
    with open(f"{output_file_name}.json", "w") as f:
        json.dump(result_overview, f, indent=2)
        print(f"\nPredictions overview saved at: {output_file_name}.json")

    logger.info(
        "\nThe avg. per graph evaluation time is {} minutes\n".format(
            avg_per_graph_time
        )
    )
    logger.info("\nThe total evaluation took {} minutes\n".format(execution_time))


if __name__ == "__main__":
    # parser = BlinkParser(add_model_args=True)
    # parser.add_training_args()
    # args = parser.parse_args()
    # print(args)
    # main(args.__dict__)

    ontology = "medic"
    model = "arboel"
    dataset = "ncbi_disease"
    abs_path = "/home2/cye73/data"
    data_path = os.path.join(abs_path, model, dataset)
    print(data_path)
    abs_path2 = "/home2/cye73/results"
    model_output_path = os.path.join(abs_path2, model, dataset)
    ontology_type = "medic"
    ontology_dir = "/mitchell/entity-linking/kbs/medic.tsv"

    best_model = "epoch_2/pytorch_model.bin"
    path_to_biencoder_model = os.path.join(abs_path2, model, dataset, best_model)

    params_test = {
        "model_output_path": model_output_path,
        "output_path": model_output_path,
        "pickle_src_path": data_path,
        "data_path": data_path,
        "knn": 10,
        "use_types": False,
        "max_context_length": 64,
        "max_cand_length": 64,
        "context_key": "context",  # to specify context_left or context_right
        "debug": True,
        "gold_arbo_knn": None,
        "within_doc": False,
        "within_doc_skip_strategy": False,
        "batch_size": 64,  # batch_size = train_batch_size
        "train_batch_size": 64,
        "filter_unlabeled": False,
        "type_optimization": "all_encoder_layers",
        # 'additional_layers', 'top_layer', 'top4_layers', 'all_encoder_layers', 'all'
        "learning_rate": 3e-5,
        "warmup_proportion": 0.1,
        "fp16": False,
        "embed_batch_size": 3500,
        "force_exact_search": True,
        "probe_mult_factor": 1,
        "pos_neg_loss": True,
        "use_types_for_eval": False,
        "drop_entities": False,
        "drop_set": False,
        "farthest_neighbor": True,
        "rand_gold_arbo": True,
        "bert_model": "michiyasunaga/BioLinkBERT-base",  # "bert-base-uncased",
        "out_dim": 768,
        "pull_from_layer": 11,  # 11 for base and 23 for large
        "add_linear": True,
        "max_grad_norm": 0,
        "gradient_accumulation_steps": 8,
        "no_cuda": False,
        "lowercase": True,
        "seed": 37,
        "only_evaluate": False,
        "shuffle": True,
        "data_parallel": True,
        "silent": False,
        "debug": False,
        "max_grad_norm": 1,
        "use_types_for_eval": True,
        "eval_interval": 30,
        "print_interval": 30,
        "path_to_model": None,
        "path_to_biencoder_model": None,
        "num_train_epochs": 5,
        "path_to_biencoder_model": path_to_biencoder_model,
        "path_to_model": None,
    }

    main(params_test)

# if __name__ == "__main__":
#     parser = BlinkParser(add_model_args=True)
#     parser.add_eval_args()
#     args = parser.parse_args()
#     print(args)
#     main(args.__dict__)
