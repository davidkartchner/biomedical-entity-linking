# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import random
import math
import time
import torch
import numpy as np
from tqdm import tqdm
import pickle
import copy

import bioel.models.arboel.biencoder.data.data_process as data_process
from bioel.models.arboel.biencoder.model.biencoder import BiEncoderRanker
from bioel.models.arboel.crossencoder.model.crossencoder import CrossEncoderRanker
from bioel.models.arboel.biencoder.model.common.params import BlinkParser
from bioel.models.arboel.biencoder.data.BiEncoderLightningDataModule import (
    read_dataset,
)

from IPython import embed

from bioel.logger import setup_logger


def load_data(
    data_split,
    bi_tokenizer,
    knn,
    params,
    logger,
    return_dict_only=False,
):
    entity_dictionary_loaded = False
    entity_dictionary_pkl_path = os.path.join(
        params["data_path"], "entity_dictionary.pickle"
    )
    if os.path.isfile(entity_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(entity_dictionary_pkl_path, "rb") as read_handle:
            entity_dictionary = pickle.load(read_handle)
        entity_dictionary_loaded = True

    if return_dict_only and entity_dictionary_loaded:
        return entity_dictionary

    # Load data
    tensor_data_pkl_path = os.path.join(
        params["data_path"], f"{data_split}_tensor_data.pickle"
    )
    processed_data_pkl_path = os.path.join(
        params["data_path"], f"{data_split}_processed_data.pickle"
    )
    if os.path.isfile(tensor_data_pkl_path) and os.path.isfile(processed_data_pkl_path):
        print("Loading stored processed data...")
        with open(tensor_data_pkl_path, "rb") as read_handle:
            tensor_data = pickle.load(read_handle)
        with open(processed_data_pkl_path, "rb") as read_handle:
            processed_data = pickle.load(read_handle)
    else:
        data_samples = read_dataset(data_split, params["data_path"])
        if not entity_dictionary_loaded:
            with open(
                os.path.join(params["data_path"], "dictionary.pickle"), "rb"
            ) as read_handle:
                entity_dictionary = pickle.load(read_handle)

        # Check if dataset has multiple ground-truth labels
        mult_labels = "labels" in data_samples[0].keys()
        # Filter samples without gold entities
        data_samples = list(
            filter(
                lambda sample: (
                    (len(sample["labels"]) > 0)
                    if mult_labels
                    else (sample["label"] is not None)
                ),
                data_samples,
            )
        )
        logger.info("Read %d data samples." % len(data_samples))

        processed_data, entity_dictionary, tensor_data = (
            data_process.process_mention_data(
                data_samples,
                entity_dictionary,
                bi_tokenizer,
                max_context_length=params["max_context_length"],
                max_cand_length=params["max_cand_length"],
                context_key=params["context_key"],
                multi_label_key="labels" if mult_labels else None,
                silent=params["silent"],
                logger=logger,
                debug=params["debug"],
                knn=knn,
                dictionary_processed=entity_dictionary_loaded,
            )
        )
        print("Saving processed data...")
        if not entity_dictionary_loaded:
            with open(entity_dictionary_pkl_path, "wb") as write_handle:
                pickle.dump(
                    entity_dictionary, write_handle, protocol=pickle.HIGHEST_PROTOCOL
                )
        with open(tensor_data_pkl_path, "wb") as write_handle:
            pickle.dump(tensor_data, write_handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(processed_data_pkl_path, "wb") as write_handle:
            pickle.dump(processed_data, write_handle, protocol=pickle.HIGHEST_PROTOCOL)

    if return_dict_only:
        return entity_dictionary
    return entity_dictionary, tensor_data, processed_data


def save_topk_biencoder_cands(
    bi_reranker,
    logger,
    params,
    bi_tokenizer,
    topk=64,
):
    entity_dictionary = load_data(
        data_split="train",
        bi_tokenizer=bi_tokenizer,
        knn=1,
        params=params,
        logger=logger,
        return_dict_only=True,
    )
    entity_dict_vecs = torch.tensor(
        list(map(lambda x: x["ids"], entity_dictionary)), dtype=torch.long
    )

    logger.info("Biencoder: Embedding and indexing entity dictionary")
    if params["use_types"]:
        _, dict_indexes, dict_idxs_by_type = data_process.embed_and_index(
            bi_reranker,
            entity_dict_vecs,
            encoder_type="candidate",
            corpus=entity_dictionary,
            force_exact_search=True,
            batch_size=params["embed_batch_size"],
        )
    else:
        _, dict_index = data_process.embed_and_index(
            bi_reranker,
            entity_dict_vecs,
            encoder_type="candidate",
            force_exact_search=True,
            batch_size=params["embed_batch_size"],
        )
    logger.info("Biencoder: Embedding and indexing finished")

    print("NCCL_TIMEOUT:", os.environ.get("NCCL_TIMEOUT"))

    for mode in ["train", "valid", "test"]:
        logger.info(
            f"Biencoder: Fetching top-{topk} biencoder candidates for {mode} set"
        )
        _, tensor_data, processed_data = load_data(
            data_split=mode,
            bi_tokenizer=bi_tokenizer,
            knn=1,
            params=params,
            logger=logger,
        )
        men_vecs = tensor_data[:][0]

        logger.info("Biencoder: Embedding mention data")
        if params["use_types"]:
            men_embeddings, _, men_idxs_by_type = data_process.embed_and_index(
                bi_reranker,
                men_vecs,
                encoder_type="context",
                corpus=processed_data,
                force_exact_search=True,
                batch_size=params["embed_batch_size"],
            )
        else:
            men_embeddings = data_process.embed_and_index(
                bi_reranker,
                men_vecs,
                encoder_type="context",
                force_exact_search=True,
                batch_size=params["embed_batch_size"],
                only_embed=True,
            )
        logger.info("Biencoder: Embedding finished")

        logger.info("Biencoder: Finding nearest entities for each mention...")
        if not params["use_types"]:
            _, bi_dict_nns = dict_index.search(men_embeddings, topk)
        else:
            bi_dict_nns = np.zeros((len(men_embeddings), topk), dtype=int)
            for entity_type in men_idxs_by_type:
                men_embeds_by_type = men_embeddings[men_idxs_by_type[entity_type]]
                _, dict_nns_by_type = dict_indexes[entity_type].search(
                    men_embeds_by_type, topk
                )
                dict_nns_idxs = np.array(
                    list(
                        map(
                            lambda x: dict_idxs_by_type[entity_type][x],
                            dict_nns_by_type,
                        )
                    )
                )
                for i, idx in enumerate(men_idxs_by_type[entity_type]):
                    bi_dict_nns[idx] = dict_nns_idxs[i]
        logger.info("Biencoder: Search finished")

        labels = [-1] * len(bi_dict_nns)
        for men_idx in range(len(bi_dict_nns)):
            gold_idx = processed_data[men_idx]["label_idxs"][0]
            for i in range(len(bi_dict_nns[men_idx])):
                if bi_dict_nns[men_idx][i] == gold_idx:
                    labels[men_idx] = i
                    break

        logger.info(f"Biencoder: Saving top-{topk} biencoder candidates for {mode} set")
        save_data_path = os.path.join(
            params["data_path"], f"candidates_{mode}_top{topk}.t7"
        )
        torch.save(
            {"mode": mode, "candidates": bi_dict_nns, "labels": labels}, save_data_path
        )
        logger.info("Biencoder: Saved")

    return {"mode": mode, "candidates": bi_dict_nns, "labels": labels}
