# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import random
import time
import pickle
import math
import json
import copy

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from pytorch_transformers.optimization import WarmupLinearSchedule
from tqdm import tqdm, trange

from special_partition.special_partition import cluster_linking_partition
import blink.biencoder.data_process_mult as data_process
import blink.biencoder.eval_cluster_linking as eval_cluster_linking
import blink.candidate_ranking.utils as utils
from blink.biencoder.biencoder import BiEncoderRanker
from blink.crossencoder.crossencoder import CrossEncoderRanker
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser

from IPython import embed

SCORING_BATCH_SIZE = 64

def concat_for_crossencoder(context_inputs, candidate_inputs, max_seq_length):
    """
    Parameters
    ----------
    context_inputs: N x D
    candidate_inputs: N x k x D
    max_seq_length: int

    Returns
    -------
    concat_input: N x k x 2D-1
    """
    new_input = []
    context_inputs = context_inputs.tolist()
    candidate_inputs = candidate_inputs.tolist()
    for i in range(len(context_inputs)):
        cur_input = context_inputs[i]
        cur_candidates = candidate_inputs[i]
        mod_input = []
        for j in range(len(cur_candidates)):
            # remove [CLS] token from candidate
            sample = cur_input + cur_candidates[j][1:]
            sample = sample[:max_seq_length]
            mod_input.append(sample)
        new_input.append(mod_input)
    return torch.LongTensor(new_input)


def score_in_batches(cross_reranker, max_context_length, cross_inputs, is_context_encoder, silent=False, scoring_batch_size=None):
    batch_size = scoring_batch_size if scoring_batch_size is not None else SCORING_BATCH_SIZE
    n_inputs, scores, reshaped = len(cross_inputs), None, False
    _iter = [cross_inputs]
    if n_inputs > 1:
        sampler = SequentialSampler(cross_inputs)
        dataloader = DataLoader(
            cross_inputs, sampler=sampler, batch_size=batch_size
        )
        _iter = dataloader if silent else tqdm(dataloader, desc=f"Scoring in batches (size={batch_size})")
    for batch in _iter:
        batch_shape = batch.size()
        if batch_shape[0] < cross_reranker.n_gpu:
            reshaped = True
            # Repeat to get length of 0th dimension >= n_gpu
            # NOTE: Needed to avoid CUDA error with data parallel
            repeat_shape = [math.ceil(cross_reranker.n_gpu / float(batch_shape[0])), *[1]*(len(batch_shape) - 1)]
            batch = batch.repeat(repeat_shape)[:cross_reranker.n_gpu]
        batch_scores = cross_reranker.score_candidate(batch.cuda(),
                                                      max_context_length,
                                                      is_context_encoder=is_context_encoder)
        if reshaped:
            reshaped = False
            batch_scores = batch_scores[:batch_shape[0]]
        scores = batch_scores if scores is None else torch.cat((scores, batch_scores), dim=0)
    return scores


def evaluate(cross_reranker,
             max_context_length,
             entity_dictionary,
             valid_processed_data,
             biencoder_valid_idxs,
             bi_valid_nn_count,
             valid_men_inputs,
             valid_ent_inputs,
             logger,
             max_k=8,
             k_biencoder=64):
    """
        - get 64 nearest bi-encoder mentions and entities
        - score using cross-encoder
        - create k-nn graphs
        - prune & predict
    """
    n_entities = len(entity_dictionary)
    n_mentions = len(valid_processed_data)

    bi_men_idxs = biencoder_valid_idxs['men_nns'][:, :k_biencoder]
    bi_ent_idxs = biencoder_valid_idxs['dict_nns'][:, :k_biencoder]

    joint_graphs = {}
    for k in ([0] + [2 ** i for i in range(int(math.log(max_k, 2)) + 1)]):
        joint_graphs[k] = {
            'rows': np.array([]),
            'cols': np.array([]),
            'data': np.array([]),
            'shape': (n_entities + n_mentions, n_entities + n_mentions)
        }
        max_k = k  # Stores the number of mention edges to add per query mention

    torch.cuda.empty_cache()
    cross_reranker.model.eval()
    with torch.no_grad():
        logger.info('Eval: Scoring mention-mention edges using cross-encoder...')
        cross_men_scores = score_in_batches(cross_reranker, max_context_length, valid_men_inputs,
                                            is_context_encoder=True)
        for i in range(len(cross_men_scores)):
            # Set scores for all invalid nearest neighbours to -infinity (due to variable NN counts of mentions)
            cross_men_scores[i][bi_valid_nn_count[i]:] = float('-inf')
        cross_men_topk_scores, cross_men_topk_idxs = torch.sort(cross_men_scores, dim=1, descending=True)
        cross_men_topk_idxs = cross_men_topk_idxs.cpu()[:, :max_k]
        cross_men_topk_scores = cross_men_topk_scores.cpu()[:, :max_k]
        logger.info('Eval: Scoring done')

        logger.info('Eval: Scoring mention-entity edges using cross-encoder...')
        cross_ent_scores = score_in_batches(cross_reranker, max_context_length, valid_ent_inputs,
                                            is_context_encoder=False)
        cross_ent_top1_score, cross_ent_top1_idx = torch.sort(cross_ent_scores, dim=1, descending=True)
        cross_ent_top1_idx = cross_ent_top1_idx.cpu()[:, 0]
        cross_ent_top1_score = cross_ent_top1_score.cpu()[:, 0]
        logger.info('Eval: Scoring done')

    bi_recall = 0.
    # TODO: Add cross-encoder recall metrics

    for men_idx in tqdm(range(len(valid_processed_data)), total=len(valid_processed_data), desc="Eval: Building graphs"):
        # Track biencoder recall@<k_biencoder>
        gold_idx = valid_processed_data[men_idx]["label_idxs"][0]
        if gold_idx in bi_ent_idxs[men_idx]:
            bi_recall += 1.
        # Get nearest entity
        m_e_idx = bi_ent_idxs[men_idx, cross_ent_top1_idx[men_idx]]
        m_e_score = cross_ent_top1_score[men_idx]
        if bi_valid_nn_count[men_idx] > 0:
            # Get nearest mentions
            topk_defined_nn_idxs = cross_men_topk_idxs[men_idx][:bi_valid_nn_count[men_idx]]
            m_m_idxs = bi_men_idxs[men_idx, topk_defined_nn_idxs] + n_entities  # Mentions added at an offset of maximum entities
            if len(topk_defined_nn_idxs) == 1:
                m_m_idxs = [m_m_idxs]
            m_m_scores = cross_men_topk_scores[men_idx][:bi_valid_nn_count[men_idx]]
        # Add edges to the graphs
        for k in joint_graphs:
            # Add mention-entity edge
            joint_graphs[k]['rows'] = np.append(
                joint_graphs[k]['rows'], [n_entities + men_idx])  # Mentions added at an offset of maximum entities
            joint_graphs[k]['cols'] = np.append(
                joint_graphs[k]['cols'], m_e_idx)
            joint_graphs[k]['data'] = np.append(
                joint_graphs[k]['data'], m_e_score)
            if k > 0 and bi_valid_nn_count[men_idx] > 0:
                # Add mention-mention edges
                joint_graphs[k]['rows'] = np.append(
                    joint_graphs[k]['rows'], [n_entities + men_idx] * len(m_m_idxs[:k]))
                joint_graphs[k]['cols'] = np.append(
                    joint_graphs[k]['cols'], m_m_idxs[:k])
                joint_graphs[k]['data'] = np.append(
                    joint_graphs[k]['data'], m_m_scores[:k])

    # Partition graphs and analyze clusters for final predictions
    max_eval_acc = {
        'directed': -1.,
        'directed_k': 0,
        'undirected': -1.,
        'undirected_k': 0
    }
    for k in joint_graphs:
        for mode in ['directed', 'undirected']:
            if k == 0 and mode == 'undirected':
                continue  # Since @k=0, both modes are equivalent
            logger.info(f"\nEval: Graph (k={k}, mode={mode}):")
            # Partition graph based on cluster-linking constraints
            partitioned_graph, clusters = eval_cluster_linking.partition_graph(
                joint_graphs[k], n_entities, directed=(mode == 'directed'), return_clusters=True)
            # Infer predictions from clusters
            result = eval_cluster_linking.analyzeClusters(clusters, entity_dictionary, valid_processed_data, k)
            acc = float(result['accuracy'].split(' ')[0])
            if acc > max_eval_acc[mode]:
                max_eval_acc[mode] = acc
                max_eval_acc[f'{mode}_k'] = k
                if k == 0 and mode == 'directed':
                    max_eval_acc['undirected'] = acc
            logger.info(f"Eval: accuracy for graph@(k={k}, mode={mode}): {acc}%")
    # Compute biencoder recall
    bi_recall /= len(valid_processed_data)
    logger.info(f"Eval: Biencoder recall@{k_biencoder} = {bi_recall * 100}%")
    logger.info(f"Eval: Best cross-encoder accuracy = {json.dumps(max_eval_acc)}")
    return max_eval_acc


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )

def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def load_optimizer_scheduler(params, logger):
    optim_sched = None
    model_path = params["path_to_model"]
    if model_path is not None:
        model_dir = os.path.dirname(model_path)
        optim_sched_fpath = os.path.join(model_dir, utils.OPTIM_SCHED_FNAME)
        if os.path.isfile(optim_sched_fpath):
            logger.info(f'Loading stored optimizer and scheduler from {optim_sched_fpath}')
            optim_sched = torch.load(optim_sched_fpath)
    return optim_sched


def get_context_doc_ids(split, params):
    samples = utils.read_dataset(split, params["data_path"])
    # Filter unlabeled data
    has_mult_labels = "labels" in samples[0].keys()
    samples = list(
        filter(lambda sample: (len(sample["labels"]) > 0) if has_mult_labels else (sample["label"] is not None),
               samples))
    context_doc_ids = [s['context_doc_id'] for s in samples]
    return context_doc_ids


def filter_by_context_doc_id(mention_idxs, doc_id, doc_id_list, return_numpy=False):
    mask = [doc_id_list[i] == doc_id for i in mention_idxs]
    if isinstance(mention_idxs, list):
        mention_idxs = np.array(mention_idxs)
    mention_idxs = mention_idxs[mask]
    if not return_numpy:
        mention_idxs = list(mention_idxs)
    return mention_idxs, mask


def get_biencoder_nns(bi_reranker, biencoder_indices_path, entity_dictionary, entity_dict_vecs, train_men_vecs,
                      train_processed_data, train_gold_clusters, valid_men_vecs, valid_processed_data,
                      use_types, logger, n_gpu, params, train_context_doc_ids=None, valid_context_doc_ids=None):
    """
    Train:
        Store sorted entities and mention idxs to disk
            <k_dict_nns> nearest non-gold (negative) entities
            <k_men_nns> nearest negative mentions
            min(<k_men_nns>, len(gold_cluster)-1) nearest gold mentions
    Dev:
        Store sorted entities and mention idxs to disk
            <k_dict_nns> nearest entities
            <k_men_nns> nearest mentions
    """
    dict_embeddings = None
    biencoder_train_idxs, biencoder_valid_idxs = None, None
    # Number of nearest neighbors to fetch and persist
    k_dict_nns, k_men_nns = 128, 128
    within_doc = params["within_doc"]

    if not params["only_evaluate"]:
        # Train set
        biencoder_train_idxs_pkl_path = os.path.join(biencoder_indices_path, 'biencoder_train_idxs.pickle')
        if os.path.isfile(biencoder_train_idxs_pkl_path):
            logger.info("Loading stored sorted biencoder train indices...")
            with open(biencoder_train_idxs_pkl_path, 'rb') as read_handle:
                biencoder_train_idxs = pickle.load(read_handle)
        else:
            logger.info('Biencoder: Embedding and indexing training data')
            if use_types:
                dict_embeddings, dict_indexes, dict_idxs_by_type = data_process.embed_and_index(
                    bi_reranker, entity_dict_vecs, encoder_type="candidate", n_gpu=n_gpu, corpus=entity_dictionary,
                    force_exact_search=True, batch_size=params['embed_batch_size'])
                train_men_embeddings, train_men_indexes, men_idxs_by_type = data_process.embed_and_index(
                    bi_reranker, train_men_vecs, encoder_type="context", n_gpu=n_gpu, corpus=train_processed_data,
                    force_exact_search=True, batch_size=params['embed_batch_size'])
            else:
                dict_embeddings, dict_index = data_process.embed_and_index(bi_reranker, entity_dict_vecs,
                    encoder_type="candidate", n_gpu=n_gpu, force_exact_search=True, batch_size=params['embed_batch_size'])
                train_men_embeddings, train_men_index = data_process.embed_and_index(bi_reranker, train_men_vecs,
                    encoder_type="context", n_gpu=n_gpu, force_exact_search=True, batch_size=params['embed_batch_size'])
            logger.info('Biencoder: Embedding and indexing finished')

            logger.info("Biencoder: Finding nearest mentions and entities for each mention...")
            bi_dict_nns = np.zeros((len(train_men_embeddings), k_dict_nns), dtype=int)  # negatives
            bi_men_nns = -1 * np.ones((len(train_men_embeddings), k_men_nns), dtype=int)  # negatives
            bi_men_gold_nns = -1 * np.ones((len(train_men_embeddings), k_men_nns), dtype=int)  # positives
            if not use_types:
                _, bi_dict_nns_np = dict_index.search(train_men_embeddings, k_dict_nns+1)
                _, bi_men_nns_np = train_men_index.search(train_men_embeddings, len(train_men_embeddings))
                for i in range(len(bi_dict_nns_np)):
                    gold_idx = train_processed_data[i]['label_idxs'][0]
                    bi_dict_nns[i] = bi_dict_nns_np[i][bi_dict_nns_np[i] != gold_idx][:k_dict_nns]
                    men_neg_mask = ~np.isin(bi_men_nns_np[i], train_gold_clusters[gold_idx])
                    neg_nns = bi_men_nns_np[i][men_neg_mask]
                    if within_doc:
                        neg_nns, _ = filter_by_context_doc_id(neg_nns, train_context_doc_ids[i], train_context_doc_ids,
                                                           return_numpy=True)
                    bi_men_nns[i][:len(neg_nns[:k_men_nns])] = neg_nns[:k_men_nns]
                    men_gold_mask = (np.isin(bi_men_nns_np[i], train_gold_clusters[gold_idx])) & (bi_men_nns_np[i] != i)
                    gold_nns = bi_men_nns_np[i][men_gold_mask]
                    if within_doc:
                        gold_nns, _ = filter_by_context_doc_id(gold_nns, train_context_doc_ids[i], train_context_doc_ids,
                                                            return_numpy=True)
                    # If len(gold_nns) < k_men_nns then set bi_men_gold_nns[i] = [...gold_nns, -1, -1, ...]
                    bi_men_gold_nns[i][:len(gold_nns[:k_men_nns])] = gold_nns[:k_men_nns]
            else:
                for entity_type in train_men_indexes:
                    men_embeds_by_type = train_men_embeddings[men_idxs_by_type[entity_type]]
                    _, dict_nns_by_type = dict_indexes[entity_type].search(men_embeds_by_type, k_dict_nns+1)
                    _, men_nns_by_type = train_men_indexes[entity_type].search(men_embeds_by_type, len(men_embeds_by_type))
                    dict_nns_idxs = np.array(list(map(lambda x: dict_idxs_by_type[entity_type][x], dict_nns_by_type)))
                    men_nns_idxs = np.array(list(map(lambda x: men_idxs_by_type[entity_type][x], men_nns_by_type)))
                    for i, idx in enumerate(men_idxs_by_type[entity_type]):
                        gold_idx = train_processed_data[idx]['label_idxs'][0]
                        bi_dict_nns[idx] = dict_nns_idxs[i][dict_nns_idxs[i] != gold_idx][:k_dict_nns]
                        men_neg_mask = ~np.isin(men_nns_idxs[i], train_gold_clusters[gold_idx])
                        neg_nns = men_nns_idxs[i][men_neg_mask]
                        if within_doc:
                            neg_nns, _ = filter_by_context_doc_id(neg_nns, train_context_doc_ids[idx],
                                                               train_context_doc_ids,
                                                               return_numpy=True)
                        bi_men_nns[idx][:len(neg_nns[:k_men_nns])] = neg_nns[:k_men_nns]
                        men_gold_mask = (np.isin(men_nns_idxs[i], train_gold_clusters[gold_idx])) & (men_nns_idxs[i] != idx)
                        gold_nns = men_nns_idxs[i][men_gold_mask]
                        if within_doc:
                            gold_nns, _ = filter_by_context_doc_id(gold_nns, train_context_doc_ids[idx],
                                                                train_context_doc_ids,
                                                                return_numpy=True)
                        # If len(gold_nns) < k_men_nns then set bi_men_gold_nns[i] = [...gold_nns, -1, -1, ...]
                        bi_men_gold_nns[idx][:len(gold_nns[:k_men_nns])] = gold_nns[:k_men_nns]
            logger.info("Biencoder: Search finished")

            logger.info("Biencoder: Saving sorted biencoder train indices...")
            biencoder_train_idxs = {
                'dict_nns': bi_dict_nns,  # Nearest negative entity idxs
                'men_nns': bi_men_nns,  # Nearest negative mention idxs
                'men_gold_nns': bi_men_gold_nns  # Nearest gold mention idxs
            }
            with open(biencoder_train_idxs_pkl_path, 'wb') as write_handle:
                pickle.dump(biencoder_train_idxs, write_handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Biencoder: Saved")

    # Dev set
    biencoder_valid_idxs_pkl_path = os.path.join(biencoder_indices_path, 'biencoder_valid_idxs.pickle')
    if os.path.isfile(biencoder_valid_idxs_pkl_path):
        logger.info("Loading stored sorted biencoder dev indices...")
        with open(biencoder_valid_idxs_pkl_path, 'rb') as read_handle:
            biencoder_valid_idxs = pickle.load(read_handle)
    else:
        logger.info('Biencoder: Embedding and indexing valid data')
        if use_types:
            if dict_embeddings is None:
                dict_embeddings, dict_indexes, dict_idxs_by_type = data_process.embed_and_index(
                    bi_reranker, entity_dict_vecs, encoder_type="candidate", n_gpu=n_gpu, corpus=entity_dictionary,
                    force_exact_search=True, batch_size=params['embed_batch_size'])
            valid_men_embeddings, valid_men_indexes, valid_men_idxs_by_type = data_process.embed_and_index(
                bi_reranker, valid_men_vecs, encoder_type="context", n_gpu=n_gpu, corpus=valid_processed_data,
                force_exact_search=True, batch_size=params['embed_batch_size'])
        else:
            if dict_embeddings is None:
                dict_embeddings, dict_index = data_process.embed_and_index(bi_reranker, entity_dict_vecs,
                                                                           encoder_type="candidate", n_gpu=n_gpu,
                                                                           force_exact_search=True,
                                                                           batch_size=params['embed_batch_size'])
            valid_men_embeddings, valid_men_index = data_process.embed_and_index(bi_reranker, valid_men_vecs,
                encoder_type="context", n_gpu=n_gpu, force_exact_search=True, batch_size=params['embed_batch_size'])
        # valid_men_embeddings = valid_men_embeddings.numpy()
        logger.info('Biencoder: Embedding and indexing finished')

        logger.info("Biencoder: Finding nearest mentions and entities for each mention...")
        bi_dict_nns = np.zeros((len(valid_men_embeddings), k_dict_nns), dtype=int)
        bi_men_nns = -1 * np.ones((len(valid_men_embeddings), k_men_nns), dtype=int)
        if not use_types:
            _, bi_dict_nns_np = dict_index.search(valid_men_embeddings, k_dict_nns)
            n_mens_to_fetch = len(valid_men_embeddings) if within_doc else k_men_nns+1
            _, bi_men_nns_np = valid_men_index.search(valid_men_embeddings, n_mens_to_fetch)
            for i in range(len(bi_men_nns_np)):
                bi_dict_nns[i] = bi_dict_nns_np[i]
                men_nns = bi_men_nns_np[i][bi_men_nns_np[i] != i]
                if within_doc:
                    men_nns, _ = filter_by_context_doc_id(men_nns, valid_context_doc_ids[i], valid_context_doc_ids,
                                                       return_numpy=True)
                men_nns = men_nns[:k_men_nns]
                bi_men_nns[i][:len(men_nns)] = men_nns
        else:
            for entity_type in valid_men_indexes:
                men_embeds_by_type = valid_men_embeddings[valid_men_idxs_by_type[entity_type]]
                _, dict_nns_by_type = dict_indexes[entity_type].search(men_embeds_by_type, k_dict_nns)
                _, men_nns_by_type = valid_men_indexes[entity_type].search(men_embeds_by_type, len(men_embeds_by_type))
                dict_nns_idxs = np.array(list(map(lambda x: dict_idxs_by_type[entity_type][x], dict_nns_by_type)))
                men_nns_idxs = np.array(list(map(lambda x: valid_men_idxs_by_type[entity_type][x], men_nns_by_type)))
                for i, idx in enumerate(valid_men_idxs_by_type[entity_type]):
                    bi_dict_nns[idx] = dict_nns_idxs[i]
                    men_nns = men_nns_idxs[i][men_nns_idxs[i] != idx]
                    if within_doc:
                        men_nns, _ = filter_by_context_doc_id(men_nns, valid_context_doc_ids[idx],
                                                           valid_context_doc_ids,
                                                           return_numpy=True)
                    men_nns = men_nns[:k_men_nns]
                    bi_men_nns[idx][:len(men_nns)] = men_nns
        logger.info("Biencoder: Search finished")

        biencoder_valid_idxs = {
            'dict_nns': bi_dict_nns,  # Nearest entity idxs
            'men_nns': bi_men_nns  # Nearest mention idxs
        }

        logger.info("Biencoder: Saving sorted biencoder valid indices...")
        with open(biencoder_valid_idxs_pkl_path, 'wb') as write_handle:
            pickle.dump(biencoder_valid_idxs, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Biencoder: Saved")

    return biencoder_train_idxs, biencoder_valid_idxs


def drop_entities_for_discovery_training(entity_dictionary, drop_set_fname, logger, ent_drop_prop = 0.1):
    # Drop a proportion of unique entities seen in the dev/test set from the entity dictionary
    # in order to train a new model without information leakage
    # drop_set_fname: Expects either test_processed_data.pickle or valid_process_data.pickle

    if not os.path.isfile(drop_set_fname):
        raise ValueError("Invalid --drop_set path provided to dev/test mention data")
    with open(drop_set_fname, 'rb') as read_handle:
        drop_set_data = pickle.load(read_handle)
    drop_set_mention_gold_cui_idxs = list(map(lambda x: x['label_idxs'][0], drop_set_data))
    ents_in_data = np.unique(drop_set_mention_gold_cui_idxs)
    logger.info(f"Dropping {ent_drop_prop * 100}% of {len(ents_in_data)} entities found in drop set")
    # Get entity indices to drop
    n_ents_dropped = int(ent_drop_prop * len(ents_in_data))
    rng = np.random.default_rng(seed=17)
    dropped_ent_idxs = rng.choice(ents_in_data, size=n_ents_dropped, replace=False)
    # Drop entities from dictionary
    # (subsequent call to process_mention_data will automatically drop corresponding mentions)
    keep_mask = np.ones(len(entity_dictionary), dtype='bool')
    keep_mask[dropped_ent_idxs] = False
    entity_dictionary = np.array(entity_dictionary)[keep_mask]
    return entity_dictionary


def load_training_data(bi_tokenizer,
                       max_context_length,
                       max_cand_length,
                       knn,
                       pickle_src_path,
                       params,
                       logger,
                       return_dict_only=False):
    # Load training data
    entity_dictionary_loaded = False
    entity_dictionary_pkl_path = os.path.join(pickle_src_path, 'entity_dictionary.pickle')
    if os.path.isfile(entity_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(entity_dictionary_pkl_path, 'rb') as read_handle:
            entity_dictionary = pickle.load(read_handle)
        entity_dictionary_loaded = True

    if return_dict_only and entity_dictionary_loaded:
        return entity_dictionary

    # Load train data
    train_tensor_data_pkl_path = os.path.join(pickle_src_path, 'train_tensor_data.pickle')
    train_processed_data_pkl_path = os.path.join(pickle_src_path, 'train_processed_data.pickle')
    if not params["drop_entities"] and os.path.isfile(train_tensor_data_pkl_path) and os.path.isfile(train_processed_data_pkl_path):
        print("Loading stored processed train data...")
        with open(train_tensor_data_pkl_path, 'rb') as read_handle:
            train_tensor_data = pickle.load(read_handle)
        with open(train_processed_data_pkl_path, 'rb') as read_handle:
            train_processed_data = pickle.load(read_handle)
    else:
        train_samples = utils.read_dataset("train", params["data_path"])
        if not entity_dictionary_loaded:
            with open(os.path.join(params["data_path"], 'dictionary.pickle'), 'rb') as read_handle:
                entity_dictionary = pickle.load(read_handle)

        # Check if dataset has multiple ground-truth labels
        mult_labels = "labels" in train_samples[0].keys()
        # Filter samples without gold entities
        train_samples = list(
            filter(lambda sample: (len(sample["labels"]) > 0) if mult_labels else (sample["label"] is not None),
                   train_samples))
        logger.info("Read %d train samples." % len(train_samples))

        # For discovery experiment: Drop entities used in training that were dropped randomly from dev/test set
        if params["drop_entities"]:
            assert entity_dictionary is not None
            entity_dictionary = drop_entities_for_discovery_training(entity_dictionary,
                                                                     params["drop_set"],
                                                                     logger)

        train_processed_data, entity_dictionary, train_tensor_data = data_process.process_mention_data(
            train_samples,
            entity_dictionary,
            bi_tokenizer,
            max_context_length,
            max_cand_length,
            context_key=params["context_key"],
            multi_label_key="labels" if mult_labels else None,
            silent=params["silent"],
            logger=logger,
            debug=params["debug"],
            knn=knn,
            dictionary_processed=entity_dictionary_loaded
        )
        print("Saving processed train data...")
        if not entity_dictionary_loaded:
            with open(entity_dictionary_pkl_path, 'wb') as write_handle:
                pickle.dump(entity_dictionary, write_handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        with open(train_tensor_data_pkl_path, 'wb') as write_handle:
            pickle.dump(train_tensor_data, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(train_processed_data_pkl_path, 'wb') as write_handle:
            pickle.dump(train_processed_data, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    if return_dict_only:
        return entity_dictionary
    return entity_dictionary, train_tensor_data, train_processed_data


def load_validation_data(bi_tokenizer,
                         max_context_length,
                         max_cand_length,
                         entity_dictionary,
                         knn,
                         pickle_src_path,
                         params,
                         logger):
    valid_tensor_data_pkl_path = os.path.join(pickle_src_path, 'valid_tensor_data.pickle')
    valid_processed_data_pkl_path = os.path.join(pickle_src_path, 'valid_processed_data.pickle')
    if os.path.isfile(valid_tensor_data_pkl_path) and os.path.isfile(valid_processed_data_pkl_path):
        print("Loading stored processed valid data...")
        with open(valid_tensor_data_pkl_path, 'rb') as read_handle:
            valid_tensor_data = pickle.load(read_handle)
        with open(valid_processed_data_pkl_path, 'rb') as read_handle:
            valid_processed_data = pickle.load(read_handle)
    else:
        valid_samples = utils.read_dataset("valid", params["data_path"])
        # Check if dataset has multiple ground-truth labels
        mult_labels = "labels" in valid_samples[0].keys()
        # Filter samples without gold entities
        valid_samples = list(
            filter(lambda sample: (len(sample["labels"]) > 0) if mult_labels else (sample["label"] is not None),
                   valid_samples))
        logger.info("Read %d valid samples." % len(valid_samples))

        valid_processed_data, _, valid_tensor_data = data_process.process_mention_data(
            valid_samples,
            entity_dictionary,
            bi_tokenizer,
            max_context_length,
            max_cand_length,
            context_key=params["context_key"],
            multi_label_key="labels" if mult_labels else None,
            silent=params["silent"],
            logger=logger,
            debug=params["debug"],
            knn=knn,
            dictionary_processed=True
        )
        print("Saving processed valid data...")
        with open(valid_tensor_data_pkl_path, 'wb') as write_handle:
            pickle.dump(valid_tensor_data, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(valid_processed_data_pkl_path, 'wb') as write_handle:
            pickle.dump(valid_processed_data, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    return valid_tensor_data, valid_processed_data


def get_gold_clusters(train_processed_data):
    train_gold_clusters = data_process.compute_gold_clusters(train_processed_data)
    max_gold_cluster_len = 0
    for ent in train_gold_clusters:
        if len(train_gold_clusters[ent]) > max_gold_cluster_len:
            max_gold_cluster_len = len(train_gold_clusters[ent])
    return train_gold_clusters, max_gold_cluster_len


def get_gold_arbo_links(cross_reranker,
                        max_context_length,
                        entity_dict_vecs,
                        train_men_vecs,
                        train_processed_data,
                        gold_men_nns,
                        max_seq_length,
                        knn,
                        within_doc=False,
                        train_context_doc_ids=None,
                        debug=False):
    cross_reranker.model.eval()
    with torch.no_grad():
        gold_links = {}
        n_entities, n_mentions = len(entity_dict_vecs), len(train_men_vecs)
        _iter = tqdm(range(len(train_men_vecs) if not debug else 200), desc="Computing gold arbos")
        for mention_idx in _iter:
            # Assuming that there is only 1 gold label
            cluster_ent = train_processed_data[mention_idx]['label_idxs'][0]
            if mention_idx not in gold_links:
                # Get <knn> nearest (biencoder) gold mentions
                gold_nns = gold_men_nns[mention_idx][gold_men_nns[mention_idx] != -1]
                if within_doc:
                    gold_nns, _ = filter_by_context_doc_id(gold_nns,
                                                        train_context_doc_ids[mention_idx],
                                                        train_context_doc_ids,
                                                        return_numpy=True)
                cluster_mens = np.concatenate(([mention_idx], gold_nns[:knn]))
                # Simply link to the entity if cluster is singleton
                if len(cluster_mens) == 1:
                    gold_links[mention_idx] = cluster_ent
                    continue
                # Run MST on the current query mention's (biencoder) k-NN gold sub-cluster to find positive edges
                rows, cols, data, shape = [], [], [], (n_entities + n_mentions, n_entities + n_mentions)
                to_ent_input = concat_for_crossencoder(train_men_vecs[cluster_mens],
                                                       entity_dict_vecs[cluster_ent].expand(len(cluster_mens), 1,
                                                                                            entity_dict_vecs.size(1)),
                                                       max_seq_length)  # Shape: N x 1 x 2D
                try:
                    to_ent_data = score_in_batches(cross_reranker, max_context_length, to_ent_input,
                                                   is_context_encoder=False, silent=True)
                except:
                    raise ValueError(f"Probably a batch size error. The current cluster size was {len(cluster_mens)}")
                to_men_input = concat_for_crossencoder(train_men_vecs[cluster_mens],
                                                       train_men_vecs[cluster_mens].expand(len(cluster_mens),
                                                                                           len(cluster_mens),
                                                                                           train_men_vecs.size(1)),
                                                       max_seq_length)  # Shape: N x N x 2D
                try:
                    to_men_data = score_in_batches(cross_reranker, max_context_length, to_men_input,
                                                   is_context_encoder=True, silent=True)
                except:
                    raise ValueError(f"Probably a batch size error. The current cluster size was {len(cluster_mens)}")
                for i in range(len(cluster_mens)):
                    from_node = n_entities + cluster_mens[i]
                    to_node = cluster_ent
                    # Add mention-entity link
                    rows.append(from_node)
                    cols.append(to_node)
                    data.append(to_ent_data[i, 0].item())
                    # Add mention-mention links
                    for j in range(len(cluster_mens)):
                        # Skip diagonal elements from to_men_data because they are scores of elements with themselves
                        if i == j:
                            continue
                        to_node = n_entities + cluster_mens[j]
                        rows.append(from_node)
                        cols.append(to_node)
                        data.append(to_men_data[i, j].item())
                # Partition graph with entity constraint
                rows, cols, data = cluster_linking_partition(np.array(rows),
                                                             np.array(cols),
                                                             np.array(data),
                                                             n_entities,
                                                             directed=True,
                                                             silent=True)
                assert np.array_equal(np.sort(rows - n_entities), np.sort(cluster_mens))
                for i in range(len(rows)):
                    men_idx = rows[i] - n_entities
                    assert men_idx >= 0
                    if men_idx in gold_links:
                        continue
                    # Only add gold link for current mention's arborescence computation
                    if men_idx == mention_idx:
                        gold_links[men_idx] = cols[i]
                        break
    return gold_links


def construct_train_batch(cross_reranker,
                          mention_idxs,
                          gold_links,
                          entity_dict_vecs,
                          train_men_vecs,
                          neg_ent_topk_inputs,
                          neg_men_topk_inputs,
                          max_seq_length,
                          max_context_length):
    n_entities = len(entity_dict_vecs)
    batch_positive_scores = None
    # Construct the batch matrix
    for mention_idx in mention_idxs:
        mention_idx = mention_idx.item()
        # Forward-propagate the positive edge
        # Case: Gold edge is an entity
        if gold_links[mention_idx] < n_entities:
            is_context_encoder = False
            to_idx = gold_links[mention_idx]
            to_input = torch.unsqueeze(entity_dict_vecs[to_idx:to_idx + 1], dim=0)
        # Case: Gold edge is a mention
        else:
            is_context_encoder = True
            to_idx = gold_links[mention_idx] - n_entities
            to_input = torch.unsqueeze(train_men_vecs[to_idx:to_idx + 1], dim=0)
        pos_label_input = concat_for_crossencoder(train_men_vecs[mention_idx:mention_idx + 1],
                                                  to_input,
                                                  max_seq_length)  # Shape: 1 x 1 x 2D
        pos_score = score_in_batches(cross_reranker, max_context_length, pos_label_input.cuda(),
                                     is_context_encoder=is_context_encoder)
        batch_positive_scores = pos_score if batch_positive_scores is None else \
            torch.cat((batch_positive_scores, pos_score), dim=0)
    batch_negative_ent_inputs = neg_ent_topk_inputs[mention_idxs]
    batch_negative_men_inputs = neg_men_topk_inputs[mention_idxs]

    return batch_positive_scores, batch_negative_ent_inputs.cuda(), batch_negative_men_inputs.cuda()


def build_cross_concat_input(biencoder_idxs,
                             men_vecs,
                             entity_dict_vecs,
                             max_seq_length,
                             k_biencoder):
    men_men_inputs = []
    men_ent_inputs = []
    for mention_idx in tqdm(range(len(men_vecs))):
        # Get nearest biencoder mentions and entities
        bi_men_idxs = biencoder_idxs['men_nns'][mention_idx][:k_biencoder]
        bi_ent_idxs = biencoder_idxs['dict_nns'][mention_idx][:k_biencoder]
        bi_men_inputs = None
        for bi_men_idx in bi_men_idxs:
            bi_men_inputs = men_vecs[bi_men_idx].unsqueeze(dim=0) if bi_men_inputs is None else \
                torch.cat((bi_men_inputs, men_vecs[bi_men_idx].unsqueeze(dim=0)), dim=0)
        bi_men_inputs = torch.unsqueeze(bi_men_inputs, dim=0)
        bi_ent_inputs = None
        for bi_ent_idx in bi_ent_idxs:
            bi_ent_inputs = entity_dict_vecs[bi_ent_idx].unsqueeze(dim=0) if bi_ent_inputs is None else \
                torch.cat((bi_ent_inputs, entity_dict_vecs[bi_ent_idx].unsqueeze(dim=0)), dim=0)
        bi_ent_inputs = torch.unsqueeze(bi_ent_inputs, dim=0)
        # Concatenate for cross-encoder
        cross_men_inputs = concat_for_crossencoder(men_vecs[mention_idx:mention_idx + 1],
                                                   bi_men_inputs,
                                                   max_seq_length)  # Shape: 1 x knn x 2D
        cross_ent_inputs = concat_for_crossencoder(men_vecs[mention_idx:mention_idx + 1],
                                                   bi_ent_inputs,
                                                   max_seq_length)  # Shape: 1 x knn x 2D
        men_men_inputs.append(cross_men_inputs)
        men_ent_inputs.append(cross_ent_inputs)
    men_men_inputs = torch.cat(men_men_inputs)
    men_ent_inputs = torch.cat(men_ent_inputs)
    return men_men_inputs, men_ent_inputs


def get_train_neg_cross_inputs(cross_reranker,
                               max_context_length,
                               train_men_concat_inputs,
                               train_ent_concat_inputs,
                               n_knn_men_negs,
                               bi_train_nn_count,
                               n_knn_ent_negs,
                               logger,
                               debug=False):
    def _helper(concat_inputs, n_knn, is_context_encoder):
        if debug:
            concat_inputs = concat_inputs[:200]
        scores = score_in_batches(cross_reranker, max_context_length, concat_inputs,
                                  is_context_encoder=is_context_encoder)
        if is_context_encoder:
            for i in range(len(scores)):
                # Set scores for all invalid nearest neighbours to -infinity (due to variable NN counts of mentions)
                scores[i][bi_train_nn_count[i]:] = float('-inf')
        topk_idxs = torch.argsort(scores, dim=1, descending=True)[:, :n_knn]
        stacked = []
        for r in range(topk_idxs.size(0)):
            stacked.append(concat_inputs[r][topk_idxs[r]])
        topk_inputs = torch.stack(stacked)
        logger.info('Scoring done')
        return topk_inputs

    # Score biencoder negatives using cross-encoder and store nearest-k for current epoch
    cross_reranker.model.eval()
    with torch.no_grad():
        logger.info('Scoring mention-mention negative edges using cross-encoder...')
        neg_men_topk_inputs = _helper(concat_inputs=train_men_concat_inputs, n_knn=n_knn_men_negs,
                                      is_context_encoder=True)
        logger.info('Scoring mention-entity negative edges using cross-encoder...')
        neg_ent_topk_inputs = _helper(concat_inputs=train_ent_concat_inputs, n_knn=n_knn_ent_negs,
                                      is_context_encoder=False)
    return neg_men_topk_inputs, neg_ent_topk_inputs


def execute_training_step(mention_idxs,
                          bi_train_nn_count,
                          n_knn_men_negs,
                          batch_positive_scores,
                          batch_negative_men_inputs,
                          batch_negative_ent_inputs,
                          cross_reranker,
                          max_context_length,
                          grad_acc_steps,
                          n_gpu,
                          no_sigmoid=False):
    # Handle variable mention-negatives lengths for different training queries in the batch
    min_negs = float('inf')
    dual_negs_mask = np.array([True] * len(mention_idxs))  # Mask to keep queries with both ent and men negs
    skipped = 0
    for i, m_idx in enumerate(mention_idxs):
        if bi_train_nn_count[m_idx] == 0:  # If no negative mentions neighbours
            dual_negs_mask[i] = False
            skipped += 1
            continue
        min_negs = min(min_negs, bi_train_nn_count[m_idx], n_knn_men_negs)
    loss_dual_negs = loss_ent_negs = 0
    assert min_negs != float('inf') or skipped > 0
    if min_negs != float('inf'):  # i.e. at least 1 query has dual negs
        positive_scores = batch_positive_scores[dual_negs_mask]
        negative_men_inputs = batch_negative_men_inputs[dual_negs_mask][:, :min_negs]
        negative_ent_inputs = batch_negative_ent_inputs[dual_negs_mask]
        # FIX: for error scenario of less number of examples than number of GPUs while using Data Parallel
        data_parallel_batch_size_check = negative_men_inputs.shape[0] * negative_men_inputs.shape[1] >= n_gpu and \
                                         negative_ent_inputs.shape[0] * negative_ent_inputs.shape[1] >= n_gpu
        if data_parallel_batch_size_check:
            loss_dual_negs = cross_reranker(positive_scores, negative_men_inputs,
                                            negative_ent_inputs, max_context_length,
                                            no_sigmoid=no_sigmoid)
    if skipped > 0:
        positive_scores = batch_positive_scores[~dual_negs_mask]
        negative_men_inputs = None
        negative_ent_inputs = batch_negative_ent_inputs[~dual_negs_mask]
        data_parallel_batch_size_check = negative_ent_inputs.shape[0] * negative_ent_inputs.shape[1] >= n_gpu
        if data_parallel_batch_size_check:
            loss_ent_negs = cross_reranker(positive_scores, negative_men_inputs,
                                           negative_ent_inputs, max_context_length,
                                           no_sigmoid=no_sigmoid)
    loss = ((loss_dual_negs * (len(mention_idxs) - skipped) + loss_ent_negs * skipped) / len(mention_idxs)) / grad_acc_steps
    loss.backward()
    return loss.item()


def main(params):
    # Parameter initializations
    logger = utils.get_logger(params["output_path"])
    debug = params["debug"]
    global SCORING_BATCH_SIZE
    SCORING_BATCH_SIZE = params["scoring_batch_size"]
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    pickle_src_path = params["pickle_src_path"]
    if pickle_src_path is None or not os.path.exists(pickle_src_path):
        pickle_src_path = model_output_path
    biencoder_indices_path = params["biencoder_indices_path"]
    if biencoder_indices_path is None:
        biencoder_indices_path = model_output_path
    elif not os.path.exists(biencoder_indices_path):
        os.makedirs(biencoder_indices_path)
    knn = params["knn"]  # Number of (positive+negatives) in each row of the training batch
    use_types = params["use_types"]
    within_doc = params["within_doc"]

    # Bi-encoder model
    biencoder_params = copy.deepcopy(params)
    biencoder_params['add_linear'] = False
    bi_reranker = BiEncoderRanker(biencoder_params)
    bi_tokenizer = bi_reranker.tokenizer
    bi_knn = params["bi_knn"]  # Number of biencoder nearest-neighbors to fetch for cross-encoder scoring (default: 64)

    # Cross-encoder model
    cross_reranker = CrossEncoderRanker(params)
    cross_model = cross_reranker.model
    cross_tokenizer = cross_reranker.tokenizer
    device = cross_reranker.device
    n_gpu = cross_reranker.n_gpu
    no_sigmoid_train = params["no_sigmoid_train"]

    # Training params
    grad_acc_steps = params["gradient_accumulation_steps"]
    if grad_acc_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(grad_acc_steps))
    # Gradient accumulation:
    # Model's batch size `z` = `x / y`. Effective batch size of `x` when accumulating gradients across `y` batches
    params["train_batch_size"] = (
        params["train_batch_size"] // grad_acc_steps
    )
    train_batch_size = params["train_batch_size"]

    # Input lengths
    max_seq_length = params["max_seq_length"]
    max_context_length = params["max_context_length"]
    max_cand_length = params["max_cand_length"]

    # Fix random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cross_reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Load data
    if params["only_evaluate"]:
        entity_dictionary = load_training_data(bi_tokenizer,
                                               max_context_length,
                                               max_cand_length,
                                               knn,
                                               pickle_src_path,
                                               params,
                                               logger,
                                               return_dict_only=True)
    else:
        # Load training data
        entity_dictionary, train_tensor_data, train_processed_data = load_training_data(bi_tokenizer,
                                                                                        max_context_length,
                                                                                        max_cand_length,
                                                                                        knn,
                                                                                        pickle_src_path,
                                                                                        params,
                                                                                        logger)
        # Store query mention vectors
        train_men_vecs = train_tensor_data[:][0]
    # Store entity dictionary vectors
    entity_dict_vecs = torch.tensor(list(map(lambda x: x['ids'], entity_dictionary)), dtype=torch.long)

    # Load validation data
    valid_tensor_data, valid_processed_data = load_validation_data(bi_tokenizer,
                                                                   max_context_length,
                                                                   max_cand_length,
                                                                   entity_dictionary,
                                                                   knn,
                                                                   pickle_src_path,
                                                                   params,
                                                                   logger)
    # Store query mention vectors
    valid_men_vecs = valid_tensor_data[:][0]

    train_context_doc_ids = valid_context_doc_ids = None
    if within_doc:
        # Get context_document_ids for each mention in training and validation
        train_context_doc_ids = get_context_doc_ids("train", params)
        valid_context_doc_ids = get_context_doc_ids("valid", params)

    if params["only_evaluate"]:
        _, biencoder_valid_idxs = get_biencoder_nns(bi_reranker=bi_reranker,
                                                    biencoder_indices_path=biencoder_indices_path,
                                                    entity_dictionary=entity_dictionary,
                                                    entity_dict_vecs=entity_dict_vecs,
                                                    train_men_vecs=None,
                                                    train_processed_data=None,
                                                    train_gold_clusters=None,
                                                    valid_men_vecs=valid_men_vecs,
                                                    valid_processed_data=valid_processed_data,
                                                    use_types=use_types,
                                                    logger=logger,
                                                    n_gpu=n_gpu,
                                                    params=params,
                                                    train_context_doc_ids=train_context_doc_ids,
                                                    valid_context_doc_ids=valid_context_doc_ids)
        bi_valid_nn_count = np.sum(biencoder_valid_idxs['men_nns'] != -1, axis=1)
        # Compute and store the concatenated cross-encoder inputs for validation
        valid_men_concat_inputs, valid_ent_concat_inputs = build_cross_concat_input(biencoder_valid_idxs,
                                                                                    valid_men_vecs,
                                                                                    entity_dict_vecs,
                                                                                    max_seq_length,
                                                                                    bi_knn)
        evaluate(cross_reranker,
                 max_context_length,
                 entity_dictionary,
                 valid_processed_data,
                 biencoder_valid_idxs,
                 bi_valid_nn_count,
                 valid_men_concat_inputs,
                 valid_ent_concat_inputs,
                 logger,
                 max_k=8,
                 k_biencoder=64)
        exit()

    # Get clusters of gold mentions that map to a unique entity
    train_gold_clusters, max_gold_cluster_len = get_gold_clusters(train_processed_data)

    # Get indices of nearest mentions and entities from the trained biencoder
    # (to reduce candidates to score by cross-encoder)
    biencoder_train_idxs, biencoder_valid_idxs = get_biencoder_nns(bi_reranker=bi_reranker,
                                                                   biencoder_indices_path=biencoder_indices_path,
                                                                   entity_dictionary=entity_dictionary,
                                                                   entity_dict_vecs=entity_dict_vecs,
                                                                   train_men_vecs=train_men_vecs,
                                                                   train_processed_data=train_processed_data,
                                                                   train_gold_clusters=train_gold_clusters,
                                                                   valid_men_vecs=valid_men_vecs,
                                                                   valid_processed_data=valid_processed_data,
                                                                   use_types=use_types,
                                                                   logger=logger,
                                                                   n_gpu=n_gpu,
                                                                   params=params,
                                                                   train_context_doc_ids=train_context_doc_ids,
                                                                   valid_context_doc_ids=valid_context_doc_ids
                                                                   )
    bi_train_nn_count = np.sum(biencoder_train_idxs['men_nns'] != -1, axis=1)
    bi_valid_nn_count = np.sum(biencoder_valid_idxs['men_nns'] != -1, axis=1)

    # Compute and store the concatenated cross-encoder inputs for validation
    logger.info('Computing the concatenated cross-encoder inputs for validation...')
    valid_men_concat_inputs, valid_ent_concat_inputs = build_cross_concat_input(biencoder_valid_idxs,
                                                                                valid_men_vecs,
                                                                                entity_dict_vecs,
                                                                                max_seq_length,
                                                                                bi_knn)
    logger.info('Done')
    # Compute and store the concatenated cross-encoder inputs for the negatives in training
    logger.info('Computing the concatenated cross-encoder negative inputs for training...')
    train_men_concat_inputs, train_ent_concat_inputs = build_cross_concat_input(biencoder_train_idxs,
                                                                                train_men_vecs,
                                                                                entity_dict_vecs,
                                                                                max_seq_length,
                                                                                bi_knn)
    logger.info('Done')

    if not params["skip_initial_eval"]:
        logger.info("Evaluating dev set on untrained model...")
        # Evaluate cross-encoder before training
        evaluate(cross_reranker,
                 max_context_length,
                 entity_dictionary,
                 valid_processed_data,
                 biencoder_valid_idxs,
                 bi_valid_nn_count,
                 valid_men_concat_inputs,
                 valid_ent_concat_inputs,
                 logger,
                 max_k=8,
                 k_biencoder=64)

    # Reduce dataset size for quick debugging
    if debug:
        train_tensor_data = train_tensor_data[:200]

    # Initialize training data loader
    train_sampler = RandomSampler(train_tensor_data) if params["shuffle"] else SequentialSampler(train_tensor_data)
    train_dataloader = DataLoader(train_tensor_data, sampler=train_sampler, batch_size=train_batch_size)
    if not params["silent"]:
        train_dataloader = tqdm(train_dataloader, desc="Batch")

    # Start training
    time_start = time.time()
    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )
    checkpoint_pkl_path = os.path.join(model_output_path, 'epoch_checkpoint.pickle')
    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, data_parallel: {}".format(device, n_gpu, params["data_parallel"])
    )

    optim_sched, optimizer, scheduler = load_optimizer_scheduler(params, logger), None, None
    if optim_sched is None:
        optimizer = get_optimizer(cross_model, params)
        scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)
    else:
        optimizer = optim_sched['optimizer']
        scheduler = optim_sched['scheduler']

    best_score = {
        'directed': {
            'acc': -1.,
            'epoch': 0,
            'k': 0
        },
        'undirected': {
            'acc': -1.,
            'epoch': 0,
            'k': 0
        }
    }
    n_knn_negs = params["knn_negs"]  # Number of k-NN negatives in each row of the training batch (default: 8)
    n_knn_ent_negs, n_knn_men_negs = n_knn_negs // 2, n_knn_negs // 2

    for epoch_idx in range(params["num_train_epochs"]):
        logger.info(f"***** Starting EPOCH {epoch_idx}/{params['num_train_epochs']-1} *****")
        torch.cuda.empty_cache()

        checkpoint_data = None
        if params["checkpoint_epoch_data"]:
            if os.path.isfile(checkpoint_pkl_path):
                logger.info("Loading stored epoch checkpoint data...")
                with open(checkpoint_pkl_path, 'rb') as read_handle:
                    checkpoint_data = pickle.load(read_handle)
                    gold_links = checkpoint_data['gold_links']
                    neg_men_topk_inputs = checkpoint_data['neg_men_topk_inputs']
                    neg_ent_topk_inputs = checkpoint_data['neg_ent_topk_inputs']

        if checkpoint_data is None:
            # Compute arborescences per gold k-NN cluster for each mention and store the ground-truth positive edges
            logger.info("Computing gold arborescence links for positive training labels")
            gold_links = get_gold_arbo_links(cross_reranker,
                                             max_context_length,
                                             entity_dict_vecs,
                                             train_men_vecs,
                                             train_processed_data,
                                             biencoder_train_idxs['men_gold_nns'],
                                             max_seq_length,
                                             knn=params.get("gold_arbo_knn", 1),
                                             debug=debug,
                                             within_doc=within_doc,
                                             train_context_doc_ids=train_context_doc_ids)
            logger.info("Done")

            # Score biencoder negatives using cross-encoder and store nearest-k for this epoch
            logger.info("Computing cross-encoder inputs for hard-negatives")
            neg_men_topk_inputs, neg_ent_topk_inputs = get_train_neg_cross_inputs(cross_reranker,
                                                                                  max_context_length,
                                                                                  train_men_concat_inputs,
                                                                                  train_ent_concat_inputs,
                                                                                  n_knn_men_negs,
                                                                                  bi_train_nn_count,
                                                                                  n_knn_ent_negs,
                                                                                  logger,
                                                                                  debug=debug)
            logger.info("Done")

        if params["checkpoint_epoch_data"] and checkpoint_data is None:
            logger.info("Checkpointing epoch data...")
            with open(checkpoint_pkl_path, 'wb') as write_handle:
                pickle.dump({
                    'gold_links': gold_links,
                    'neg_men_topk_inputs': neg_men_topk_inputs,
                    'neg_ent_topk_inputs': neg_ent_topk_inputs
                }, write_handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Checkpoint saved!")

        save_intervals_done = 0
        tr_loss = 0
        cross_model.train()
        for step, batch in enumerate(train_dataloader):
            _, _, _, mention_idxs = batch

            batch_positive_scores, batch_negative_ent_inputs, batch_negative_men_inputs = construct_train_batch(
                cross_reranker=cross_reranker,
                mention_idxs=mention_idxs,
                gold_links=gold_links,
                entity_dict_vecs=entity_dict_vecs,
                train_men_vecs=train_men_vecs,
                neg_ent_topk_inputs=neg_ent_topk_inputs,
                neg_men_topk_inputs=neg_men_topk_inputs,
                max_seq_length=max_seq_length,
                max_context_length=max_context_length)

            total_loss = execute_training_step(mention_idxs,
                                               bi_train_nn_count,
                                               n_knn_men_negs,
                                               batch_positive_scores,
                                               batch_negative_men_inputs,
                                               batch_negative_ent_inputs,
                                               cross_reranker,
                                               max_context_length,
                                               grad_acc_steps,
                                               n_gpu,
                                               no_sigmoid=no_sigmoid_train)
            tr_loss += total_loss

            n_print_iters = params["print_interval"] * grad_acc_steps
            if (step + 1) % n_print_iters == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}".format(
                        step,
                        epoch_idx,
                        tr_loss / n_print_iters,
                    )
                )
                tr_loss = 0

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    cross_model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if params["eval_interval"] != -1:
                if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                    logger.info("Evaluation on the dev set")
                    evaluate(cross_reranker,
                             max_context_length,
                             entity_dictionary,
                             valid_processed_data,
                             biencoder_valid_idxs,
                             bi_valid_nn_count,
                             valid_men_concat_inputs,
                             valid_ent_concat_inputs,
                             logger,
                             max_k=8,
                             k_biencoder=64)
                    cross_model.train()
                    logger.info("\n")

            if params["save_interval"] != -1:
                train_prop_done = (step + 1) / len(train_dataloader)
                # Skip saving the model at 100% completion since we do that regardless in the outer loop
                if train_prop_done != 1 and train_prop_done >= (save_intervals_done + 1) * params["save_interval"]:
                    epoch_output_folder_path = os.path.join(
                        model_output_path, f"epoch_{epoch_idx}_{save_intervals_done}"
                    )
                    save_intervals_done += 1
                    logger.info(
                        f"***** Saving fine-tuned model (at {save_intervals_done * params['save_interval'] * 100}%) *****")
                    utils.save_model(cross_model, cross_tokenizer, epoch_output_folder_path)
                    logger.info(f"Model saved at {epoch_output_folder_path}")

        logger.info("***** Saving fine-tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(cross_model, cross_tokenizer, epoch_output_folder_path, scheduler, optimizer)
        logger.info(f"Model saved at {epoch_output_folder_path}")

        if not params["skip_epoch_eval"]:
            eval_accuracy = evaluate(cross_reranker,
                                     max_context_length,
                                     entity_dictionary,
                                     valid_processed_data,
                                     biencoder_valid_idxs,
                                     bi_valid_nn_count,
                                     valid_men_concat_inputs,
                                     valid_ent_concat_inputs,
                                     logger,
                                     max_k=8,
                                     k_biencoder=64)
            for mode in ['directed', 'undirected']:
                ls = [best_score[mode]['acc'], eval_accuracy[mode]]
                li = [best_score[mode]['epoch'], epoch_idx]
                best_score[mode]['acc'] = ls[np.argmax(ls)]
                best_score[mode]['epoch'] = li[np.argmax(ls)]
                if np.argmax(ls) == 0:
                    best_score[mode]['k'] = eval_accuracy[f'{mode}_k']
            logger.info("\n")

        if params["checkpoint_epoch_data"] and epoch_idx < params["num_train_epochs"] - 1:
            if os.path.isfile(checkpoint_pkl_path):
                logger.info("Clearing stored epoch checkpoint data...")
                os.remove(checkpoint_pkl_path)

    execution_time = (time.time() - time_start) / 60
    logger.info("The training took {} minutes".format(execution_time))
    logger.info("Best performance:")
    logger.info(json.dumps(best_score))
    logger.info("***** Training finished *****")


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_joint_train_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)
