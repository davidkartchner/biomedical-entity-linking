# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import collections
import os
import io
import copy
import torch
import logging
import random
import time
import numpy as np
import pickle
import json
from tqdm import tqdm, trange
from bioel.models.arboel.biencoder.model.common.optimizer import get_bert_optimizer
from pytorch_transformers.optimization import WarmupLinearSchedule
from bioel.models.arboel.crossencoder.model.crossencoder import CrossEncoderRanker
import bioel.models.arboel.biencoder.data.data_process as data_process
from IPython import embed

from bioel.logger import setup_logger

import lightning as L
from bioel.models.arboel.biencoder.model.common.params import BlinkParser

# logger = setup_logger()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pytorch_transformers.modeling_utils import CONFIG_NAME, WEIGHTS_NAME

OPTIM_SCHED_FNAME = "optim_sched.pth"


def create_versioned_filename(base_path, base_name, extension=".json"):
    """Create a versioned filename to avoid overwriting existing files.
    Params:
    - base_path (str):
        The directory where the file will be saved.
    base_name (str):
        The base name of the file without the extension.
    extension (str):
        The file extension.
    Returns:
        str: A versioned filename that does not exist in the base path.
    """
    version = 1
    while True:
        new_filename = f"{base_name}{version}{extension}"
        full_path = os.path.join(base_path, new_filename)
        if not os.path.exists(full_path):
            return full_path
        version += 1


def convert_defaultdict(d):
    if isinstance(d, collections.defaultdict):
        d = {k: convert_defaultdict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        return {k: convert_defaultdict(v) for k, v in d.items()}
    elif isinstance(d, np.int64):
        return int(d)
    elif isinstance(d, np.float64):
        return float(d)
    return d


def merge_dicts(dict1, dict2):
    """Merge two dictionaries with support for nested structures and summing numeric values"""
    merged_dict = dict1.copy()
    for key, value in dict2.items():
        if key in merged_dict:
            if isinstance(value, collections.defaultdict) and isinstance(
                merged_dict[key], collections.defaultdict
            ):
                for k, v in value.items():
                    merged_dict[key][k] = merged_dict[key].get(k, 0) + v
            elif isinstance(value, (int, float)) and isinstance(
                merged_dict[key], (int, float)
            ):
                merged_dict[key] += value
            else:
                if isinstance(value, list) and isinstance(merged_dict[key], list):
                    merged_dict[key].extend(value)
                else:
                    merged_dict[key] = value
        else:
            merged_dict[key] = copy.deepcopy(value)

    return merged_dict


def accuracy(out, labels, return_bool_arr=False):
    outputs = np.argmax(out, axis=1)
    if return_bool_arr:
        return outputs == labels, outputs
    return np.sum(outputs == labels)


def evaluate_single_batch(
    reranker,
    batch,
    logger,
    context_length,
    mention_data=None,
    compute_macro_avg=False,
    store_failure_success=False,
):
    context_input, label_input, mention_idxs = batch
    results = {}
    eval_accuracy = 0
    nb_eval_examples = 0

    if mention_data is not None:
        processed_mention_data = mention_data["mention_data"]
        dictionary = mention_data["entity_dictionary"]
        stored_candidates = mention_data["stored_candidates"]
        if store_failure_success:
            failsucc = {"failure": [], "success": []}

    n_evaluated_per_type = collections.defaultdict(int)
    n_hits_per_type = collections.defaultdict(int)

    with torch.no_grad():
        eval_loss, logits = reranker(context_input, label_input, context_length)

    logits = logits.detach().cpu().numpy()
    label_ids = label_input.cpu().numpy()

    tmp_eval_hits, predicted = accuracy(logits, label_ids, return_bool_arr=True)
    tmp_eval_accuracy = np.sum(tmp_eval_hits)

    eval_accuracy = tmp_eval_accuracy
    nb_eval_examples = context_input.size(0)

    if compute_macro_avg:
        for i, m_idx in enumerate(mention_idxs):
            mention_type = processed_mention_data[m_idx]["type"]
            n_evaluated_per_type[mention_type] += 1
            is_hit = tmp_eval_hits[i]
            n_hits_per_type[mention_type] += is_hit

    if store_failure_success:
        for i, m_idx in enumerate(mention_idxs):
            m_idx = m_idx.item()
            men_query = processed_mention_data[m_idx]
            dict_pred = dictionary[stored_candidates["candidates"][m_idx][predicted[i]]]
            report_obj = {
                "mention_id": men_query["mention_id"],
                "mention_name": men_query["mention_name"],
                "mention_gold_cui": "|".join(men_query["label_cuis"]),
                "mention_gold_cui_name": "|".join(
                    [
                        dictionary[i]["title"]
                        for i in men_query["label_idxs"][: men_query["n_labels"]]
                    ]
                ),
                "predicted_name": dict_pred["title"],
                "predicted_cui": dict_pred["cui"],
            }
            failsucc["success" if tmp_eval_hits[i] else "failure"].append(report_obj)

    results["nb_samples_evaluated"] = nb_eval_examples
    results["correct_pred"] = int(eval_accuracy)

    print(f"Eval: Accuracy: {eval_accuracy/nb_eval_examples*100}%")

    if compute_macro_avg:
        results["n_hits_per_type"] = n_hits_per_type
        results["n_evaluated_per_type"] = n_evaluated_per_type

    if store_failure_success:
        results["failure"] = failsucc["failure"]
        results["success"] = failsucc["success"]

    if not store_failure_success:
        logger.info(json.dumps(results))

    return results


class LitCrossEncoder(L.LightningModule):
    """ """

    def __init__(self, params):
        super(LitCrossEncoder, self).__init__()
        self.save_hyperparameters(params)
        self.reranker = CrossEncoderRanker(params)
        self.model = self.reranker.model
        self.val_results = {}

    def training_step(self, batch, batch_idx):
        context_input, label_input, _ = batch
        train_loss, _ = self.reranker(
            context_input, label_input, self.hparams["max_context_length"]
        )

        print("training loss :", train_loss)
        self.log(
            "train loss :", train_loss, on_step=True, on_epoch=True, sync_dist=True
        )

        return train_loss

    def configure_optimizers(self):
        # Define optimizer
        optimizer = get_bert_optimizer(
            models=[self.model],
            type_optimization=self.hparams["type_optimization"],
            learning_rate=self.hparams["learning_rate"],
            fp16=self.hparams.get("fp16"),
        )

        # Define scheduler
        num_train_steps = (
            int(
                len(self.trainer.datamodule.train_data["mention_data"])
                / self.hparams["train_batch_size"]
                / self.trainer.accumulate_grad_batches
            )
            * self.trainer.max_epochs
        )
        num_warmup_steps = int(num_train_steps * self.hparams["warmup_proportion"])

        scheduler = WarmupLinearSchedule(
            optimizer=optimizer,
            warmup_steps=num_warmup_steps,
            t_total=num_train_steps,
        )
        logger.info(" Num optimization steps = %d" % num_train_steps)
        logger.info(" Num warmup steps = %d", num_warmup_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def validation_step(self, batch, batch_idx):

        results = evaluate_single_batch(
            reranker=self.reranker,
            batch=batch,
            logger=logger,
            context_length=self.hparams["max_context_length"],
        )

        self.log(
            "Accuracy",
            results["correct_pred"] / results["nb_samples_evaluated"],
            on_epoch=True,
            sync_dist=True,
        )

    def on_test_epoch_start(self):

        self.n_evaluated_per_type = {}
        self.n_mentions_per_type = {}

        self.test_results = {
            "normalized_accuracy": 0,
            "unnormalized_accuracy": 0,
            "normalized_macro_avg_acc": 0,
            "unnormalized_macro_avg_acc": 0,
            "filtered_length": int(
                len(self.trainer.datamodule.test_data["mention_data"])
                - self.trainer.datamodule.nb_no_label_test
            ),
            "unfiltered_length": len(self.trainer.datamodule.test_data["mention_data"]),
            "n_mentions_per_type": collections.defaultdict(int),
            "n_hits_per_type": collections.defaultdict(int),
            "n_evaluated_per_type": collections.defaultdict(int),
        }

    def test_step(self, batch, batch_idx):

        results = evaluate_single_batch(
            reranker=self.reranker,
            batch=batch,
            logger=logger,
            context_length=self.hparams["max_context_length"],
            mention_data=self.trainer.datamodule.test_data,
            compute_macro_avg=True,
            store_failure_success=True,
        )
        self.test_results = merge_dicts(self.test_results, results)

    def on_test_epoch_end(self):

        self.test_results["normalized_accuracy"] = float(
            self.test_results["correct_pred"]
            / self.test_results["nb_samples_evaluated"]
            * 100
        )

        self.test_results["unnormalized_accuracy"] = float(
            self.test_results["normalized_accuracy"]
            * self.test_results["filtered_length"]
            / self.test_results["unfiltered_length"]
        )

        for mention_type in self.test_results["n_evaluated_per_type"]:
            self.test_results["normalized_macro_avg_acc"] += float(
                self.test_results["n_hits_per_type"][mention_type]
                / self.test_results["n_evaluated_per_type"][mention_type]
                if self.test_results["n_evaluated_per_type"][mention_type] > 0
                else 0
            )

        self.test_results["normalized_macro_avg_acc"] = (
            self.test_results["normalized_macro_avg_acc"]
            / len(self.test_results["n_evaluated_per_type"])
            * 100
        )

        self.log(
            "Number of samples evaluated in this test",
            self.test_results["nb_samples_evaluated"],
            sync_dist=True,
        )

        self.log(
            "Normalized Accuracy : Only considered mentions with the correct cui in top candidates",
            self.test_results["normalized_accuracy"],
            sync_dist=True,
        )

        self.log(
            "Number of samples with the correct cui in top candidates in the test set",
            self.test_results["filtered_length"],
            sync_dist=True,
        )
        self.log(
            "Number of samples with and without the correct cui in top candidatesin the test set",
            self.test_results["unfiltered_length"],
            sync_dist=True,
        )

        self.log(
            "Unnormalized Accuracy : Include mentions without the correct cui in top candidates",
            self.test_results["unnormalized_accuracy"],
            sync_dist=True,
        )
        self.log(
            "Normalized macro_avg_acc : Average accuracy across all types (Only mentions with the correct cui in top candidates).",
            self.test_results["normalized_macro_avg_acc"],
            sync_dist=True,
        )

        if (
            self.test_results["filtered_length"]
            == self.test_results["nb_samples_evaluated"]
        ):
            for men in self.trainer.datamodule.test_data["mention_data"]:
                self.test_results["n_mentions_per_type"][men["type"]] += 1

            for mention_type in self.test_results["n_mentions_per_type"]:
                self.test_results["unnormalized_macro_avg_acc"] += float(
                    self.test_results["n_hits_per_type"][mention_type]
                    / self.test_results["n_mentions_per_type"][mention_type]
                    if self.test_results["n_mentions_per_type"][mention_type] > 0
                    else 0
                )

            self.test_results["unnormalized_macro_avg_acc"] = (
                self.test_results["unnormalized_macro_avg_acc"]
                / len(self.test_results["n_mentions_per_type"])
                * 100
            )
            self.log(
                "Unnormalized macro_avg_acc : Average accuracy across all types (Including mentions without the correct cui in top candidates)",
                self.test_results["unnormalized_macro_avg_acc"],
                sync_dist=True,
            )

        self.test_results = convert_defaultdict(self.test_results)

        # Store results for evaluation object
        output_path = self.hparams["output_path"]
        base_filename = "crossencoder_output_eval"
        file_to_save = create_versioned_filename(output_path, base_filename)

        # Now, write to the file
        with open(file_to_save, "w") as f:
            json.dump(self.test_results, f, indent=2)
            print(f"\nPredictions overview saved at: {file_to_save}")

    def on_train_start(self):
        self.start_time = time.time()

    def on_train_end(self):
        execution_time = (time.time() - self.start_time) / 60
        logging.info(f"The training took {execution_time} minutes")
