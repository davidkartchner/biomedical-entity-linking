# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import collections
import os
import sys
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
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from bioel.models.arboel.model.common.optimizer import get_bert_optimizer
from pytorch_transformers.optimization import WarmupLinearSchedule
from bioel.models.arboel.crossencoder.original.crossencoder import CrossEncoderRanker
import bioel.models.arboel.data.data_process as data_process
from IPython import embed

from bioel.logger import setup_logger

from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import lightning as L
from datetime import datetime
from bioel.models.arboel.model.common.params import BlinkParser

# logger = setup_logger()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pytorch_transformers.modeling_utils import CONFIG_NAME, WEIGHTS_NAME

OPTIM_SCHED_FNAME = "optim_sched.pth"


def read_dataset(dataset_name, preprocessed_json_data_parent_folder, debug=False):
    file_name = "{}.jsonl".format(dataset_name)
    txt_file_path = os.path.join(preprocessed_json_data_parent_folder, file_name)

    samples = []

    with io.open(txt_file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            samples.append(json.loads(line.strip()))
            if debug and len(samples) > 200:
                break

    print(f"Read {len(samples)} samples.")
    return samples


def accuracy(out, labels, return_bool_arr=False):
    outputs = np.argmax(out, axis=1)
    if return_bool_arr:
        return outputs == labels, outputs
    return np.sum(outputs == labels)


def modify(context_input, candidate_input, max_seq_length):
    new_input = []
    context_input = context_input.tolist()
    candidate_input = candidate_input.tolist()

    for i in tqdm(range(len(context_input)), desc="Concatenating"):
        cur_input = context_input[i]
        cur_candidate = candidate_input[i]
        mod_input = []
        for j in range(len(cur_candidate)):
            # remove [CLS] token from candidate
            sample = cur_input + cur_candidate[j][1:]
            sample = sample[:max_seq_length]
            mod_input.append(sample)

        new_input.append(mod_input)

    return torch.LongTensor(new_input)


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


def load_data(
    data_split,
    bi_tokenizer,
    max_context_length,
    max_cand_length,
    knn,
    pickle_src_path,
    params,
    logger,
    return_dict_only=False,
):
    entity_dictionary_loaded = False
    entity_dictionary_pkl_path = os.path.join(
        pickle_src_path, "entity_dictionary.pickle"
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
        pickle_src_path, f"{data_split}_tensor_data.pickle"
    )
    processed_data_pkl_path = os.path.join(
        pickle_src_path, f"{data_split}_processed_data.pickle"
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
                max_context_length,
                max_cand_length,
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
    return processed_data, entity_dictionary, tensor_data


def get_data_loader(
    params,
    data_split,
    tokenizer,
    context_length,
    candidate_length,
    max_seq_length,
    pickle_src_path,
    logger,
    inject_ground_truth=False,
    max_n=None,
    shuffle=True,
    return_data=False,
    custom_cand_set=None,
):
    # Load the top-64 indices for each mention query and the ground truth label if it exists in the candidate set
    logger.info(f"Loading {data_split} data...")
    cand_name = data_split
    if custom_cand_set is not None:
        logger.info(f"Loading custom candidate set: {custom_cand_set}...")
        cand_name = custom_cand_set
    fname = os.path.join(
        params["biencoder_indices_path"], f"candidates_{cand_name}_top64.t7"
    )
    stored_data = torch.load(fname)
    processed_data, entity_dictionary, tensor_data = load_data(
        data_split,
        tokenizer,
        context_length,
        candidate_length,
        1,
        pickle_src_path,
        params,
        logger,
    )
    logger.info("Loaded")
    dict_vecs = list(map(lambda x: x["ids"], entity_dictionary))

    mention_idxs = torch.tensor([i for i in range(len(stored_data["labels"]))])
    candidate_input = []
    keep_mask = [True] * len(stored_data["labels"])
    for i in tqdm(range(len(stored_data["labels"])), desc="Processing"):
        if stored_data["labels"][i] == -1:
            # If ground truth not in candidates, replace the last candidate with the ground truth
            if inject_ground_truth:
                gold_idx = processed_data[i]["label_idxs"][0]
                stored_data["labels"][i] = len(stored_data["candidates"][i]) - 1
                stored_data["candidates"][i][-1] = gold_idx
            else:
                keep_mask[i] = False
                continue
        cands = list(map(lambda x: dict_vecs[x], stored_data["candidates"][i]))
        candidate_input.append(cands)
    candidate_input = np.array(candidate_input)
    context_input = tensor_data[:][0][keep_mask]
    label_input = torch.tensor(stored_data["labels"])[keep_mask]
    mention_idxs = mention_idxs[keep_mask]

    n_no_label = len(stored_data["labels"]) - np.sum(keep_mask)

    if max_n is not None:
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]

    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]

    context_input = modify(context_input, candidate_input, max_seq_length)
    tensor_data = TensorDataset(context_input, label_input, mention_idxs)
    dataloader = DataLoader(
        tensor_data,
        shuffle=shuffle,
        batch_size=params[
            "train_batch_size" if data_split == "train" else "eval_batch_size"
        ],
    )
    if return_data:
        return (
            dataloader,
            n_no_label,
            {
                "entity_dictionary": entity_dictionary,
                "mention_data": processed_data,
                "stored_candidates": stored_data,
            },
        )
    return dataloader, n_no_label


class CrossEncoderDataModule(L.LightningDataModule):
    """ """

    def __init__(self, params):
        """
        Parameters
        ----------
        - params : dict(str)
        Contains configuration options
        """
        super(CrossEncoderDataModule, self).__init__()
        self.save_hyperparameters(params)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams["model_name_or_path"]
        )

    def setup(self, stage=None):
        """
        For processing and splitting. Called at the beginning of fit (train + validate), test, or predict.
        """

        # Fit
        if stage == "fit" or stage is None:

            # Training set
            self.train_loader, _, self.train_data = get_data_loader(
                params=self.hparams,
                data_split="train",
                tokenizer=self.tokenizer,
                context_length=self.hparams["max_context_length"],
                candidate_length=self.hparams["max_context_length"],
                max_seq_length=self.hparams["max_seq_length"],
                pickle_src_path=self.hparams["data_path"],
                logger=logger,
                inject_ground_truth=self.hparams["inject_train_ground_truth"],
                return_data=True,
            )

            # Validation set
            self.valid_loader, _, self.valid_data = get_data_loader(
                params=self.hparams,
                data_split="valid",
                tokenizer=self.tokenizer,
                context_length=self.hparams["max_context_length"],
                candidate_length=self.hparams["max_context_length"],
                max_seq_length=self.hparams["max_seq_length"],
                pickle_src_path=self.hparams["data_path"],
                logger=logger,
                inject_ground_truth=self.hparams["inject_eval_ground_truth"],
                return_data=True,
            )

        # Test
        if stage == "test":
            # Test set
            self.test_loader, self.nb_no_label_test, self.test_data = get_data_loader(
                params=self.hparams,
                data_split="test",
                tokenizer=self.tokenizer,
                context_length=self.hparams["max_context_length"],
                candidate_length=self.hparams["max_context_length"],
                max_seq_length=self.hparams["max_seq_length"],
                pickle_src_path=self.hparams["data_path"],
                logger=logger,
                inject_ground_truth=self.hparams["inject_eval_ground_truth"],
                return_data=True,
            )

    def train_dataloader(self):
        # Return the training DataLoader
        return self.train_loader

    def val_dataloader(self):
        # Return the training DataLoader
        return self.valid_loader

    def test_dataloader(self):
        # Return the training DataLoader
        return self.test_loader


class CrossEncoder(L.LightningModule):
    """ """

    def __init__(self, params):
        super(CrossEncoder, self).__init__()
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
            "max_acc",
            results["normalized_accuracy"],
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
        eval_file_name = os.path.join(
            self.hparams["output_path"],
            f"crossencoder_output_eval", 
            self.hparams["experiment"],
        )
        with open(f"{eval_file_name}.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
            print(f"\nPredictions overview saved at: {eval_file_name}.json")

    def on_train_start(self):
        self.start_time = time.time()

    def on_train_end(self):
        execution_time = (time.time() - self.start_time) / 60
        logging.info(f"The training took {execution_time} minutes")


def main(args):
    print("Current seed:", args["seed"])

    data_module = CrossEncoderDataModule(params=args)

    model = CrossEncoder(params=args)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_checkpoint = ModelCheckpoint(
        monitor="max_acc",  # Metric to monitor
        dirpath=args["output_path"],  # Directory to save the model
        filename=f"{current_time}-{{epoch}}-{{max_acc:.2f}}",  # Saves the model with epoch and val_loss in the filename
        save_top_k=1,  # Number of best models to save; -1 means save all of them
        mode="max",  # 'max' means the highest max_acc will be considered as the best model
        verbose=True,  # Logs a message whenever a model checkpoint is saved
    )

    # wandb_logger = WandbLogger(project=args["experiment"])

    trainer = L.Trainer(
        limit_val_batches=1,
        # num_sanity_val_steps=0,
        # fast_dev_run=True,
        max_epochs=args["num_train_epochs"],
        devices=args["devices"],
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        enable_progress_bar=True,
        callbacks=[model_checkpoint],
        accumulate_grad_batches=args["gradient_accumulation_steps"],
        precision="16-mixed",
        check_val_every_n_epoch=1,
        # logger=wandb_logger,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)
