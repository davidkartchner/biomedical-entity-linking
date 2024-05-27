import os
import io
import json
import pickle
import logger
import numpy as np
import torch

import lightning as L
from transformers import AutoTokenizer
from typing import Optional

from torch.utils.data import DataLoader
import bioel.models.arboel.biencoder.data.data_process as data_process
from bioel.ontology import BiomedicalOntology
from bioel.models.arboel.biencoder.data.data_utils import process_mention_dataset
from bioel.models.arboel.biencoder.model.eval_cluster_linking import (
    filter_by_context_doc_id,
)

import logging

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def read_data(split, params, logger):
    """
    Description
    -----------
    Loads dataset samples from a specified path
    Optionally filters out samples without labels
    Checks if the dataset supports multiple labels per sample
    "has_mult_labels" : bool

    Parameters
    ----------
    split : str
        Indicates the portion of the dataset to load ("train", "test", "valid"), used by utils.read_dataset to determine which data to read.
    params : dict(str)
        Contains configuration options
    logger :
        An object used for logging messages about the process, such as the number of samples read.
    """
    samples = read_dataset(split, params["data_path"])
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
    logger.info(f"Read %d {split} samples.." % len(samples))
    return samples, has_mult_labels


"Data module"


class ArboelDataModule(L.LightningDataModule):
    """
    Attributes
    ----------

    - entity_dictionary : list of dict
    Stores the initial and raw entity dictionary
    - train_tensor_data : TensorDataset(context_vecs, label_idxs, n_labels, mention_idx) with :
        - “context_vecs” : tensor containing IDs of (mention + surrounding context) tokens
        - “label_idxs” : tensor with indices pointing to the entities in the entity dictionary that are considered correct labels for the mention.
        - “n_labels” : Number of labels (=entities) associated with the mention
        - “mention_idx” : tensor containing a sequence of integers from 0 to N-1 (N = number of mentions in the dataset) serving as a unique identifier for each mention.
    - train_processed_data : list of dict
    Contains information about mentions (mention_id, mention_name, context, etc…)
    - valid_tensor_data : TensorDataset
    Same as "train_tensor_dataset" but for validation set
    - max_gold_cluster_len : int
    Maximum length of clusters inside gold_cluster
    - train_context_doc_ids : list
    # Store the context_doc_id (=context document indice) for every mention in the train set
    """

    def __init__(self, params):
        """
        Parameters
        ----------
        - params : dict(str)
        Contains configuration options
        """
        super().__init__()
        self.save_hyperparameters(params)

        self.data_path = self.hparams.data_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams["model_name_or_path"]
        )

        self.batch_size = self.hparams.get("train_batch_size")

        self.train_processed_data = None
        self.valid_processed_data = None
        self.test_processed_data = None
        self.train_tensor_data = None
        self.valid_tensor_data = None
        self.test_tensor_data = None
        self.train_samples = None
        self.valid_samples = None
        self.test_samples = None
        self.entity_dict_vecs = None

    def prepare_data(self):
        """
        Use this to download and prepare data.
        - Create the entity data files: dictionary.pickle
        - Create the mentions data files:  train.jsonl, valid.jsonl, test.jsonl
        """

        # path to a file where the training data is stored
        self.train_data = os.path.join(self.data_path, "train.jsonl")

        # if the full path to file exist, no need to prepare them, they are already ready
        if not os.path.isfile(self.train_data):

            # Load the ontology object
            with open(
                os.path.join(
                    self.hparams["data_path"],
                    f"{self.hparams['ontology']}_object.pickle",
                ),
                "rb",
            ) as f:
                self.ontology_object = pickle.load(f)

            process_mention_dataset(
                ontology=self.ontology_object,
                dataset=self.hparams["dataset"],
                data_path=self.data_path,
                path_to_abbrev=self.hparams["path_to_abbrev"],
                tax2name_filepath=self.hparams["tax2name_filepath"],
            )

    def setup(self, stage=None):
        """
        For processing and splitting. Called at the beginning of fit (train + validate), test, or predict.
        """

        "entity dictionary"
        # if entity dictionary already tokenized (=add tokens and idx keys), load it
        self.entity_dictionary_pkl_path = os.path.join(
            self.data_path, "entity_dictionary.pickle"
        )
        self.entity_dictionary_loaded = False
        if os.path.isfile(self.entity_dictionary_pkl_path):
            print("Loading stored processed entity dictionary...")
            with open(self.entity_dictionary_pkl_path, "rb") as read_handle:
                self.entity_dictionary = pickle.load(read_handle)
            self.entity_dictionary_loaded = True

        else:  # else load the one not processed yet
            with open(
                os.path.join(self.data_path, "dictionary.pickle"), "rb"
            ) as read_handle:
                self.entity_dictionary = pickle.load(read_handle)

        "Entity dict : drop entity for discovery"
        # For discovery experiment: Drop entities used in training that were dropped randomly from dev/test set
        if self.hparams["drop_entities"]:
            assert self.entity_dictionary
            drop_set_path = (
                self.hparams["drop_set"]
                if self.hparams["drop_set"] is not None
                else os.path.join(self.data_path, "drop_set_mention_data.pickle")
            )
            if not os.path.isfile(drop_set_path):
                raise ValueError(
                    "Invalid or no --drop_set path provided to dev/test mention data"
                )
            with open(drop_set_path, "rb") as read_handle:
                drop_set_data = pickle.load(read_handle)
            # gold cuis indices for each mention in drop_set_data
            drop_set_mention_gold_cui_idxs = list(
                map(lambda x: x["label_idxs"][0], drop_set_data)
            )
            # Make the set unique
            ents_in_data = np.unique(drop_set_mention_gold_cui_idxs)
            # % of drop
            ent_drop_prop = 0.1
            logger.info(
                f"Dropping {ent_drop_prop*100}% of {len(ents_in_data)} entities found in drop set"
            )
            # Number of entity indices to drop
            n_ents_dropped = int(ent_drop_prop * len(ents_in_data))
            # Random selection drop
            rng = np.random.default_rng(seed=17)
            # Indices of all entities that are dropped
            dropped_ent_idxs = rng.choice(
                ents_in_data, size=n_ents_dropped, replace=False
            )

            # Drop entities from dictionary (subsequent processing will automatically drop corresponding mentions)
            keep_mask = np.ones(len(self.entity_dictionary), dtype="bool")
            keep_mask[dropped_ent_idxs] = False
            self.entity_dictionary = np.array(self.entity_dictionary)[keep_mask]

        # fit
        if stage == "fit" or stage is None:
            "training mention data"
            # path to a file where the training data, already processed into tensors is saved
            self.train_tensor_data_pkl_path = os.path.join(
                self.data_path, "train_tensor_data.pickle"
            )
            # path to a file where metadata / additional information about the training data is stored
            self.train_processed_data_pkl_path = os.path.join(
                self.data_path, "train_processed_data.pickle"
            )

            # if the full path to file exist, load the file
            if os.path.isfile(self.train_tensor_data_pkl_path) and os.path.isfile(
                self.train_processed_data_pkl_path
            ):
                print("Loading stored processed train data...")
                with open(self.train_tensor_data_pkl_path, "rb") as read_handle:
                    self.train_tensor_data = pickle.load(read_handle)
                with open(self.train_processed_data_pkl_path, "rb") as read_handle:
                    self.train_processed_data = pickle.load(read_handle)

            else:  # Load and Process train data if not done yet
                # train_samples = list of dict. Each dict contains information about a mention (id, name, context, etc…).
                # Each key can have a dictionary itself. Ex : mention["context"]["tokens"] or mention["context"]["ids"]
                self.train_samples, self.train_mult_labels = read_data(
                    "train", self.hparams, logger
                )

                # train_processed_data = (mention + surrounding context) tokens
                # entity_dictionary = tokenized entities
                # tensor_train_dataset = Dataset containing several tensors (IDs of mention + context / indices of correct entities etc..) # Go check "process_mention_data" for more info
                (
                    self.train_processed_data,
                    self.entity_dictionary,
                    self.train_tensor_data,
                ) = data_process.process_mention_data(
                    samples=self.train_samples,
                    entity_dictionary=self.entity_dictionary,
                    tokenizer=self.tokenizer,
                    max_context_length=self.hparams["max_context_length"],
                    max_cand_length=self.hparams["max_cand_length"],
                    silent=self.hparams["silent"],
                    dictionary_processed=self.entity_dictionary_loaded,
                    context_key=self.hparams["context_key"],
                    multi_label_key="labels" if self.train_mult_labels else None,
                    logger=logger,
                    debug=self.hparams["debug"],
                    knn=self.hparams["knn"],
                )

                # Save the entity dictionary if not already done
                if not self.entity_dictionary_loaded:
                    print("Saving entity dictionary...")
                    with open(self.entity_dictionary_pkl_path, "wb") as write_handle:
                        pickle.dump(
                            self.entity_dictionary,
                            write_handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )

                self.entity_dictionary_loaded = True

                print("Saving processed train data...")
                with open(self.train_tensor_data_pkl_path, "wb") as write_handle:
                    pickle.dump(
                        self.train_tensor_data,
                        write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                with open(self.train_processed_data_pkl_path, "wb") as write_handle:
                    pickle.dump(
                        self.train_processed_data,
                        write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

            # Prepare tensor containing only ID of (mention + surrounding context) tokens of training set'
            self.train_men_vecs = self.train_tensor_data[:][0]

            "Validation mention data"
            self.valid_tensor_data_pkl_path = os.path.join(
                self.data_path, "valid_tensor_data.pickle"
            )
            self.valid_processed_data_pkl_path = os.path.join(
                self.data_path, "valid_processed_data.pickle"
            )
            # Same as training data :
            # if the full path to file exist, load the file
            if os.path.isfile(self.valid_tensor_data_pkl_path) and os.path.isfile(
                self.valid_processed_data_pkl_path
            ):
                print("Loading stored processed valid data...")
                with open(self.valid_tensor_data_pkl_path, "rb") as read_handle:
                    self.valid_tensor_data = pickle.load(read_handle)
                with open(self.valid_processed_data_pkl_path, "rb") as read_handle:
                    self.valid_processed_data = pickle.load(read_handle)

            else:
                # Load and Process validation data if not done yet
                self.valid_samples, self.valid_mult_labels = read_data(
                    "valid", self.hparams, logger
                )
                self.valid_processed_data, _, self.valid_tensor_data = (
                    data_process.process_mention_data(
                        samples=self.valid_samples,
                        entity_dictionary=self.entity_dictionary,
                        tokenizer=self.tokenizer,
                        max_context_length=self.hparams["max_context_length"],
                        max_cand_length=self.hparams["max_cand_length"],
                        silent=self.hparams["silent"],
                        context_key=self.hparams["context_key"],
                        multi_label_key="labels" if self.valid_mult_labels else None,
                        logger=logger,
                        debug=self.hparams["debug"],
                        knn=self.hparams["knn"],
                        dictionary_processed=self.entity_dictionary_loaded,
                    )
                )

                # Save the entity dictionary if not already done
                if not self.entity_dictionary_loaded:
                    print("Saving entity dictionary...")
                    with open(self.entity_dictionary_pkl_path, "wb") as write_handle:
                        pickle.dump(
                            self.entity_dictionary,
                            write_handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )

                self.entity_dictionary_loaded = True

                print("Saving processed valid data...")
                with open(self.valid_tensor_data_pkl_path, "wb") as write_handle:
                    pickle.dump(
                        self.valid_tensor_data,
                        write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                with open(self.valid_processed_data_pkl_path, "wb") as write_handle:
                    pickle.dump(
                        self.valid_processed_data,
                        write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
            # Prepare tensor containing only ID of (mention + surrounding context) tokens of validation data
            self.valid_men_vecs = self.valid_tensor_data[:][0]

            # Get clusters of mentions that map to a gold entity
            self.train_gold_clusters = data_process.compute_gold_clusters(
                self.train_processed_data
            )
            # Maximum length of clusters inside gold_cluster
            self.max_gold_cluster_len = 0
            for ent in self.train_gold_clusters:
                if len(self.train_gold_clusters[ent]) > self.max_gold_cluster_len:
                    self.max_gold_cluster_len = len(self.train_gold_clusters[ent])

        # test
        if stage == "test" or stage is None:

            if self.hparams.get("transductive"):
                "training mention data"
                # path to a file where the training data, already processed into tensors is saved
                self.train_tensor_data_pkl_path = os.path.join(
                    self.data_path, "train_tensor_data.pickle"
                )
                # path to a file where metadata / additional information about the training data is stored
                self.train_processed_data_pkl_path = os.path.join(
                    self.data_path, "train_processed_data.pickle"
                )

                # if the full path to file exist, load the file
                if os.path.isfile(self.train_tensor_data_pkl_path) and os.path.isfile(
                    self.train_processed_data_pkl_path
                ):
                    print("Loading stored processed train data...")
                    with open(self.train_tensor_data_pkl_path, "rb") as read_handle:
                        self.train_tensor_data = pickle.load(read_handle)
                    with open(self.train_processed_data_pkl_path, "rb") as read_handle:
                        self.train_processed_data = pickle.load(read_handle)

                else:  # Load and Process train data if not done yet
                    # train_samples = list of dict. Each dict contains information about a mention (id, name, context, etc…).
                    # Each key can have a dictionary itself. Ex : mention["context"]["tokens"] or mention["context"]["ids"]
                    self.train_samples, self.train_mult_labels = read_data(
                        "train", self.hparams, logger
                    )

                    # train_processed_data = (mention + surrounding context) tokens
                    # entity_dictionary = tokenized entities
                    # tensor_train_dataset = Dataset containing several tensors (IDs of mention + context / indices of correct entities etc..) # Go check "process_mention_data" for more info
                    (
                        self.train_processed_data,
                        self.entity_dictionary,
                        self.train_tensor_data,
                    ) = data_process.process_mention_data(
                        samples=self.train_samples,
                        entity_dictionary=self.entity_dictionary,
                        tokenizer=self.tokenizer,
                        max_context_length=self.hparams["max_context_length"],
                        max_cand_length=self.hparams["max_cand_length"],
                        silent=self.hparams["silent"],
                        dictionary_processed=self.entity_dictionary_loaded,
                        context_key=self.hparams["context_key"],
                        multi_label_key="labels" if self.train_mult_labels else None,
                        logger=logger,
                        debug=self.hparams["debug"],
                        knn=self.hparams["knn"],
                    )

                    # Save the entity dictionary if not already done
                    if not self.entity_dictionary_loaded:
                        print("Saving entity dictionary...")
                        with open(
                            self.entity_dictionary_pkl_path, "wb"
                        ) as write_handle:
                            pickle.dump(
                                self.entity_dictionary,
                                write_handle,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            )

                    self.entity_dictionary_loaded = True

                    print("Saving processed train data...")
                    with open(self.train_tensor_data_pkl_path, "wb") as write_handle:
                        pickle.dump(
                            self.train_tensor_data,
                            write_handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    with open(self.train_processed_data_pkl_path, "wb") as write_handle:
                        pickle.dump(
                            self.train_processed_data,
                            write_handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )

                # Prepare tensor containing only ID of (mention + surrounding context) tokens of training set'
                self.train_men_vecs = self.train_tensor_data[:][0]

            "test mention data"
            self.test_tensor_data_pkl_path = os.path.join(
                self.data_path, "test_tensor_data.pickle"
            )
            self.test_processed_data_pkl_path = os.path.join(
                self.data_path, "test_processed_data.pickle"
            )
            # Same as training data :
            # if the full path to file exist, load the file
            if os.path.isfile(self.test_tensor_data_pkl_path) and os.path.isfile(
                self.test_processed_data_pkl_path
            ):
                print("Loading stored processed test data...")
                with open(
                    self.test_tensor_data_pkl_path, "rb"
                ) as read_handle:  # CC7 'rb' = binary read mode
                    self.test_tensor_data = pickle.load(read_handle)
                with open(self.test_processed_data_pkl_path, "rb") as read_handle:
                    self.test_processed_data = pickle.load(read_handle)

            else:
                # Load and Process test data if not done yet
                self.test_samples, self.test_mult_labels = read_data(
                    "test", self.hparams, logger
                )
                self.test_processed_data, _, self.test_tensor_data = (
                    data_process.process_mention_data(
                        samples=self.test_samples,
                        entity_dictionary=self.entity_dictionary,
                        tokenizer=self.tokenizer,
                        max_context_length=self.hparams["max_context_length"],
                        max_cand_length=self.hparams["max_cand_length"],
                        context_key=self.hparams["context_key"],
                        multi_label_key="labels" if self.test_mult_labels else None,
                        silent=self.hparams["silent"],
                        logger=logger,
                        debug=self.hparams["debug"],
                        knn=self.hparams["knn"],
                        dictionary_processed=self.entity_dictionary_loaded,
                    )
                )

                print("Saving processed test data...")
                with open(self.test_tensor_data_pkl_path, "wb") as write_handle:
                    pickle.dump(
                        self.test_tensor_data,
                        write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                with open(self.test_processed_data_pkl_path, "wb") as write_handle:
                    pickle.dump(
                        self.test_processed_data,
                        write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

            # Prepare tensor containing only ID of (mention + surrounding context) tokens of validation data'
            self.test_men_vecs = self.test_tensor_data[:][0]

        # Store the IDs of the entity in entity_dictionary # It's the equivalent of train_men_vecs for entities
        # (Done here because data_process.process_mention_data will tokenize the entities in entity_dict)
        self.entity_dict_vecs = torch.tensor(
            list(map(lambda x: x["ids"], self.entity_dictionary)), dtype=torch.long
        )

        # Within_doc search
        # Consider if it’s within_doc (=search only within the document)'
        self.train_context_doc_ids = None
        self.valid_context_doc_ids = None
        self.test_context_doc_ids = None

        if self.hparams["within_doc"]:
            logger.info("within_doc")
            # Store the context_doc_id for every mention in the train and valid sets
            if self.train_samples is None:
                self.train_samples, _ = read_data("train", self.hparams, logger)
            self.train_context_doc_ids = [
                s["context_doc_id"] for s in self.train_samples
            ]
            if self.valid_samples is None:
                self.valid_samples, _ = read_data("valid", self.hparams, logger)
            self.valid_context_doc_ids = [
                s["context_doc_id"] for s in self.valid_samples
            ]
            if self.test_samples is None:
                self.test_samples, _ = read_data("test", self.hparams, logger)
            self.test_context_doc_ids = [s["context_doc_id"] for s in self.test_samples]

    def train_dataloader(self):
        # Return the training DataLoader
        return DataLoader(
            self.train_tensor_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=11,
        )

    def val_dataloader(self):
        # Return the validation DataLoader
        return DataLoader(
            self.valid_tensor_data,
            batch_size=self.batch_size,
            num_workers=11,
        )

    def test_dataloader(self):
        # Return the validation DataLoader
        return DataLoader(
            self.test_tensor_data,
            batch_size=self.batch_size,
            num_workers=11,
        )


def main():

    model = "arboel"

    dataset = "ncbi_disease"
    abs_path = "/home2/cye73/data_test2"
    data_path = os.path.join(abs_path, model, dataset)
    os.makedirs(data_path, exist_ok=True)

    # dataset = "bc5cdr"
    # abs_path = "/home2/cye73/data"
    # data_path = os.path.join(abs_path, model, dataset)

    # dataset = "medmentions_st21pv"
    # abs_path = "/home2/cye73/data"
    # data_path = os.path.join(abs_path, model, dataset)

    params_test = {
        "data_path": data_path,
        "batch_size": 64,
        "max_context_length": 128,
        "max_cand_length": 128,
        "context_key": "context",
        "debug": False,
        "knn": 4,
        "model_name_or_path": "dmis-lab/biobert-base-cased-v1.2",
        "out_dim": 768,
        "pull_from_layer": 11,
        "add_linear": True,
        "use_types": True,
        "force_exact_search": True,
        "probe_mult_factor": 1,
        "embed_batch_size": 768,
        "drop_entities": False,
        "within_doc": True,
        "filter_unlabeled": False,
        "model": "arboel",
        # "ontology": "umls",
        # "dataset": "medmentions_st21pv",
        # "ontology_dir": "/mitchell/entity-linking/2017AA/META/",
        "ontology": "medic",
        "dataset": "ncbi_disease",
        "ontology_dir": "/mitchell/entity-linking/kbs/medic.tsv",
        # "ontology": "mesh",
        # "dataset": "bc5cdr",
        # "ontology_dir": "/mitchell/entity-linking/2017AA/META/",
        "silent": False,
        "abbrevs": True,
        "path_to_abbrev": "/home2/cye73/data_test2/abbreviations.json",
    }

    data_module = ArboelDataModule(params=params_test)

    data_module.prepare_data()

    data_module.setup()


if __name__ == "__main__":
    main()
