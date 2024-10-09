import sys

import os
import io
import json
import pickle
import logger
import numpy as np
import torch
import glob
import pandas as pd
from tqdm import tqdm

import lightning as L
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Optional
from pytorch_metric_learning import samplers

from torch.utils.data import DataLoader

# import bioel.models.arboel.data.data_process as data_process
# from bioel.models.arboel.data.data_utils import process_mention_dataset
# from bioel.models.arboel.model.eval_cluster_linking import filter_by_context_doc_id
from bioel.ontology import BiomedicalOntology
from bioel.models.sapbert.data.utils import (
    generate_positive_pairs,
    MetricLearningDataset,
    MetricLearningDataset_pairwise,
    SapBertBigBioDataset,
    SapBertDictionaryDataset,
)

from bioel.utils import bigbio_utils

import logging

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SapbertDataModule(L.LightningDataModule):
    """ """

    def __init__(self, params):
        """
        Parameters
        ----------
        - params : dict(str)
        Contains Configuration options
        """
        super().__init__()
        self.save_hyperparameters(params)

        self.data_path = self.hparams.train_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_dir)

        self.batch_size = self.hparams.train_batch_size

    def prepare_data(self):
        """
        Prepare the data
        """
        if self.hparams.mode == "pretrain":
            print(self.data_path)
            ontology_config = {
                "filepath": self.data_path,
                "name": "umls",
                "abbrev": None,
            }
            ontology = BiomedicalOntology.load_umls(**ontology_config)

            self.positive_pairs = generate_positive_pairs(ontology.entities)
        elif self.hparams.mode == "finetune":
            logger.info(f"Loading the BigBio Dataset {self.hparams.train_dir}")

            curie_dict = SapBertDictionaryDataset(self.hparams.train_dir).data
            self.positive_pairs = generate_positive_pairs(curie_dict)

        else:
            raise ValueError(f"Invalid mode: {self.hparams.mode}")

    def setup(self):
        """
        Setup the data
        """
        if self.hparams.pairwise:
            self.train_set = MetricLearningDataset_pairwise(
                training_pairs=self.positive_pairs, tokenizer=self.tokenizer
            )
        else:
            self.train_set = MetricLearningDataset(
                training_pairs=self.positive_pairs, tokenizer=self.tokenizer
            )

    def collate_fn_batch_encoding(self, batch):
        query1, query2, query_id = zip(*batch)
        query_encodings1 = self.tokenizer(
            list(query1),
            max_length=self.hparams.max_length,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt",
        )
        query_encodings2 = self.tokenizer(
            list(query2),
            max_length=self.hparams.max_length,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt",
        )
        query_ids = torch.tensor(list(query_id))
        return query_encodings1, query_encodings2, query_ids

    def train_dataloader(self):
        """
        Returns the training dataloader
        """
        if self.hparams.pairwise:
            print(len(self.train_set))
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=True,
                collate_fn=self.collate_fn_batch_encoding,
            )
        else:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                num_workers=self.hparams.num_workers,
                sampler=samplers.MPerClassSampler(
                    self.train_set.query_ids,
                    m=2,
                    length_before_new_iter=100000,
                ),
            )


if __name__ == "__main__":
    data_module = SapbertDataModule(
        params={
            "train_dir": "/mitchell/entity-linking/2017AA/META/",
            "model_dir": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "train_batch_size": 32,
            "num_workers": 4,
            "pairwise": True,
            "max_length": 128,
            "mode": "pretrain",
        }
    )
    data_module.prepare_data()
    data_module.setup()
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        print(batch)
        break
