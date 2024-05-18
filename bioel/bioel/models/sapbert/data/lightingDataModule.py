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


import lightning as L
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Optional

from torch.utils.data import DataLoader
# import bioel.models.arboel.data.data_process as data_process
# from bioel.models.arboel.data.data_utils import process_mention_dataset
# from bioel.models.arboel.model.eval_cluster_linking import filter_by_context_doc_id
from bioel.ontology import BiomedicalOntology
from bioel.models.sapbert.data.utils import generate_pretraining_data
from bioel import bigbio_utils

import logging

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SapbertDataModule(L.LightningDataModule):
    """
    
    
    """
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.model_dir
        )

        self.batch_size = self.hparams.train_batch_size

    
    def prepare_data(self):
        """
        Prepare the data
        """
        if self.hparams.mode == "pretrain":
            ontology_config = {
                "filepath": self.data_path,
                "name": "UMLS",
                "abbrev": None,
            }
            ontology = BiomedicalOntology.load_umls(**ontology_config)

            pos_pairs = generate_pretraining_data(ontology)
    
    def setup(self):
        """
        Setup the data
        """
        if self.hparams.mode == "pretrain":
            if self.hparams.pairwise:
                self.train_dataset = MetricLearningDataset_pairwise(self.data_path, self.tokenizer)
            else:
                self.train_dataset = MetricLearningDataset(self.data_path, self.tokenizer)
        elif self.hparams.mode == "pretrain_pairwise":
            self.train_dataset = MetricLearningDataset_pairwise(self.data_path, self.tokenizer)
        self.val_dataset = MetricLearningDataset(self.data_path, self.tokenizer)
        self.test_dataset = MetricLearningDataset(self.data_path, self.tokenizer)

            

