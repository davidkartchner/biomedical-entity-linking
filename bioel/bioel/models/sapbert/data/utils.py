from tqdm import tqdm
import itertools
import random
from typing import List, Optional
import torch
import logging
import os
import glob
import numpy as np
import json
import pandas as pd
import ujson
import warnings
from itertools import combinations
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from bioel.utils.bigbio_utils import (
    load_bigbio_dataset,
    dataset_to_df,
    DATASET_NAMES,
    CUIS_TO_EXCLUDE,
    CUIS_TO_REMAP,
    resolve_abbreviation,
    add_deabbreviations,
)

from transformers import AutoTokenizer

from bioel.ontology import BiomedicalOntology
from bioel.logger import setup_logger

logger = setup_logger()


"""
Generate Pretraining Data for SAPBERT from the UMLS Ontology
"""
def generate_positive_pairs(alias_mapping: dict, max_pairs_per_cui: int = 50) -> List[str]:
    """
    Generate pretraining data from UMLS for the SAPBERT Contrastive Learning.
    """
    positive_pairs = []
    for cui, entity in tqdm(alias_mapping.items(), desc = "Generating pretraining data for SAPBERT"):
        if isinstance(entity, list):
            pairs = list(combinations(entity, r = 2))
        else:
            pairs = list(combinations(entity.aliases, r = 2))
        if len(pairs) > max_pairs_per_cui: # If > 50 pairs, sample 50 pairs
            pairs = random.sample(pairs, k=max_pairs_per_cui)
        
        for p in pairs:
            if p[0] and p[1]:
                line = str(cui) + "||" + p[0].lower() + "||" + p[1].lower()
                
                positive_pairs.append(line)
    return positive_pairs

class DictionaryDataset:
    """
    A class used to load dictionary data
    """

    def __init__(self, dictionary_path):
        """
        Parameters
        ----------
        dictionary_path : str
            The path of the dictionary
        draft : bool
            use only small subset
        """
        logger.info("DictionaryDataset! dictionary_path={}".format(dictionary_path))
        self.data = self.load_data(dictionary_path)

    def load_data(self, dictionary_path):
        name_cui_map = {}
        data_dict = defaultdict(list)
        with open(dictionary_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip()
                if not line or len(line.split("||")) != 2:
                    continue
                if line == "":
                    continue
                cui, name = line.split("||")
                name = name.lower()
                if cui.lower() == "cui-less":
                    continue

                data_dict[name].append(cui)
                # data.append((name,cui))

        # LOGGER.info("concerting loaded dictionary data to numpy array...")
        # data = np.array(data)
        data = [(name, "|".join(cuis)) for name, cuis in data_dict.items()]
        return data

class SapBertDictionaryDataset(Dataset):
    def __init__(self, dictionary_path: str):
        logger.info("Loading from the Alias Mapping of the Dataset")
        self.data = self.read_examples(dictionary_path)

    
    def read_examples(self, dictionary_path):
        """
        Read examples from the file
        """
        with open(dictionary_path, "r") as f:
            lines = f.read().split("\n")
        
        # Construt the UMLS Dict for finetuning the model
        umls_dict = {}
        for line in tqdm(lines, desc="Reading examples from the file"):
            if not line or len(line.split("||")) != 2:
                continue
            cui, name = line.split("||")
            name = name.lower()
            if cui in umls_dict:
                umls_dict[cui].add(name)
            else:
                umls_dict[cui] = set([name])
        
        for key, value in umls_dict.items():
            umls_dict[key] = list(value)
        
        return umls_dict
        

        
    
class SapBertBigBioDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "medmentions_st21pv",
        splits_to_include: List[str] = ["train"],
        resolve_abbreviations: bool = True,
        path_to_abbreviation_dict: Optional[str] = None,
    ):
        """
        Load initial BigBio dataset
        """
        # Pull in data
        self.data = load_bigbio_dataset(dataset_name)
        self.splits_to_include = splits_to_include

        # Resolve abbreviations if desired
        self.resolve_abbreviations = resolve_abbreviations
        if self.resolve_abbreviations:
            self.data = add_deabbreviations(self.data, path_to_abbreviation_dict)
        #     self.abbreviations = ujson.load(open(path_to_abbreviation_dict, "r"))

        self.cuis_to_exclude = CUIS_TO_EXCLUDE[dataset_name]
        self.cuis_to_remap = CUIS_TO_REMAP[dataset_name]

        # Put examples into list for retrieval
        self._data_to_flat_instances()



    def _data_to_flat_instances(self):
        """
        Convert dataset into flat set of positive Examples to use with dataloader
        """
        df = dataset_to_df(
            self.data,
            self.splits_to_include,
            entity_remapping_dict=self.cuis_to_remap,
            cuis_to_exclude=self.cuis_to_exclude,
        )
        # if self.resolve_abbreviations:
        #     df["text"] = df[["document_id", "text"]].apply(
        #         lambda x: resolve_abbreviation(x[0], x[1], self.abbreviations), axis=1
        #     )
        self.flat_instances = df.to_dict(orient="records")

    

    def __len__(self):
        return len(self.flat_instances)

    def __getitem__(self, idx):
        return self.flat_instances[idx]


# def sapbert_collate_fn(batch):
#     mentions = [x["text"] for x in batch]
#     # labels = [x["cuis"] for x in batch]
#     labels = [x["db_ids"] for x in batch]
#     metadata = batch

#     return mentions, labels, metadata


class MetricLearningDataset_pairwise(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(
            self, 
            training_pairs: List[str], 
            tokenizer): #d_ratio, s_score_matrix, s_candidate_idxs):
        self.query_ids = []
        self.query_names = []
        for i, line in tqdm(enumerate(training_pairs), desc="Loading training pairs"):
            line = line.rstrip("\n")
            if (len(line.split("||")) == 3):
                query_id, name1, name2 = line.split("||")
                self.query_ids.append(query_id)
                self.query_names.append((name1, name2))
        print("Lengths of the dataset: ", len(self.query_ids), len(self.query_names))
        self.tokenizer = tokenizer
        self.query_id_2_index_id = {k: v for v, k in enumerate(list(set(self.query_ids)))}
    
    def __getitem__(self, query_idx):

        query_name1 = self.query_names[query_idx][0]
        query_name2 = self.query_names[query_idx][1]
        query_id = self.query_ids[query_idx]
        query_id = int(self.query_id_2_index_id[query_id])

        return query_name1, query_name2, query_id


    def __len__(self):
        return len(self.query_names)



class MetricLearningDataset(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, path, tokenizer): #d_ratio, s_score_matrix, s_candidate_idxs):
        logger.info("Initializing metric learning data set! ...")
        with open(path, 'r') as f:
            lines = f.readlines()
        self.query_ids = []
        self.query_names = []
        cuis = []
        for line in lines:
            cui, _ = line.split("||")
            cuis.append(cui)

        self.cui2id = {k: v for v, k in enumerate(cuis)}
        for line in lines:
            line = line.rstrip("\n")
            cui, name = line.split("||")
            query_id = self.cui2id[cui]
            #if query_id.startswith("C"):
            #    query_id = query_id[1:]
            #query_id = int(query_id)
            self.query_ids.append(query_id)
            self.query_names.append(name)
        self.tokenizer = tokenizer
    
    def __getitem__(self, query_idx):

        query_name = self.query_names[query_idx]
        query_id = self.query_ids[query_idx]
        query_token = self.tokenizer.transform([query_name])[0]

        return torch.tensor(query_token), torch.tensor(query_id)

    def __len__(self):
        return len(self.query_names)


if __name__ == "__main__":
    dataset = SapBertBigBioDataset(
        path_to_abbreviation_dict="/home/pbathala3/entity_linking/biomedical-entity-linking/bioel/bioel/utils/solve_abbreviation/abbreviations.json"
    )
    #dataloader = DataLoader(dataset, collate_fn=sapbert_collate_fn, batch_size=1)
    # for i, batch in enumerate(tqdm(dataloader)):
    #     if i < 1:
    #         print(batch)
    
    ontology_config = {
        "filepath": "/mitchell/entity-linking/2022AA/META/",
        "name": "UMLS",
        "abbrev": None,
    }
    ontology = BiomedicalOntology.load_umls(**ontology_config)
    pos_pairs = generate_positive_pairs(ontology.entities)
    for i, pair in enumerate(tqdm(pos_pairs)):
        if i < 1:
            print(pair)