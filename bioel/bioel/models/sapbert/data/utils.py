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

from torch.utils.data import Dataset, DataLoader
from bioel.utils.bigbio_utils import (
    load_bigbio_dataset,
    dataset_to_df,
    DATASET_NAMES,
    CUIS_TO_EXCLUDE,
    CUIS_TO_REMAP,
    resolve_abbreviation,
)

from bioel.ontology import BiomedicalOntology
from bioel.logger import setup_logger

logger = setup_logger()


"""
Generate Pretraining Data for SAPBERT from the UMLS Ontology
"""
def gen_pairs(input_list: List) -> List:
    """
    Generate all possible pairs from a list of items.
    """
    return list(itertools.combinations(input_list, r = 2))

def generate_pretraining_data(ontology: BiomedicalOntology):
    """
    Generate pretraining data from UMLS for the SAPBERT model.
    """
    

    pos_pairs = []
    for k, v in tqdm(ontology.entities.items(), desc = "Generating pretraining data for SAPBERT"):
        pairs = gen_pairs(v.aliases)
        if len(pairs) > 50: # If > 50 pairs, sample 50 pairs
            pairs = random.sample(pairs, 50)
        for p in pairs:
            print(k, p[0], p[1])
            line = str(k) + "||" + p[0] + "||" + p[1]
            pos_pairs.append(line)
    
    # Save the pos_pairs into a .txt file
    with open('./training_file_umls_2022AA_en_uncased_no_dup_pairwise_pair_th50.txt', 'w') as file:
        for pair in pos_pairs:
            file.write(pair + '\n')
    
    return pos_pairs

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
            self.abbreviations = ujson.load(open(path_to_abbreviation_dict, "r"))

        self.cuis_to_exclude = CUIS_TO_EXCLUDE[dataset_name]
        self.cuis_to_remap = CUIS_TO_REMAP[dataset_name]

        # Put examples into list for retrieval
        self._data_to_flat_instances()


    def _data_to_flat_instances(self):
        """
        Convert dataset into flat set of examples to use with dataloader
        """
        df = dataset_to_df(
            self.data,
            self.splits_to_include,
            entity_remapping_dict=self.cuis_to_remap,
            cuis_to_exclude=self.cuis_to_exclude,
        )
        if self.resolve_abbreviations:
            df["text"] = df[["document_id", "text"]].apply(
                lambda x: resolve_abbreviation(x[0], x[1], self.abbreviations), axis=1
            )
        self.flat_instances = df.to_dict(orient="records")

    def __len__(self):
        return len(self.flat_instances)

    def __getitem__(self, idx):
        return self.flat_instances[idx]


def sapbert_collate_fn(batch):
    mentions = [x["text"] for x in batch]
    # labels = [x["cuis"] for x in batch]
    labels = [x["db_ids"] for x in batch]
    metadata = batch

    return mentions, labels, metadata


class MetricLearningDataset_pairwise(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, path, tokenizer): #d_ratio, s_score_matrix, s_candidate_idxs):
        with open(path, 'r') as f:
            lines = f.readlines()
        self.query_ids = []
        self.query_names = []
        for line in lines:
            line = line.rstrip("\n")
            query_id, name1, name2 = line.split("||")
            self.query_ids.append(query_id)
            self.query_names.append((name1, name2))
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
    dataloader = DataLoader(dataset, collate_fn=sapbert_collate_fn, batch_size=64)
    for i, batch in enumerate(tqdm(dataloader)):
        if i < 3:
            print(batch)
    
    # ontology_config = {
    #     "filepath": "/mitchell/entity-linking/2022AA/META/",
    #     "name": "UMLS",
    #     "abbrev": None,
    # }
    # ontology = BiomedicalOntology.load_umls(**ontology_config)
    # print(generate_pretraining_data(ontology))