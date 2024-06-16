import os
import sys
import numpy as np
from tqdm import tqdm
import pickle
import json
import argparse
import pickle
import pandas as pd
from transformers import BartTokenizer, AutoTokenizer
from bioel.models.biogenel.trie import Trie

sys.setrecursionlimit(8000)


def create_trie(data_dir: str, use_biobart_tokenizer=False):
    """
    Function that creates the Trie given a repository director containing the target knowledge base (ontology) and a tokenizer
    """
    if use_biobart_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-v2-large")
    else:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    with open(f"{data_dir}/target_kb.json", "r") as f:
        cui2str = json.load(f)

    entities = []
    for cui in cui2str:
        entities += cui2str[cui]

    trie = Trie(
        [16] + list(tokenizer(" " + entity.lower())["input_ids"][1:])
        for entity in tqdm(entities)
    ).trie_dict

    if use_biobart_tokenizer:
        with open(f"{data_dir}/biobart_trie.pkl", "wb") as w_f:
            pickle.dump(trie, w_f)
    else:
        with open(f"{data_dir}/trie.pkl", "wb") as w_f:
            pickle.dump(trie, w_f)

    print("finish running!")
