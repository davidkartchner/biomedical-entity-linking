from typing import List, Optional, Union, Dict
import time
import logging
import json
from dataclasses import dataclass
import ujson

import torch
from torch import Tensor as T
from transformers import PreTrainedTokenizer
from bioel.utils.bigbio_utils import (
    load_bigbio_dataset,
    add_deabbreviations,
    dataset_to_df,
    resolve_abbreviation,
    dataset_to_documents,
    get_left_context,
    get_right_context,
)
from bioel.utils.dataset_consts import (
    DATASET_NAMES,
    CUIS_TO_EXCLUDE,
    CUIS_TO_REMAP,
)
from tqdm.auto import tqdm

tqdm.pandas()


logger = logging.getLogger()


@dataclass
class Mention:
    cui: str
    start: int
    end: int
    text: str
    types: List[str]
    deabbreviated_text: Optional[str]


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        splits: Optional[List[str]] = ["train", "valid", "test"],
        abbreviations_path=None,
    ):
        """
        Parameters
        ---------
        - dataset_name: str
        Name of the dataset to load. Example : "bc5cdr", "ncbi-disease", etc...
        - splits: List[str]
        List of splits to load. Default: ["train", "valid", "test"]
        - abbreviations_path: str
        Path to the abbreviations dictionary. Default: None
        """
        self.dataset_name = dataset_name
        self.data = load_bigbio_dataset(dataset_name)
        self.splits = splits
        self.abbreviations_path = abbreviations_path
        self.name_to_cuis = {}

        exclude = CUIS_TO_EXCLUDE[dataset_name]
        remap = CUIS_TO_REMAP[dataset_name]

        if self.abbreviations_path:
            self.data = add_deabbreviations(
                dataset=self.data, path_to_abbrev=self.abbreviations_path
            )
            print("Resolved abbreviations")
        else:
            print("No abbreviations dictionary found.")

        df = dataset_to_df(
            self.data, cuis_to_exclude=exclude, entity_remapping_dict=remap
        )
        df["start"] = df["offsets"].map(lambda x: x[0][0])
        df["end"] = df["offsets"].map(lambda x: x[-1][-1])
        # df = df[df.split.isin(splits)]
        # print(dataset_name, splits, df.split.unique())

        self.df = df

        self.documents = dataset_to_documents(self.data)


if __name__ == "__main__":
    # Testss
    abbreviations_path = "/home2/cye73/data_test/abbreviations.json"
    bc5cdr = Dataset("bc5cdr", ["train", "validation", "test"], abbreviations_path)
    bc5cdr.data
