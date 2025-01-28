import os
import ujson
import warnings
from collections import defaultdict
import pandas as pd

# from bigbio.dataloader import BigBioConfigHelpers
import json
import joblib
import numpy as np
import datetime

from tqdm.auto import tqdm

tqdm.pandas()

from datasets import load_dataset
from bioel.utils.bigbio_utils import (
    dataset_to_documents,
    dataset_to_df,
    resolve_abbreviation,
    load_bigbio_dataset,
)
from bioel.utils.dataset_consts import (
    CUIS_TO_REMAP,
    CUIS_TO_EXCLUDE,
    VALIDATION_DOCUMENT_IDS,
)
from sklearn.feature_extraction.text import TfidfVectorizer

from bioel.ontology import BiomedicalOntology
from typing import List


def cuis_to_aliases(ontology: BiomedicalOntology, save_dir: str, dataset_name: str):
    """
    Creates the .txt file that maps the Cuis to the aliases and canonical name for each entities
    from the provided ontology.

    Parameters
    ----------
    ontology_dir: str, required.
        The path where the ontology is stored.
    save_dir : str, required.
        The path where the mapping file "{dataset_name}_aliases.txt" will be stored
    dataset_name : str, required.
        The name of the dataset for which the mapping will be created.
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    import re

    file_path = os.path.join(save_dir, f"{dataset_name}_aliases.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        for cui, entity in ontology.entities.items():
            file.write(f"{cui}||{entity.name}\n")
            if len(entity.aliases) != 0:
                if isinstance(entity.aliases, list):
                    for word in entity.aliases:
                        if word is not None:
                            # print(word)
                            word = word.strip()
                            file.write(f"{cui}||{word}\n")
                else:
                    # words = entity.aliases.split('; , ')
                    words = re.split("[;|]", entity.aliases)
                    for word in words:
                        word = word.strip()
                        file.write(f"{cui}||{word}\n")
            if entity.equivalant_cuis:
                for eqcui in entity.equivalant_cuis:
                    if eqcui != entity.cui:
                        file.write(f"{eqcui}||{entity.name}\n")
                        if entity.aliases:
                            words = re.split("[;|]", entity.aliases)
                            for word in words:
                                word = word.strip()
                                file.write(f"{eqcui}||{word}\n")

    return file_path


if __name__ == "__main__":
    # data_preprocess(
    #     dataset_name = "bc5cdr",
    #     save_dir = "data2/",
    #     ontology_dir = "Y:/mitchell/entity-linking/2017AA/META/",
    #     ontology_name = "mesh",
    #     resolve_abbrevs=False)
    cuis_to_aliases(
        ontology=BiomedicalOntology.load_medic(
            filepath="/mitchell/entity-linking/kbs/medic.tsv", name="medic"
        ),
        save_dir="/home2/cye73/data_test2/sapbert/ncbi_disease/",
        dataset_name="ncbi_disease",
    )
    # cuis_to_aliases(
    #     ontology=BiomedicalOntology.load_entrez(
    #         filepath="/mitchell/entity-linking/el-robustness-comparison/data/gene_info.tsv",
    #         dataset="gnormplus",
    #     ),
    #     save_dir="/home2/cye73/data_test2/sapbert/ncbi_disease/",
    #     dataset_name="gnormplus",
    # )
    # cuis_to_aliases(
    #     ontology=BiomedicalOntology.load_entrez(
    #         filepath="/mitchell/entity-linking/el-robustness-comparison/data/gene_info.tsv",
    #         dataset="nlm_gene",
    #     ),
    #     save_dir="/home2/cye73/data_test2/sapbert/ncbi_disease/",
    #     dataset_name="nlm_gene",
    # )
    # cuis_to_aliases(
    #     ontology=BiomedicalOntology.load_mesh(
    #         filepath="/mitchell/entity-linking/2017AA/META/", name="mesh"
    #     ),
    #     save_dir="/home2/cye73/data_test2/sapbert/ncbi_disease/",
    #     dataset_name="bc5cdr",
    # )
    # cuis_to_aliases(
    #     ontology=BiomedicalOntology.load_mesh(
    #         filepath="/mitchell/entity-linking/2017AA/META/", name="mesh"
    #     ),
    #     save_dir="/home2/cye73/data_test2/sapbert/ncbi_disease/",
    #     dataset_name="nlm_chem",
    # )

    # cuis_to_aliases(
    #     ontology=BiomedicalOntology.load_umls(
    #         filepath="/mitchell/entity-linking/2017AA/META/", name="umls"
    #     ),
    #     save_dir="/home2/cye73/data_test2/sapbert/ncbi_disease/",
    #     dataset_name="medmentions_full",
    # )

    # cuis_to_aliases(
    #     ontology=BiomedicalOntology.load_umls(
    #         filepath="/mitchell/entity-linking/2017AA/META/", name="umls"
    #     ),
    #     save_dir="/home2/cye73/data_test2/sapbert/ncbi_disease/",
    #     dataset_name="medmentions_st21pv",
    # )

    # onto = BiomedicalOntology.load_entrez(
    #     filepath="/mitchell/entity-linking/el-robustness-comparison/data/gene_info.tsv",
    #     dataset="gnormplus",
    # )
