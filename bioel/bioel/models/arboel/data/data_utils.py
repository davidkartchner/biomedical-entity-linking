import pickle
import ujson
import sys
import os

import csv

import numpy as np

from tqdm import tqdm
from typing import Optional

from bioel.utils.umls_utils import UmlsMappings
from bioel.utils.bigbio_utils import (
    CUIS_TO_REMAP,
    CUIS_TO_EXCLUDE,
    VALIDATION_DOCUMENT_IDS,
    dataset_to_documents,
    dataset_to_df,
)

from bioel.utils.bigbio_utils import load_bigbio_dataset, resolve_abbreviation
from bioel.ontology import BiomedicalOntology


def process_medic_ontology(ontology, data_path, ontology_dir):
    """
    This function prepares the entity data : dictionary.pickle

    Parameters
    ----------
    - ontology : str (only umls for now)
        Ontology associated with the dataset
    - data_path : str
        Path where to load and save dictionary.pickle
    - ontology_dir : str
        Path to medic ontology
    """

    # Use the class method to load the MEDIC ontology and get a new instance of BiomedicalOntology
    ontology = BiomedicalOntology.load_medic(filepath=ontology_dir, name="medic")

    # Check if equivalent CUIs are present for the first entity
    first_entity_cui = next(iter(ontology.entities))
    equivalent_cuis = bool(ontology.entities[first_entity_cui].equivalant_cuis)

    "If dictionary already processed, load it else process and load it"
    entity_dictionary_pkl_path = os.path.join(data_path, "dictionary.pickle")

    if os.path.isfile(entity_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(entity_dictionary_pkl_path, "rb") as read_handle:
            entities = pickle.load(read_handle)

        return entities, equivalent_cuis

    ontology_entities = []
    for cui, entity in tqdm(ontology.entities.items()):
        if entity.aliases != "":
            if entity.definition != "":
                new_entity = {
                    "type": "Disease",
                    "cui": entity.cui,
                    "title": entity.name,
                    "cuis": entity.equivalant_cuis,
                    "description": f"{entity.name} ( Disease : {entity.aliases} ) [{entity.definition}]",
                }
            else:
                new_entity = {
                    "type": "Disease",
                    "cui": entity.cui,
                    "title": entity.name,
                    "cuis": entity.equivalant_cuis,
                    "description": f"{entity.name} ( Disease : {entity.aliases} )",
                }

        else:
            if entity.definition != "":
                new_entity = {
                    "type": "Disease",
                    "cui": entity.cui,
                    "title": entity.name,
                    "cuis": entity.equivalant_cuis,
                    "description": f"{entity.name} ( Disease) [{entity.definition}]",
                }
            else:
                new_entity = {
                    "type": "Disease",
                    "cui": entity.cui,
                    "title": entity.name,
                    "cuis": entity.equivalant_cuis,
                    "description": f"{entity.name} ( Disease)",
                }
        ontology_entities.append(new_entity)

    # Check if the directory exists, and create it if it does not
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Save entities to pickle file
    with open(os.path.join(data_path, "dictionary.pickle"), "wb") as f:
        pickle.dump(ontology_entities, f)

    entities = pickle.load(open(os.path.join(data_path, "dictionary.pickle"), "rb"))
    return entities, equivalent_cuis


# Function for retrieving the title + description of entities in obo_ontology
def process_obo_ontology(ontology, data_path, obo_dict):
    """
    This function prepares the entity data : dictionary.pickle

    Parameters
    ----------
    - ontology : str (only umls for now)
        Ontology associated with the dataset
    - data_path : str
        Path where to load and save dictionary.pickle
    - obo_dict : dict
        dictionary with the parameter expected by load_obo method :
        filepath, name, prefix_to_keep, entity_type, abbrev
    """

    ontology = BiomedicalOntology.load_obo(**obo_dict)

    equivalent_cuis = False
    if ontology.entities[0].equivalant_cuis is not None:
        equivalent_cuis = True

    "If dictionary already processed, load it else process and load it"
    entity_dictionary_pkl_path = os.path.join(data_path, "dictionary.pickle")

    if os.path.isfile(entity_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(entity_dictionary_pkl_path, "rb") as read_handle:
            entities = pickle.load(read_handle)

        return entities, equivalent_cuis

    ontology_entities = []
    for entity in tqdm(ontology.entities):
        if entity.aliases != "":
            if entity.definition != "":
                new_entity = {
                    "type": entity.types,
                    "cui": entity.cui,
                    "title": entity.name,
                    "description": f"{entity.name} ( {entity.types} : {entity.aliases} ) [{entity.definition}]",
                }
            else:
                new_entity = {
                    "type": entity.types,
                    "cui": entity.cui,
                    "title": entity.name,
                    "description": f"{entity.name} ( {entity.types} : {entity.aliases} )",
                }

        else:
            if entity.definition != "":
                new_entity = {
                    "type": entity.types,
                    "cui": entity.cui,
                    "title": entity.name,
                    "description": f"{entity.name} ( {entity.types} ) [{entity.definition}]",
                }
            else:
                new_entity = {
                    "type": entity.types,
                    "cui": entity.cui,
                    "title": entity.name,
                    "description": f"{entity.name} ( {entity.types} )",
                }
        ontology_entities.append(new_entity)

    # Check if the directory exists, and create it if it does not
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Save entities to pickle file
    with open(os.path.join(data_path, "dictionary.pickle"), "wb") as f:
        pickle.dump(ontology_entities, f)

    entities = pickle.load(open(os.path.join(data_path, "dictionary.pickle"), "rb"))
    return entities, equivalent_cuis


# Function for retrieving the title + description of entities of umls_ontology
def process_umls_ontology(ontology, data_path, ontology_dir):
    """
    This function prepares the entity data : dictionary.pickle

    Parameters
    ----------
    - ontology : str (only umls for now)
        Ontology associated with the dataset
    - data_path : str
        Path where to load and save dictionary.pickle
    - ontology_dir : str
        Path to UMLS ontology
    """

    # Use the class method to load the MEDIC ontology and get a new instance of BiomedicalOntology
    ontology = BiomedicalOntology.load_umls(filepath=ontology_dir, name="umls")

    # Check if equivalent CUIs are present for the first entity
    first_entity_cui = next(iter(ontology.entities))
    equivalent_cuis = bool(ontology.entities[first_entity_cui].equivalant_cuis)

    "If dictionary already processed, load it else process and load it"
    entity_dictionary_pkl_path = os.path.join(data_path, "dictionary.pickle")

    if os.path.isfile(entity_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(entity_dictionary_pkl_path, "rb") as read_handle:
            entities = pickle.load(read_handle)

        return entities, equivalent_cuis

    ontology_entities = []
    for cui, entity in tqdm(ontology.entities.items()):
        if entity.aliases != "":
            if entity.definition != "":
                new_entity = {
                    "type": entity.types[0],
                    "cui": entity.cui,
                    "title": entity.name,
                    "description": f"{entity.name} ( {entity.types[0]} : {entity.aliases} ) [{entity.definition}]",
                }
            else:
                new_entity = {
                    "type": entity.types[0],
                    "cui": entity.cui,
                    "title": entity.name,
                    "description": f"{entity.name} ( {entity.types[0]} : {entity.aliases} )",
                }

        else:
            if entity.definition != "":
                new_entity = {
                    "type": entity.types[0],
                    "cui": entity.cui,
                    "title": entity.name,
                    "description": f"{entity.name} ( {entity.types[0]} ) [{entity.definition}]",
                }
            else:
                new_entity = {
                    "type": entity.types[0],
                    "cui": entity.cui,
                    "title": entity.name,
                    "description": f"{entity.name} ( {entity.types[0]} )",
                }
        ontology_entities.append(new_entity)

    # Check if the directory exists, and create it if it does not
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Save entities to pickle file
    with open(os.path.join(data_path, "dictionary.pickle"), "wb") as f:
        pickle.dump(ontology_entities, f)

    entities = pickle.load(open(os.path.join(data_path, "dictionary.pickle"), "rb"))
    return entities, equivalent_cuis


def process_mesh_ontology(ontology, data_path, ontology_dir):
    """
    This function prepares the entity data : dictionary.pickle

    Parameters
    ----------
    - ontology : str (only umls for now)
        Ontology associated with the dataset
    - data_path : str
        Path where to load and save dictionary.pickle
    - ontology_dir : str
        Path to medic ontology
    """

    ontology = BiomedicalOntology.load_mesh(filepath=ontology_dir, name="mesh")

    # Check if equivalent CUIs are present for the first entity
    first_entity_cui = next(iter(ontology.entities))
    equivalent_cuis = bool(ontology.entities[first_entity_cui].equivalant_cuis)

    "If dictionary already processed, load it else process and load it"
    entity_dictionary_pkl_path = os.path.join(data_path, "dictionary.pickle")

    if os.path.isfile(entity_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(entity_dictionary_pkl_path, "rb") as read_handle:
            entities = pickle.load(read_handle)

        return entities, equivalent_cuis

    ontology_entities = []
    for cui, entity in tqdm(ontology.entities.items()):
        if entity.aliases != "":
            if entity.definition != "":
                new_entity = {
                    "type": "Disease",
                    "cui": entity.cui,
                    "title": entity.name,
                    "description": f"{entity.name} ( Disease : {entity.aliases} ) [{entity.definition}]",
                }
            else:
                new_entity = {
                    "type": "Disease",
                    "cui": entity.cui,
                    "title": entity.name,
                    "description": f"{entity.name} ( Disease : {entity.aliases} )",
                }

        else:
            if entity.definition != "":
                new_entity = {
                    "type": "Disease",
                    "cui": entity.cui,
                    "title": entity.name,
                    "description": f"{entity.name} ( Disease) [{entity.definition}]",
                }
            else:
                new_entity = {
                    "type": "Disease",
                    "cui": entity.cui,
                    "title": entity.name,
                    "description": f"{entity.name} ( Disease)",
                }
        ontology_entities.append(new_entity)

    # Check if the directory exists, and create it if it does not
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Save entities to pickle file
    with open(os.path.join(data_path, "dictionary.pickle"), "wb") as f:
        pickle.dump(ontology_entities, f)

    entities = pickle.load(open(os.path.join(data_path, "dictionary.pickle"), "rb"))
    return entities, equivalent_cuis


# Function for preparing the mentions in the dataset into the right format for each model
def process_mention_dataset(
    ontology,
    dataset,
    data_path,
    resolve_abbrevs=False,
    path_to_abbrev=None,
    obo_dict: Optional[dict] = None,
    ontology_dir: Optional[str] = None,
):
    """
    This function prepares the mentions data :  Creates the train.jsonl, valid.jsonl, test.jsonl
    Each .jsonl contains data in the following format :
    {'mention': mention,
    'mention_id': ID of the mention, (optional)
    'context_left': context before mention,
    'context_right': context after mention,
    'context_doc_id': ID of the doc, (optional)
    'type': type
    'label_id': label_id,
    'label': entity description, (optional)
    'label_title': entity title

    Parameters
    ----------
    - ontology : str
    Ontology associated with the dataset
    - dataset : str
    Name of the dataset
    - data_path : str
    Path where to load and save dictionary.pickle
    - ontology_dir : str
    Path to the ontology (umls, medic etc...)
    """
    data = load_bigbio_dataset(
        dataset_name=dataset, abbrev=resolve_abbrevs, path_to_abbrev=path_to_abbrev
    )
    exclude = CUIS_TO_EXCLUDE[dataset]
    remap = CUIS_TO_REMAP[dataset]

    if ontology == "obo":
        entities, equivalant_cuis = process_obo_ontology(ontology, data_path, obo_dict)
    elif ontology == "medic":
        entities, equivalant_cuis = process_medic_ontology(
            ontology, data_path, ontology_dir
        )
    elif ontology == "umls":
        entities, equivalant_cuis = process_umls_ontology(
            ontology, data_path, ontology_dir
        )
    elif ontology == "mesh":
        entities, equivalant_cuis = process_mesh_ontology(
            ontology, data_path, ontology_dir
        )
    else:
        print("ERROR!")

    entity_dictionary = {d["cui"]: d for d in tqdm(entities)}  # CC1

    # For ontology with multiples cuis
    if equivalant_cuis:
        # Need to redo this since we have multiple synonymous CUIs for ncbi_disease
        entity_dictionary = {cui: d for d in tqdm(entities) for cui in d["cuis"]}
        cui_synsets = {}
        for subdict in tqdm(entities):
            for cui in subdict["cuis"]:
                if cui in subdict:
                    print(cui, cui_synsets[cui], subdict["cuis"])
                cui_synsets[cui] = subdict["cuis"]
        with open(os.path.join(data_path, "cui_synsets.json"), "w") as f:
            f.write(ujson.dumps(cui_synsets, indent=2))

    if dataset in VALIDATION_DOCUMENT_IDS:
        validation_pmids = VALIDATION_DOCUMENT_IDS[dataset]
    else:
        print("ERROR!!!")

    # Convert BigBio dataset to pandas DataFrame
    df = dataset_to_df(
        data,
        entity_remapping_dict=remap,
        cuis_to_exclude=exclude,
        val_split_ids=validation_pmids,
    )
    # Return dictionary of documents in BigBio dataset
    docs = dataset_to_documents(data)
    label_len = df["db_ids"].map(lambda x: len(x)).max()
    print("Max labels on one doc:", label_len)

    abbrev_dict = ujson.load(open(path_to_abbrev))

    for split in df.split.unique():
        ents_in_split = []
        for d in tqdm(
            df.query("split == @split").to_dict(orient="records"),
            desc=f"Creating correct mention format for {split} dataset",
        ):
            abbrev_resolved = False
            offsets = d["offsets"]
            doc_id = d["document_id"]
            doc = docs[doc_id]
            mention = d["text"]

            # Resolve abbreviaions if desired
            if resolve_abbrevs and abbrev_dict is not None:
                deabbreviated_mention = resolve_abbreviation(
                    doc_id, mention, abbrev_dict
                )
                abbrev_resolved = True

            # Get offsets and context
            start = offsets[0][0]  # start on the mention
            end = offsets[-1][-1]  # end of the mention
            before_context = doc[:start]  # left context
            after_context = doc[end:]  # right context

            # ArboEL can't handle multi-labels, so we randomly choose one.
            if len(d["db_ids"]) == 1:
                label_id = d["db_ids"][0]

            # For ontology with multiples cuis
            elif equivalant_cuis:
                labels = []
                used_cuis = set([])
                choosable_ids = []
                for db_id in d["db_ids"]:
                    if db_id in used_cuis:
                        continue
                    else:
                        used_cuis.update(set(entity_dictionary[db_id]["cuis"]))
                    choosable_ids.append(db_id)

                label_id = np.random.choice(choosable_ids)

            else:
                label_id = np.random.choice(d["db_ids"])

            # Check if we missed something
            if label_id not in entity_dictionary:
                # print(label_id)
                continue

            output = [
                {
                    "mention": mention,
                    "mention_id": d["mention_id"],
                    "context_left": before_context,
                    "context_right": after_context,
                    "context_doc_id": doc_id,
                    "type": d["type"][0],
                    "label_id": label_id,
                    "label": entity_dictionary[label_id]["description"],
                    "label_title": entity_dictionary[label_id]["title"],
                }
            ]

            if abbrev_resolved:
                output.append(
                    {
                        "mention": deabbreviated_mention,
                        "mention_id": d["mention_id"] + ".abbr_resolved",
                        "context_left": before_context,
                        "context_right": after_context,
                        "context_doc_id": doc_id,
                        "type": d["type"][0],
                        "label_id": label_id,
                        "label": entity_dictionary[label_id]["description"],
                        "label_title": entity_dictionary[label_id]["title"],
                    }
                )

            ents_in_split.extend(output)

        split_name = split
        if split == "validation":
            split_name = "valid"
        with open(os.path.join(data_path, f"{split_name}.jsonl"), "w") as f:
            f.write("\n".join([ujson.dumps(x) for x in ents_in_split]))
