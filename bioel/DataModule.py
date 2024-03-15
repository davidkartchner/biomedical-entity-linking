import pickle
import ujson
import sys
import os

import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from typing import Optional, Union

from bigbio.dataloader import BigBioConfigHelpers

sys.path.append('..')
from umls_utils import UmlsMappings
from bigbio_utils import CUIS_TO_REMAP, CUIS_TO_EXCLUDE, DATASET_NAMES, VALIDATION_DOCUMENT_IDS
from bigbio_utils import dataset_to_documents, dataset_to_df, resolve_abbreviation, get_left_context, get_right_context
from bioel.ontology import BiomedicalOntology



conhelps = BigBioConfigHelpers()

# utilities functions for medic ontology
def medic_get_canonical_name(entities):
    '''
    Get name of entities in the ontology
    data: list of dict
    '''
    canonical_names = {entity['DiseaseID']: entity['DiseaseName'] for entity in entities}
    return canonical_names

def medic_get_aliases(entities):
    '''
    Get aliases of entities in the ontology
    data: list of dict
    '''
    aliases = {entity['DiseaseID']: entity['Synonyms'] for entity in entities}
    return aliases

def medic_get_definition(entities):
    '''
    Get definition of entities in the ontology
    data: list of dict
    '''
    definitions_dict = {entity['DiseaseID']: entity['Definition'] for entity in entities if entity['Definition'] is not None}
    return definitions_dict

def medic_get_types(entities):
    '''
    Get type of entities in the ontology
    data: list of dict
    '''
    # Extract tuples of CUI and types from the Data
    types = {entity['DiseaseID']: entity['SlimMappings'] for entity in entities}
    return types

# Function for retrieving the title + description of entities in medic_ontology
def process_medic_ontology(ontology, 
                        data_path,
                        ):
    '''
    This function prepares the entity data : dictionary.pickle
    
    Parameters 
    ----------
    - ontology : str (only umls for now)
    Ontology associated with the dataset
    - data_path : str
    Path where to load and save dictionary.pickle
    '''
    
    # Get canonical name of entities in the ontology
    cui2name = medic_get_canonical_name(ontology)
    # Get aliases of entities in the ontology
    cui2alias = medic_get_aliases(ontology)
    # Get definition of entities in the ontology
    cui2definition = medic_get_definition(ontology)
    # Get types of entities in the ontology
    cui2tui = medic_get_types(ontology)


    # Check if the directory exists, and create it if it does not
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    ontology_entities = []
    for cui, name in tqdm(cui2name.items()):
        d = {}
        ent_type = cui2tui[cui]
        d['type'] = ent_type
        # other_aliases = [x for x in cui2alias[cui] if x != name]
        # joined_aliases = ' ; '.join(other_aliases)
        d['cui'] = f"{cui}"
        d['title'] = name
        if cui2definition[cui] != "":
            definition = cui2definition[cui]
        else:
            definition = None

        if cui2alias[cui] is not None:
            if definition is not None:
                d['description'] = f"{name} ( {ent_type} : {cui2alias[cui]} ) [ {definition} ]"
            else:
                d['description'] = f"{name} ( {ent_type} : {cui2alias[cui]} )"
        else:
            if definition is not None:
                d['description'] = f"{name} ( {ent_type} ) [ {definition} ]"
            else:
                d['description'] = f"{name} ( {ent_type} )"

        ontology_entities.append(d)

    pickle.dump(ontology_entities, open(os.path.join(data_path, 'dictionary.pickle'), 'wb'))
    entities = pickle.load(open(os.path.join(data_path, 'dictionary.pickle'), 'rb'))
    return entities


# Function for retrieving the title + description of entities in obo_ontology
def process_obo_ontology(ontology, 
                     data_path):
    '''
    This function prepares the entity data : dictionary.pickle
    
    Parameters 
    ----------
    - ontology : str (only umls for now)
    Ontology associated with the dataset
    - data_path : str
    Path where to load and save dictionary.pickle
    '''
    
    # Get canonical name of entities in the ontology
    cui2name = ontology.get_canonical_name()
    # Get aliases of entities in the ontology
    cui2alias = ontology.get_aliases()
    # Get definition of entities in the ontology
    cui2definition = ontology.get_definition()
    # Get types of entities in the ontology
    cui2tui = ontology.get_types()


    # Check if the directory exists, and create it if it does not
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    ontology_entities = []
    for cui, name in tqdm(cui2name.items(), desc="Creating correct entities format"):
        d = {}
        ent_type = cui2tui[cui]
        d['type'] = ent_type
        other_aliases = [x for x in cui2alias[cui] if x != name]
        joined_aliases = ' ; '.join(other_aliases)
        d['cui'] = f"{cui}"
        d['title'] = name
        if cui in cui2definition:
            definition = cui2definition[cui]
        else:
            definition = None

        if len(other_aliases) > 0:
            if definition is not None:
                d['description'] = f"{name} ( {ent_type} : {joined_aliases} ) [ {definition} ]"
            else:
                d['description'] = f"{name} ( {ent_type} : {joined_aliases} )"
        else:
            if definition is not None:
                d['description'] = f"{name} ( {ent_type} ) [ {definition} ]"
            else:
                d['description'] = f"{name} ( {ent_type} )"

        ontology_entities.append(d)

    pickle.dump(ontology_entities, open(os.path.join(data_path, 'dictionary.pickle'), 'wb'))
    entities = pickle.load(open(os.path.join(data_path, 'dictionary.pickle'), 'rb'))
    return entities


# Function for retrieving the title + description of entities of umls_ontology
def process_umls_ontology(ontology, # not used
                     data_path,
                     umls_dir):
    '''
    This function prepares the entity data : dictionary.pickle
    
    Parameters 
    ----------
    - ontology : str
    Ontology associated with the dataset
    - data_path : str
    Path where to load and save dictionary.pickle
    - umls_dir : str
    Path to umls data
    '''
    umls = UmlsMappings(
    umls_dir=umls_dir, debug=False, force_reprocess=False,
    )
    
    umls_to_name = umls.get_canonical_name(ontologies_to_include="all",
    use_umls_curies=True,
    lowercase=False)
    
    umls_to_alias = umls.get_aliases(
    ontologies_to_include="all",
    use_umls_curies=True,
    lowercase=False)
    
    umls_cui2definition = umls.get_definition(
    ontologies_to_include="all",
    use_umls_curies=True,
    lowercase=False)
    
    umls_cui2types = umls.umls.groupby('cui').tui.first().to_dict()

    # Check if the directory exists, and create it if it does not
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    umls_entities = []
    for cui, name in tqdm(umls_to_name.items(), "Creating correct entities format"):
        d = {}
        ent_type = umls_cui2types[cui][0]
        other_aliases = [x for x in umls_to_alias[cui] if x != name]
        joined_aliases = ' ; '.join(other_aliases)
        d['cui'] = f"UMLS:{cui}"
        d['title'] = name
        d['type'] = ent_type
        if cui in umls_cui2definition:
            definition = umls_cui2definition[cui]
        else:
            definition = None

        if len(other_aliases) > 0:
            if definition is not None:
                d['description'] = f"{name} ( {ent_type} : {joined_aliases} ) [ {definition} ]"
            else:
                d['description'] = f"{name} ( {ent_type} : {joined_aliases} )"
        else:
            if definition is not None:
                d['description'] = f"{name} ( {ent_type} ) [ {definition} ]"
            else:
                d['description'] = f"{name} ( {ent_type} )"

        umls_entities.append(d)

    pickle.dump(umls_entities, open(os.path.join(data_path, 'dictionary.pickle'), 'wb'))
    entities = pickle.load(open(os.path.join(data_path, 'dictionary.pickle'), 'rb'))
    return entities

# Function for preparing the mentions in the dataset into the right format for each model

def process_mention_dataset(ontology,
                            dataset,
                            data_path,
                            ontology_type,
                            umls_dir: Optional[str] = None,
                            mention_id: Optional[bool] = True,
                            context_doc_id: Optional[bool] = True,
                            label: Optional[bool] = True
                            ): 
    '''
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
    - ontology : str (only umls for now)
    Ontology associated with the dataset
    - dataset : str
    Name of the dataset
    - data_path : str
    Path where to load and save dictionary.pickle
    - ontology_type : str
    'obo' or 'umls' and possibly others
    - umls_dir : str
    Path to umls data
    '''
    data = conhelps.for_config_name(f'{dataset}_bigbio_kb').load_dataset()
    exclude = CUIS_TO_EXCLUDE[dataset]
    remap = CUIS_TO_REMAP[dataset]

    'If dictionary already processed, load it else process and load it'
    entity_dictionary_pkl_path = os.path.join(data_path, 'dictionary.pickle')
    
    if os.path.isfile(entity_dictionary_pkl_path): 
        print("Loading stored processed entity dictionary...")
        with open(entity_dictionary_pkl_path, 'rb') as read_handle:
            entities = pickle.load(read_handle)
    else :
        if ontology_type == "obo" :
            entities = process_obo_ontology(ontology, data_path)
        elif ontology_type == "medic" : 
            entities = process_medic_ontology(ontology, data_path)
        elif ontology_type == "umls" : 
            entities = process_umls_ontology(ontology, data_path, umls_dir)
        else : 
            print("ERROR!")

    entity_dictionary = {d['cui']:d for d in tqdm(entities)} #CC1

    'specific to ncbi_disease'
    # if dataset == 'ncbi_disease': #CC2
    #     # Need to redo this since we have multiple synonymous CUIs for ncbi_disease
    #     entity_dictionary = {cui:d for d in tqdm(entities) for cui in d['cui']}
    #     cui_synsets = {}
    #     for subdict in tqdm(entities): 
    #         for cui in subdict['cui']:
    #             if cui in subdict:
    #                 print(cui, cui_synsets[cui], subdict['cui'])
    #             cui_synsets[cui] = subdict['cui'] 
    #     with open(os.path.join(data_path, 'cui_synsets.json'), 'w') as f:
    #         f.write(ujson.dumps(cui_synsets, indent=2))

    if dataset in VALIDATION_DOCUMENT_IDS:
        validation_pmids = VALIDATION_DOCUMENT_IDS[dataset]
    else:
        print("ERROR!!!")
        
    # Convert BigBio dataset to pandas DataFrame
    df = dataset_to_df(data, entity_remapping_dict=remap, cuis_to_exclude=exclude, val_split_ids=validation_pmids)
    # Return dictionary of documents in BigBio dataset
    docs = dataset_to_documents(data)
    label_len = df['db_ids'].map(lambda x: len(x)).max()
    print("Max labels on one doc:", label_len)

    for split in df.split.unique():
        ents_in_split = []
        for d in tqdm(df.query("split == @split").to_dict(orient='records'), desc=f"Creating correct mention format for {split} dataset"):
            abbrev_resolved = False
            offsets = d['offsets']
            doc_id = d['document_id']
            doc = docs[doc_id]
            mention = d['text']
            
            # Get offsets and context
            start = offsets[0][0] # start on the mention
            end = offsets[-1][-1] # end of the mention
            before_context = doc[:start] # left context
            after_context = doc[end:] # right context
            
            
            # ArboEL can't handle multi-labels, so we randomly choose one.
            if len(d['db_ids']) == 1:
                label_id = d['db_ids'][0]

            # 'specific to ncbi_disease'
            # # ncbi_disease is a special case that requires extra care
            # elif dataset == 'ncbi_disease':
            #     labels = []
            #     used_cuis = set([])
            #     choosable_ids = []
            #     for db_id in d['db_ids']:
            #         if db_id in used_cuis:
            #             continue
            #         else:
            #             used_cuis.update(set(entity_dictionary[db_id]['cuis']))
            #         choosable_ids.append(db_id)

            #     label_id = np.random.choice(choosable_ids)
            
            else:
                label_id = np.random.choice(d['db_ids'])

            # Check if we missed something
            if label_id not in entity_dictionary:
                # print(label_id)
                continue
            

            
            output = [{
                'mention': mention, 
                'context_left': before_context,
                'context_right': after_context, 
                'type': d['type'][0],
                'label_id': label_id,
                'label_title': entity_dictionary[label_id]['title'],
            }]
            
            if mention_id:
                output[0]['mention_id'] = d.get('mention_id', None)
        
            if context_doc_id:
                output[0]['context_doc_id'] = d.get('context_doc_id', None)
                
            if context_doc_id:
                output[0]['label'] = d.get(entity_dictionary[label_id]['description'], None)

            ents_in_split.extend(output)

        split_name = split
        if split =='validation':
            split_name = 'valid'
        with open(os.path.join(data_path, f'{split_name}.jsonl'), 'w') as f:
            f.write('\n'.join([ujson.dumps(x) for x in ents_in_split]))
            







