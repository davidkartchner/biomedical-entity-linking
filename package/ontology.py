from dataclasses import dataclass
from typing import List, Optional, Union
from tqdm import tqdm

import obonet

from .utils import _obo_extract_definition, _obo_extract_synonyms
from .logger import setup_logger

logger = setup_logger()



@dataclass
class BiomedicalEntity:
    """
    Class for keeping track of all relevant fields in an ontology
    """
    cui: str
    name: str
    types: List[str]
    aliases: List[str] = []
    definition: Optional[str]
    equivalant_cuis: Optional[List[str]]
    taxonomy: Optional[str]
    extra_data: Optional[dict]

@dataclass
class BiomedicalOntology:
    name: str
    types: List[str]
    entities: List[BiomedicalEntity] # Dict mapping CUI: BiomedicalEntity
    abbrev: Optional[str] # Abbreviated name of ontology if different than name
    # typed_entities: dict

    def get_aliases(self, cui=None):
        '''
        Get aliases for a particular CUI.  If cui=None, provide a mapping of {cui: [aliases]}
        '''
        pass

    def get_entities_with_alias(self, alias=None):
        '''
        Get all entities sharing a particular alias.  If alias=None, return a mapping of {alias: [cuis]}
        '''
        pass

    def get_definitions(self, cui):
        pass

    def load_obo(self, filepath, name=None, prefix_to_keep=None, entity_type=None, abbrev=None):
        '''
        Read an ontology in .obo format

        Parameters:
        ----------------------
            filepath: str (optional)
                Path to .obo formatted ontology.  Can be a URL.  
            prefix_to_keep: str (optional)
        '''

        entities = []
        if entity_type:
            types = [entity_type]
        else:
            types = []
        ontology = obonet.read_obo(filepath)

        for curie, data in tqdm(ontology.nodes(data=True)):
            # Exclude CUIs that are from cross-referenced ontologies
            if prefix_to_keep is not None:
                if not curie.startswith(prefix_to_keep):
                    continue

            # Extract name and synonyms
            if 'name' not in data:
                synonyms = _obo_extract_synonyms(data)
            else:
                synonyms = [data['name']] + _obo_extract_synonyms(data)
            
            # Skip entities with no name/aliases
            if len(synonyms) == 0:
                logger.warning(f"Data entry for CUI {curie} has no listed names/aliases.  Skipping.")
                continue
            name = synonyms[0]
            if len(synonyms) > 1:
                other_synonyms = synonyms[1:]
            else:
                other_synonyms = []

            # Get definition if it exists
            definition = _obo_extract_definition(data)

            
            ent = BiomedicalEntity(cui=curie, name=name, aliases=other_synonyms, types=types, definition=definition)
            entities.append(ent)

        self.entities = entities
        self.types = types
        if name:
            self.name = name
        elif filepath.startswith('http'):
            parsed_name = filepath.split('/')[-1].split('.')[0]
            logger.warning(f"No ontology name provided.  Using name from provided URL: {parsed_name}")
            self.name = parsed_name
        if abbrev:
            self.abbrev = abbrev

        
            
        # synonyms = _obo_term_to_synonyms(ontology, filter_prefix=prefix_to_keep)
        # definitions = _obo_term_to_definitions(ontology, filter_prefix=prefix_to_keep)
        # for key 
        pass

    def load_umls(self, umls_dir):
        pass

    def load_mesh(self, mesh_dir):
        pass

    def load_ncbi_taxon(self):
        pass

    def load_csv(self):
        pass

    def load_json(self):
        pass


@dataclass
class CompositeOntology:
    ontologies: dict # Dict of each ontology used

    def load_from_config(self, config=None, config_path=None):
        pass