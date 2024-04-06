from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union
from tqdm import tqdm

import obonet

from .utils.obo_utils import _obo_extract_definition, _obo_extract_synonyms
from .logger import setup_logger

from .utils.umls_utils import UmlsMappings

logger = setup_logger()


@dataclass
class BiomedicalEntity:
    """
    Class for keeping track of all relevant fields in an ontology
    """
    cui: str
    name: str
    types: List[str]
    aliases: List[str] = field(default_factory=list)
    definition: Optional[str] = None
    equivalant_cuis: Optional[List[str]] = None
    taxonomy: Optional[str] = None
    metadata: Optional[dict] = None


@dataclass
class BiomedicalOntology:
    name: str
    types: List[str] = field(default_factory=list)
    entities: List[BiomedicalEntity] = field(default_factory=list) # Dict mapping CUI: BiomedicalEntity
    abbrev: Optional[str] = None # Abbreviated name of ontology if different than name
    metadata: Optional[dict] = None

    def get_canonical_name(self):
        '''
        Get name of entities in the ontology
        data: list of dict
        '''
        canonical_names = {entity.cui: entity.name for entity in self.entities}
        return canonical_names
    
    def get_aliases(self):
        '''
        Get aliases of entities in the ontology
        data: list of dict
        '''
        aliases = {entity.cui: entity.aliases for entity in self.entities}
        return aliases
    
    def get_definition(self):
        '''
        Get definition of entities in the ontology
        data: list of dict
        '''
        definitions_dict = {entity.cui: entity.definition for entity in self.entities if entity.definition is not None}
        return definitions_dict
    
    def get_types(self):
        '''
        Get type of entities in the ontology
        data: list of dict
        '''
        # Extract tuples of CUI and types from the Data
        types = {entity.cui: entity.types for entity in self.entities}
        return types

    @classmethod
    def load_obo(cls, filepath, name=None, prefix_to_keep=None, entity_type=None, abbrev=None):
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
        
        logger.info(f'Reading OBO ontology from {filepath}')
        ontology = obonet.read_obo(filepath)

        data_keys = set([])
        for curie, data in tqdm(ontology.nodes(data=True)):
            # Exclude CUIs that are from cross-referenced ontologies
            data_keys.update(set(data.keys()))


            if prefix_to_keep is not None:
                if not curie.startswith(prefix_to_keep):
                    continue

            # Extract name and synonyms
            if 'name' not in data:
                synonyms = _obo_extract_synonyms(data)
            else:
                synonyms = [data['name']] + _obo_extract_synonyms(data)

            # Include deprecated CUIs as alternative ids
            alt_cuis = None
            if 'alt_id' in data and len(data['alt_id']) > 0:
                if prefix_to_keep:
                    alt_cuis = [x for x in data['alt_id'] if x.startswith(prefix_to_keep)]
                else:
                    alt_cuis = data['alt_id']
            
            # Skip entities with no name/aliases
            if len(synonyms) == 0:
                logger.warning(f"Data entry for CUI {curie} has no listed names/aliases.  Skipping.")
                continue
            ent_name = synonyms[0]
            if len(synonyms) > 1:
                other_synonyms = synonyms[1:]
            else:
                other_synonyms = []

            # Get definition if it exists
            definition = _obo_extract_definition(data)

            
            ent = BiomedicalEntity(cui=curie, name=ent_name, aliases=other_synonyms, types=types, definition=definition, equivalant_cuis=alt_cuis)
            entities.append(ent)

        if not name :
            if filepath.startswith('http'):
                parsed_name = filepath.split('/')[-1].split('.')[0]
                logger.warning(f"No ontology name provided.  Using name from provided URL: {parsed_name}")
                name = parsed_name

        return cls(entities=entities, types=types, name=name, abbrev=abbrev)

    @classmethod
    def load_umls(cls, filepath, name = None, abbrev = None, api_key = ""):
        '''
        Read an ontology from the UMLS Directory

        Parameters:
        ----------------------
            filepath: str (Pointing to the UMLS directory)
            name: str (optional)
            abbrev: str (optional)
            api_key: str (optional)
        '''

        entities = []
        types = []

        logger.info(f'Reading UMLS from {filepath}')
        umls = UmlsMappings(umls_dir = filepath, umls_api_key=api_key)

        # Get the Canonial Names
        lowercase = False
        umls_to_name = umls.get_canonical_name(
            ontologies_to_include="all",
            use_umls_curies=True,
            lowercase=lowercase,
        )

        # Group by the canonical names to group the alias and types 
        all_umls_df = umls.umls.query('lang == "ENG"').groupby('cui').agg({'alias': lambda x: list(set(x)), 'tui':'first', 'group': 'first', 'def':'first'}).reset_index()
        all_umls_df['name'] = all_umls_df.cui.map(umls_to_name)
        all_umls_df['alias'] = all_umls_df[['name','alias']].apply(lambda x: list(set(x[1]) - set([x[0]])) , axis=1)
        all_umls_df['cui'] = all_umls_df['cui'].map(lambda x: 'UMLS:' + x)
        all_umls_df['has_definition'] = all_umls_df['def'].map(lambda x: x is not None)
        all_umls_df['num_aliases'] = all_umls_df['alias'].map(lambda x: len(x))

        for index, row in all_umls_df.iterrows():
            entity = BiomedicalEntity(
                cui = row['cui'],
                name = row['name'],
                types = row['tui'],
                aliases = row['alias'],
                definition = row['def'],
                metadata = {
                    'group': row['group'],
                }
            )
            entities.append(entity)
            types.append(row['tui'])
        
        return cls(entities=entities, types=types, name=name, abbrev=abbrev)
        


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