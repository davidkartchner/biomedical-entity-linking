from dataclasses import dataclass
from typing import List, Optional, Union

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
    extra_data: Optional(dict)

@dataclass
class BiomedicalOntology:
    name: str
    types: List[str]
    abbrev: Optional[str] # Abbreviated name of ontology if different than name
    entities: dict # Dict mapping CUI: BiomedicalEntity
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

    def from_obo(self, filepath=None):
        pass

    def from_umls(self):
        pass

    def from_mesh(self):
        pass

    def from_ncbi_taxon(self):
        pass

    def from_csv(self):
        pass

    def from_json(self):
        pass


@dataclass
class CompositeOntology:
    ontologies: dict # Dict of each ontology used