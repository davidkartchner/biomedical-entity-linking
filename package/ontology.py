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


@dataclass
class BiomedicalOntology:
    entities: List[BiomedicalEntity]
    # typed_entities: dict

    def from_obo(self, entity_type=None):
        pass
