from dataclasses import dataclass
from collections import defaultdict
from typing import List, Optional, Union
from tqdm.auto import tqdm

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

    def load_data(self, path):
        with open(path, "r") as f:
            self.raw_data = f.readlines()

    def preprocess(self):
        """
        Extract only English Names, Process to extract the cui, synonyms.
        """
        processed = []
        for l in tqdm(self.raw_data):
            l_data = l.rstrip('\n').split("|")
            cui, lang, synonym = l_data[0], l_data[1], l_data[14]
            if lang != 'ENG':
                continue
            row = cui + "||" + synonym.lower()
            processed.append(row)
        
        self.alias_to_cui = defaultdict(list)
        processed = list(set(processed)) # Remove duplicates
        for line in tqdm(processed):
            cui, name = line.split('||')
            if cui in self.entities:
                self.entities.append(name)
            else:
                self.entities[cui] = [name]
        

            self.alias_to_cui[name].append(cui)


            
    def get_aliases(self, cui=None):
        '''
        Get aliases for a particular CUI.  If cui=None, provide a mapping of {cui: [aliases]}
        '''
        return self.entities[cui]

    def get_entities_with_alias(self, alias=None):
        '''
        Get all entities sharing a particular alias.  If alias=None, return a mapping of {alias: [cuis]}
        '''
        return self.alias_to_cui[alias]

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


ontology = BiomedicalOntology()
ontology.load_data("/mitchell/entity-linking/2022AA/mitchell/entity-linking/2022AA/META/MRCONSO.RRF")
ontology.preprocess()
