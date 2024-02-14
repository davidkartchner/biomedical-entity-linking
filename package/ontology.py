import sys
sys.path.append('/home/pbathala3/entity_linking/biomedical-entity-linking')
from dataclasses import dataclass, field
from typing import List, Optional, Union
from umls_utils import UmlsMappings

@dataclass
class BiomedicalEntity:
    """
    Class for keeping track of all relevant fields in an ontology
    """
    cui: str
    name: str
    types: List[str]
    aliases: List[str]
    definition: Optional[str]
    equivalant_cuis: Optional[List[str]] = None
    taxonomy: Optional[str] = None
    extra_data: Optional[dict] = None

@dataclass
class BiomedicalOntology:
    name: str
    abbrev: Optional[str] = None                                                                  # Abbreviated name of ontology if different than name
    types: List[str] = field(default_factory=list)                                          # List of all types in the ontology                                        
    entities: List[BiomedicalEntity] = field(default_factory=list)                          # List Containing all Biomedical Entity Objects
    mappings: dict = field(default_factory=dict)                                            # Dict mapping a cui to the index in entities

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

    def load_umls(self, path = None, api_key = ""):
        umls = UmlsMappings(umls_dir = path, umls_api_key=api_key)

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
        all_umls_df['cui'] = all_umls_df['cui'].map(lambda x: 'UMLS' + x)
        all_umls_df['has_definition'] = all_umls_df['def'].map(lambda x: x is not None)
        all_umls_df['num_aliases'] = all_umls_df['alias'].map(lambda x: len(x))

        for index, row in all_umls_df.iterrows():
            entity = BiomedicalEntity(
                cui = row['cui'],
                name = row['name'],
                types = row['tui'],
                aliases = row['alias'],
                definition = row['def'],
                extra_data = {
                    'group': row['group'],
                }
            )
            self.entities.append(entity)
            self.mappings[row['cui']] = index 


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

if __name__ == "__main__":
    ontology = BiomedicalOntology(name="UMLS")
    ontology.load_umls(path="/mitchell/entity-linking/2017AA/META/", api_key="")
    print(ontology.entities[0])
    print(ontology.entities[0].__dict__)
    print(ontology.entities[0].cui)
    print(ontology.mappings[ontology.entities[0].cui])