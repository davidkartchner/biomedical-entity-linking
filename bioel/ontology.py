from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union
from tqdm import tqdm

import obonet
import csv

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
    aliases: List[str] = field(default_factory=list)
    definition: Optional[str] = None
    equivalant_cuis: Optional[List[str]] = None
    taxonomy: Optional[str] = None
    metadata: Optional[dict] = None


@dataclass
class BiomedicalOntology:
    name: str
    types: List[str] = field(default_factory=list)
    entities: List[BiomedicalEntity] = field(
        default_factory=list
    )  # Dict mapping CUI: BiomedicalEntity
    abbrev: Optional[str] = None  # Abbreviated name of ontology if different than name
    metadata: Optional[dict] = None

    def get_canonical_name(self):
        """
        Get name of entities in the ontology
        data: list of dict
        """
        canonical_names = {entity.cui: entity.name for entity in self.entities}
        return canonical_names

    def get_aliases(self):
        """
        Get aliases of entities in the ontology
        data: list of dict
        """
        aliases = {entity.cui: entity.aliases for entity in self.entities}
        return aliases

    def get_definition(self):
        """
        Get definition of entities in the ontology
        data: list of dict
        """
        definitions_dict = {
            entity.cui: entity.definition
            for entity in self.entities
            if entity.definition is not None
        }
        return definitions_dict

    def get_types(self):
        """
        Get type of entities in the ontology
        data: list of dict
        """
        # Extract tuples of CUI and types from the Data
        types = {entity.cui: entity.types for entity in self.entities}
        return types

    @classmethod
    def load_obo(
        cls, filepath, name=None, prefix_to_keep=None, entity_type=None, abbrev=None
    ):
        """
        Read an ontology in .obo format

        Parameters:
        ----------------------
            filepath: str (optional)
                Path to .obo formatted ontology.  Can be a URL.
            prefix_to_keep: str (optional)
        """

        entities = []
        if entity_type:
            types = [entity_type]
        else:
            types = []

        logger.info(f"Reading OBO ontology from {filepath}")
        ontology = obonet.read_obo(filepath)

        data_keys = set([])
        for curie, data in tqdm(ontology.nodes(data=True)):
            # Exclude CUIs that are from cross-referenced ontologies
            data_keys.update(set(data.keys()))

            if prefix_to_keep is not None:
                if not curie.startswith(prefix_to_keep):
                    continue

            # Extract name and synonyms
            if "name" not in data:
                synonyms = _obo_extract_synonyms(data)
            else:
                synonyms = [data["name"]] + _obo_extract_synonyms(data)

            # Include deprecated CUIs as alternative ids
            alt_cuis = None
            if "alt_id" in data and len(data["alt_id"]) > 0:
                if prefix_to_keep:
                    alt_cuis = [
                        x for x in data["alt_id"] if x.startswith(prefix_to_keep)
                    ]
                else:
                    alt_cuis = data["alt_id"]

            # Skip entities with no name/aliases
            if len(synonyms) == 0:
                logger.warning(
                    f"Data entry for CUI {curie} has no listed names/aliases.  Skipping."
                )
                continue
            ent_name = synonyms[0]
            if len(synonyms) > 1:
                other_synonyms = synonyms[1:]
            else:
                other_synonyms = []

            # Get definition if it exists
            definition = _obo_extract_definition(data)

            ent = BiomedicalEntity(
                cui=curie,
                name=ent_name,
                aliases=other_synonyms,
                types=types,
                definition=definition,
                equivalant_cuis=alt_cuis,
            )
            entities.append(ent)

        if not name:
            if filepath.startswith("http"):
                parsed_name = filepath.split("/")[-1].split(".")[0]
                logger.warning(
                    f"No ontology name provided.  Using name from provided URL: {parsed_name}"
                )
                name = parsed_name

        return cls(entities=entities, types=types, name=name, abbrev=abbrev)

        # synonyms = _obo_term_to_synonyms(ontology, filter_prefix=prefix_to_keep)
        # definitions = _obo_term_to_definitions(ontology, filter_prefix=prefix_to_keep)
        # for key
        pass

    def load_medic(self, path):
        """
        path : str
            Path to medic.tsv dataset
        """

        # Attributes of the medic ontology
        key_dict = [
            "DiseaseName",
            "DiseaseID",
            "AltDiseaseIDs",
            "Definition",
            "ParentIDs",
            "TreeNumbers",
            "ParentTreeNumbers",
            "Synonyms",
            "SlimMappings",
        ]

        # Open the TSV file
        with open(path, newline="") as tsvfile:
            reader = csv.reader(tsvfile, delimiter="\t")

            counter = 0  # First entity in the tsv file appears in line 29

            ontology = []
            for row in reader:
                dict = {}
                if counter > 28:
                    for i, elements in enumerate(row):
                        dict[key_dict[i]] = elements
                    ontology.append(dict)
                counter += 1

        for element in ontology:
            equivalant_cuis = [element["DiseaseID"]]
            alt_ids = (
                element["AltDiseaseIDs"].split("|") if element["AltDiseaseIDs"] else []
            )
            for alt_id in alt_ids:
                if alt_id not in equivalant_cuis and alt_id[:2] != "DO":
                    equivalant_cuis.append(alt_id)

            entity = BiomedicalEntity(
                cui=element["DiseaseID"],
                name=element["DiseaseName"],
                types="Disease",
                aliases=element["Synonyms"],
                definition=element["Definition"],
                equivalant_cuis=equivalant_cuis,
            )
            self.entities.append(entity)

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
    ontologies: dict  # Dict of each ontology used

    def load_from_config(self, config=None, config_path=None):
        pass
