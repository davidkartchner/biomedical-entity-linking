from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union
from tqdm import tqdm

import obonet
import pandas as pd
import numpy as np

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

    def _create_mapping(self):
        mapping = {}
        for i, ent in enumerate(self.entities):
            mapping[ent.cui] = i

        self.mapping = mapping

    def __getitem__(self, key):
        try:
            index = self.mapping[key]
            return self.entities[index]
        except:
            raise KeyError(f"CUI '{key}' not found in ontology")

    def get_aliases(self, cui=None):
        """
        Get aliases for a particular CUI.  If cui=None, provide a mapping of {cui: [aliases]}
        """
        pass

    def get_entities_with_alias(self, alias=None):
        """
        Get all entities sharing a particular alias.  If alias=None, return a mapping of {alias: [cuis]}
        """
        pass

    def get_definitions(self, cui):
        pass

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

    def load_umls(self, umls_dir):
        pass

    def load_mesh(self, mesh_dir):
        raise NotImplementedError

    @classmethod
    def load_ncbi_taxonomy(cls):
        raise NotImplementedError

    @classmethod
    def load_ncbi_gene(
        cls,
        ncbigene_path: str = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/All_Data.gene_info.gz",
        taxa: Optional[List[int]] = None,
        types_to_remove: Optional[List[str]] = ["unknown", "tRNA", "biological-region"],
        remove_hypothetical: bool = True,
        remove_predicted: bool = True,
        # canonicalize_names=False,
        debug: bool = False,
    ):
        """

        Parameters:
        ----------------------
        taxa:
            NCBI Taxonomy ids to keep
        """

        if debug:
            # Load in tiny subset of fungi to test that data processing works
            ncbigene_path = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"

        # Load data
        logger.info(
            "Loading NCBI Gene (Entrez).  This ontology is large and may take a few minutes."
        )
        entrez = (
            pd.read_csv(
                ncbigene_path,
                delimiter="\t",
                usecols=[
                    "#tax_id",
                    "GeneID",
                    "Symbol",
                    "Synonyms",
                    "Symbol_from_nomenclature_authority",
                    "Full_name_from_nomenclature_authority",
                    "Other_designations",
                    "type_of_gene",
                    "description",
                    "dbXrefs",
                ],
                na_filter=False,
            )
            .rename(
                {
                    "Symbol_from_nomenclature_authority": "official_symbol",
                    "Full_name_from_nomenclature_authority": "official_name",
                    "#tax_id": "tax_id",
                },
                axis=1,
            )
            .convert_dtypes()
        )
        entrez.columns = [x.lower() for x in entrez.columns]

        for col in entrez.columns:
            if str(entrez.dtypes[col]) == "string":
                entrez[col] = entrez[col].map(lambda x: x.strip())

        entrez = entrez.replace("-", pd.NA)

        # Filter to desired subset of taxonomic categories, gene types, etc.
        logger.info("Filtering to desired taxa and gene types")
        filter_mask = np.ones(entrez.shape[0]).astype(bool)

        if taxa:
            filter_mask = (filter_mask) & ((entrez.tax_id.isin(taxa)))
        if types_to_remove:
            filter_mask = (filter_mask) & (~entrez.type_of_gene.isin(types_to_remove))
        if remove_hypothetical:
            filter_mask = (filter_mask) & (entrez.description != "hypothetical protein")
        if remove_predicted:
            filter_mask = (filter_mask) & (
                ~entrez.official_name.map(
                    lambda x: type(x) == str and x.lower().startswith("predicted")
                )
            )

        if filter_mask.sum() != entrez.shape[0]:
            filtered = entrez[filter_mask]

        else:
            filtered

        # Consolidate aliases in single column
        filtered.loc[:, "all_aliases"] = filtered.loc[
            :, ["synonyms", "official_symbol", "official_name", "other_designations"]
        ].progress_apply(
            lambda x: "|".join(
                [i.strip() for i in x if type(i) == str and i.strip() != "-"]
            ),
            axis=1,
        )

        filtered["aliases"] = filtered["all_aliases"].progress_map(
            lambda x: list(set(x.split("|"))) if x.strip() != "" else []
        )

        filtered = filtered.rename(
            columns={
                "geneid": "cui",
                "symbol": "name",
                "type_of_gene": "types",
                "description": "definition",
                "tax_id": "taxonomy",
            }
        )

        all_types = filtered["types"].unique()

        filtered["types"] = filtered["types"].map(lambda x: [x])
        records = filtered[
            ["cui", "name", "types", "aliases", "definition", "taxonomy"]
        ].to_dict(orient="records")

        # Initialize Ontology
        entities = [BiomedicalEntity(**x) for x in records]
        name = "ncbigene"
        abbrev = "ncbigene"

        return cls(entities=entities, types=all_types, name="NCBI Gene", abbrev=abbrev)

    def to_df(self):
        records = self.to_records()
        df = pd.DataFrame(records)

        if self.abbrev:
            df["kb"] = self.abbrev
        else:
            df["kb"] = self.name

        return df

    def to_records(self):
        return [asdict(x for x in self.entities)]

    def load_csv(self):
        raise NotImplementedError

    def load_json(self):
        raise NotImplementedError


@dataclass
class CompositeOntology:
    ontologies: dict  # Dict of each ontology used

    def load_from_config(self, config=None, config_path=None):
        pass
