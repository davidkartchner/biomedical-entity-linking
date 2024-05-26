import os
import pandas as pd
import logging


from tqdm import tqdm
from typing import Union, Optional
from logging import getLogger

tqdm.pandas()

logger = logging.getLogger(__file__)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
)

# from config import umls_api_key


class UmlsMappings(object):
    def __init__(
        self,
        umls_dir="/Users/david/umls/",
        umls_api_key="",
        force_reprocess=False,
        debug=False,
    ):
        self.umls_dir = umls_dir
        self.debug = debug
        self.force_reprocess = force_reprocess
        if not os.path.isfile(umls_dir + "MRCONSO.RRF"):
            self._download_umls(umls_api_key)

        self._load_umls()
        if self.using_cached:
            self._load_type_abbrevs()

        else:
            self._load_types()
            self._merge_types()
            self._add_definitions()

        # self.cui_to_aliases = self.get_aliases()
        # self.cui_to_types = self._process_types()
        # self.cui_to_groups = self._process_groups()

        self._cache_umls()

    def _download_umls(self, api_key):
        if api_key == "":
            raise ValueError(
                """Please add valid UTS API key to config.py! \
                (For instructions on how to obtain a UTS API key,\
                please see https://documentation.uts.nlm.nih.gov/automating-downloads.html"""
            )
        else:
            pass

    def _download_semtypes(self, api_key):
        """
        Download semantic type files from UMLS
        """
        # Semantic network files
        # https://lhncbc.nlm.nih.gov/semanticnetwork/download/sn_current.tgz

        # Semantic Group Files
        # https://lhncbc.nlm.nih.gov/semanticnetwork/download/SemGroups.txt
        pass

    def _add_definitions(self):
        '''
        Add definitions of entities where available
        '''
        umls_definitions = pd.read_csv(
            os.path.join(self.umls_dir, "MRDEF.RRF"),
            sep="|",
            usecols=[0,1,4,5,],
            names=["cui", "aui", "sab", 'def',],
        )

        self.umls = self.umls.join(umls_definitions.set_index(['cui','aui','sab']), on=['cui','aui','sab'])

    def _load_umls(self):
        """
        Load UMLS MRCONSO.RRF
        """
        self.using_cached = False

        cache_path = os.path.join(self.umls_dir, ".cached_df.feather")
        if os.path.isfile(cache_path) and not self.force_reprocess:
            print(f"Loading cached UMLS data from {cache_path}")
            self.umls = pd.read_feather(cache_path)
            self.using_cached = True
            return

        print("Loading UMLS dataset.  This may take a few minutes")
        col_names = [
            "cui",
            "lang",
            "term_status",
            "lui",
            "stt",
            "sui",
            "ispref",
            "aui",
            "saui",
            "scui",
            "sdui",
            "sab",
            "tty",
            "code",
            "alias",
            "srl",
            "suppress",
            "cvf",
            "null_col",
        ]

        cols_to_use = [
            "cui",
            "sab",
            "sdui",
            "scui",
            "alias",
            "lang",
            "ispref",
            "tty",
            # "lui",
            # "sui",
            "aui",
            # "saui",
            "suppress",
        ]

        if self.debug:
            df = next(
                pd.read_csv(
                    os.path.join(self.umls_dir, "MRCONSO.RRF"),
                    delimiter="|",
                    names=col_names,
                    iterator=True,
                    # usecols=['cui','lang','sab','sdui','lui','sui','aui','saui','scui']
                    usecols=cols_to_use,
                    low_memory=False,
                    chunksize=10000,
                )
            )
        else:
            df = pd.read_csv(
                os.path.join(self.umls_dir, "MRCONSO.RRF"),
                delimiter="|",
                names=col_names,
                # usecols=['cui','lang','sab','sdui','lui','sui','aui','saui','scui']
                usecols=cols_to_use,
                low_memory=False,
                # engine='pyarrow',
            )

        print("Load rankings")
        rankings = pd.read_csv(
            os.path.join(self.umls_dir, "MRRANK.RRF"),
            delimiter="|",
            names=["rank", "sab", "tty", "_", "_1"],
            usecols=["rank", "sab", "tty"],
        )
        rankings["rank"] = -rankings["rank"] + 850

        print("Merge rankings")
        df = df.merge(rankings, on=["sab", "tty"])

        self.umls = df

    def _cache_umls(self):
        """
        Cache processed UMLS dataframe for faster reloading
        """
        if not self.debug and not self.using_cached:
            cache_path = os.path.join(self.umls_dir, ".cached_df.feather")
            print(f"Caching processed UMLS data to {cache_path}")
            print(self.umls.index)
            self.umls.reset_index(drop=True).to_feather(cache_path)

    def get_canonical_name(
        self,
        ontologies_to_include: Union[str, list] = "all",
        types_to_include: Union[str, list] = "all",
        groups_to_include: Union[str, list] = "all",
        use_umls_curies: bool = True,
        mapping_cols: Optional[dict] = None,
        prefixes: Optional[dict] = None,
        remove_multigroup=False,
        reverse=False,
        lowercase=False,
    ):
        """
        Get canonical name of entities in UMLS Metathesaurus
        """
        df = self.filter_ontologies_and_types(
            ontologies_to_include=ontologies_to_include,
            types_to_include=types_to_include,
            groups_to_include=groups_to_include,
            remove_multigroup=remove_multigroup,
        )

        # Pick which set of CURIEs to use
        if use_umls_curies:
            df["identifier"] = df["cui"]
            filtered = df[["identifier", "rank", "alias"]]

        else:
            subsets = []
            for ontology in df.sab.unique():
                mapping_col = ""
                prefix = ""
                if ontology in mapping_cols:
                    mapping_col = mapping_cols[ontology]
                if ontology in prefixes:
                    prefix = prefixes[ontology]

                subset = self._get_alias_single_subset(
                    df, ontology, mapping_col=mapping_col, prefix=prefix
                )
                subsets.append(subset[["identifier", "rank", "alias"]])
            filtered = pd.concat(subsets).drop_duplicates()

        if lowercase:
            filtered["alias"] = filtered.alias.map(lambda x: x.lower())
            filtered = filtered.drop_duplicates()

        # Get output dict of CUI to Name mappings
        output_dict = (
            filtered.loc[filtered.groupby("identifier")["rank"].idxmin()]
            .set_index("identifier")["alias"]
            .to_dict()
        )
        return output_dict

    def get_definition(
        self,
        ontologies_to_include: Union[str, list] = "all",
        types_to_include: Union[str, list] = "all",
        groups_to_include: Union[str, list] = "all",
        use_umls_curies: bool = True,
        mapping_cols: Optional[dict] = None,
        prefixes: Optional[dict] = None,
        remove_multigroup=False,
        reverse=False,
        lowercase=False,
    ):
        """
        Get canonical name of entities in UMLS Metathesaurus
        """
        print("Filtering by ontologies")
        df = self.filter_ontologies_and_types(
            ontologies_to_include=ontologies_to_include,
            types_to_include=types_to_include,
            groups_to_include=groups_to_include,
            remove_multigroup=remove_multigroup,
        )

        print("Removing null definitions")
        df = df.loc[~df['def'].isnull(),:]

        # Pick which set of CURIEs to use
        if use_umls_curies:
            df["identifier"] = df["cui"]
            filtered = df[["identifier", "rank", "def"]]

        else:
            subsets = []
            for ontology in df.sab.unique():
                mapping_col = ""
                prefix = ""
                if ontology in mapping_cols:
                    mapping_col = mapping_cols[ontology]
                if ontology in prefixes:
                    prefix = prefixes[ontology]

                subset = self._get_alias_single_subset(
                    df, ontology, mapping_col=mapping_col, prefix=prefix
                )
                subsets.append(subset[["identifier", "rank", "def"]])
            filtered = pd.concat(subsets).drop_duplicates()

        if lowercase:
            filtered["def"] = filtered['def'].map(lambda x: x.lower())
            filtered = filtered.drop_duplicates()

        # Get output dict of CUI to Name mappings
        output_dict = (
            filtered.loc[filtered.groupby("identifier")["rank"].idxmin()]
            .set_index("identifier")["def"]
            .to_dict()
        )
        return output_dict

    def get_types_and_groups(
        self,
        ontologies_to_include: Union[str, list] = "all",
        types_to_include: Union[str, list] = "all",
        groups_to_include: Union[str, list] = "all",
        use_umls_curies: bool = True,
        mapping_cols: Optional[dict] = None,
        prefixes: Optional[dict] = None,
        remove_multigroup=False,
        reverse=False,
        lowercase=False,
    ):

        df = self.filter_ontologies_and_types(
            ontologies_to_include=ontologies_to_include,
            types_to_include=types_to_include,
            groups_to_include=groups_to_include,
            remove_multigroup=remove_multigroup,
        )

        # Pick which set of CURIEs to use
        if use_umls_curies:
            df["identifier"] = df["cui"]
            filtered = df[["identifier", "rank", "tui", "group"]]

        else:
            subsets = []
            for ontology in df.sab.unique():
                mapping_col = ""
                prefix = ""
                if ontology in mapping_cols:
                    mapping_col = mapping_cols[ontology]
                if ontology in prefixes:
                    prefix = prefixes[ontology]

                subset = self._get_alias_single_subset(
                    df, ontology, mapping_col=mapping_col, prefix=prefix
                )
                subsets.append(subset[["identifier", "rank", "tui", "group"]])
            filtered = pd.concat(subsets)

        grouped = filtered.groupby('identifier')
        id2types = grouped.tui.first().to_dict()
        id2groups = grouped.group.first().to_dict()

        return id2types, id2groups

    def get_mapping(
        self,
        ontology_abbreviation="MSH",
        reverse=False,
        mapping_col="sdui",
        umls_prefix="",
        other_prefix="",
    ):
        """
        Get cross-references between UMLS and another ongology
        """
        # Load main UMLS file (MRCONSO.RRF)

        # Filter down to desired ontology
        filtered = (
            self.umls.query("sab == @ontology_abbreviation")
            .loc[:, ["cui", mapping_col]]
            .dropna()
            .drop_duplicates()
            .reset_index(drop=True)
        )
        if umls_prefix != "":
            filtered["cui"] = filtered["cui"].map(lambda x: umls_prefix + ":" + x)

        if other_prefix != "":
            filtered[mapping_col] = filtered[mapping_col].map(
                lambda x: other_prefix + ":" + x
            )

        if reverse:
            output = filtered.set_index(mapping_col).to_dict()["cui"]
        else:
            output = filtered.set_index("cui").to_dict()[mapping_col]

        return output

    def interalias_mapping(self, type_subset="gngm"):
        """
        Get mapping between all terms with the same alias
        """
        pass
        # if hasattr(self.cui2type):

    def _load_types(self):
        """
        Load mapping from CUI to UMLS semantic type
        """
        # Load type dataframe
        print("Loading semantic types")
        cui_type_df = pd.read_csv(
            os.path.join(self.umls_dir, "MRSTY.RRF"),
            usecols=[0, 1],
            names=["cui", "tui"],
            delimiter="|",
            # engine='pyarrow'
        )

        self._load_type_abbrevs()

        cui_type_df["group"] = cui_type_df["tui"].map(self.tui2group)

        grouped = cui_type_df.groupby("cui")

        # Get mapping from CUI to types
        print("Mapping CUIs to types")
        self.cui2types = grouped["tui"].progress_apply(list).to_dict()

        # Get mapping from CUI to groups
        print("Mapping CUIs to groups")
        self.cui2groups = (
            grouped["group"].progress_apply(lambda x: list(set(x))).to_dict()
        )

    def _load_type_abbrevs(self):
        """
        Get type abbreviations and semantic groups
        """
        # Map each TUI to a semantic group
        self.tui2group = (
            pd.read_csv(
                os.path.join(self.umls_dir, "SemGroups.txt"),
                sep="|",
                usecols=[0, 2],
                names=["group", "tui"],
            )
            .set_index("tui")
            .to_dict()["group"]
        )

        self.semantic_groups = list(set(self.tui2group.values()))

        # Mape each TUI to its abbreviaion/name
        # Semantic network abbreviations
        sem_network_cols = [
            "row_type",
            "tui",
            "name",
            "tree_index",
            "desc",
            "_1",
            "clarification",
            "_2",
            "abbrev",
            "inverse_relation",
            "_3",
        ]

        self.tui_type_abbreviations = pd.read_csv(
            os.path.join(self.umls_dir, "semantic_network/SRDEF"),
            sep="|",
            names=sem_network_cols,
            usecols=["row_type", "tui", "name", "desc", "abbrev"],
        )
        self.type2abbrev = (
            self.tui_type_abbreviations[["tui", "abbrev"]]
            .set_index("tui")
            .to_dict()["abbrev"]
        )

    def _merge_types(self):
        """
        Merge type information into UMLS dataframe
        """
        print("Adding type information to UMLS dataframe")
        self.umls["tui"] = self.umls["cui"].map(self.cui2types)

        print("Adding semantic group informaiton to UMLS dataframe")
        self.umls["group"] = self.umls["cui"].map(self.cui2groups)

    def _load_type2name(self, umls_dir):
        """
        Load mapping from semantic types to names/abbreviations
        """
        pass

    def _get_alias_single_subset(self, df, ontology, mapping_col="", prefix=""):
        """
        Get CURIE -> alias mapping for a single subset of UMLS
        """
        filtered = df.query("sab == @ontology")
        inds = filtered.loc[:, [mapping_col, "alias"]].dropna().drop_duplicates().index
        filtered = filtered.loc[inds, :].reset_index(drop=True)
        filtered["identifier"] = filtered[mapping_col].map(lambda x: prefix + ":" + x)
        return filtered


    def list_groups(self):
        """
        List available semantic groups
        """
        return self.semantic_groups

    def filter_ontologies_and_types(
        self,
        ontologies_to_include: Union[str, list] = "all",
        types_to_include: Union[str, list] = "all",
        groups_to_include: Union[str, list] = "all",
        remove_multigroup=False,
    ):

        # Pull correct subset of UMLS, if subsetting at all
        # print(self.umls.columns)
        if ontologies_to_include == "all":
            df = self.umls.query('lang == "ENG"')
        elif type(ontologies_to_include) == str:
            df = self.umls.query('sab == @ontologies_to_include & lang == "ENG"')
        else:
            df = self.umls[self.umls.sab.isin(ontologies_to_include)].query(
                'lang == "ENG"'
            )

        # Remove concepts with more than one group
        if remove_multigroup:
            df = df[df["group"].map(lambda x: len(x) == 1)]

        # Filter by semantic group
        if type(groups_to_include) == str:
            if groups_to_include != "all":
                raise ValueError('groups_to_include must be a list of strings or "all"')
        elif type(groups_to_include) == list:
            mask = df["group"].map(lambda x: any([y in groups_to_include for y in x]))
            df = df[mask]
        else:
            raise TypeError("groups_to_include must be list or string")

        # Filter by semantic type
        if type(types_to_include) == str:
            if types_to_include != "all":
                raise ValueError('types_to_include must be a list of strings or "all"')
        elif type(types_to_include) == list:
            mask = df["tui"].map(lambda x: any([y in types_to_include for y in x]))
            df = df[mask]
        else:
            raise TypeError("types_to_include must be list or string")

        return df

    def get_types(
        self,
        ontologies_to_include: Union[str, list] = "all",
        types_to_include: Union[str, list] = "all",
        groups_to_include: Union[str, list] = "all",
        remove_multigroup=False,
    ):
        """
        Get semantic types for UMLS entities
        """

        df = self.filter_ontologies_and_types(
            ontologies_to_include=ontologies_to_include,
            types_to_include=types_to_include,
            groups_to_include=groups_to_include,
            remove_multigroup=remove_multigroup,
        )

        return df.groupby("cui").tui.first().to_dict()

    def get_aliases(
        self,
        ontologies_to_include: Union[str, list] = "all",
        types_to_include: Union[str, list] = "all",
        groups_to_include: Union[str, list] = "all",
        use_umls_curies: bool = True,
        mapping_cols: Optional[dict] = None,
        prefixes: Optional[dict] = None,
        remove_multigroup=False,
        reverse: bool = False,
        lowercase: bool = False,
    ):
        """
        Get mapping of CURIE -> aliases for all or a subset of UMLS entities

        Parameters:
        -----------------------------
            ontologies_to_include:
                Ontologies in UMLS to include in subset, default 'all'

            types_to_include:
                Semantic types to include in subset, default 'all'

            groups_to_include:
                Semantic groups to include in subset, default: 'all'
                Semantic groups are more general than types and typicially
                include multiple types.

            use_umls_curies: bool, default True
                Whether to use UMLS CUIs as unique identifiers.
                If false, use CUIs subset ontologies.

            mapping_cols:
                Dict of which column to use for CURIEs for each ontology.  If none
                specified, sdui is used.

            prefixes:
                Dict of prefixes to use for each ontology.  If none given, no prefixes
                are used

            reverse: bool, default False
                If true, return mapping from aliases to CURIEs instead of vice versa

            lowercase: bool, default False
                If true, lowercase all aliases before deduplicating/returning

            remove_multigroup: bool, default True
                If true, remove entities that belong to multiple groups
        """
        df = self.filter_ontologies_and_types(
            ontologies_to_include=ontologies_to_include,
            types_to_include=types_to_include,
            groups_to_include=groups_to_include,
            remove_multigroup=remove_multigroup,
        )

        # Pick which set of CURIEs to use
        if use_umls_curies:
            filtered = (
                df.loc[:, ["cui", "alias"]]
                .dropna()
                .drop_duplicates()
                .reset_index(drop=True)
            ).rename({"cui": "identifier"}, axis=1)

        else:
            subsets = []
            for ontology in df.sab.unique():
                mapping_col = ""
                prefix = ""
                if ontology in mapping_cols:
                    mapping_col = mapping_cols[ontology]
                if ontology in prefixes:
                    prefix = prefixes[ontology]

                subset = self._get_alias_single_subset(
                    df, ontology, mapping_col=mapping_col, prefix=prefix
                )
                subsets.append(subset[["identifier", "alias"]])
            filtered = pd.concat(subsets).drop_duplicates()

        if lowercase:
            filtered["alias"] = filtered.alias.map(lambda x: x.lower())
            filtered = filtered.drop_duplicates()

        if reverse:
            return filtered.groupby("alias")["identifier"].apply(list).to_dict()
        else:
            return filtered.groupby("identifier")["alias"].apply(list).to_dict()

    def get_synsets(self, preferred_type="gngm"):
        """
        Get mapping between all terms with the same alias

        This is mostly to get mapping from proteins to their corresponding gene and vice-versa
        """
        df = self.umls
        df["type"] = df.tui.map(self.type2abbrev)

        df["has_preferred_type"] = df["type"] == preferred_type

        grouped = self.umls.groupby("alias")


# Semantic network abbreviations
# def get_semantic_type_hierarchy(umls_dir, tui):
#     sem_network_cols = [
#         "row_type",
#         "tui",
#         "name",
#         "tree_index",
#         "desc",
#         "_1",
#         "clarification",
#         "_2",
#         "abbrev",
#         "inverse_relation",
#         "_3",
#     ]

#     semantic_network = pd.read_csv(
#         os.path.join(umls_dir, "semantic_network/SRDEF"),
#         sep="|",
#         names=sem_network_cols,
#         usecols=["row_type", "tui", "name", "tree_index"],
#     )

#     tree_index_to_name = semantic_network.set_index('tree_index')['name'].to_dict()


# def read_sem_group():
#     pass


# def download_umls():
#     pass
