from typing import List, Dict, NamedTuple, Optional, Set, Tuple, Type
import json
from collections import defaultdict
from scispacy.file_cache import cached_path
import datetime
import scipy
import numpy
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import nmslib
from nmslib.dist import FloatIndex



class Entity(NamedTuple):
    """
    A class reprensenting an Entity
    """
    
    concept_id: str
    canonical_name: str
    aliases: List[str]
    types: List[str] = []
    definition: Optional[str] = None

    def __repr__(self):
        rep = ""
        num_aliases = len(self.aliases)
        rep = rep + f"CUI: {self.concept_id}, Name: {self.canonical_name}\n"
        rep = rep + f"Definition: {self.definition}\n"
        rep = rep + f"TUI(s): {', '.join(self.types)}\n"
        if num_aliases > 10:
            rep = (
                rep
                + f"Aliases (abbreviated, total: {num_aliases}): \n\t {', '.join(self.aliases[:10])}"
            )
        else:
            rep = (
                rep + f"Aliases: (total: {num_aliases}): \n\t {', '.join(self.aliases)}"
            )
        return rep

class LinkerPaths(NamedTuple):
    """
    Encapsulates all the (possibly remote) paths to data for a scispacy CandidateGenerator.
    ann_index: str
        Path to the approximate nearest neighbours index.
    tfidf_vectorizer: str
        Path to the joblib serialized sklearn TfidfVectorizer.
    tfidf_vectors: str
        Path to the float-16 encoded tf-idf vectors for the entities in the KB.
    concept_aliases_list: str
        Path to the indices mapping concepts to aliases in the index.
    alias_to_cuis: Dict[str, Set[str]]
        Dictionary which maps aliases to their unique concept ids
    cui_to_entity : Dict[str, Entity]
        Dictionary which maps unique concept ids to the corresponding entities
    """

    ann_index: str
    tfidf_vectorizer: str
    tfidf_vectors: str
    concept_aliases_list: str
    alias_to_cuis : Dict[str, Set[str]]
    cui_to_entity : Dict[str, Entity]

class KnowledgeBase:
    """
    A class representing two commonly needed views of a Knowledge Base:
    1. A mapping from concept_id to an Entity NamedTuple with more information.
    2. A mapping from aliases to the sets of concept ids for which they are aliases.

    Parameters
    ----------
    file_path: str, required.
        The file path to the json/jsonl representation of the KB to load.
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
    ):
        if file_path is None:
            raise ValueError(
                "Do not use the default arguments to KnowledgeBase. "
                "Instead, use a subclass (e.g UmlsKnowledgeBase) or pass a path to a kb."
            )
        if file_path.endswith("jsonl"):
            raw = (json.loads(line) for line in open(cached_path(file_path)))
        else:
            raw = json.load(open(cached_path(file_path)))

        alias_to_cuis: Dict[str, Set[str]] = defaultdict(set)
        self.cui_to_entity: Dict[str, Entity] = {}

        for concept in raw:
            unique_aliases = set(concept["aliases"])
            unique_aliases.add(concept["canonical_name"])
            for alias in unique_aliases:
                alias_to_cuis[alias].add(concept["concept_id"])
            self.cui_to_entity[concept["concept_id"]] = Entity(**concept)

        self.alias_to_cuis: Dict[str, Set[str]] = {**alias_to_cuis}

        self.tfidf_vectorizer_path = None
        self.ann_index_path = None 
        self.tfidf_vectors_path = None
        self.umls_concept_aliases_path = None

    def create_tfidf_ann_index(
    self, out_path: str
    ) -> Tuple[List[str], TfidfVectorizer, FloatIndex]:
        """
        Build tfidf vectorizer and ann index.
    
        Parameters
        ----------
        out_path: str, required.
            The path where the various model pieces will be saved.
        kb : KnowledgeBase, optional.
            The kb items to generate the index and vectors for.
    
        """
        tfidf_vectorizer_path = f"{out_path}/tfidf_vectorizer.joblib"
        ann_index_path = f"{out_path}/nmslib_index.bin"
        tfidf_vectors_path = f"{out_path}/tfidf_vectors_sparse.npz"
        umls_concept_aliases_path = f"{out_path}/concept_aliases.json"

        self.tfidf_vectorizer_path = tfidf_vectorizer_path
        self.ann_index_path = ann_index_path 
        self.tfidf_vectors_path = tfidf_vectors_path
        self.umls_concept_aliases_path = umls_concept_aliases_path
    
        #kb = kb or UmlsKnowledgeBase()
    
        # nmslib hyperparameters (very important)
        # guide: https://github.com/nmslib/nmslib/blob/master/manual/methods.md
        # Default values resulted in very low recall.
    
        # set to the maximum recommended value. Improves recall at the expense of longer indexing time.
        # We use the HNSW (Hierarchical Navigable Small World Graph) representation which is constructed
        # by consecutive insertion of elements in a random order by connecting them to M closest neighbours
        # from the previously inserted elements. These later become bridges between the network hubs that
        # improve overall graph connectivity. (bigger M -> higher recall, slower creation)
        # For more details see:  https://arxiv.org/pdf/1603.09320.pdf?
        m_parameter = 100
        # `C` for Construction. Set to the maximum recommended value
        # Improves recall at the expense of longer indexing time
        construction = 2000
        num_threads = 60  # set based on the machine
        index_params = {
            "M": m_parameter,
            "indexThreadQty": num_threads,
            "efConstruction": construction,
            "post": 0,
        }
    
        print(
            f"No tfidf vectorizer on {tfidf_vectorizer_path} or ann index on {ann_index_path}"
        )
        concept_aliases = list(self.alias_to_cuis.keys())
    
        # NOTE: here we are creating the tf-idf vectorizer with float32 type, but we can serialize the
        # resulting vectors using float16, meaning they take up half the memory on disk. Unfortunately
        # we can't use the float16 format to actually run the vectorizer, because of this bug in sparse
        # matrix representations in scipy: https://github.com/scipy/scipy/issues/7408
        print(f"Fitting tfidf vectorizer on {len(concept_aliases)} aliases")
        tfidf_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 3), min_df=10, dtype=numpy.float32
        )
        start_time = datetime.datetime.now()
        concept_alias_tfidfs = tfidf_vectorizer.fit_transform(concept_aliases)
        print(f"Saving tfidf vectorizer to {tfidf_vectorizer_path}")
        joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print(f"Fitting and saving vectorizer took {total_time.total_seconds()} seconds")
    
        print("Finding empty (all zeros) tfidf vectors")
        empty_tfidfs_boolean_flags = numpy.array(
            concept_alias_tfidfs.sum(axis=1) != 0
        ).reshape(-1)
        number_of_non_empty_tfidfs = sum(empty_tfidfs_boolean_flags == False)  # noqa: E712
        total_number_of_tfidfs = numpy.size(concept_alias_tfidfs, 0)
    
        print(
            f"Deleting {number_of_non_empty_tfidfs}/{total_number_of_tfidfs} aliases because their tfidf is empty"
        )
        # remove empty tfidf vectors, otherwise nmslib will crash
        concept_aliases = [
            alias
            for alias, flag in zip(concept_aliases, empty_tfidfs_boolean_flags)
            if flag
        ]
        concept_alias_tfidfs = concept_alias_tfidfs[empty_tfidfs_boolean_flags]
        assert len(concept_aliases) == numpy.size(concept_alias_tfidfs, 0)
    
        print(
            f"Saving list of concept ids and tfidfs vectors to {umls_concept_aliases_path} and {tfidf_vectors_path}"
        )
        json.dump(concept_aliases, open(umls_concept_aliases_path, "w"))
        scipy.sparse.save_npz(
            tfidf_vectors_path, concept_alias_tfidfs.astype(numpy.float16)
        )
    
        print(f"Fitting ann index on {len(concept_aliases)} aliases (takes 2 hours)")
        start_time = datetime.datetime.now()
        ann_index = nmslib.init(
            method="hnsw",
            space="cosinesimil_sparse",
            data_type=nmslib.DataType.SPARSE_VECTOR,
        )
        ann_index.addDataPointBatch(concept_alias_tfidfs)
        ann_index.createIndex(index_params, print_progress=True)
        ann_index.saveIndex(ann_index_path)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print(f"Fitting ann index took {elapsed_time.total_seconds()} seconds")
    
        return concept_aliases, tfidf_vectorizer, ann_index

    def serialized_kb(self) -> LinkerPaths:
        tupled_kb = LinkerPaths(
            ann_index= self.ann_index_path,  # noqa
            tfidf_vectorizer=self.tfidf_vectorizer_path,  # noqa
            tfidf_vectors=self.tfidf_vectors_path,  # noqa
            concept_aliases_list=self.umls_concept_aliases_path,  # noqa
            alias_to_cuis = self.alias_to_cuis,
            cui_to_entity = self.cui_to_entity
        )
        return tupled_kb

def load_approximate_nearest_neighbours_index(
    serialized_kb : LinkerPaths,
    ef_search: int = 200,
) -> FloatIndex:
    """
    Load an approximate nearest neighbours index from disk.

    Parameters
    ----------
    linker_paths: LinkerPaths, required.
        Contains the paths to the data required for the entity linker.
    ef_search: int, optional (default = 200)
        Controls speed performance at query time. Max value is 2000,
        but reducing to around ~100 will increase query speed by an order
        of magnitude for a small performance hit.
    """
    concept_alias_tfidfs = scipy.sparse.load_npz(
        cached_path(serialized_kb[2])
    ).astype(numpy.float32)
    ann_index = nmslib.init(
        method="hnsw",
        space="cosinesimil_sparse",
        data_type=nmslib.DataType.SPARSE_VECTOR,
    )
    ann_index.addDataPointBatch(concept_alias_tfidfs)
    ann_index.loadIndex(cached_path(serialized_kb[0]))
    query_time_params = {"efSearch": ef_search}
    ann_index.setQueryTimeParams(query_time_params)

    return ann_index
