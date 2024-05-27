

import os
import pickle
import logging
import time
import glob
import math


import multiprocessing
from typing import List, Tuple, Dict, Iterator, Set
from functools import partial
from multiprocessing.dummy import Pool

import numpy as np
import faiss

import torch
import torch.nn as nn
from torch import Tensor as T
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizer,
)

from bioel.models.krissbert.data.utils import BigBioDataset, generate_vectors

# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter(
    "[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)

class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def init_index(self, vector_sz: int):
        raise NotImplementedError

    def index_data(self, data: List[Tuple[object, np.array]]):
        raise NotImplementedError

    def get_index_name(self):
        raise NotImplementedError

    def search_knn(
        self, query_vectors: np.array, top_docs: int
    ) -> List[Tuple[List[object], List[float]]]:
        raise NotImplementedError

    def serialize(self, file: str):
        logger.info("Serializing index to %s", file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + ".index.dpr"
            meta_file = file + ".index_meta.dpr"

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode="wb") as f:
            pickle.dump(self.index_id_to_db_id, f)

    def get_files(self, path: str):
        if os.path.isdir(path):
            index_file = os.path.join(path, "index.dpr")
            meta_file = os.path.join(path, "index_meta.dpr")
        else:
            index_file = path + ".index.dpr"
            meta_file = path + ".index_meta.dpr"
        return index_file, meta_file

    def index_exists(self, path: str):
        index_file, meta_file = self.get_files(path)
        return os.path.isfile(index_file) and os.path.isfile(meta_file)

    def deserialize(self, path: str):
        logger.info("Loading index from %s", path)
        index_file, meta_file = self.get_files(path)

        self.index = faiss.read_index(index_file)
        logger.info(
            "Loaded index of type %s and size %d", type(self.index), self.index.ntotal
        )

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def _update_id_mapping(self, db_ids: List) -> int:
        self.index_id_to_db_id.extend(db_ids)
        return len(self.index_id_to_db_id)


class DenseFlatIndexer(DenseIndexer):
    def __init__(self, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)

    def init_index(self, vector_sz: int):
        self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, data: List[Tuple[object, np.array]]):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            db_ids = [t[0] for t in data[i : i + self.buffer_size]]
            vectors = [
                np.reshape(t[1], (1, -1)) for t in data[i : i + self.buffer_size]
            ]
            vectors = np.concatenate(vectors, axis=0)
            total_data = self._update_id_mapping(db_ids)
            self.index.add(vectors)
            logger.info("data indexed %d", total_data)

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info("Total data indexed %d", indexed_cnt)

    def search_knn(
        self,
        query_vectors: np.array,
        top_docs: int,
        batch_size: int = 4096,
    ) -> List[Tuple[List[object], List[float]]]:
        num_queries = query_vectors.shape[0]
        scores, indexes = [], []
        for start in range(0, num_queries, batch_size):
            logger.info(f"Searched {start} queries.")
            batch_vectors = query_vectors[start : start + batch_size]
            batch_scores, batch_indexes = self.index.search(batch_vectors, top_docs)
            scores.extend(batch_scores)
            indexes.extend(batch_indexes)
        # convert to external ids
        db_ids = [
            [self.index_id_to_db_id[i] for i in query_top_idxs]
            for query_top_idxs in indexes
        ]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

    def get_index_name(self):
        return "flat_index"


def iterate_encoded_files(
    vector_files: list,
    candidate_ids: Set = None,
    umls_data: Dict = None,
) -> Iterator:
    logger.info("Loading encoded prototype embeddings...")
    proto_data = {}
    for file in vector_files:
        logger.info("Reading file %s", file)
        with open(file, "rb") as reader:
            for meta, vec in pickle.load(reader):
                cuis = meta["cuis"]
                if candidate_ids and all(c not in candidate_ids for c in cuis):
                    continue
                for cui in cuis:
                    proto_data.setdefault(cui, []).append((meta, vec))
    # Concatenate prototype embs with additional knowledge embs from UMLS.
    if umls_data is not None:
        for cui, (meta, vec) in umls_data.items():
            if cui in proto_data:
                for _, _vec in proto_data.pop(cui):
                    extended_vec = np.concatenate((vec, _vec), axis=0)
                    yield (meta, extended_vec)
            else:
                extended_vec = np.concatenate((vec, np.zeros_like(vec)), axis=0)
                yield (meta, extended_vec)
    for cui in list(proto_data.keys()):
        for meta, vec in proto_data.pop(cui):
            extended_vec = np.concatenate((np.zeros_like(vec), vec), axis=0)
            yield (meta, extended_vec)
    assert len(proto_data) == 0

class Krissbert(nn.Module):
    def __init__(self, model_name_or_path="microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.index = DenseFlatIndexer()
        self.index.init_index(self.config.hidden_size * 2)

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)

    def generate_prototypes(self, dataset, output_dir, batch_size=256, max_length=64):
        self.encoder.eval()
        data = generate_vectors(self.encoder, self.tokenizer, dataset, batch_size, max_length, is_prototype=True)
        output_dir = output_dir.rstrip("/")
        if isinstance(dataset, BigBioDataset):
            output_prototypes = f'{output_dir}/{dataset.dataset_name}_embeddings.pickle'
            output_name_cuis = f'{output_dir}/{dataset.dataset_name}_name_cuis.txt'
        else:
            output_prototypes = f'{output_dir}/embeddings.pickle'
            output_name_cuis = f'{output_dir}/name_cuis.txt'

        os.makedirs(os.path.dirname(output_prototypes), exist_ok=True)
        logger.info("Writing results to %s", output_prototypes)
        with open(output_prototypes, mode="wb") as f:
            pickle.dump(data, f)
        with open(output_name_cuis, "w") as f:
            for name, cuis in dataset.name_to_cuis.items():
                f.write("|".join(cuis) + "||" + name + "\n")
        logger.info("Total data processed %d. Written to %s", len(data), output_prototypes)

    def generate_mention_vectors(self, ds, batch_size=256, max_length=64):
        self.encoder.eval()
        return generate_vectors(
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            dataset=ds,
            batch_size=batch_size,
            max_length=max_length,
        )
    
    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int,
        candidate_ids: Set = None,
        umls_data: Dict = None,
    ):
        """
        Indexes encoded data takes form a list of files
        :param vector_files: a list of files
        :param buffer_size: size of a buffer to send for the indexing at once
        :return:
        """
        buffer = []
        for i, item in enumerate(
            iterate_encoded_files(vector_files, candidate_ids, umls_data)
        ):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_hits(
        self, mention_vectors: np.array, top_k: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching given the mention vectors batch
        """
        time0 = time.time()
        search = partial(
            self.index.search_knn,
            top_docs=top_k,
        )
        results = []
        num_processes = multiprocessing.cpu_count()
        shard_size = math.ceil(mention_vectors.shape[0] / num_processes)
        shards = []
        for i in range(0, mention_vectors.shape[0], shard_size):
            shards.append(mention_vectors[i : i + shard_size])
        with Pool(processes=num_processes) as pool:
            it = pool.map(search, shards)
            for ret in it:
                results += ret
            # results = self.index.search_knn(mention_vectors, top_k)
        logger.info("index search time: %f sec.", time.time() - time0)
        self.index = None
        return results