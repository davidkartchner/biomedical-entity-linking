import argparse
import os
import glob
from typing import List, Tuple, Dict, Iterator, Set
import torch
import numpy as np
import ujson
from bioel.models.krissbert.model.model import Krissbert
from bioel.models.krissbert.data.utils import BigBioDataset
from transformers import set_seed

import logging

# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter(
    "[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)

def load_umls_data(files_patterns: List[str], candidate_ids: Dict = None) -> Dict:
    input_paths = []
    for pattern in files_patterns:
        pattern_files = glob.glob(pattern)
        input_paths.extend(pattern_files)
    umls_data = {}
    for file in sorted(input_paths):
        logger.info("Reading encoded UMLS data from file %s", file)
        with open(file, "rb") as reader:
            for meta, vec in pickle.load(reader):
                assert len(meta["cuis"]) == 1, breakpoint()
                cui = meta["cuis"][0]
                if candidate_ids and cui not in candidate_ids:
                    continue
                umls_data[cui] = (meta, vec)
    logger.info(f"Loaded UMLS data = {len(umls_data)}.")
    return umls_data

def hit(pred: List[str], gold: List[str]) -> bool:
    return all(p in gold for p in pred)

def dedup_ids(ids: List[Dict]) -> List[Dict]:
    deduped_ids = []
    seen_cuis = set()
    for d in ids:
        if all(cui in seen_cuis for cui in d["cuis"]):
            continue
        seen_cuis.update(d["cuis"])
        deduped_ids.append(d)
    return deduped_ids

def evaluate(
    ds: torch.utils.data.Dataset,
    result_ent_ids: List[Tuple[List[object], List[float]]],
    lookup_table: str,
    output_path: str,
    top_ks: List[int] = (1, 5, 50, 100),
) -> List[Dict]:
    """
    Evaluate model
    """
    # Load lookup table
    # Each entity name is mapped to
    lut = {}
    with open(lookup_table, encoding="utf-8") as f:
        for ln in f:
            cuis, name = ln.strip().split("||")
            cuis = cuis.split("|")
            lut[name] = cuis

    n = len(ds)
    top_k_hits = {top_k: 0 for top_k in top_ks}


    all_output = []
    for i in range(len(result_ent_ids)):
        d = ds[i]
        ids, _ = result_ent_ids[i]
        # logger.info(ids)
        ids = dedup_ids(ids)
        ids = ids[: max(top_ks)]
        candidates = [
            {"cuis": eid["cuis"], "hit": int(hit(pred=eid["cuis"], gold=d.cuis))}
            for eid in ids
        ]
        lut_cuis = lut.get(d.mention, [])
        if len(lut_cuis) == 1:
            # If the mention only has one ID in the look up table,
            # we use the ID as the top prediction.
            candidates.insert(
                0, {"cuis": lut_cuis, "hit": int(hit(pred=lut_cuis, gold=d.cuis))}
            )

        output_candidates = [x['cuis'] for x in candidates]
        output = d.to_dict()
        # print(output)
        output['candidates'] = output_candidates
        # output['hits'] = [x['hit'] for x in candidates]
        all_output.append(output)

    

        for top_k in top_k_hits:
            if any(c["hit"] for c in candidates[:top_k]):
                top_k_hits[top_k] += 1

    with open(output_path, "w") as f:
        f.write(ujson.dumps(all_output, indent=2))

    top_k_acc = {top_k: v / n for top_k, v in top_k_hits.items()}
    logger.info("Top-k accuracy %s", top_k_acc)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Krissbert model")
    parser = argparse.ArgumentParser(description="Generate Krissbert prototypes")
    parser.add_argument("--model_name_or_path", type=str,
                        default="microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL",
                        help="Path to the pre-trained model")
    parser.add_argument("--dataset_name", type=str, default="nlmchem", help="Name of the dataset")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for generating prototypes")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default='./results/', help="Directory to save the results")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--num_retrievals", type=int, default=100, help="Number of retrievals")
    parser.add_argument("--index_path", type=str, default=None, help="Path to save the Faiss index")

    top_ks = [1, 5, 50, 100]
    encoded_umls_files = []
    encoded_files = []
    entity_list_names = None
    entity_list_ids = None
    

    args = parser.parse_args()

    set_seed(args.seed)

    # Load the saved Krissbert model
    model = Krissbert(args.model_name_or_path)
    model.cuda()

    # Load the dataset
    ds = BigBioDataset(args.dataset_name, splits=["test"])

    # candidate ids
    candidate_ids = None
    if entity_list_ids:
        with open(entity_list_ids, encoding="utf-8") as f:
            candidate_ids = set(f.read().split("\n"))

    # Start indexing
    input_paths = []
    for pattern in encoded_files:
        pattern_files = glob.glob(pattern)
        input_paths.extend(pattern_files)
    input_paths = sorted(set(input_paths))

    if len(input_paths) == 0:
        input_paths.append(f'prototypes/{args.dataset_name}_embeddings.pickle')

    mentions_tensor = model.generate_mention_vectors(ds)

    # Load UMLS knowledge
    umls_data = None
    if encoded_umls_files:
        umls_data = load_umls_data(encoded_umls_files, candidate_ids)

    index_path = args.index_path
    if index_path and model.index.index_exists(index_path):
        logger.info("Index path: %s", index_path)
        model.index.deserialize(index_path)
    else:
        logger.info("Indexing encoded data from files: %s", input_paths)
        model.index_encoded_data(
            vector_files=input_paths,
            buffer_size=model.index.buffer_size,
            candidate_ids=candidate_ids,
            umls_data=umls_data,
        )
        if index_path:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            model.index.serialize(index_path)

    # Encode test data.
    mentions_tensor = torch.cat([mentions_tensor, mentions_tensor], dim=1)

    # To get k different entities, we retrieve 32 * k mentions and then dedup.
    top_ids_and_scores = model.get_top_hits(
        mentions_tensor.numpy(), args.num_retrievals * 32
    )

    if entity_list_names:
        entity_list_names = entity_list_names
    else:
        entity_list_names = f'prototypes/{args.dataset_name}_name_cuis.txt'

    output_path = f"{args.output_dir.rstrip('/')}/{args.dataset_name}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    evaluate(ds, top_ids_and_scores, entity_list_names, output_path=output_path, top_ks=top_ks)

if __name__ == "__main__":
    main()