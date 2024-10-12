import argparse
import logging
import os
import torch
import pickle
import ujson
from tqdm.auto import tqdm

from bioel.models.sapbert.data.utils import DictionaryDataset


LOGGER = logging.getLogger()


def parse_args_bigbio():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="sapbert evaluation")

    # Required
    parser.add_argument("--model_dir", required=True, help="Directory for model")
    parser.add_argument(
        "--dictionary_path", type=str, required=True, help="dictionary path"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=[
            "medmentions_full",
            "medmentions_st21pv",
            "bc5cdr",
            "nlmchem",
            "nlm_gene",
            "gnormplus",
            "ncbi_disease",
        ],
        help="data set to evaluate",
    )
    parser.add_argument("--dict_cache_path", type=str)

    # Run settings
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument(
        "--output_dir", type=str, default="./output/", help="Directory for output"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dict_batch_size", type=int, default=8192)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--split",
        default="test",
        type=str,
        help="Which datasplit to evaluate model performance on",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run model with small data subset as a test of functionality",
    )

    parser.add_argument(
        "--abbreviations_path",
        type=str,
        help="Path to .json file of abbreviations in each article",
    )

    # Tokenizer settings
    parser.add_argument("--max_length", default=25, type=int)

    # Classification settings
    parser.add_argument(
        "--agg_mode", type=str, default="cls", help="{cls|mean_pool|nospec}"
    )

    args = parser.parse_args()
    return args


def make_unique_model_savepath(dir, model, dataset, file_ext):
    # Ensure the directory exists
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Create the base filename
    base_filename = f"{model}_{dataset}.{file_ext}"
    unique_filename = base_filename
    counter = 1

    # Append a number to the filename until a unique one is found
    while os.path.exists(os.path.join(dir, unique_filename)):
        unique_filename = f"{model}_{dataset}_{counter}.{file_ext}"
        counter += 1

    return os.path.join(dir, unique_filename)


def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s: [ %(message)s ]", "%m/%d/%Y %I:%M:%S %p")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


def load_dictionary(dictionary_path):
    dictionary = DictionaryDataset(dictionary_path=dictionary_path)
    return dictionary.data


def sapbert_predict(
    model_wrapper,
    eval_dictionary,
    dataloader,
    batch_size: int = 64,
    dict_batch_size: int = 8192,
    topk: int = 25,
    agg_mode: str = "cls",
    dict_cache_filepath=None,
    db_name: str = "UMLS",
    debug=False,
):
    """
    Make predictions for SapBERT given a particular dict of medical concepts
    """
    # # Quick test of dataloader
    # for i, dat in enumerate(dataloader):
    #     print(ujson.dumps(dat))

    print("[start embedding dictionary...]")
    # embed dictionary
    dict_names = [row[0] for row in eval_dictionary]

    if debug:
        dict_names = dict_names[:4096]

    if dict_cache_filepath is not None:
        if os.path.isfile(dict_cache_filepath):
            print(f"Found cached dict at {dict_cache_filepath}.  Loading.")
            dict_dense_embeds = torch.load(
                dict_cache_filepath,
            )
        else:
            dict_dense_embeds = model_wrapper.embed_dense(
                names=dict_names,
                show_progress=True,
                batch_size=dict_batch_size,
                agg_mode=agg_mode,
            )
            print(f"Saving dict to {dict_cache_filepath}")
            torch.save(
                dict_dense_embeds,
                dict_cache_filepath,
                pickle_protocol=pickle.HIGHEST_PROTOCOL,
            )
    else:
        dict_dense_embeds = model_wrapper.embed_dense(
            names=dict_names,
            show_progress=True,
            batch_size=dict_batch_size,
            agg_mode=agg_mode,
        )

    # print ("dict_dense_embeds.shape:", dict_dense_embeds.shape)

    print("[computing rankings...]")

    results = []
    for batch in tqdm(dataloader):
        # for i in tqdm(np.arange(0,len(eval_queries),batch_size), total=len(eval_queries)//batch_size+1):
        # mentions = list(eval_queries[i:i+batch_size][:,0])
        mentions = batch[0]
        mention_dense_embeds = model_wrapper.embed_dense(
            names=mentions, agg_mode=agg_mode
        )
        metadata = batch[-1]

        # get score matrix
        dense_score_matrix = model_wrapper.get_score_matrix(
            query_embeds=mention_dense_embeds, dict_embeds=dict_dense_embeds
        )
        score_matrix = dense_score_matrix
        candidate_idxs_batch = model_wrapper.retrieve_candidate_cuda(
            score_matrix=score_matrix,
            topk=topk,
            batch_size=batch_size,
            show_progress=False,
        )

        for i, idx_list in enumerate(candidate_idxs_batch):
            np_candidates = [eval_dictionary[ind] for ind in idx_list.tolist()]
            # print(np_candidates)
            candidates_extra = [{"text": x[0], "db_id": x[1]} for x in np_candidates]
            candidates = [x[1].split("|") for x in np_candidates]
            metadata[i]["candidates"] = candidates
            metadata[i]["candidates_metadata"] = candidates_extra
            if len(results) == 0 and debug:
                LOGGER.debug(metadata)
        results.extend(metadata)
        # print(f"results : {results[:10]}")

        if debug:
            break

    return results
