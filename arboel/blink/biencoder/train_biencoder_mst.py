import os
import random
import time
import pickle
import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)  # DD1 Create mini batches of your data and can sample it sequentially or shuffled it
from pytorch_transformers.optimization import (
    WarmupLinearSchedule,
)  # D Used for learning rate scheduling
from scipy.sparse.csgraph import (
    minimum_spanning_tree,
)  # J Compute the MST of an undirected graph represented as a sparse matrix

# csgraph = compressed sparse graph
from scipy.sparse import (
    csr_matrix,
)  # J  Memory-efficient format for representing sparse matrices. Stores the non-zero values of the matrix along their indexes (row,column=)

# csr_matrix = compressed sparse row matrices
from collections import (
    Counter,
)  # J Used for counting the occurrences of elements in an iterable, such as a list, tuple, or string

# Import from another repo (need to check later)
import sys

sys.path.append("../..")
import blink.biencoder.data_process_mult as data_process  # Process data into a suitable format for transformer-based model like BERT
import blink.biencoder.eval_cluster_linking as eval_cluster_linking
import blink.candidate_ranking.utils as utils
from blink.biencoder.biencoder import BiEncoderRanker
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser

from IPython import embed  # J Create an interactive shell using embed()

# For example, you can print the values of variables, test expressions, and modify variables to understand and diagnose issues in your code.


def evaluate(
    reranker,
    valid_dict_vecs,
    valid_men_vecs,
    device,
    logger,
    knn,
    n_gpu,
    entity_data,
    query_data,
    silent=False,
    use_types=False,
    embed_batch_size=768,
    force_exact_search=False,
    probe_mult_factor=1,
    within_doc=False,
    context_doc_ids=None,
):
    """
    Description
    -----------
    1) Computes embeddings and indexes for entities and mentions.
    2) Performs k-nearest neighbors (k-NN) search to establish relationships between them.
    3) Constructs graphs based on these relationships.
    4) Evaluates the model's accuracy by analyzing how effectively the model can link mentions to the correct entities.

    Parameters
    ----------
    reranker : BiEncoderRanker
        NN-based ranking model
    valid_dict_vec : list or ndarray
        Ground truth dataset containing the entities
    valid_men_vecs : list or ndarray
        Dataset containing mentions
    device : str
        cpu or gpu
    logger : 'Logger' object
        Logging object used to record messages
    knn : int
        Number of neighbors
    n_gpu : int
        Number of gpu
    entity_data : list or dict
        Entities from the data
    query_data : list or dict
        Queries / mentions against which the entities are evaluated
    silent=False : bool
        When set to "True", likely suppresses the output or logging of progress updates to keep the console output clean.
    use_types=False : bool
        A boolean flag that indicates whether or not to use type-specific indexes for entities and mentions
    embed_batch_size=768 : int
        The batch size to use when processing embeddings.
    force_exact_search=False : bool
        force the embedding process to use exact search methods rather than approximate methods.
    probe_mult_factor=1 : int
        A multiplier factor used in index building for probing in case of approximate search (bigger = better but slower)
    within_doc=False : bool
        Boolean flag that indicates whether the evaluation should be constrained to within-document contexts
    context_doc_ids=None : bool
        This would be used in conjunction with within_doc to limit evaluations within the same document.
    """
    torch.cuda.empty_cache()  # Empty the CUDA cache to free up GPU memory

    reranker.model.eval()  # DD2 Puts the reranker model in evaluation mode
    n_entities = len(valid_dict_vecs)  # total number of entities
    n_mentions = len(valid_men_vecs)  # total number of mentions
    max_knn = 8  # max number of neighbors

    joint_graphs = (
        {}
    )  # Store results of the NN search and distance between entities and mentions

    for k in [0, 1, 2, 4, 8]:
        joint_graphs[k] = {  # DD3
            "rows": np.array([]),
            "cols": np.array([]),
            "data": np.array([]),
            "shape": (n_entities + n_mentions, n_entities + n_mentions),
        }

    "1) Computes embeddings and indexes for entities and mentions. "
    """
    This block is preparing the data for evaluation by transforming raw vectors into a format that can be efficiently used for retrieval and comparison operations
    """
    if use_types:  # corpus = entity data
        # corpus is a collection of entities, which is used to build type-specific search indexes if provided.
        """
        With a Corpus : Multiple type-specific indexes are created, allowing for more targeted and efficient searches within specific categories of entities.
        'dict_embeds' and 'men_embeds': The resulting entity and mention embeddings.
        'dict_indexes' and 'men_indexes': Dictionary that will store search indexes (!= indices)for each unique entity type found in the corpus
        'dict_idxs_by_type' and 'men_idxs_by_type': Dictionary to store indices of the corpus elements, grouped by their entity type.
        !!! idxs = indices / indexes = indexes !!!
        """
        logger.info("Eval: Dictionary: Embedding and building index")  # For entities
        dict_embeds, dict_indexes, dict_idxs_by_type = data_process.embed_and_index(
            reranker,
            valid_dict_vecs,
            encoder_type="candidate",
            n_gpu=n_gpu,
            corpus=entity_data,
            force_exact_search=force_exact_search,
            batch_size=embed_batch_size,
            probe_mult_factor=probe_mult_factor,
        )
        logger.info("Eval: Queries: Embedding and building index")  # For mentions
        men_embeds, men_indexes, men_idxs_by_type = data_process.embed_and_index(
            reranker,
            valid_men_vecs,
            encoder_type="context",
            n_gpu=n_gpu,
            corpus=query_data,
            force_exact_search=force_exact_search,
            batch_size=embed_batch_size,
            probe_mult_factor=probe_mult_factor,
        )
    else:  # corpus = None
        """
        Without a Corpus: A single, general index is created for all embeddings, suitable for broad searches across the entire dataset.
        'dict_embeds' and 'men_embeds': The resulting entity and mention embeddings.
        'dict_index' and 'men_index': Dictionary that will store search index
        """
        logger.info("Eval: Dictionary: Embedding and building index")
        dict_embeds, dict_index = data_process.embed_and_index(
            reranker,
            valid_dict_vecs,
            "candidate",
            n_gpu=n_gpu,
            force_exact_search=force_exact_search,
            batch_size=embed_batch_size,
            probe_mult_factor=probe_mult_factor,
        )
        logger.info("Eval: Queries: Embedding and building index")
        men_embeds, men_index = data_process.embed_and_index(
            reranker,
            valid_men_vecs,
            "context",
            n_gpu=n_gpu,
            force_exact_search=force_exact_search,
            batch_size=embed_batch_size,
            probe_mult_factor=probe_mult_factor,
        )

    "2) Performs k-nearest neighbors (k-NN) search to establish relationships between mentions and entities."
    logger.info(
        "Eval: Starting KNN search..."
    )  # An informational message is logged to indicate that the k-NN search is starting.
    # Fetch recall_k (default 16) knn entities for all mentions
    # Fetch (k+1) NN mention candidates; fetching all mentions for within_doc to filter down later
    n_men_to_fetch = (
        len(men_embeds) if within_doc else max_knn + 1
    )  # Number of mentions to fetch
    if not use_types:  # Only one index so only need one search
        nn_ent_dists, nn_ent_idxs = dict_index.search(
            men_embeds, 1
        )  # DD4/DD5 #return the distance and the indice of the closest entity for all mentions in men_embeds
        nn_men_dists, nn_men_idxs = men_index.search(
            men_embeds, n_men_to_fetch
        )  # return the distances and the indices of the k closest mentions for all mentions in men_embeds
    else:  # C Several indexes corresponding to the different entities in entity_data so we can use the specific search index
        # DD6
        # DD7
        nn_ent_idxs = -1 * np.ones(
            (len(men_embeds), 1), dtype=int
        )  # Indice of the closest entity for all mentions in men_embeds
        nn_ent_dists = -1 * np.ones(
            (len(men_embeds), 1), dtype="float64"
        )  # Distance of the closest entity for all mentions in men_embeds
        nn_men_idxs = -1 * np.ones(
            (len(men_embeds), n_men_to_fetch), dtype=int
        )  # Indice of k closest mentions for all mentions in men_embeds
        nn_men_dists = -1 * np.ones(
            (len(men_embeds), n_men_to_fetch), dtype="float64"
        )  # Distance of the k closest mentions for all mentions in men_embeds
        for entity_type in men_indexes:
            # CC3 Creates a new list only containing the mentions for which type = entity_types
            men_embeds_by_type = men_embeds[
                men_idxs_by_type[entity_type]
            ]  # Only want to search the mentions that belongs to a specific type of entity.
            # Returns the distance and the indice of the closest entity for all mentions in men_embeds by entity type
            nn_ent_dists_by_type, nn_ent_idxs_by_type = dict_indexes[
                entity_type
            ].search(men_embeds_by_type, 1)
            nn_ent_idxs_by_type = np.array(  # CC4 DD8
                list(  # DD9
                    map(  # lambda x : acts as a function
                        lambda x: dict_idxs_by_type[entity_type][x], nn_ent_idxs_by_type
                    )  # nn_ent_idxs_by_type is the iterable being processed by the map function
                    # Each element within nn_ent_idxs_by_type is passed to the lambda function as x.
                )  # map alone would return an object, that's why need a list
            )
            # Returns the distance and the indice of the k closest mentions for all mention in men_embeds by entity type
            # Note that here we may not necessarily have k mentions in each entity type which is why we use min(k,len(men_embeds_by_type))
            nn_men_dists_by_type, nn_men_idxs_by_type = men_indexes[entity_type].search(
                men_embeds_by_type, min(n_men_to_fetch, len(men_embeds_by_type))
            )
            nn_men_idxs_by_type = np.array(
                list(
                    map(lambda x: men_idxs_by_type[entity_type][x], nn_men_idxs_by_type)
                )
            )
            for i, idx in enumerate(men_idxs_by_type[entity_type]):  # CC5
                nn_ent_idxs[idx] = nn_ent_idxs_by_type[i]
                nn_ent_dists[idx] = nn_ent_dists_by_type[i]
                nn_men_idxs[idx][: len(nn_men_idxs_by_type[i])] = nn_men_idxs_by_type[i]
                nn_men_dists[idx][: len(nn_men_dists_by_type[i])] = (
                    nn_men_dists_by_type[i]
                )
    logger.info(
        "Eval: Search finished"
    )  # An informational message is logged to indicate that the k-NN search is finished

    "3) Constructs graphs based on these relationships."
    """
    nn_ent_dists contain information about distance of the closest entity
    nn_ent_idxs contain information about indice of the closest entity
    nn_men_dists contain information about distance of the k nearest mentions
    nn_men_idxs contain information about indice of the k nearest mentions
    - We can fill in the "rows" part (=start nodes) of the graph in the order of the mentions
    - We can fill in the "cols" part (=end nodes) of the graph with nn_ent_idxs and nn_men_idxs
    - We can fill in the "data" part (=weights) of the graph with nn_ent_dists and nn_men_dists
    """
    logger.info("Eval: Building graphs")
    for men_query_idx, men_embed in enumerate(
        tqdm(men_embeds, total=len(men_embeds), desc="Eval: Building graphs")
    ):
        # Get nearest entity candidate
        dict_cand_idx = nn_ent_idxs[men_query_idx][
            0
        ]  # Use of [0] to retrieve a scalar and not an 1D array
        dict_cand_score = nn_ent_dists[men_query_idx][0]

        # Filter candidates to remove -1s, mention query, within doc (if reqd.), and keep only the top k candidates
        filter_mask_neg1 = (
            nn_men_idxs[men_query_idx] != -1
        )  # bool ndarray. Ex : np.array([True, False, True, False])
        men_cand_idxs = nn_men_idxs[men_query_idx][
            filter_mask_neg1
        ]  # Only keep the elements != -1
        men_cand_scores = nn_men_dists[men_query_idx][filter_mask_neg1]

        if within_doc:
            men_cand_idxs, wd_mask = filter_by_context_doc_id(
                men_cand_idxs,
                context_doc_ids[men_query_idx],
                context_doc_ids,
                return_numpy=True,
            )
            men_cand_scores = men_cand_scores[wd_mask]

        # Filter self-reference + limits the number of candidate to 'max_knn'
        filter_mask = men_cand_idxs != men_query_idx
        men_cand_idxs, men_cand_scores = (
            men_cand_idxs[filter_mask][:max_knn],
            men_cand_scores[filter_mask][:max_knn],
        )

        # Add edges to the graphs
        for k in joint_graphs:
            joint_graph = joint_graphs[
                k
            ]  # There is no "s" in "joint_graph", it's not the same !
            # Add mention-entity edge
            joint_graph["rows"] = (
                np.append(  # Mentions are offset by the total number of entities to differentiate mention nodes from entity nodes
                    joint_graph["rows"], [n_entities + men_query_idx]
                )
            )
            joint_graph["cols"] = np.append(joint_graph["cols"], dict_cand_idx)
            joint_graph["data"] = np.append(joint_graph["data"], dict_cand_score)
            if k > 0:
                # Add mention-mention edges
                joint_graph["rows"] = np.append(
                    joint_graph["rows"],
                    [n_entities + men_query_idx]
                    * len(
                        men_cand_idxs[:k]
                    ),  # creates an array where the starting node (current mention) is repeated len(men_cand_idxs[:k]) times
                )
                joint_graph["cols"] = np.append(
                    joint_graph["cols"], n_entities + men_cand_idxs[:k]
                )
                joint_graph["data"] = np.append(
                    joint_graph["data"], men_cand_scores[:k]
                )

    "4) Evaluates the model's accuracy by analyzing how effectively the model can link mentions to the correct entities."
    max_eval_acc = -1.0
    for k in joint_graphs:
        logger.info(f"\nEval: Graph (k={k}):")
        # Partition graph based on cluster-linking constraints (inference procedure)
        partitioned_graph, clusters = eval_cluster_linking.partition_graph(
            joint_graphs[k], n_entities, directed=True, return_clusters=True
        )
        # Infer predictions from clusters
        result = eval_cluster_linking.analyzeClusters(
            clusters, entity_data, query_data, k
        )
        acc = float(result["accuracy"].split(" ")[0])
        max_eval_acc = max(acc, max_eval_acc)
        logger.info(f"Eval: accuracy for graph@k={k}: {acc}%")
    logger.info(f"Eval: Best accuracy: {max_eval_acc}%")
    return max_eval_acc, (
        {
            "dict_embeds": dict_embeds,
            "dict_indexes": dict_indexes,
            "dict_idxs_by_type": dict_idxs_by_type,
        }
        if use_types
        else {"dict_embeds": dict_embeds, "dict_index": dict_index}
    )


def get_optimizer(model, params):
    """
    Description
    -----------
    Constructs and returns an optimizer configured for the provided model, using the specified optimization type and learning rate.

    Parameters
    ----------
    model : Bert encoder
        Encoder
    params : dict(str)
        dictionary containing configuration options that affect how the optimizer is set up.
    """
    return get_bert_optimizer(
        [model],
        params["type_optimization"],  # A31
        params["learning_rate"],  # A32
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    """
    Description
    -----------
    Creates a learning rate scheduler for the training process, based on the provided parameters and the optimizer

    Parameters
    ----------
    params : dict(str)
        dictionary containing configuration options that affect how the scheduler is set up.
    optimizer :
        The optimizer for which the scheduler will adjust the learning rate for.
    len_train_data : int
        The total number of training data points.
    logger :
        An object used for logging messages during the execution of the function.
    """
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])  # A33

    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=num_warmup_steps,
        t_total=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def load_optimizer_scheduler(params, logger):
    """
    Description
    -----------
    Attempts to load and return a previously saved optimizer and scheduler state from disk, based on the model path provided in the parameters

    Parameters
    ----------
    params : dict(str)
        dictionary containing configuration options.
    logger :
        An object used for logging messages during the execution of the function, including whether the loading operation was successful.
    """
    optim_sched = None
    model_path = params["path_to_model"]
    if model_path is not None:
        model_dir = os.path.dirname(model_path)
        optim_sched_fpath = os.path.join(model_dir, utils.OPTIM_SCHED_FNAME)
        if os.path.isfile(optim_sched_fpath):
            logger.info(
                f"Loading stored optimizer and scheduler from {optim_sched_fpath}"
            )
            optim_sched = torch.load(optim_sched_fpath)
    return optim_sched


def read_data(split, params, logger):
    """
    Description
    -----------
    Loads dataset samples from a specified path
    Optionally filters out samples without labels
    Checks if the dataset supports multiple labels per sample
    "has_mult_labels" : bool

    Parameters
    ----------
    split : str
        Indicates the portion of the dataset to load ("train", "test", "valid"), used by utils.read_dataset to determine which data to read.
    params : dict(str)
        Contains configuration options
    logger :
        An object used for logging messages about the process, such as the number of samples read.
    """
    samples = utils.read_dataset(split, params["data_path"])  # DD21
    # Check if dataset has multiple ground-truth labels
    has_mult_labels = "labels" in samples[0].keys()
    if params["filter_unlabeled"]:
        # Filter samples without gold entities
        samples = list(
            filter(
                lambda sample: (
                    (len(sample["labels"]) > 0)
                    if has_mult_labels
                    else (sample["label"] is not None)
                ),
                samples,
            )
        )
    logger.info(f"Read %d {split} samples." % len(samples))
    return samples, has_mult_labels


def filter_by_context_doc_id(mention_idxs, doc_id, doc_id_list, return_numpy=False):
    # CC6
    """
    Description
    -----------
    Filters and returns mention indices that belong to a specific document identified by "doc_id".
    Ensures that the analysis are constrained within the context of that particular document.

    Parameters
    ----------
    - mention_idxs : ndarray(int) of dim = (number of mentions)
    Represents the indices of mentions
    - doc_id : int
    Indice of the target document
    - doc_id_list : ndarray(int) of dim = (number of mentions)
    Array of integers, where each element is a document ID associated with the corresponding mention in mention_idxs.
    The length of doc_id_list should match the total number of mentions referenced in mention_idxs.
    - return_numpy : bool
    A flag indicating whether to return the filtered list of mention indices as a NumPy array.
    If True, the function returns a NumPy array; otherwise, it returns a list
    -------
    Outputs:
    - mask : ndarray(bool) of dim = (number of mentions)
    Mask indicating where each mention's document ID (from doc_id_list) matches the target doc_id
    - mention_idxs :
    Only contains mention indices that belong to the target document (=doc_id).
    """
    mask = [doc_id_list[i] == doc_id for i in mention_idxs]
    if isinstance(mention_idxs, list):  # Test if mention_idxs = list. Return a bool
        mention_idxs = np.array(mention_idxs)
    mention_idxs = mention_idxs[
        mask
    ]  # possible only if mention_idxs is an array, not a list
    if not return_numpy:
        mention_idxs = list(mention_idxs)
    return mention_idxs, mask


def main(params):

    "I. Define the paths where all the files are located (in pickle format)"
    ontology = "medic"
    model = "arboel"
    dataset = "ncbi_disease"
    abs_path = "/home2/cye73/data"
    data_path = os.path.join(abs_path, model, dataset)
    print(data_path)
    abs_path2 = "/home2/cye73/results"
    model_output_path = os.path.join(abs_path2, model, dataset)

    ontology_type = "umls"
    umls_dir = "/mitchell/entity-linking/2017AA/META/"

    params_test = {
        "model_output_path": model_output_path,
        "data_path": data_path,
        "knn": 4,
        "use_types": False,
        "max_context_length": 64,
        "max_cand_length": 64,
        "context_key": "context",  # to specify context_left or context_right
        "debug": True,
        "gold_arbo_knn": 4,
        "within_doc": True,
        "within_doc_skip_strategy": False,
        "batch_size": 128,  # batch_size = embed_batch_size
        "train_batch_size": 128,
        "filter_unlabeled": False,
        "type_optimization": "all",
        # 'additional_layers', 'top_layer', 'top4_layers', 'all_encoder_layers', 'all'
        "learning_rate": 3e-5,
        "warmup_proportion": 464,
        "fp16": False,
        "embed_batch_size": 128,
        "force_exact_search": True,
        "probe_mult_factor": 1,
        "pos_neg_loss": True,
        "use_types_for_eval": True,
        "drop_entities": False,
        "drop_set": False,
        "farthest_neighbor": True,
        "rand_gold_arbo": True,
        "bert_model": "michiyasunaga/BioLinkBERT-base",
        "out_dim": 768,
        "pull_from_layer": 11,  # 11 for base and 23 for large
        "add_linear": True,
        "max_grad_norm": 0,
        "gradient_accumulation_steps": 3,
    }
    "I.1) Path where the model outputs and logs should be saved"
    # RR2 : model_output_path no longer needed in lightning
    model_output_path = params[
        "output_path"
    ]  # A1 Path where the model outputs and logs should be saved
    if not os.path.exists(model_output_path):
        os.makedirs(
            model_output_path
        )  # If it doesn't exists, create the path specified in params["output_path"]
    logger = utils.get_logger(params["output_path"])  # DD10

    "I.2). Path to the directory where all the preprocessed data are (pickle format)"
    pickle_src_path = params[
        "pickle_src_path"
    ]  # A2 #DD11 Path to the directory containing preprocessed data in pickle format
    if pickle_src_path is None or not os.path.exists(pickle_src_path):
        pickle_src_path = model_output_path

    knn = params["knn"]  # A3
    use_types = params["use_types"]  # A4
    gold_arbo_knn = params[
        "gold_arbo_knn"
    ]  # A5  # Number of gold nearest neighbors to consider

    within_doc = params["within_doc"]  # A6

    "II. Model Initialization"
    # Init model
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    # RR1
    device = reranker.device
    n_gpu = reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:  # A7
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    params["train_batch_size"] = (  # A8
        params["train_batch_size"] // params["gradient_accumulation_steps"]  # DD12
    )
    train_batch_size = params["train_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # RR3 : Done in the Trainer
    # Fix the random seeds
    seed = params["seed"]  # A9
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # RR3 : I removed this part where we save the processed data so that it's faster if we want to use it in the future
    entity_dictionary_loaded = False
    # Full path to entity dictionary file
    entity_dictionary_pkl_path = os.path.join(
        pickle_src_path, "entity_dictionary.pickle"
    )  # DD12A
    train_samples = valid_samples = None

    "III) "
    "Load the files containing the already preprocessed data if they exist, else load the raw data and preprocess it yourself."
    "If you have to preprocess it, it saves the preprocess version so that  it’s no longer needed next time."

    if os.path.isfile(entity_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(entity_dictionary_pkl_path, "rb") as read_handle:
            entity_dictionary = pickle.load(read_handle)  # DD12B
        entity_dictionary_loaded = True

    if not params["only_evaluate"]:  # A10
        "III.1) Prepare training data (mention with context + entities with description)"

        # path to a file where the training data, already processed into tensors is saved
        train_tensor_data_pkl_path = os.path.join(
            pickle_src_path, "train_tensor_data.pickle"
        )
        # path to a file where metadata / additional information about the training data is stored
        train_processed_data_pkl_path = os.path.join(
            pickle_src_path, "train_processed_data.pickle"
        )

        # if the full path to file exist, load the file
        if os.path.isfile(train_tensor_data_pkl_path) and os.path.isfile(
            train_processed_data_pkl_path
        ):
            print("Loading stored processed train data...")
            with open(train_tensor_data_pkl_path, "rb") as read_handle:
                train_tensor_data = pickle.load(read_handle)
            with open(train_processed_data_pkl_path, "rb") as read_handle:
                train_processed_data = pickle.load(read_handle)

        # else : Pre-processed data files do not exist. Proceeds to load and process the raw training data.
        else:
            if not entity_dictionary_loaded:
                with open(
                    os.path.join(params["data_path"], "dictionary.pickle"), "rb"
                ) as read_handle:  # A11
                    entity_dictionary = pickle.load(read_handle)

            "III.1A) Load training data"
            # train_samples = list of dict. Each dict contains information about a mention (id, name, context, etc…).
            # Each key can have a dictionary itself. Ex : mention["context"]["tokens"] or mention["context"]["ids"]
            train_samples, mult_labels = read_data("train", params, logger)

            # For discovery experiment: Drop entities used in training that were dropped randomly from dev/test set
            if params["drop_entities"]:  # A12
                assert entity_dictionary  # It's not assert entity_dictionary_loaded !
                drop_set_path = (
                    params["drop_set"]
                    if params["drop_set"] is not None
                    else os.path.join(pickle_src_path, "drop_set_mention_data.pickle")
                )  # A12
                if not os.path.isfile(drop_set_path):
                    raise ValueError(
                        "Invalid or no --drop_set path provided to dev/test mention data"
                    )
                with open(drop_set_path, "rb") as read_handle:
                    drop_set_data = pickle.load(read_handle)
                # gold cuis indices for each mention in drop_set_data
                drop_set_mention_gold_cui_idxs = list(
                    map(lambda x: x["label_idxs"][0], drop_set_data)
                )
                # Make the set unique
                ents_in_data = np.unique(drop_set_mention_gold_cui_idxs)
                # % of drop
                ent_drop_prop = 0.1
                logger.info(
                    f"Dropping {ent_drop_prop*100}% of {len(ents_in_data)} entities found in drop set"
                )
                # Number of entity indices to drop
                n_ents_dropped = int(ent_drop_prop * len(ents_in_data))
                # Random selection drop
                rng = np.random.default_rng(seed=17)
                # Indices of all entities that are dropped
                dropped_ent_idxs = rng.choice(
                    ents_in_data, size=n_ents_dropped, replace=False
                )

                # Drop entities from dictionary (subsequent processing will automatically drop corresponding mentions)
                keep_mask = np.ones(len(entity_dictionary), dtype="bool")
                keep_mask[dropped_ent_idxs] = False
                entity_dictionary = np.array(entity_dictionary)[keep_mask]

            "III.B) Process training data"
            # train_processed_data = (mention + surrounding context) tokens
            # entity_dictionary = tokenized entities
            # tensor_train_dataset = Dataset containing several tensors (IDs of mention + context / indices of correct entities etc..) # Go check "process_mention_data" for more info
            train_processed_data, entity_dictionary, train_tensor_data = (
                data_process.process_mention_data(
                    train_samples,
                    entity_dictionary,
                    tokenizer,
                    params["max_context_length"],  # A14
                    params["max_cand_length"],  # A15
                    context_key=params["context_key"],  # A16
                    multi_label_key="labels" if mult_labels else None,
                    silent=params["silent"],  # A17
                    logger=logger,
                    debug=params["debug"],  # A18
                    knn=knn,
                    dictionary_processed=entity_dictionary_loaded,
                )
            )

            "III.2) If you have to preprocess it, it saves the preprocess version so that it’s no longer needed next time."
            print("Saving processed train data...")
            if not entity_dictionary_loaded:
                with open(entity_dictionary_pkl_path, "wb") as write_handle:
                    pickle.dump(
                        entity_dictionary,
                        write_handle,  # DD13 serialize an object hierarchy and write it to a file != to dispose of
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
            with open(train_tensor_data_pkl_path, "wb") as write_handle:
                pickle.dump(
                    train_tensor_data, write_handle, protocol=pickle.HIGHEST_PROTOCOL
                )
            with open(train_processed_data_pkl_path, "wb") as write_handle:
                pickle.dump(
                    train_processed_data, write_handle, protocol=pickle.HIGHEST_PROTOCOL
                )

        "III.1.C) Prepare tensor containing only ID of (mention + surrounding context) tokens of training set"
        train_men_vecs = train_tensor_data[:][0]

        "III.1.D) Prepare train DataLoader"
        if params["shuffle"]:  # A19
            train_sampler = RandomSampler(train_tensor_data)
        else:
            train_sampler = SequentialSampler(train_tensor_data)

        train_dataloader = DataLoader(
            train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
        )

    "III.1.E) Prepare entity dictionary"
    # Store the IDs of the entity in entity_dictionary # It's the equivalent of train_men_vecs for entities
    entity_dict_vecs = torch.tensor(
        list(map(lambda x: x["ids"], entity_dictionary)), dtype=torch.long
    )

    "III.3) Prepare the validation data (Used for evaluation during and after training)"

    "III.3.A) Load validation data"
    # Load VALIDATION data : It's used for evaluation during and after training
    valid_tensor_data_pkl_path = os.path.join(
        pickle_src_path, "valid_tensor_data.pickle"
    )
    valid_processed_data_pkl_path = os.path.join(
        pickle_src_path, "valid_processed_data.pickle"
    )

    # Same as training data :
    # if the full path to file exist, load the file
    if os.path.isfile(valid_tensor_data_pkl_path) and os.path.isfile(
        valid_processed_data_pkl_path
    ):
        print("Loading stored processed valid data...")
        with open(
            valid_tensor_data_pkl_path, "rb"
        ) as read_handle:  # CC7 'rb' = binary read mode
            valid_tensor_data = pickle.load(read_handle)
        with open(valid_processed_data_pkl_path, "rb") as read_handle:
            valid_processed_data = pickle.load(read_handle)

    # III.3.B) Process validation data
    # else : Pre-processed data files do not exist. Proceeds to load and process the raw validation data.

    else:
        valid_samples, mult_labels = read_data("valid", params, logger)
        valid_processed_data, _, valid_tensor_data = data_process.process_mention_data(
            valid_samples,
            entity_dictionary,
            tokenizer,
            params["max_context_length"],
            params["max_cand_length"],
            context_key=params["context_key"],
            multi_label_key="labels" if mult_labels else None,
            silent=params["silent"],  # RR7
            logger=logger,
            debug=params["debug"],
            knn=knn,
            dictionary_processed=True,
        )
        print("Saving processed valid data...")
        with open(valid_tensor_data_pkl_path, "wb") as write_handle:
            pickle.dump(
                valid_tensor_data, write_handle, protocol=pickle.HIGHEST_PROTOCOL
            )
        with open(valid_processed_data_pkl_path, "wb") as write_handle:
            pickle.dump(
                valid_processed_data, write_handle, protocol=pickle.HIGHEST_PROTOCOL
            )

    "III.3.C) Prepare tensor containing only ID of (mention + surrounding context) tokens of validation data"
    # Store the query mention vectors # Used for evaluation during and after training
    valid_men_vecs = valid_tensor_data[:][0]

    "III.4) Consider if it’s within_doc (=search only within the document)"
    train_context_doc_ids = valid_context_doc_ids = None
    if within_doc:
        # Store the context_doc_id for every mention in the train and valid sets
        if train_samples is None:
            train_samples, _ = read_data("train", params, logger)
        train_context_doc_ids = [s["context_doc_id"] for s in train_samples]
        if valid_samples is None:
            valid_samples, _ = read_data("valid", params, logger)
        valid_context_doc_ids = [
            s["context_doc_id"] for s in train_samples
        ]  # Should be "valid_context_doc_ids = [s['context_doc_id'] for s in valid_samples]"

    if params["only_evaluate"]:
        evaluate(
            reranker,
            entity_dict_vecs,
            valid_men_vecs,
            device=device,
            logger=logger,
            knn=knn,
            n_gpu=n_gpu,
            entity_data=entity_dictionary,
            query_data=valid_processed_data,
            silent=params["silent"],
            use_types=use_types or params["use_types_for_eval"],
            embed_batch_size=params["embed_batch_size"],
            force_exact_search=use_types
            or params["use_types_for_eval"]
            or params["force_exact_search"],
            probe_mult_factor=params["probe_mult_factor"],
            within_doc=within_doc,
            context_doc_ids=valid_context_doc_ids,
        )  # A20 #A21 #A22 #A23
        exit()  # DD14 Terminates the Script

    # Get clusters of mentions that map to a gold entity
    train_gold_clusters = data_process.compute_gold_clusters(train_processed_data)
    # Maximum length of clusters inside gold_cluster
    max_gold_cluster_len = 0
    for ent in train_gold_clusters:
        if len(train_gold_clusters[ent]) > max_gold_cluster_len:
            max_gold_cluster_len = len(train_gold_clusters[ent])

    "III.5) Set up parameters for training (optimizer, scheduler etc…)"
    n_entities = len(entity_dictionary)
    n_mentions = len(train_processed_data)

    time_start = time.time()
    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )
    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, data_parallel: {}".format(
            device, n_gpu, params["data_parallel"]
        )  # A24
    )

    # Set model to training mode
    optim_sched, optimizer, scheduler = (
        load_optimizer_scheduler(params, logger),
        None,
        None,
    )
    if optim_sched is None:
        optimizer = get_optimizer(model, params)
        scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)
    else:
        optimizer = optim_sched["optimizer"]
        scheduler = optim_sched["scheduler"]

    best_epoch_idx = -1
    best_score = -1
    num_train_epochs = params["num_train_epochs"]  # A25

    # DD15 Flag indicating whether a base model should be initialized from scratch (True) or use a pre-loaded model (False).
    init_base_model_run = True if params.get("path_to_model", None) is None else False
    # DD16 Constructs a file path where initialization data (ex : embedding and indexing) is expected to be loaded from.
    init_run_pkl_path = os.path.join(
        pickle_src_path, f'init_run_{"type" if use_types else "notype"}.t7'
    )

    dict_embed_data = None
    "IV. Training the model"
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):  # CC8
        model.train()
        torch.cuda.empty_cache()
        tr_loss = 0

        # DD17 # Check if embeddings and index can be loaded
        init_run_data_loaded = False
        if init_base_model_run:
            if os.path.isfile(init_run_pkl_path):
                logger.info("Loading init run data")
                init_run_data = torch.load(init_run_pkl_path)
                # Flag indicating whether initialization data (precomputed embeddings and indexes) was successfully loaded
                init_run_data_loaded = True

        # If fresh training session and precomputed initialization data has been successfully loaded
        load_stored_data = init_base_model_run and init_run_data_loaded

        "IV.1) Compute mention and entity embeddings and indexes at the start of each epoch"
        # Compute mention and entity embeddings and indexes at the start of each epoch
        if use_types:  # type-specific indexes
            if load_stored_data:
                train_dict_embeddings, dict_idxs_by_type = (
                    init_run_data["train_dict_embeddings"],
                    init_run_data["dict_idxs_by_type"],
                )
                train_dict_indexes = data_process.get_index_from_embeds(
                    train_dict_embeddings,
                    dict_idxs_by_type,
                    force_exact_search=params["force_exact_search"],
                    probe_mult_factor=params["probe_mult_factor"],
                )
                train_men_embeddings, men_idxs_by_type = (
                    init_run_data["train_men_embeddings"],
                    init_run_data["men_idxs_by_type"],
                )
                train_men_indexes = data_process.get_index_from_embeds(
                    train_men_embeddings,
                    men_idxs_by_type,
                    force_exact_search=params["force_exact_search"],
                    probe_mult_factor=params["probe_mult_factor"],
                )
            else:
                # We will always go through this else the first time. Only purpose of the "if" is when we already have a model ready for evaluation
                # That's where the edges weights will change through reranker that has its parameter updated after each epoch
                logger.info("Embedding and indexing")
                if dict_embed_data is not None:
                    train_dict_embeddings, train_dict_indexes, dict_idxs_by_type = (
                        dict_embed_data["dict_embeds"],
                        dict_embed_data["dict_indexes"],
                        dict_embed_data["dict_idxs_by_type"],
                    )
                else:
                    train_dict_embeddings, train_dict_indexes, dict_idxs_by_type = (
                        data_process.embed_and_index(
                            reranker,
                            entity_dict_vecs,
                            encoder_type="candidate",
                            n_gpu=n_gpu,
                            corpus=entity_dictionary,
                            force_exact_search=params["force_exact_search"],
                            batch_size=params["embed_batch_size"],
                            probe_mult_factor=params["probe_mult_factor"],
                        )
                    )
                train_men_embeddings, train_men_indexes, men_idxs_by_type = (
                    data_process.embed_and_index(
                        reranker,
                        train_men_vecs,
                        encoder_type="context",
                        n_gpu=n_gpu,
                        corpus=train_processed_data,
                        force_exact_search=params["force_exact_search"],
                        batch_size=params["embed_batch_size"],
                        probe_mult_factor=params["probe_mult_factor"],
                    )
                )
        else:  # general indexes
            if load_stored_data:
                train_dict_embeddings = init_run_data["train_dict_embeddings"]
                train_dict_index = data_process.get_index_from_embeds(
                    train_dict_embeddings,
                    force_exact_search=params["force_exact_search"],
                    probe_mult_factor=params["probe_mult_factor"],
                )
                train_men_embeddings = init_run_data["train_men_embeddings"]
                train_men_index = data_process.get_index_from_embeds(
                    train_men_embeddings,
                    force_exact_search=params["force_exact_search"],
                    probe_mult_factor=params["probe_mult_factor"],
                )
            else:
                logger.info("Embedding and indexing")
                if dict_embed_data is not None:
                    train_dict_embeddings, train_dict_index = (
                        dict_embed_data["dict_embeds"],
                        dict_embed_data["dict_index"],
                    )
                else:
                    train_dict_embeddings, train_dict_index = (
                        data_process.embed_and_index(
                            reranker,
                            entity_dict_vecs,
                            encoder_type="candidate",
                            n_gpu=n_gpu,
                            force_exact_search=params["force_exact_search"],
                            batch_size=params["embed_batch_size"],
                            probe_mult_factor=params["probe_mult_factor"],
                        )
                    )
                train_men_embeddings, train_men_index = data_process.embed_and_index(
                    reranker,
                    train_men_vecs,
                    encoder_type="context",
                    n_gpu=n_gpu,
                    force_exact_search=params["force_exact_search"],
                    batch_size=params["embed_batch_size"],
                    probe_mult_factor=params["probe_mult_factor"],
                )

        "IV.2) Save the initial embeddings and index if this is the first run and data isn't persistent"
        if init_base_model_run and not load_stored_data:
            init_run_data = {}
            init_run_data["train_dict_embeddings"] = train_dict_embeddings
            init_run_data["train_men_embeddings"] = train_men_embeddings
            if use_types:
                init_run_data["dict_idxs_by_type"] = dict_idxs_by_type
                init_run_data["men_idxs_by_type"] = men_idxs_by_type
            # NOTE: Cannot pickle faiss index because it is a SwigPyObject
            torch.save(
                init_run_data,
                init_run_pkl_path,
                pickle_protocol=pickle.HIGHEST_PROTOCOL,
            )

        init_base_model_run = False

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        # Store golden MST links
        gold_links = {}

        # Calculate the number of negative entities and mentions to fetch # Divides the k-nn evenly between entities and mentions
        knn_dict = knn // 2
        knn_men = knn - knn_dict

        "IV.3) knn search : indice and distance of k closest mentions and entities"
        logger.info("Starting KNN search...")
        # INFO: Fetching all sorted mentions to be able to filter to within-doc later=
        n_men_to_fetch = (
            len(train_men_embeddings) if within_doc else knn_men + max_gold_cluster_len
        )
        n_ent_to_fetch = (
            knn_dict + 1
        )  # +1 accounts for the possibility of self-reference
        if not use_types:
            _, dict_nns = train_dict_index.search(train_men_embeddings, n_ent_to_fetch)
            _, men_nns = train_men_index.search(train_men_embeddings, n_men_to_fetch)
        else:
            dict_nns = -1 * np.ones((len(train_men_embeddings), n_ent_to_fetch))
            men_nns = -1 * np.ones((len(train_men_embeddings), n_men_to_fetch))
            for entity_type in train_men_indexes:
                men_embeds_by_type = train_men_embeddings[men_idxs_by_type[entity_type]]
                _, dict_nns_by_type = train_dict_indexes[entity_type].search(
                    men_embeds_by_type, n_ent_to_fetch
                )
                _, men_nns_by_type = train_men_indexes[entity_type].search(
                    men_embeds_by_type, min(n_men_to_fetch, len(men_embeds_by_type))
                )
                dict_nns_idxs = np.array(
                    list(
                        map(
                            lambda x: dict_idxs_by_type[entity_type][x],
                            dict_nns_by_type,
                        )
                    )
                )
                men_nns_idxs = np.array(
                    list(
                        map(lambda x: men_idxs_by_type[entity_type][x], men_nns_by_type)
                    )
                )
                for i, idx in enumerate(men_idxs_by_type[entity_type]):
                    dict_nns[idx] = dict_nns_idxs[i]
                    men_nns[idx][: len(men_nns_idxs[i])] = men_nns_idxs[i]
        logger.info("Search finished")

        total_skipped = total_knn_men_negs = 0

        "IV.4) for every batch in batches"
        for step, batch in enumerate(
            iter_
        ):  # Already organized into batches by the DataLoader in PyTorch

            "IV.4.A)  Initialize the parameters"
            knn_men = knn - knn_dict
            batch = tuple(t.to(device) for t in batch)
            # batch is a subsample from tensor_dataset
            batch_context_inputs, candidate_idxs, n_gold, mention_idxs = batch
            # mentions within the batch
            mention_embeddings = train_men_embeddings[mention_idxs.cpu()]

            if len(mention_embeddings.shape) == 1:
                mention_embeddings = np.expand_dims(mention_embeddings, axis=0)  # CC9

            # batch_context_inputs: Shape: batch x token_len
            # candidate_inputs = []
            # candidate_inputs = np.array([], dtype=np.long) # Shape: (batch*knn) x token_len
            # label_inputs = (candidate_idxs >= 0).type(torch.float32) # Shape: batch x knn

            positive_idxs = []
            negative_dict_inputs = []
            negative_men_inputs = []

            skipped_positive_idxs = []
            skipped_negative_dict_inputs = []

            min_neg_mens = float("inf")
            skipped = 0
            context_inputs_mask = [True] * len(batch_context_inputs)

            "IV.4.B) For each mention within the batch"
            # For each mention within the batch
            for m_embed_idx, m_embed in enumerate(mention_embeddings):
                mention_idx = int(mention_idxs[m_embed_idx])
                # CC11 ground truth entities of the mention "mention_idx"
                gold_idxs = set(
                    train_processed_data[mention_idx]["label_idxs"][
                        : n_gold[m_embed_idx]
                    ]
                )

                # TEMPORARY: Assuming that there is only 1 gold label, TODO: Incorporate multiple case
                assert n_gold[m_embed_idx] == 1

                if mention_idx in gold_links:
                    gold_link_idx = gold_links[mention_idx]
                else:
                    "IV.4.B.a) Create the graph with positive edges"
                    # This block creates all the positive edges of the mention in this iteration
                    # Run MST on mention clusters of all the gold entities of the current query mention to find its positive edge
                    rows, cols, data, shape = (
                        [],
                        [],
                        [],
                        (n_entities + n_mentions, n_entities + n_mentions),
                    )
                    seen = set()

                    # Set whether the gold edge should be the nearest or the farthest neighbor
                    sim_order = 1 if params["farthest_neighbor"] else -1  # A26

                    for cluster_ent in gold_idxs:
                        # CC12 IDs of all the mentions inside the gold cluster with entity id = "cluster_ent"
                        cluster_mens = train_gold_clusters[cluster_ent]

                        if within_doc:
                            # Filter the gold cluster to within-doc
                            cluster_mens, _ = filter_by_context_doc_id(
                                cluster_mens,
                                train_context_doc_ids[mention_idx],
                                train_context_doc_ids,
                            )

                        # weights for all the mention-entity links inside the cluster of the current mention
                        to_ent_data = (
                            train_men_embeddings[cluster_mens]
                            @ train_dict_embeddings[cluster_ent].T
                        )

                        # weights for all the mention-mention links inside the cluster of the current mention
                        to_men_data = (
                            train_men_embeddings[cluster_mens]
                            @ train_men_embeddings[cluster_mens].T
                        )

                        if gold_arbo_knn is not None:
                            # Descending order of similarity if nearest-neighbor, else ascending order
                            sorti = np.argsort(sim_order * to_men_data, axis=1)
                            sortv = np.take_along_axis(to_men_data, sorti, axis=1)
                            if params["rand_gold_arbo"]:
                                randperm = np.random.permutation(sorti.shape[1])
                                sortv, sorti = sortv[:, randperm], sorti[:, randperm]

                        for i in range(len(cluster_mens)):
                            from_node = n_entities + cluster_mens[i]
                            to_node = cluster_ent
                            # Add mention-entity link
                            rows.append(from_node)
                            cols.append(to_node)
                            data.append(-1 * to_ent_data[i])
                            if gold_arbo_knn is None:
                                # Add forward and reverse mention-mention links over the entire MST
                                for j in range(i + 1, len(cluster_mens)):
                                    to_node = n_entities + cluster_mens[j]
                                    if (from_node, to_node) not in seen:
                                        score = to_men_data[i, j]
                                        rows.append(from_node)
                                        cols.append(to_node)
                                        data.append(
                                            -1 * score
                                        )  # Negatives needed for SciPy's Minimum Spanning Tree computation
                                        seen.add((from_node, to_node))
                                        seen.add((to_node, from_node))
                            else:
                                # Approximate the MST using <gold_arbo_knn> nearest mentions from the gold cluster
                                added = 0
                                approx_k = min(gold_arbo_knn + 1, len(cluster_mens))
                                for j in range(approx_k):
                                    if added == approx_k - 1:
                                        break
                                    to_node = n_entities + cluster_mens[sorti[i, j]]
                                    if to_node == from_node:
                                        continue
                                    added += 1
                                    if (from_node, to_node) not in seen:
                                        score = sortv[i, j]
                                        rows.append(from_node)
                                        cols.append(to_node)
                                        data.append(
                                            -1 * score
                                        )  # Negatives needed for SciPy's Minimum Spanning Tree computation
                                        seen.add((from_node, to_node))

                    "IV.4.B.b) Fine tuning with inference procedure to get a mst"
                    # Creates MST with entity constraint (inference procedure)
                    csr = csr_matrix(
                        (-sim_order * np.array(data), (rows, cols)), shape=shape
                    )
                    # Note: minimum_spanning_tree expects distances as edge weights
                    mst = minimum_spanning_tree(csr).tocoo()
                    # Note: cluster_linking_partition expects similarities as edge weights # Convert directed to undirected graph
                    rows, cols, data = cluster_linking_partition(
                        np.concatenate((mst.row, mst.col)),
                        np.concatenate((mst.col, mst.row)),
                        np.concatenate((sim_order * mst.data, sim_order * mst.data)),
                        n_entities,
                        directed=True,
                        silent=True,
                    )
                    assert np.array_equal(rows - n_entities, cluster_mens)

                    for i in range(len(rows)):
                        men_idx = rows[i] - n_entities
                        if men_idx in gold_links:
                            continue
                        assert men_idx >= 0
                        add_link = True
                        # Store the computed positive edges for the mentions in the clusters only if they have the same gold entities as the query mention
                        for l in train_processed_data[men_idx]["label_idxs"][
                            : train_processed_data[men_idx]["n_labels"]
                        ]:
                            if l not in gold_idxs:
                                add_link = False
                                break
                        if add_link:
                            gold_links[men_idx] = cols[i]
                    gold_link_idx = gold_links[mention_idx]

                "IV.4.B.c) Retrieve the pre-computed nearest neighbors"
                knn_dict_idxs = dict_nns[mention_idx]
                knn_dict_idxs = knn_dict_idxs.astype(np.int64).flatten()
                knn_men_idxs = men_nns[mention_idx][men_nns[mention_idx] != -1]
                knn_men_idxs = knn_men_idxs.astype(np.int64).flatten()
                if within_doc:
                    knn_men_idxs, _ = filter_by_context_doc_id(
                        knn_men_idxs,
                        train_context_doc_ids[mention_idx],
                        train_context_doc_ids,
                        return_numpy=True,
                    )
                "IV.4.B.d) Add negative examples"
                neg_mens = list(
                    knn_men_idxs[
                        ~np.isin(
                            knn_men_idxs,
                            np.concatenate(
                                [train_gold_clusters[gi] for gi in gold_idxs]
                            ),
                        )
                    ][:knn_men]
                )

                # Track queries with no valid mention negatives
                if len(neg_mens) == 0:
                    context_inputs_mask[m_embed_idx] = False
                    skipped_negative_dict_inputs += list(
                        knn_dict_idxs[~np.isin(knn_dict_idxs, list(gold_idxs))][
                            :knn_dict
                        ]
                    )
                    skipped_positive_idxs.append(gold_link_idx)
                    skipped += 1
                    continue
                else:
                    min_neg_mens = min(min_neg_mens, len(neg_mens))
                negative_men_inputs.append(
                    knn_men_idxs[
                        ~np.isin(
                            knn_men_idxs,
                            np.concatenate(
                                [train_gold_clusters[gi] for gi in gold_idxs]
                            ),
                        )
                    ][:knn_men]
                )
                negative_dict_inputs += list(
                    knn_dict_idxs[~np.isin(knn_dict_idxs, list(gold_idxs))][:knn_dict]
                )
                # Add the positive example
                positive_idxs.append(gold_link_idx)

            "IV.4.C) Skip this iteration if no suitable negative examples found"
            if len(negative_men_inputs) == 0:
                continue

            # Sets the minimum number of negative mentions found across all processed mentions in the current batch
            knn_men = min_neg_mens

            # This step ensures that each mention is compared against a uniform number of negative mentions
            filtered_negative_men_inputs = []
            for row in negative_men_inputs:
                filtered_negative_men_inputs += list(row[:knn_men])
            negative_men_inputs = filtered_negative_men_inputs

            # Assertions for Data Integrity
            assert (
                len(negative_dict_inputs)
                == (len(mention_embeddings) - skipped) * knn_dict
            )
            assert (
                len(negative_men_inputs)
                == (len(mention_embeddings) - skipped) * knn_men
            )

            total_skipped += skipped
            total_knn_men_negs += knn_men

            # Convert to tensors
            negative_dict_inputs = torch.tensor(
                list(map(lambda x: entity_dict_vecs[x].numpy(), negative_dict_inputs))
            )
            negative_men_inputs = torch.tensor(
                list(map(lambda x: train_men_vecs[x].numpy(), negative_men_inputs))
            )

            # Labels indicating the correct candidates. Used for computing loss.
            positive_embeds = []
            for pos_idx in positive_idxs:
                if pos_idx < n_entities:
                    pos_embed = reranker.encode_candidate(
                        entity_dict_vecs[pos_idx : pos_idx + 1].cuda(),
                        requires_grad=True,
                    )
                else:
                    pos_embed = reranker.encode_context(
                        train_men_vecs[
                            pos_idx - n_entities : pos_idx - n_entities + 1
                        ].cuda(),
                        requires_grad=True,
                    )
                positive_embeds.append(pos_embed)
            positive_embeds = torch.cat(positive_embeds)

            # Remove mentions with no negative examples
            context_inputs = batch_context_inputs[context_inputs_mask]
            context_inputs = context_inputs.cuda()

            # Tensor containing binary values that act as indicator variables in the paper:
            # Contains Indicator variable such that I_{u,m_i} = 1 if(u,mi) ∈ E'_{m_i} and I{u,m_i} = 0 otherwise.
            label_inputs = torch.tensor(
                [[1] + [0] * (knn_dict + knn_men)] * len(context_inputs),
                dtype=torch.float32,
            ).cuda()

            "IV.4.D) Compute the loss"
            loss_dual_negs = loss_ent_negs = 0

            # RR4 distributed_data_parallel no longer needed
            # FIX: for error scenario of less number of examples than number of GPUs while using Data Parallel
            data_parallel_batch_size_check = (
                negative_men_inputs.shape[0] >= n_gpu
                and negative_dict_inputs.shape[0] >= n_gpu
            )
            if data_parallel_batch_size_check:
                # loss of a batch includes both negative mention and entity inputs (alongside positive examples ofc)
                loss_dual_negs, _ = reranker(
                    context_inputs,
                    label_input=label_inputs,
                    mst_data={
                        "positive_embeds": positive_embeds.cuda(),
                        "negative_dict_inputs": negative_dict_inputs.cuda(),
                        "negative_men_inputs": negative_men_inputs.cuda(),
                    },
                    pos_neg_loss=params["pos_neg_loss"],
                )  # A27

            skipped_context_inputs = []
            if skipped > 0 and not params["within_doc_skip_strategy"]:  # A28
                skipped_negative_dict_inputs = torch.tensor(
                    list(
                        map(
                            lambda x: entity_dict_vecs[x].numpy(),
                            skipped_negative_dict_inputs,
                        )
                    )
                )
                skipped_positive_embeds = []
                for pos_idx in skipped_positive_idxs:
                    if pos_idx < n_entities:
                        pos_embed = reranker.encode_candidate(
                            entity_dict_vecs[pos_idx : pos_idx + 1].cuda(),
                            requires_grad=True,
                        )
                    else:
                        pos_embed = reranker.encode_context(
                            train_men_vecs[
                                pos_idx - n_entities : pos_idx - n_entities + 1
                            ].cuda(),
                            requires_grad=True,
                        )
                    skipped_positive_embeds.append(pos_embed)
                skipped_positive_embeds = torch.cat(skipped_positive_embeds)
                skipped_context_inputs = batch_context_inputs[
                    ~np.array(context_inputs_mask)
                ]
                skipped_context_inputs = skipped_context_inputs.cuda()
                skipped_label_inputs = torch.tensor(
                    [[1] + [0] * (knn_dict)] * len(skipped_context_inputs),
                    dtype=torch.float32,
                ).cuda()

                data_parallel_batch_size_check = (
                    skipped_negative_dict_inputs.shape[0] >= n_gpu
                )
                if data_parallel_batch_size_check:
                    # DD18 loss of a batch that only includes negative entity inputs.
                    loss_ent_negs, _ = reranker(
                        skipped_context_inputs,
                        label_input=skipped_label_inputs,
                        mst_data={
                            "positive_embeds": skipped_positive_embeds.cuda(),
                            "negative_dict_inputs": skipped_negative_dict_inputs.cuda(),
                            "negative_men_inputs": None,
                        },
                        pos_neg_loss=params["pos_neg_loss"],
                    )

            # RR8
            # len(context_input) = Number of mentions in the batch that successfully found negative entities and mentions.
            # len(skipped_context_inputs): Number of mentions in the batch that only found negative entities.
            loss = (
                (
                    loss_dual_negs * len(context_inputs)
                    + loss_ent_negs * len(skipped_context_inputs)
                )
                / (len(context_inputs) + len(skipped_context_inputs))
            ) / grad_acc_steps

            if isinstance(loss, torch.Tensor):  # safety protocol
                tr_loss += (
                    loss.item()
                )  # loss.item() is used to extract the scalar value from a tensor that contains a single value.
                loss.backward()  # CC10 #DD19

            "IV.4.E) Information about the training (step, epoch, average_loss)"
            n_print_iters = params["print_interval"] * grad_acc_steps  # 29
            if (step + 1) % n_print_iters == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}".format(
                        step,
                        epoch_idx,
                        tr_loss / n_print_iters,
                    )
                )
                if total_skipped > 0:
                    logger.info(
                        f"Queries per batch w/o mention negs={total_skipped / n_print_iters}/{len(mention_embeddings)}; Negative mentions per query per batch={total_knn_men_negs / n_print_iters} "
                    )
                total_skipped = 0
                total_knn_men_negs = 0
                tr_loss = 0

            "IV.4.F) Model updates"
            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(  # DD20
                    model.parameters(), params["max_grad_norm"]  # A30
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # RR6 : Trainer(val_check_interval= 10)
            # Regular checks on model performance against a validation dataset without interrupting the training more often than desired
            if (
                params["eval_interval"] != -1
            ):  # A31  #Evaluation every "params["eval_interval"] * grad_acc_steps" steps
                if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                    logger.info("Evaluation on the development dataset")
                    evaluate(
                        reranker,
                        entity_dict_vecs,
                        valid_men_vecs,
                        device=device,
                        logger=logger,
                        knn=knn,
                        n_gpu=n_gpu,
                        entity_data=entity_dictionary,
                        query_data=valid_processed_data,
                        silent=params["silent"],
                        use_types=use_types or params["use_types_for_eval"],
                        embed_batch_size=params["embed_batch_size"],
                        force_exact_search=use_types
                        or params["use_types_for_eval"]
                        or params["force_exact_search"],
                        probe_mult_factor=params["probe_mult_factor"],
                        within_doc=within_doc,
                        context_doc_ids=valid_context_doc_ids,
                    )
                    model.train()  # Switch back to training model. There is model.eval() inside the evaluate function
                    logger.info("\n")

        "IV.5) Save the models for each epoch"
        logger.info("***** Saving fine-tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(
            model, tokenizer, epoch_output_folder_path, scheduler, optimizer
        )
        logger.info(f"Model saved at {epoch_output_folder_path}")

        "IV.6) Evaluate final accuracy"
        eval_accuracy, dict_embed_data = evaluate(
            reranker,
            entity_dict_vecs,
            valid_men_vecs,
            device=device,
            logger=logger,
            knn=knn,
            n_gpu=n_gpu,
            entity_data=entity_dictionary,
            query_data=valid_processed_data,
            silent=params["silent"],
            use_types=use_types or params["use_types_for_eval"],
            embed_batch_size=params["embed_batch_size"],
            force_exact_search=use_types
            or params["use_types_for_eval"]
            or params["force_exact_search"],
            probe_mult_factor=params["probe_mult_factor"],
            within_doc=within_doc,
            context_doc_ids=valid_context_doc_ids,
        )

        ls = [best_score, eval_accuracy]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    "V. Some log stuffs (training time, Best performance in epochs number_i, Best model saved at path)"
    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # Save the best model in the parent_dir
    # RR2 : No longer needed; it's in the Trainer's callback
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = os.path.join(
        model_output_path, "epoch_{}".format(best_epoch_idx)
    )
    utils.save_model(reranker.model, tokenizer, model_output_path, scheduler, optimizer)
    logger.info(f"Best model saved at {model_output_path}")


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)
