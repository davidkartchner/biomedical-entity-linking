import spacy
from bioel.ontology import BiomedicalOntology
from bioel.models.scispacy.candidate_generation import CandidateGenerator
from bioel.models.scispacy.scispacy_embeddings import KnowledgeBaseEmbeddings
from bioel.models.scispacy.entity_linking import EntityLinker
from bioel.utils.bigbio_utils import (
    load_bigbio_dataset,
    add_deabbreviations,
    dataset_to_documents,
    dataset_to_df,
    load_dataset_df,
    resolve_abbreviation,
    CUIS_TO_REMAP,
    CUIS_TO_EXCLUDE,
    DATASET_NAMES,
    VALIDATION_DOCUMENT_IDS,
)
from bioel.utils.dataset_consts import (
    dataset_to_pretty_name,
    model_to_pretty_name,
    model_to_color,
)
import ujson
import os
import pickle


def evaluate_model(
    params,
    model,
):
    """
    Params : dict
    ---------
    dataset: str
        Name of the dataset to evaluate
    ontology: BiomedicalOntology
        Ontology to use for candidate generation
    k: int
        Number of candidates to generate
    path_to_save: str
        Path to save the serialized knowledge base
    output_path: str
        Path to save the output
    equivalent_cuis: bool
        Whether the ontology has equivalent cuis or not
    path_to_abbrev: str
        Path to the abbreviations file
    """
    print("type of params :", type(params))
    print("params :", params)
    if hasattr(BiomedicalOntology, params["load_function"]):
        load_func = getattr(BiomedicalOntology, params["load_function"])
        if params["ontology_dict"]:
            ontology_object = load_func(**params["ontology_dict"])
            print(f"Ontology loaded successfully. Name: {ontology_object.name}")
        else:
            raise ValueError("No ontology data provided.")
    else:
        raise ValueError(
            f"Error: {params['load_function']} is not a valid function for BiomedicalOntology."
        )

    myembed = KnowledgeBaseEmbeddings(ontology_object)
    myembed.create_tfidf_ann_index(params["path_to_save"])
    kb = myembed.serialized_kb()
    cand_gen = CandidateGenerator(kb)
    df = load_dataset_df(
        name=params["dataset"], path_to_abbrev=params["path_to_abbrev"]
    )
    df = df.query("split == 'test'")

    output = list()

    if params["equivalant_cuis"]:
        cui_synsets = {}
        for cui, entity in ontology_object.entities.items():
            cui_synsets[cui] = entity.equivalant_cuis

    for index, row in df.iterrows():
        list_dbid = list(row.db_ids)  # Accessing db_ids as a list
        if params["path_to_abbrev"]:
            mention = row.deabbreviated_text  # Accessing the text of the current row
        else:
            mention = row.text
        candidates = cand_gen([mention], 3 * params["k"])  # Generating candidates
        predicted = []

        for cand in candidates[0]:
            score = max(cand.similarities)
            predicted.append((cand.concept_id, score))

        sorted_predicted = sorted(predicted, reverse=True, key=lambda x: x[1])
        sorted_predicted = sorted_predicted[: params["k"]]  # Taking top k predictions

        cui_result = [[cui[0]] for cui in sorted_predicted]
        score = [[cui[1]] for cui in sorted_predicted]

        if params["equivalant_cuis"]:
            cui_result = [cui_synsets[y[0]] for y in cui_result]

        if params["path_to_abbrev"]:
            output.append(
                {
                    "document_id": row.document_id,
                    "offsets": row.offsets,
                    "text": row.text,
                    "type": row.type,
                    "db_ids": list_dbid,
                    "split": row.split,
                    "deabbreviated_text": row.deabbreviated_text,
                    "mention_id": row.mention_id + ".abbr_resolved",
                    "candidates": cui_result,
                    "scores": score,
                }
            )
        else:
            output.append(
                {
                    "document_id": row.document_id,
                    "offsets": row.offsets,
                    "text": row.text,
                    "type": row.type,
                    "db_ids": list_dbid,
                    "split": row.split,
                    "mention_id": row.mention_id,
                    "candidates": cui_result,
                    "scores": score,
                }
            )

    output_dir = params["path_to_save"]
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    output_path = os.path.join(output_dir, "output.json")
    with open(output_path, "w") as f:
        f.write(ujson.dumps(output, indent=2))
