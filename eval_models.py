from xmlrpc.client import FastMarshaller
import spacy
import numpy as np
import pandas as pd
import pickle
import ujson

from tqdm.auto import tqdm

# from scispacy.abbreviation import AbbreviationDetector
# from scispacy.linking import EntityLinker
# from scispacy.umls_utils import UmlsKnowledgeBase
# from bigbio.dataloader import BigBioConfigHelpers
from typing import List, Optional
from collections import defaultdict
from argparse import ArgumentParser


def _hit_at_k_scispacy(gold_entity, candidate_entities, k):
    """
    Determine if correct entity link is in top k entity candidates
    """
    gold_cui = gold_entity["db_id"]
    top_candidates = sorted(candidate_entities, key=lambda x: x["score"])[::-1][:k]
    if gold_cui in [x["db_id"] for x in top_candidates]:
        return True
    else:
        return False


def hits_by_type_scispacy(
    gold_lookup, model_lookup, ks=[1, 5, 10, 25], custom_types=None
):
    """
    Get hits@k grouped by entity type.

    By default, bases type annotations on
    """
    # Hits @ k
    hits_dict = {}
    entity_types = set([])
    for k in ks:
        hits = defaultdict(int)
        # Get hits @ k for each entity type
        for key, gold_entity in tqdm(gold_lookup.items()):
            ent_type = gold_entity["type"]

            if custom_types is not None:
                cui = gold_entity["db_id"].split(":")[-1]

                # Make sure CUI is in SciSpacy KB
                if cui in kb.cui_to_entity:
                    pred_types = kb.cui_to_entity[cui].types
                    for group_name, vals in custom_types.items():
                        if any([x in vals for x in pred_types]):
                            # TODO: Handle case where an entity falls in more than 1 category in custom dict
                            ent_type = group_name
                            entity_types.add(ent_type)
                            break
            else:
                entity_types.add(ent_type)

            if key in model_lookup:
                hits[ent_type] += _hit_at_k_scispacy(
                    gold_entity, model_lookup[key]["candidates"], k
                )
                hits[ent_type + "_total"] += 1
            else:
                hits[ent_type + "_missing"] += 1

        overall_hits = np.sum(hits[x] for x in entity_types) / np.sum(
            hits[x + "_total"] for x in entity_types
        )
        overall_recall = np.sum(hits[x] for x in entity_types) / np.sum(
            hits[x + "_total"] + hits[x + "_missing"] for x in entity_types
        )
        hits["overall (exclude missing)"] = overall_hits
        hits["overall (include missing)"] = overall_recall
        hits_dict[k] = hits

    hits_dict["types"] = entity_types
    return hits_dict


def evaluate_linking_scispacy(processed_data, model="scispacy", custom_types=None):
    """
    Evaluate a number of metrics on entity linking models:
        * Hits@k for k = 1, 5, 10
    """
    # preformat data into dict with keys of form (pmid, offset_start, offset_end) for faster access

    gold_lookup = {
        (doc["pmid"], x["offsets"][0][0], x["offsets"][0][1]): {
            "db_id": x["normalized"][0]["db_id"],
            "type": x["type"],
        }
        for doc in processed_data
        for x in doc["gold_entities"]
    }
    model_lookup = {
        (doc["pmid"], x["offsets"][0][0], x["offsets"][0][1]): {
            "normalized": x["normalized"][0]["db_id"],
            "candidates": x["candidates"],
        }
        for doc in processed_data
        for x in doc["predictions"][model]
    }

    # for doc in processed_data:
    #     pmid = doc['pmid']
    #     for x in doc['predictions']['model']

    hits_dict = hits_by_type_scispacy(
        gold_lookup, model_lookup, custom_types=custom_types
    )

    return hits_dict


def format_output_scispacy(hits_dict, type_to_name_mapping=None}):
    """
    Format entity linking results into pandas DataFrame
    """
    entity_types = hits_dict["types"]
    ks = [k for k in hits_dict.keys() if type(k) == int]
    rows = []
    for t in entity_types:
        if t not in type_to_name_mapping:
            name = t.upper()
        else:
            name = type_to_name_mapping[t]
        rows.append(
            {
                **{"Semantic Type": name},
                **{
                    f"Hits @ {k}": hits_dict[k][t] / hits_dict[k][t + "_total"]
                    for k in ks
                },
                **{
                    "Spans Matched": hits_dict[ks[0]][t + "_total"],
                    "Missing": hits_dict[ks[0]][t + "_missing"],
                },
            }
        )

    # Add overall scores
    rows.append(
        {
            **{"Semantic Type": "**Overall**"},
            **{f"Hits @ {k}": hits_dict[k]["overall (exclude missing)"] for k in ks},
            **{
                "Spans Matched": np.sum(
                    [hits_dict[ks[0]][t + "_total"] for t in entity_types]
                ),
                "Missing": np.sum(
                    [hits_dict[ks[0]][t + "_missing"] for t in entity_types]
                ),
            },
        }
    )
    rows.append(
        {
            **{"Semantic Type": "**Overall (including missing spans)**"},
            **{f"Hits @ {k}": hits_dict[k]["overall (include missing)"] for k in ks},
            **{
                "Spans Matched": np.sum(
                    [hits_dict[ks[0]][t + "_total"] for t in entity_types]
                ),
                "Missing": np.sum(
                    [hits_dict[ks[0]][t + "_missing"] for t in entity_types]
                ),
            },
        }
    )

    results = pd.DataFrame.from_records(rows)
    results["Total"] = results[["Spans Matched", "Missing"]].sum(axis=1)
    results["Missing Proportion"] = results["Missing"] / (results["Total"])
    return results.round(4)


def _hit_at_k(gold_db_id, candidates, k):
    '''
    Calculate if one of the top k ranked entities was the correct one
    '''
    if gold_db_id in [x['db_id'] for x in candidates[:k]]:
        return True
    else:
        return False

def evaluate_linking(preds, type2name, ks=[2**i for i in range(7)], custom_types=None):
    '''
    Evaluate entity linking for "normal" entity linking models (i.e. models that make predictions for every span in the dataset)
    '''
    ks = np.array(ks)
    pred_summary = {semtype: {'pred': np.zeros_like(ks), 'total':np.zeros_like(ks)} for semtype in type2name.keys()}
#     pred_summary['overall'] = {'pred': np.zeros_like(ks), 'total':np.zeros_like(ks)}
    # errors = defaultdict(list)
    # successes = defaultdict(list)
    for pred_dict in preds:
        sem_type = pred_dict['type']
        gold_db_id = pred_dict['normalized'][0]['db_id']
        candidates = pred_dict['candidates']
        pred_summary[sem_type]['pred'] += np.array([_hit_at_k(gold_db_id, candidates, k) for k in ks])
        pred_summary[sem_type]['total'] += np.ones_like(ks)
        
        
        
    filtered_summary = {k:v for k, v in pred_summary.items() if v['total'][0] > 0}

    return filtered_summary

def format_pred_summary(pred_summary, type2name, ks=[2**i for i in range(7)]):
    '''
    Format summary of predictions
    '''
    data = {type2name[k]: v['pred']/v['total'] for k, v in pred_summary.items()}
    pred_counts =  np.array([x['pred'] for x in pred_summary.values()]).sum(axis=0)
    total_counts =  np.array([x['total'] for x in pred_summary.values()]).sum(axis=0)   
    data['overall'] = pred_counts/total_counts
#     data.append({})
    return pd.DataFrame.from_dict(data, orient='index', columns=[f'recall@{k}' for k in ks])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pred_path", required=True, type=str)
    parser.add_argument("--scispacy", action="store_true")
    parser.add_argument(
        "--types_path",
        type=str,
        default="/efs/davidkartchner/2021AB_semantic_network/SRDEF",
        help="Path to NLM Semantic Network file that maps each semantic type to a name.  Can be downloaded from https://lhncbc.nlm.nih.gov/semanticnetwork/download.html",
    )
    parser.add_argument
    args = parser.parse_args()
    type2name = load_types(args.types_path)
    preds = ujson.load(args.pred_path)
    summary = evaluate_linking(preds, type2name)
    print(format_pred_summary(summary, type2name))
