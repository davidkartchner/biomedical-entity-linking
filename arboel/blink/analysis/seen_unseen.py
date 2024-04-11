import pickle
import os
import json
from collections import defaultdict
from IPython import embed

BLINK_ROOT = f'{os.path.abspath(os.path.dirname(__file__))}/../..'

# MedMentions:
output_file_path = os.path.join(BLINK_ROOT, 'models/trained/medmentions', 'seen_unseen.json')
train_data_path = os.path.join(BLINK_ROOT, 'models/trained/medmentions', 'train_processed_data.pickle')
test_data_path = os.path.join(BLINK_ROOT, 'models/trained/medmentions', 'test_processed_data.pickle')
result_paths = {
    'in-batch (independent)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_blink/eval/wo_type/independent/eval_results_1621176737-directed-0.json'),
    'in-batch (directed)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_blink/eval/wo_type/eval_results_1621123985-directed-1.json'),
    'in-batch (undirected)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_blink/eval/wo_type/eval_results_1621123985-undirected-1.json'),
    'knn (independent)': os.path.join(BLINK_ROOT, 'models/trained/medmentions/eval/pos_neg_loss/no_type/wo_type/independent/eval_results_1621175775-directed-0.json'),
    'knn (directed)': os.path.join(BLINK_ROOT, 'models/trained/medmentions/eval/pos_neg_loss/no_type/wo_type/probe10/eval_results_1621123098-directed-1.json'),
    'knn (undirected)': os.path.join(BLINK_ROOT, 'models/trained/medmentions/eval/pos_neg_loss/no_type/wo_type/probe10/eval_results_1621123098-undirected-1.json'),
    'mst (independent)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_mst/eval/pos_neg_loss/no_type/wo_type/independent/eval_results_1621174536-directed-0.json'),
    'mst (directed)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_mst/eval/pos_neg_loss/no_type/wo_type/eval_results_1621123562-directed-2.json'),
    'mst (undirected)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_mst/eval/pos_neg_loss/no_type/wo_type/eval_results_1621123562-undirected-1.json'),
    '1nn (independent)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_mst/eval/pos_neg_loss/arboNN/eval_results_1634803359-directed-0.json'),
    '1nn (directed)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_mst/eval/pos_neg_loss/arboNN/eval_results_1634803359-directed-4.json'),
    '1nn (undirected)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_mst/eval/pos_neg_loss/arboNN/eval_results_1634803359-undirected-4.json'),
    '1rand (independent)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_mst/eval/pos_neg_loss/arboNN_rand/eval_results_1634829864-directed-0.json'),
    '1rand (directed)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_mst/eval/pos_neg_loss/arboNN_rand/eval_results_1634829864-directed-4.json'),
    '1rand (undirected)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_mst/eval/pos_neg_loss/arboNN_rand/eval_results_1634829864-undirected-4.json'),
    'cross_arbo': os.path.join(BLINK_ROOT, 'models/trained/medmentions/crossencoder/eval/arbo/results.json'),
    'cross_arbo (oracle)': os.path.join(BLINK_ROOT, 'models/trained/medmentions/crossencoder/eval/arbo/oracle/results.json'),
    'cross_1nn': os.path.join(BLINK_ROOT, 'models/trained/medmentions/crossencoder/eval/1nn/results.json'),
    'cross_1nn (oracle)': os.path.join(BLINK_ROOT, 'models/trained/medmentions/crossencoder/eval/1nn/oracle/results.json'),
    'cross_1rand': os.path.join(BLINK_ROOT, 'models/trained/medmentions/crossencoder/eval/1rand/results.json'),
    'cross_1rand (oracle)': os.path.join(BLINK_ROOT, 'models/trained/medmentions/crossencoder/eval/1rand/oracle/results.json'),
    'cross_knn': os.path.join(BLINK_ROOT, 'models/trained/medmentions/crossencoder/eval/knn/results.json'),
    'cross_knn (oracle)': os.path.join(BLINK_ROOT, 'models/trained/medmentions/crossencoder/eval/knn/oracle/results.json'),
    'cross_in-batch': os.path.join(BLINK_ROOT, 'models/trained/medmentions/crossencoder/eval/in_batch/results.json'),
    'cross_in-batch (oracle)': os.path.join(BLINK_ROOT, 'models/trained/medmentions/crossencoder/eval/in_batch/oracle/results.json'),
}

seen_unseen_results = {}

print("Loading train data...")
with open(train_data_path, 'rb') as read_handle:
    train_data = pickle.load(read_handle)

seen_cui_ids = set()
cui_to_men = defaultdict(set)
for mention in train_data:
    cui = mention['label_cuis'][0]
    seen_cui_ids.add(cui)
    cui_to_men[cui].add(mention['mention_id'])

n_men_per_cui = {k: len(v) for k, v in cui_to_men.items()}

print("Loading test data...")
with open(test_data_path, 'rb') as read_handle:
    test_data = pickle.load(read_handle)

seen_mention_idxs, unseen_mention_idxs = set(), set()
seen_single_mention_idxs, seen_non_single_mention_idxs = set(), set()  # singletons, non-singletons

print("Analyzing...")
for mention in test_data:
    cui = mention['label_cuis'][0]
    men_id = mention['mention_id']
    if cui in seen_cui_ids:
        seen_mention_idxs.add(men_id)
        if n_men_per_cui[cui] == 1:
            seen_single_mention_idxs.add(men_id)
        else:
            seen_non_single_mention_idxs.add(men_id)
    else:
        unseen_mention_idxs.add(men_id)

len_seen = len(seen_mention_idxs)
len_single_seen = len(seen_single_mention_idxs)
len_non_single_seen = len(seen_non_single_mention_idxs)
len_unseen = len(unseen_mention_idxs)
len_total = len_seen + len_unseen

seen_unseen_results['n_seen'] = len_seen
seen_unseen_results['n_single_seen'] = len_single_seen
seen_unseen_results['n_non_single_seen'] = len_non_single_seen
seen_unseen_results['n_unseen'] = len_unseen
seen_unseen_results['n_total'] = len_total

for mode in result_paths:
    n_success_seen, n_success_unseen = 0., 0.
    n_success_single_seen, n_success_non_single_seen = 0., 0.
    print(f"Mode: {mode}")
    with open(result_paths[mode]) as f:
        results = json.load(f)
    for m in results['success']:
        men_id = m['mention_id']
        if men_id in seen_mention_idxs:
            n_success_seen += 1.
            if men_id in seen_single_mention_idxs:
                n_success_single_seen += 1.
            else:
                n_success_non_single_seen += 1.
        else:
            n_success_unseen += 1.
    n_success_total = n_success_seen + n_success_unseen

    seen_acc = (n_success_seen / len_seen) * 100
    seen_single_acc = (n_success_single_seen / len_single_seen) * 100
    seen_non_single_acc = (n_success_non_single_seen / len_non_single_seen) * 100
    unseen_acc = (n_success_unseen / len_unseen) * 100
    overall_acc = (n_success_total / len_total) * 100

    seen_unseen_results[mode] = {
        'overall': overall_acc,
        'seen': seen_acc,
        'seen_singleton': seen_single_acc,
        'seen_non_singleton': seen_non_single_acc,
        'unseen': unseen_acc
    }

with open(output_file_path, 'w') as f:
    json.dump(seen_unseen_results, f, indent=2)
    print(f"\nAnalysis saved at: {output_file_path}")
