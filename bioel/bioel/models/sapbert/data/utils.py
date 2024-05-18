from tqdm import tqdm
import itertools
import random
from typing import List


from bioel.ontology import BiomedicalOntology

def gen_pairs(input_list: List) -> List:
    """
    Generate all possible pairs from a list of items.
    """
    return list(itertools.combinations(input_list, r = 2))

def generate_pretraining_data(ontology: BiomedicalOntology):
    """
    Generate pretraining data from UMLS for the SAPBERT model.
    """
    

    pos_pairs = []
    for k, v in tqdm(ontology.entities.items(), desc = "Generating pretraining data for SAPBERT"):
        pairs = gen_pairs(v.aliases)
        if len(pairs) > 50: # If > 50 pairs, sample 50 pairs
            pairs = random.sample(pairs, 50)
        for p in pairs:
            print(k, p[0], p[1])
            line = str(k) + "||" + p[0] + "||" + p[1]
            pos_pairs.append(line)
    
    # Save the pos_pairs into a .txt file
    with open('./training_file_umls_2022AA_en_uncased_no_dup_pairwise_pair_th50.txt', 'w') as file:
        for pair in pos_pairs:
            file.write(pair + '\n')
    
    return pos_pairs

ontology_config = {
        "filepath": "/mitchell/entity-linking/2022AA/META/",
        "name": "UMLS",
        "abbrev": None,
    }
ontology = BiomedicalOntology.load_umls(**ontology_config)
print(generate_pretraining_data(ontology))
