from tqdm.auto import tqdm

# Replace paths with the correct paths on your system
pubmed_dir = 'path/to/pubmed_dir/'
output_dir = 'path/to/output_dir/'
entity_extractor_path = 'path/to/entity_extractor.pkl'

def filter_entities(entities):
    
    def filter_spans(spans):
        spans = list(set(spans))
        # Sort the spans by start index
        sorted_spans = sorted(spans, key=lambda x: (x[0], -x[1]))
        # Create a list to store the filtered spans
        filtered_spans = []
        discard_spans = set()
        # Loop through the sorted spans
        for i, span in enumerate(sorted_spans):
            if span in discard_spans:
                continue
            # Check if this span overlaps with another span
            overlaps = False
            for other_span in sorted_spans[i+1:]:
                if span[1] > other_span[0]:
                    if span[1] < other_span[1]:
                        # partial overlap, discard both
                        discard_spans.add(other_span)
                    overlaps = True
            if overlaps:
                continue
            # Otherwise, add this span to the filtered list
            filtered_spans.append(span)
        return filtered_spans

    spans = [tuple(ent['offset']) for ent in entities]
    filtered_spans = set(filter_spans(spans))
    return [ent for ent in entities if tuple(ent['offset']) in filtered_spans]


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

class EntityExtractor:
    def __init__(self, entity_alias_map):
        self.trie = Trie()
        self.entity_alias_map = entity_alias_map
        print("Building trie...")
        # for mention in tqdm(entity_alias_map.keys()):
        #     self.trie.insert(mention)
        
        list(map(self.trie.insert, tqdm(entity_alias_map.keys())))
    
    def extract_entities(self, text):
        entities = []
        text_length = len(text)
        word_indices = [0] + [i + 1 for i, char in enumerate(text) if char.isspace()]
        for i in word_indices:
            node = self.trie.root
            for j in range(i, text_length):
                char = text[j]
                if char in node.children:
                    node = node.children[char]
                    if node.is_end_of_word:
                        # if (j+1 < len(text)) and not text[j+1].isspace():
                        if (j+1 < len(text)) and text[j+1].isalnum():
                            continue
                        entity = text[i:j + 1]
                        try:
                            ids = self.entity_alias_map[entity]
                            entities.append({
                                "cui": ids['cui'],
                                "tui": ids['tui'],
                                "mention": entity,
                                "offset": [i, j + 1],
                            })
                        except KeyError:
                            pass
                else:
                    break
        return entities
    
    def get_context(self, entities, text, window_size):
        k = 9 * (window_size//2) # 9 is 1.25 times the average word length
        context_entities = []
        text_length = len(text)
        for entity in entities:
            i, j = entity['offset']
            start_index = max(0, i - k)
            end_index = min(text_length, j + k)
            left_context = text[start_index:i].split()
            right_context = text[j:end_index].split()
            left_context = " ".join([tok for tok in left_context[-window_size//2:]])
            right_context = " ".join([tok for tok in right_context[:window_size//2]])
            entity_w_context = left_context + " " + entity['mention'] + " " + right_context
            entity['mention_w_context'] = entity_w_context
            context_entities.append(entity)
        return context_entities

import _pickle as pickle

print("Loading entity extractor ...")

with open(entity_extractor_path, 'rb') as f:
    extractor = pickle.load(f)

print("Finished loading entity extractor.")

import os
import json

from pubmed_xml import Pubmed_XML_Parser

pubmed = Pubmed_XML_Parser()

print("Finding files ...")
files = [file for file in os.listdir(pubmed_dir) if file.endswith('.xml.gz')]
print(f"Found {len(files)} pubmed files.")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

global_id_counter = 0

def process_file(file, limit=None):
    global global_id_counter
    data = []
    for i, article in enumerate(pubmed.parse_title_abstract(os.path.join('/mitchell/nlp_corpora/pubmed/', file))):
        title_offset = len(article.title) + 1
        article_data = {'id': str(global_id_counter), 'document_id': str(article.pmid), 'passages': []}
        global_id_counter += 1
        title_passage = {'id': str(global_id_counter), 'type': 'title', 'text': [article.title], 'offsets': [[0, len(article.title)]]}
        global_id_counter += 1
        abstract_passage = {'id': str(global_id_counter), 'type': 'abstract', 'text': [article.abstract], 'offsets': [[title_offset, title_offset + len(article.abstract)]]}
        global_id_counter += 1
        article_data['passages'].append(title_passage)
        article_data['passages'].append(abstract_passage)
        
        text = article.title + "\n" + article.abstract
        entities = extractor.extract_entities(text)
        entities = filter_entities(entities)
        article_data['entities'] = [{'id': str(global_id_counter+idx), 'type': entity['tui'], 'text': [entity['mention']], 'offsets': [entity['offset']], 'normalized': [{'db_name': 'UMLS', 'db_id': entity['cui']}]} for idx, entity in enumerate(entities)]
        global_id_counter += len(entities)
        article_data['events'] = []
        article_data['coreferences'] = []
        article_data['relations'] = []
        data.append(article_data)
        
        if limit and i+1 >= limit:
            print("Document processing limit reached. Terminating ...")
            break

    output_file = os.path.join(output_dir, f'{file.rstrip(".xml.gz")}.json')
    with open(output_file, 'w') as f:
        json.dump(data, f)

print("Processing files ...")
for file in tqdm(files):
    process_file(file)

print("Finished processing files.")