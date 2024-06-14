from tqdm import tqdm


def _obo_extract_synonyms(data):
    if 'synonym' in data:
        return [syn.split('"')[1]for syn in data['synonym']]
    else:
        return []

def _obo_extract_definition(data):
    if 'def' in data:
        return data['def'].strip('"').split('"')[0]

        
def _obo_term_to_synonyms(graph, filter_prefix=None):
    node_dict = {}
    for curie, data in tqdm(graph.nodes(data=True)):
        if filter_prefix is not None:
            if not curie.startswith(filter_prefix):
                continue
        if 'name' not in data:
            # print(f'Missing name.  CURIE: {curie}, data: {data}')
        # if 'synonym' not in data:
            # print(f'Missing synonym.  CURIE: {curie}, data: {data}')
            synonyms = _obo_extract_synonyms(data)
        else:
            synonyms = [data['name']] + _obo_extract_synonyms(data)
            node_dict[curie] = synonyms

    return node_dict

def _obo_term_to_definitions(graph, filter_prefix=None):
    node_dict = {}
    for curie, data in tqdm(graph.nodes(data=True)):
        if filter_prefix is not None:
            if not curie.startswith(filter_prefix):
                continue
        else:
            definition = _obo_extract_definition(data)
            node_dict[curie] = definition

    return node_dict