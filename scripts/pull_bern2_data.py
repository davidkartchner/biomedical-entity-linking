import requests
import time

import ujson
import pandas as pd

from bigbio.dataloader import BigBioConfigHelpers
from tqdm.auto import tqdm, trange
from collections import defaultdict

conhelps = BigBioConfigHelpers()


def query_pmid(pmids, url="http://bern2.korea.ac.kr/pubmed"):
    return requests.get(url + "/" + ",".join(pmids)).json()


def query_plain(text, url="http://bern2.korea.ac.kr/plain"):
    return requests.post(url, json={"text": text}).json()


def retrieve_pmid_list(pmid_list, chunksize=20):
    all_retrieved_documents = []
    for i in trange(len(pmid_list) // chunksize + 1):
        pmid_chunk = pmid_list[i * chunksize : (i + 1) * chunksize]
        retrieved_docs = query_pmid(pmid_chunk)
        if len(retrieved_docs) == 0:
            print("Error on PMIDS:", pmid_chunk)
        all_retrieved_documents.extend(retrieved_docs)
        time.sleep(100)

    return all_retrieved_documents


def retrieve_full_text_documents(all_full_text_dict, chunksize=20):
    all_annotations = []
    chunk_iter = 0
    for pmid, doc in all_full_text_dict.items():
        if chunk_iter == chunksize:
            time.sleep(100)
            chunk_iter = 0
        annotations = query_plain(doc)
        if len(annotations) == 0:
            print("Error for PMID:", pmid)
        annotations["document_id"] = pmid
        all_annotations.append(annotations)
        chunk_iter += 1

    return all_annotations


all_pmids = set([])
all_full_text = defaultdict(str)
total_docs = 0


for dataset in tqdm(
    ["medmentions_full", "bc5cdr", "gnormplus", "ncbi_disease", "nlmchem", "nlm_gene"]
):
    data = conhelps.for_config_name(f"{dataset}_bigbio_kb").load_dataset()
    for split in data.keys():
        for doc in data[split]:
            pmid = doc["document_id"]
            if pmid in all_pmids:
                continue

            all_pmids.add(pmid)
            doc_text = " ".join([" ".join(p["text"]) for p in doc["passages"]])
            all_full_text[pmid] = doc_text


# # PlantNorm
# print("Running Plant Norm")
# for subset in ['training','test','development']:
#     with open(f'../../PPRcorpus/corpus/DMCB_plant_{subset}_corpus.txt', 'r', encoding='utf-8', errors='ignore') as g:
#         all_text = g.read()
#         abstracts = all_text.strip().split('\n\n')
#         abstract_lines = [x.split('\n') for x in abstracts]
#         for abs in tqdm(abstract_lines):
#             pmid = abs[0].split('|')[0]
#             if pmid in all_pmids:
#                 continue
#             if len(abs[0].split('|')) == 1:
#                 abs.pop(0)
#             title = abs[0].split('|')[1]
#             abs_text = abs[1].split('|')[1]
#             doc_text = ' '.join([title, abs_text])

#             all_pmids.add(pmid)
#             all_full_text[pmid].add(doc_text)


all_pmids = list(all_pmids)


pulled_pmids = retrieve_pmid_list(all_pmids)
with open("data/bern2_annotations_from_pmids.json", "w") as f:
    f.write(ujson.dumps(pulled_pmids, indent=2))


pulled_full_text = retrieve_full_text_documents(all_full_text)
with open("data/bern2_annotations_from_full_text.json", "w") as f:
    f.write(ujson.dumps(pulled_full_text, indent=2))
