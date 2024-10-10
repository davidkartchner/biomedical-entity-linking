import os

print("current directory:", os.getcwd())
from bioel.utils.bigbio_utils import load_bigbio_dataset
from tqdm import tqdm
from collections import defaultdict

output_file = "all_article_text.txt"
output_dir = "/home2/cye73/data/solve_abbrev"

# Make sure the output directory exists, create it if it doesn't
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, output_file)

# Make sure we don't duplicate anything
all_pmids = set([])
all_full_text = defaultdict(set)
total_docs = 0

# Loop through all datasets
# for dataset in tqdm(
#     ["medmentions_full", "bc5cdr", "gnormplus", "ncbi_disease", "nlmchem", "nlm_gene"]
# ):
for dataset in tqdm(["ncbi_disease"]):
    data = load_bigbio_dataset(dataset)
    for split in data.keys():
        for doc in data[split]:
            pmid = doc["document_id"]
            if pmid in all_pmids:
                continue

            all_pmids.add(pmid)
            doc_text = " ".join([" ".join(p["text"]) for p in doc["passages"]])
            all_full_text[pmid].add(doc_text)


# # PlantNorm
# print("Running Plant Norm")
# for subset in ['training','test','development']:
#     with open(f'/Users/david/Downloads/DMCB_plant_{subset}_corpus.txt', 'r', encoding='utf-8', errors='ignore') as g:
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

# Remove newlines that will interfere with Ab3P
for pmid, text_set in all_full_text.items():
    text = list(text_set)[0]
    if "\n" in text:
        print(pmid)
    all_full_text[pmid] = text.replace("\n", " ")
# Write output to file
with open(output_path, "w") as f:
    output = "\n\n".join(
        [pmid + " | " + doc_text for pmid, doc_text in all_full_text.items()]
    )
    f.write(output)
