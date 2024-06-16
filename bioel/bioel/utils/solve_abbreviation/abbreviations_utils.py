import ujson
import os

from bioel.utils.bigbio_utils import load_bigbio_dataset
from tqdm import tqdm
from collections import defaultdict


def extract_document_text(output_dir, all_dataset):
    """
    Collect and consolidate textual data from various medical datasets
    ---------
    Parameter
    - output_dir : Path to directory where to save "abbreviations.json" file
    - all_dataset : Datasets for which you want to find abbreviations
    """

    # Make sure the output directory exists, create it if it doesn't
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"all_article_text.txt")

    # Make sure we don't duplicate anything
    all_pmids = set([])
    all_full_text = defaultdict(set)
    total_docs = 0

    # Loop through all datasets
    for dataset in tqdm(all_dataset):
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
    with open(output_file, "w") as f:
        output = "\n\n".join(
            [pmid + " | " + doc_text for pmid, doc_text in all_full_text.items()]
        )
        f.write(output)


def process_abbreviations(output_dir, all_dataset):
    """
    Process a list of abbreviations from a text file,
    filtering and collecting them into a JSON format based on a confidence score,
    and then saving the result to a specified output directory
    ---------
    Parameter
    - output_dir : Path to directory where to save "abbreviations.json" file
    - all_dataset : Datasets for which you want to find abbreviations
    """
    # Set up necessary variables/parameters
    all_abbreviations = {}
    min_confidence_cutoff = 0.95
    omitted = 0
    included = 0

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "abbreviations.json")

    input_file = os.path.join(output_dir, "raw_abbreviations.txt")

    with open(input_file, "r") as f:
        chunks = f.read().strip().split("\n\n")

    # Get abbreviations from each article
    for chunk in chunks:
        lines = chunk.split("\n")
        pmid = lines[0].split("|")[0].strip()
        abbrev_dict = {}
        for line in lines[1:]:
            abbrev, long_form, confidence_score = line.strip().split("|")
            confidence_score = float(confidence_score)
            if confidence_score > min_confidence_cutoff:
                abbrev_dict[abbrev] = long_form
                included += 1
            else:
                # print(abbrev, long_form, confidence_score)
                omitted += 1

        all_abbreviations[pmid] = abbrev_dict

    with open(output_file, "w") as f:
        f.write(ujson.dumps(all_abbreviations, indent=2))
