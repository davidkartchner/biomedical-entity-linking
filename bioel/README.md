# BioEL: A comprehensive package for training, evaluating, and benchmarking biomedical entity linking models.

## Installation
```bash
conda create -n bioel python=3.9
conda activate bioel
pip install -e .
```

## Development Instructions

1. Install as in editable package using `pip` as shown above.
1. Add any new dependencies to `setup.py`.
1. Add tests to `tests/` directory.

## Ontologies
Ontologies included in the package : 

- UMLS : UMLS is licensed by the National Library of Medicine and requires a free account to download. You can sign up for an account at https://uts.nlm.nih.gov/uts/signup-login. Once your account has been approved, you can download the UMLS metathesaurus at https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html.

- MeSH : MeSH is derived from UMLS

- NCBI Gene (Entrez) ontololgy. It can be downloaded at https://ftp.ncbi.nih.gov/gene/DATA/
(gene_info.gz)

- MEDIC (MErged DIsease voCabulary) : It can be downloaded at https://ctdbase.org/downloads/

## Resolving abbreviations
As a preprocessing step, we resolve abbreviations in the text using Ab3P, an abbreviation detector created for biomedical text. We ran abbreviation detection on the text of all documents in our benchmark, the results of which are stored in a large dictionary in `data/abbreviations.json`. In order to reproduce our abbreviation detection/resolution pipeline, please run the following:

```
from bioel.utils.solve_abbreviation.solve_abbreviation import create_abbrev
create_abbrev(output_dir, all_dataset)
# output_path : path where to create abbreviations.json
# all_dataset : datasets for which you want the abbreviations.
```

## Example usage
```
# Import modules
from bioel.model import Model_Wrapper
from bioel.evaluate import Evaluate

# load model
krissbert = Model_Wrapper.load_krissbert(
        name="krissbert", params_file=params,
    )
# Look at data/params.json for more information about the parameters
krissbert.training() # train
krissbert.inference() # inference

abbreviations_path = "data/abbreviations.json"
dataset_names = ["ncbi_disease"]
model_names = ["krissbert"]
path_to_result = {
    "ncbi_disease": {
        "krissbert": "results/ncbi_disease.json"
    }
}

# Results
evaluator = Evaluate(dataset_names, model_names, path_to_result, abbreviations_path)
evaluator.load_results()
evaluator.process_datasets()
evaluator.evaluate()
evaluator.plot_results()
```

## ArboEL

ArboEL operates in two stages: First, you need to train the biencoder (`load_arboel_biencoder`). Then, you use the candidate results from the biencoder to train the crossencoder (`load_arboel_crossencoder`) and perform evaluation with the crossencoder.


<!-- TODO: Add quickstart, examples -->