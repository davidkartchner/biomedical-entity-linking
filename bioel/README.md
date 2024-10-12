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
from bioel.utils.solve_abbreviation.solve_abbreviations import create_abbrev
create_abbrev(output_dir, all_dataset)
# output_path : path where to create abbreviations.json
# all_dataset : datasets for which you want the abbreviations.
```

## Example usage
```
# Import modules
from bioel.model import BioEL_Model
from bioel.evaluate import Evaluate

# load model
krissbert = BioEL_Model.load_krissbert(
        name="krissbert", params_file='path/to/params_krissbert.json",
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
eval_strategies=["basic"]

# Results
evaluator = Evaluate(dataset_names=dataset_names, 
                     model_names=model_names, 
                     path_to_result=path_to_result, 
                     abbreviations_path=abbreviations_path, 
                     eval_strategies=eval_strategies,
                     max_k=10,
                     )
evaluator.load_results()
evaluator.process_datasets()
evaluator.evaluate()
evaluator.plot_results()
evaluator.detailed_results()
```

These functions will run the evaluation for all models / datasets.
For error analysis with hit index details, use `evaluator.error_analysis_dfs` attribute.
For detailed results on failure stage, accuracy per type, recall@k per type, MAP@k, statistical significance (p_values), use `evaluator.detailed_results_analysis`.

## Load the different datasets
```
from bioel.evaluate import Evaluate
abbreviations_path = "data/abbreviations.json"
dataset_name = "bc5cdr" # Specify the desired dataset name from the BigBio collection here.
dataset = Dataset(
    dataset_name = dataset_name, abbreviations_path=abbreviations_path
)

```


## Load the different ontologies

```
from bioel.ontology import BiomedicalOntology

##### ----------------- Load medic ----------------- #####

dataset_name = 'ncbi_disease'
medic_dict = {"name" : "medic",
            "filepath" : "path/to/medic"} # medic.tsv file

ontology = BiomedicalOntology.load_medic(**medic_dict)

##### ----------------- Load entrez ----------------- #####

dataset_name = "gnormplus" # or "nlm_gene"

entrez_dict = {"name" : "entrez",
             "filepath" : "path/to/entrez", # gene_info.tsv file
             "dataset" : f"{dataset_name}",}
ontology = BiomedicalOntology.load_entrez(**entrez_dict)

##### ----------------- Load MESH ----------------- #####

dataset_name = "nlmchem"
mesh_dict = {"name" : "mesh",
             "filepath" : "path/to/umls"}
ontology = BiomedicalOntology.load_mesh(**mesh_dict)

##### ----------------- Load UMLS (st21pv subset) ----------------- #####

dataset_name = "medmentions_st21pv"
umls_dict_st21pv = {
    "name": "umls",
    "filepath": "path/to/umls",
    "path_st21pv_cui": "data/umls_cuis_st21pv.json",
}
ontology = BiomedicalOntology.load_umls(**umls_dict_st21pv)

##### ----------------- Load UMLS (full) ----------------- #####

dataset_name = "medmentions_full"
umls_dict = {
    "name": "umls",
    "filepath": "path/to/umls",
}
ontology = BiomedicalOntology.load_umls(**umls_dict)
```

## Config files
Example configuration files for the various models are available in the `data/` directory for users to reference and follow.


## ArboEL


ArboEL operates in two stages: First, you need to train the biencoder (`load_arboel_biencoder`). Then, you use the candidate results from the biencoder to train the crossencoder (`load_arboel_crossencoder`) and perform evaluation with the crossencoder.


## BioBART/BioGenEL

BioBART and BioGenEL share the same entity linking module: 
- In order to finetune from BioBART set the `model_load_path` parameter in the .json config file to `GanjinZero/biobart-v2-large`, it will load the pretrained weights from HuggingFace.
  
- In order the finetune from BioGenEL's Knowledge base guided pretrained weights, you first must download the pretrained weights from this link: https://drive.google.com/file/d/1TqvQRau1WPYE9hKfemKZr-9ptE-7USAH/view?usp=sharing and then set the `model_load_path` parameter in the .json config file to the path where you stored the pretrained weights.

There is only one config file to handle the training and the evaluation for BioBART/BioGenEL. You cannot launch the training and the evaluation in the same time, thus first set all the parameters in the config file for both the evaluation and the training. When you run training set `model_load_path` as described above.
Some important information:
- if `model_load_path` is not provided, by default the dataset folder will be `biogenel/data/`
- set `preprocess_data` to `true` during the first training for a given dataset or ontology to generate the preprocessed data. If you want to train with different parameters while using the same preprocessed data you can set it to `false`. During evaluation set it to `false`.
- `trie_path` and `dict_path` will be set to `data/abbreviationmode/datasetname/` folder for a given `data` folder. Thus, if `resolve_abbrevs` is set to `true`, make sure to include `/abbr_res/datasetname/trie.pkl` as a path extension for the trie and `/abbr_res/datasetname/target_kb.json` for the dict.
- For evaluation set `evaluation` to `true` and change the `model_load_path` to the path you provided to `model_save_path` during training. 

Here is a config file example for BioBart during training:
```
{
    "dataset_path": "/home/bnursal3/biomedical-entity-linking/bioel/bioel/models/biogenel/data/",
    "dataset_name": "bc5cdr",
    "ontology_name": "mesh",
    "ontology_dict": {
        "name": "mesh",
        "filepath": "/your/ontology/path"
    },
    "load_function": "load_mesh",
    "path_to_abbrev": "/home/bnursal3/biomedical-entity-linking/bioel/bioel/models/biogenel/abbreviations.json",
    "resolve_abbrevs": true,
    "preprocess_data": true,
    "model_save_path": "/home/bnursal3/biomedical-entity-linking/bioel/bioel/models/biogenel/model_saved/Lightning_checkpoint-20000/biobart/bc5cdr",
    "trie_path": "/home/bnursal3/biomedical-entity-linking/bioel/bioel/models/biogenel/data/abbr_res/bc5cdr/trie.pkl",
    "dict_path": "/home/bnursal3/biomedical-entity-linking/bioel/bioel/models/biogenel/data/abbr_res/bc5cdr/target_kb.json",
    "model_load_path": "/home/bnursal3/biomedical-entity-linking/bioel/bioel/models/biogenel/kb_guided_pretrain_ckpt_hf",
    "model_token_path": "facebook/bart-large",
    "logging_path": "/home/bnursal3/biomedical-entity-linking/bioel/bioel/models/biogenel/logs_bis",
    "logging_steps": 100,
    "save_steps": 20000,
    "num_train_epochs": 8,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 1,
    "warmup_steps": 500,
    "finetune": true ,
    "t5": false,
    "fairseq_loss": false,
    "evaluation": false,
    "testset": true,
    "load_prompt": true,
    "weight_decay": 0.01,
    "length_penalty": 1,
    "beam_threshold": 0,
    "unlikelihood_loss": false,
    "init_lr": 1e-5,
    "evaluation_strategy": "no",
    "prompt_tokens_enc": 0,
    "prompt_tokens_dec": 0,
    "seed": 0,
    "label_smoothing_factor": 0.1,
    "unlikelihood_weight": 0.1,
    "max_grad_norm": 0.1,
    "max_steps": 20000,
    "gradient_accumulate": 1,
    "lr_scheduler_type": "polynomial",
    "attention_dropout": 0.1,
    "dropout":0.1,
    "max_position_embeddings": 1024,
    "num_beams": 20,
    "max_length": 384,
    "min_length": 1,
    "sample_train": false,
    "prefix_prompt": false,
    "rerank": false,
    "init_from_vocab": false,
    "no_finetune_decoder": false,
    "syn_pretrain": false,
    "gold_sty": false,
    "prefix_mention_is": true,
    "rdrop": 0.0
}
```


<!-- TODO: Add quickstart, examples -->
