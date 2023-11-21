# A Comprehensive Evaluation of Biomedical Entity Linking Models
Benchmark and error analysis of biomedical entity linking models

![Structure of our comparison of biomedical entity linking models](figures/230415_entity_linking_survey_updated.png)

### Citation
If you have found our manuscript useful in your work, please consider citing:

> Kartchner D., Deng J., Lohiya S., Kopparthi T., Bathala B, Daniel Domingo-Fernández†, Cassie S. Mitchell† (2023).
A Comprehensive Evaluation of Biomedical Entity Linking Models. *Empirical Methods in Natural Language Processing (EMNLP) 2023*.

## Installation
In order to run this benchmark, please install the following conda environment:

```bash
conda env create -f el_robustness_env.yaml
conda activate el-robustness
```


If you wish to run the notebooks, also run:
```bash
ipykernel install el-robustness
```

## Datasets
<!-- | Dataset | Source Ongologies | Num. Abstracts | Num. Mentions | In context? |
| MedMentions [(Mohan and Li, 2019)](https://github.com/chanzuckerberg/MedMentions) | UMLS | 4,392 | ? | Yes |  -->
| Dataset                                                                                                            | Total Documents | Total Mentions | Unique Entities | Unique Types | Entity Overlap | Mention Overlap | Source Documents | Linked Ontology |
| :----------------------------------------------------------------------------------------------------------------- | :-------------- | :------------- | :-------------- | -----------: | -------------: | --------------: | :--------------- | :-------------- |
| MedMentions Full [(Mohan and Li, 2019)](https://github.com/chanzuckerberg/MedMentions)                             | 4,392           | 385,098        | 34,724          |          127 |         0.6199 |          0.8221 | PubMed Abstracts | UMLS            |
| MedMentions ST21PV [(Mohan and Li, 2019)](https://github.com/chanzuckerberg/MedMentions)                           | 4,392           | 203,282        | 25,419          |           21 |         0.5755 |          0.7741 | PubMed Abstracts | UMLS            |
| BC5CDR [(Li et al, 2016)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4860626)                                    | 1,500           | 29,044         | 2,348           |            2 |           0.53 |          0.7733 | PubMed Abstracts | MeSH            |
| GNormPlus [(Wei et al, 2016)](https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus)                        | 533             | 6,252          | 1,353           |            2 |         0.0789 |          0.0838 | PubMed Abstracts | Entrez          |
| NCBI Disease [(Dogan et al, 2014)](https://www.sciencedirect.com/science/article/pii/S1532046413001974?via%3Dihub) | 792             | 6,881          | 789             |            4 |           0.67 |          0.8156 | PubMed Abstracts | MeSH, OMIM      |
| NLM Chem [(Islamaj et al, 2021)](https://www.nature.com/articles/s41597-021-00875-1)                               | 150             | 37,999         | 1,787           |            1 |         0.4747 |          0.6229 | PMC Full-Text    | MeSH            |


In order to use these datasets, please install the BigScience Biomedical project:
```bash
git clone https://github.com/davidkartchner/biomedical.git
cd biomedical
pip install -e .
git checkout fix_data_inconsistencies
```

## Ontologies
Many of the ontologies in this benchmark are derived from UMLS.  UMLS is licensed by the National Library of Medicine and requires a free account to download.  You can sign up for an account at https://uts.nlm.nih.gov/uts/signup-login.  Once your account has been approved, you can download the UMLS metathesaurus at https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html.

This benchmnark also uses the NCBI Gene (Entrez) ontololgy.  It can be downloaded with the command:
```
wget https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/All_Data.gene_info.gz
```

## Resolving abbreviations
As a preprocessing step, we resolve abbreviations in the text using [Ab3P](https://github.com/ncbi-nlp/Ab3P), an abbreviation detector created for biomedical text. We ran abbreviation detection on the text of all documents in our benchmark, the results of which are stored in a large dictionary in `data/abbreviations.json`.  In order to reproduce our abbreviation detection/resolution pipeline, please run the following:

```bash
# Install submodules
git submodule init
git submodule update

# Install NCBITextLib
cd NCBITextLib/lib
make
cd ../..

# Clone Ab3P
cd Ab3P
```

Before building  will need to add the path of `NCBITextLib` to `Makefile` and `lib/Makefile` within Ab3P.  Assuming you installed NCBITextLib as above, set the value of `NCBITEXTLIB=../NCBITextLib` in the root directory `Makefile` and `NCBITEXTLIB=../../NCBITextLib` in `lib/Makefile`.

Now build the project and test that it ran correctly by running
```
make
make test
```
You can now detect abbreviations in a text file by simply running `./identify_abbr text_file_name` within the Ab3P directory.  For more data on running Ab3P, please see their [github repo](https://github.com/ncbi-nlp/Ab3P).

In order to process all abbreviations in the datasets used in this survey, simply run:
```bash
bash process_abbreviations.sh
```


## Using UMLS Utilities
This repository contains some utilities for extracting of UMLS and mappings between UMLS CUIs and unique identifiers from its constituent vocabularies.  


## Models Included
A number of models are included in this benchmark.  In order to make them as close to the original implementation as possible, we used much of the original source code modified to accomodate the our test datasets.  Instructions for installing each modified version as a submodule are provided below.

### SapBERT
**Install SapBERT**
```bash
git clone git@github.com:enveda/sapbert.git
```

#### Usage
In order to run SapBERT, you will need a dictionary mapping entity CUIs (concept unique identifiers) to their aliases.  This should be provided in the format of a text file with one CUI/alias per line as follows:
```
CUI1||alias1_1
CUI1||alias1_2
CUI1||alias1_3
CUI2||alias2_1
...
CUIn||aliasn_m
```
<!-- An example script that generates these files for various ontologies can be found in `notebooks/ -->
An example of this data can be found in `data/mesh_to_alias.txt`.


SapBERT can be used off-the-shelf with a pretrained transformer model located at https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext. You can also train your own model using an alias dictionary from the ontology to which mentions will be linked.  `sapbert/README.md` describes the required format of the training data.  Given an alias list such as `data/mesh_to_alias.txt`, you can generate training data for a model by running the command

```bash
python sapbert/generate_pretrain_data.py --input_dict_path data/mesh_to_alias.txt --output_filepath sapbert/training_data/mesh_positive_example_pairs.txt
```

In order to run SapBERT training, you will need to create a (free) [weights and biases](https://wandb.ai/site) account for automatic experiment tracking.  Then train the model by running

```bash
bash pretrain.sh 0 sapbert/training_data/mesh_positive_example_pairs.txt sapbert/trained_models/mesh/
```

Finally, the model can be evaluated by running

```bash
MODEL_DIR="trained_models/mesh/" 
DICT_PATH=/efs/davidkartchner/el-robustness-comparison/data/mesh_to_alias.txt
DATASET_NAME=bc5cdr

CUDA_VISIBLE_DEVICES=0 python3 run_bigbio_inference.py \
        --model_dir $MODEL_DIR \
        --dictionary_path $DICT_PATH \
        --dataset_name $DATASET_NAME \
        --output_dir ./output/ \
        --use_cuda \
        --max_length 25 \
        --batch_size 32 \
        --abbreviations_path /efs/davidkartchner/el-robustness-comparison/data/abbreviations.json
```

### KRISSBERT
#### Install KRISSBERT**
```bash
git lfs install
git clone https://huggingface.co/envedabio/krissbert
git submodule add krissbert
```
**Generate Prototypes**
Note: This will use a single GPU.  You can configure multi-gpu usage by modifying the `CUDA_VISIBLE_DEVICES` argument
```bash
cd krissbert/usage

# For a single dataset (medmentions_full)...
CUDA_VISIBLE_DEVICES=0 python generate_prototypes.py train_data.dataset_name=medmentions_full train_data.splits=[train]
# ...or for all datasets in the benchmark
bash get_all_embeddings.sh
```

**Run linking with generated prototypes**
```bash
# Single dataset
CUDA_VISIBLE_DEVICES=0 python run_entity_linking.py test_data.dataset_name=medmentions_full
# All datasets in benchmark
bash run_all_linking.sh
```
More information on KRISSBERT can be found in `krissbert/README.md`.



### ArboEL
ArboEL has dependencies that conflict with the rest of the packages in this repository so it needs its own conda environment.  To install this, please run the following:
```
conda env create -f arboel.yaml
```

### SciSpacy

### BERN2

## Evaluation Strategy

## References
