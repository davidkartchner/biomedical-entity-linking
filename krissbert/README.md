---
language: en
tags:
- exbert
license: mit
pipeline_tag: feature-extraction
widget:
- text: "<ENT> ER </ENT> crowding has become a wide-spread problem."
---

## KRISSBERT 
[https://arxiv.org/pdf/2112.07887.pdf](https://arxiv.org/pdf/2112.07887.pdf)

Entity linking faces significant challenges such as prolific variations and prevalent ambiguities, especially in high-value domains with myriad entities. Standard classification approaches suffer from the annotation bottleneck and cannot effectively handle unseen entities. Zero-shot entity linking has emerged as a promising direction for generalizing to new entities, but it still requires example gold entity mentions during training and canonical descriptions for all entities, both of which are rarely available outside of Wikipedia ([Logeswaran et al., 2019](https://aclanthology.org/P19-1335.pdf); [Wu et al., 2020](https://aclanthology.org/2020.emnlp-main.519.pdf)). We explore Knowledge-RIch Self-Supervision (KRISS) and train a contextual encoder (KRISSBERT) for entity linking, by leveraging readily available unlabeled text and domain knowledge.

Specifically, the KRISSBERT model is initialized with [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract) parameters, and then continuously pretrained using biomedical entity names from the [UMLS](https://www.nlm.nih.gov/research/umls/index.html) ontology to self-supervise entity linking examples from [PubMed](https://pubmed.ncbi.nlm.nih.gov/) abstracts. Experiments on seven standard biomedical entity linking datasets show that KRISSBERT attains new state of the art, outperforming prior self-supervised methods by as much as 20 absolute points in accuracy.
See [Zhang et al., 2021](https://arxiv.org/abs/2112.07887) for the details.

Note that some prior systems like [BioSyn](https://aclanthology.org/2020.acl-main.335.pdf), [SapBERT](https://aclanthology.org/2021.naacl-main.334.pdf), and their follow-up work (e.g., [Lai et al., 2021](https://aclanthology.org/2021.findings-emnlp.140.pdf)) claimed to do entity linking, but their systems completely ignore the context of an entity mention, and can only predict a surface form in the entity dictionary (See Figure 1 in [BioSyn](https://aclanthology.org/2020.acl-main.335.pdf)), _**not the canonical entity ID (e.g., CUI in UMLS)**_. Therefore, they can't disambiguate ambiguous mentions. For instance, given the entity mention "_ER_" in the sentence "*ER crowding has become a wide-spread problem*", their systems ignore the sentence context, and simply predict the closest surface form, which is just "ER". Multiple entities share this surface form as a potential name or alias, such as *Emergency Room (C0562508)*, *Estrogen Receptor Gene (C1414461)*, and *Endoplasmic Reticulum(C0014239)*. Without using the context information, their systems can't resolve such ambiguity and pinpoint the correct entity *Emergency Room (C0562508)*. More problematically, their evaluation would deem such an ambiguous prediction as correct. Consequently, the reported results in their papers do not reflect true performance on entity linking.


## Usage for Entity Linking

Here, we use the [MedMentions](https://github.com/chanzuckerberg/MedMentions) data to show you how to 1) **generate prototype embeddings**, and 2) **run entity linking**.

(We are currently unable to release the self-supervised mention examples, because they require the UMLS and PubMed licenses.)


#### 1. Create conda environment and install requirements
```bash
conda create -n kriss -y python=3.8 && conda activate kriss
pip install -r requirements.txt
```

#### 2. Switch the root dir to [usage](https://huggingface.co/microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL/tree/main/usage)
```bash
cd usage
```

#### 3. Download the MedMentions dataset

```bash
git clone https://github.com/chanzuckerberg/MedMentions.git
```

#### 4. Generate prototype embeddings
```bash
python generate_prototypes.py
```

#### 5. Run entity linking
```bash
python run_entity_linking.py
```

This will give you about `58.3%` top-1 accuracy.


## Citation

If you find KRISSBERT useful in your research, please cite the following paper:

```latex
@article{krissbert,
  author = {Sheng Zhang, Hao Cheng, Shikhar Vashishth, Cliff Wong, Jinfeng Xiao, Xiaodong Liu, Tristan Naumann, Jianfeng Gao, Hoifung Poon},
  title = {Knowledge-Rich Self-Supervision for Biomedical Entity Linking},
  year = {2021},
  url = {https://arxiv.org/abs/2112.07887},
  eprinttype = {arXiv},
  eprint = {2112.07887},
}
```