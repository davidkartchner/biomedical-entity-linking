#!/bin/bash

for DATASET in "ncbi_disease" "bc5cdr" "nlmchem" "medmentions_st21pv" "nlm_gene" "gnormplus" "medmentions_full" 
do
    cp /mitchell/entity-linking/MedLinker/data/${DATASET}_big_bio/output_test.json results/medlinker/$DATASET.json
done