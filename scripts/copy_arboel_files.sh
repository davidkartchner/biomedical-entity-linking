#!/bin/bash

for DATASET in "ncbi_disease" "bc5cdr" "nlmchem" "medmentions_st21pv" "nlm_gene" "gnormplus" "medmentions_full" 
do
    cp arboel/models/trained/${DATASET}_mst/eval_results.json results/arboel_biencoder/$DATASET.json
    cp arboel/models/trained/${DATASET}/crossencoder/eval/arbo/results.json results/arboel_crossencoder/$DATASET.json
done