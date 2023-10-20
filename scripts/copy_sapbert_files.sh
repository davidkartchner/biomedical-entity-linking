#!/bin/bash

for DATASET in "ncbi_disease" "bc5cdr" "nlmchem" "medmentions_st21pv" "nlm_gene" "gnormplus" "medmentions_full" 
do
    cp sapbert/output/$DATASET/3.json results/sapbert/with_abbr_res/$DATASET.json
    cp sapbert/output/$DATASET/2.json results/sapbert/no_abbr_res/$DATASET.json
done