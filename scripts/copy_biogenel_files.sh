#!/bin/bash

for DATASET in "ncbi_disease" "bc5cdr" "nlmchem" "medmentions_st21pv" "nlm_gene" "gnormplus" "medmentions_full" 
do
    cp biogenel/src/bigbio_/output/$DATASET.json results/biogenel/with_abbr_res/$DATASET.json
    cp biogenel/src/bigbio_/output/${DATASET}_biobart20000.json results/biobart/with_abbr_res/$DATASET.json
    # cp biogenel/src/bigbio_/output/${DATASET}_biobart20000.json results/biobart/no_abbr_res/$DATASET.json
done