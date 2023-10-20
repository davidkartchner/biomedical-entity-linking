#!/bin/bash
DEVICE=$1

# With abbreviation resolution
for DATASET in  "ncbi_disease" "bc5cdr" "nlmchem" "medmentions_st21pv" "nlm_gene" "gnormplus" "medmentions_full"
do
    CUDA_VISIBLE_DEVICES=$DEVICE python run_entity_linking.py test_data.dataset_name=$DATASET
done
# CUDA_VISIBLE_DEVICES=0 python run_entity_linking.py test_data.dataset_name=bc5cdr
# CUDA_VISIBLE_DEVICES=0 python run_entity_linking.py test_data.dataset_name=ncbi_disease
# CUDA_VISIBLE_DEVICES=0 python run_entity_linking.py test_data.dataset_name=medmentions_full
# CUDA_VISIBLE_DEVICES=0 python run_entity_linking.py test_data.dataset_name=nlm_gene
# CUDA_VISIBLE_DEVICES=0 python run_entity_linking.py test_data.dataset_name=nlmchem
# CUDA_VISIBLE_DEVICES=0 python run_entity_linking.py test_data.dataset_name=gnormplus
