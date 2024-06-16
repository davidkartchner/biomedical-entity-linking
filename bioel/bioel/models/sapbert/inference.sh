#!/bin/bash

# With abbreviation resolution
#for DATASET in  "ncbi_disease" "bc5cdr" "nlmchem" "medmentions_full" "medmentions_st21pv"
for DATASET in "nlm_gene" "gnormplus"
do
	MODEL_DIR="/home/pbathala3/entity_linking/biomedical-entity-linking/bioel/bioel/models/sapbert/finetuned_models_new_1/${DATASET}"
	#MODEL_DIR="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
	DICT_PATH="/home/pbathala3/entity_linking/biomedical-entity-linking/bioel/bioel/models/sapbert/data/data_bis/${DATASET}_aliases.txt"
    CUDA_VISIBLE_DEVICES=1 python3 inference.py \
	--model_dir $MODEL_DIR \
	--dictionary_path $DICT_PATH \
	--dataset_name $DATASET \
	--output_dir ./output/with_abbreviations_base \
	--dict_cache_path cached_dicts_new/${DATASET}_dict.pt \
	--use_cuda \
	--max_length 25 \
	--batch_size 32 \
	--abbreviations_path /home/pbathala3/entity_linking/biomedical-entity-linking/bioel/bioel/utils/solve_abbreviation/abbreviations.json \
	# --debug

done

# Without abbreviation resolution
# for DATASET in "ncbi_disease" "bc5cdr" "nlmchem" "nlm_gene" "gnormplus" "medmentions_st21pv" "medmentions_full" 
# # for DATASET in "medmentions_full"
# do
# 	MODEL_DIR="/home/pbathala3/entity_linking/biomedical-entity-linking/bioel/bioel/models/sapbert/finetuned_models/${DATASET}"
# 	DICT_PATH="/home/pbathala3/entity_linking/biomedical-entity-linking/bioel/bioel/models/sapbert/data/data_bis/${DATASET}_aliases.txt"
#     CUDA_VISIBLE_DEVICES=3 python3 inference.py \
# 	--model_dir $MODEL_DIR \
# 	--dictionary_path $DICT_PATH \
# 	--dataset_name $DATASET \
# 	--output_dir ./output/without_abbreviations \
# 	--dict_cache_path cached_dicts/${DATASET}_dict.pt \
# 	--use_cuda \
# 	--max_length 25 \
# 	--batch_size 32 \

# done