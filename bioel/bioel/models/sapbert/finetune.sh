#!/bin/bash

#for DATASET in "ncbi_disease" "bc5cdr" "nlmchem" "medmentions_st21pv" "medmentions_full" "nlm_gene" "gnormplus"
for DATASET in "ncbi_disease"
do 
    MODEL_DIR="/home/pbathala3/entity_linking/biomedical-entity-linking/bioel/bioel/models/sapbert/pretrained_model"
    TRAIN_DIR="/home/pbathala3/entity_linking/biomedical-entity-linking/bioel/bioel/models/sapbert/data/data_bis/${DATASET}_aliases.txt"
    OUTPUT_DIR="/home/pbathala3/entity_linking/biomedical-entity-linking/bioel/bioel/models/sapbert/finetuned_models_new_1/${DATASET}"
    MAX_LENGTH=25
    BATCH_SIZE=256
    NUM_EPOCHS=1
    LEARNING_RATE=2e-5
    CHECKPOINT_STEP=999999
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --model_dir $MODEL_DIR \
        --train_dir $TRAIN_DIR \
        --output_dir $OUTPUT_DIR \
        --use_cuda \
        --epoch $NUM_EPOCHS \
        --train_batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --max_length $MAX_LENGTH \
        --checkpoint_step $CHECKPOINT_STEP \
        --amp \
        --pairwise \
        --random_seed 33 \
        --loss ms_loss \
        --use_miner \
        --type_of_triplets "all" \
        --miner_margin 0.2 \
        --agg_mode "cls" \
        --project "SAPBERT" \
        --mode "finetune"
done
