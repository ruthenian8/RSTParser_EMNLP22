#!/bin/bash
set -x

# OPTIONS
DATA_DIR=./data/PCC
SAVE_DIR=./models/PCC
PARSER_TYPE=shift_reduce_v2
BERT_TYPE=microsoft/mdeberta-v3-base
LR=1e-5
NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0
for SEED in 0 1 2; do
    MODEL_NAME="PCC_SOFT"
    VERSION=$SEED

    if [ -d $SAVE_DIR/$MODEL_NAME/version_$SEED/checkpoints ]; then
        # checkpoints exist, skip TRAINING
        :
    else
        # RUN TRAINING
            python src/train.py \
                --corpus PCC \
                --train-file merged_train.json \
                --valid-file pcc_test.json \
                --test-file pcc_test.json \
                --model-type $PARSER_TYPE \
                --bert-model-name $BERT_TYPE \
                --batch-unit-type span_fast \
                --batch-size 25 \
                --accumulate-grad-batches 1 \
                --num-workers 0 \
                --disable-lr-schedule \
                --lr $LR \
                --num-gpus $NUM_GPUS \
                --data-dir $DATA_DIR \
                --save-dir $SAVE_DIR \
                --model-name $MODEL_NAME \
                --model-version $SEED \
                --seed $SEED \
                --use-soft-labels
    fi


    # RUN TEST
    if [ -d $SAVE_DIR/$MODEL_NAME/version_$SEED/checkpoints ]; then
        python src/test.py \
            --corpus PCC \
            --train-file merged_train.json \
            --valid-file pcc_test.json \
            --test-file pcc_test.json \
            --metrics OriginalParseval \
            --num-workers 0 \
            --data-dir $DATA_DIR \
            --ckpt-dir $SAVE_DIR/$MODEL_NAME/version_$SEED/checkpoints \
            --save-dir $SAVE_DIR/$MODEL_NAME/version_$SEED/trees
    else
        # No exists checkpoint dir
        :
    fi
done
