#!/bin/bash
set -x

# OPTIONS
DATA_DIR=./data/PCC
SAVE_DIR=./models/PCC
PARSER_TYPE=shift_reduce_v1
BERT_TYPE=ikim-uk-essen/geberta-base
LR=1e-5
NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0
for SEED in 0 1 2; do
    MODEL_NAME=$PARSER_TYPE.$BERT_TYPE.$LR
    VERSION=$SEED

    if [ -d $SAVE_DIR/$MODEL_NAME/version_$SEED/checkpoints ]; then
        # checkpoints exist, skip TRAINING
        :
    else
        # RUN TRAINING
            python src/train.py \
                --corpus PCC \
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
                --seed $SEED
    fi


    # RUN TEST
    if [ -d $SAVE_DIR/$MODEL_NAME/version_$SEED/checkpoints ]; then
        python src/test.py \
            --num-workers 0 \
            --data-dir $DATA_DIR \
            --ckpt-dir $SAVE_DIR/$MODEL_NAME/version_$SEED/checkpoints \
            --save-dir $SAVE_DIR/$MODEL_NAME/version_$SEED/trees
    else
        # No exists checkpoint dir
        :
    fi
done