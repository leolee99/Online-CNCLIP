#!/usr/bin/env

# Guide:
# This script supports test and ensemble model. 

DATAPATH=${1}

if [[ ${online_learning} = "None" ]];
then
    online_learning=""
fi

if [[ ${online_cache} = "None" ]];
then
    online_cache=""
fi

# data options
train_data=${DATAPATH}/datasets/MUGE/lmdb/train
val_data=${DATAPATH}/datasets/MUGE/lmdb/valid # if val_data is not specified, the validation will be automatically disabled
image_data=${DATAPATH}/datasets/MUGE/lmdb/test/imgs
text_data=${DATAPATH}/datasets/MUGE/test_texts.jsonl
image_feats=${DATAPATH}/datasets/MUGE/test_imgs.img_feat-last.jsonl
text_feats=${DATAPATH}/datasets/MUGE/test_texts.txt_feat-last.jsonl

# restore options
extract_image_feats="--extract-image-feats"
extract_text_feats="--extract-text-feats"

# restore options
resume_path=${DATAPATH}/resume # or specify your customed ckpt path to resume

# output options
checkpoint_path=${DATAPATH}/experiments
report_training_batch_acc="--report-training-batch-acc"
output=${DATAPATH}/experiments/results
npy_path=${DATAPATH}/npy_save_last

# online learning hyper-params
context_length=24
warmup=100
batch_size=128
valid_batch_size=128
lr=0.75e-5
wd=0.001
max_epochs=1
vision_model=ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese
use_augment="--use-augment"

# ensemble
python -u src/eval/make_topk_predictions_ensemble.py \
    --image-feats=${image_feats} \
    --text-feats=${text_feats} \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output=${output}/test_predictions_ensemble_last.jsonl \
    --npy_path=${npy_path} \
    --npy_save=${npy_path}/matrix-ensemble-last.npy