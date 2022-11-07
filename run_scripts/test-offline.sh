#!/usr/bin/env

# Guide:
# This script supports test and ensemble model. 

DATAPATH=${1}
online_learning=${2}
online_cache=${3}
max_epochs=${4}
model_1=${5}
model_2=${6}
model_3=${7}
model_4=${8}
model_5=${9}
model_6=${10}
model_7=${11}
model_8=${12}

if [ ${online_learning} = "None" ];
then
    online_learning=""
fi

if [ ${online_cache} = "None" ];
then
    online_cache=""
fi

# data options
train_data=${DATAPATH}/datasets/MUGE/lmdb/train
val_data=${DATAPATH}/datasets/MUGE/lmdb/valid # if val_data is not specified, the validation will be automatically disabled
image_data=${DATAPATH}/datasets/MUGE/lmdb/test/imgs
text_data=${DATAPATH}/datasets/MUGE/test_texts.jsonl
image_feats=${DATAPATH}/datasets/MUGE/test_imgs.img_feat.jsonl
text_feats=${DATAPATH}/datasets/MUGE/test_texts.txt_feat.jsonl

# restore options
extract_image_feats="--extract-image-feats"
extract_text_feats="--extract-text-feats"

# restore options
resume_path=${DATAPATH}/resume # or specify your customed ckpt path to resume

# output options
checkpoint_path=${DATAPATH}/experiments
report_training_batch_acc="--report-training-batch-acc"
output=${DATAPATH}/experiments/results
npy_path=${DATAPATH}/npy_save

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
#online_learning="--online-learning"
#online_cache="--online-cache"

# first model
python -u src/eval/extract_features.py \
    ${extract_image_feats} \
    ${extract_text_feats} \
    --image-data=${image_data} \
    --text-data=${text_data} \
    --img-batch-size=${batch_size} \
    --text-batch-size=${batch_size} \
    --context-length=${context_length} \
    --resume=${resume_path}/${model_1} \
    --lr=${lr} \
    --wd=${wd} \
    --max-epochs=${max_epochs} \
    --vision-model=${vision_model} \
    --text-model=${text_model} \
    --checkpoint_path=${checkpoint_path} \
    ${online_learning} \
    ${online_cache}

python -u src/eval/make_topk_predictions.py \
    --image-feats=${image_feats} \
    --text-feats=${text_feats} \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output=${output}/test_predictions_1.jsonl \
    --npy_save=${npy_path}/matrix-1.npy

# second model
python -u src/eval/extract_features.py \
    ${extract_image_feats} \
    ${extract_text_feats} \
    --image-data=${image_data} \
    --text-data=${text_data} \
    --img-batch-size=${batch_size} \
    --text-batch-size=${batch_size} \
    --context-length=${context_length} \
    --resume=${resume_path}/${model_2} \
    --lr=${lr} \
    --wd=${wd} \
    --max-epochs=${max_epochs} \
    --vision-model=${vision_model} \
    --text-model=${text_model} \
    --checkpoint_path=${checkpoint_path} \
    ${online_learning} \
    ${online_cache} 

python -u src/eval/make_topk_predictions.py \
    --image-feats=${image_feats} \
    --text-feats=${text_feats} \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output=${output}/test_predictions_2.jsonl \
    --npy_save=${npy_path}/matrix-2.npy

# third model
python -u src/eval/extract_features.py \
    ${extract_image_feats} \
    ${extract_text_feats} \
    --image-data=${image_data} \
    --text-data=${text_data} \
    --img-batch-size=${batch_size} \
    --text-batch-size=${batch_size} \
    --context-length=${context_length} \
    --resume=${resume_path}/${model_3} \
    --lr=${lr} \
    --wd=${wd} \
    --max-epochs=${max_epochs} \
    --vision-model=${vision_model} \
    --text-model=${text_model} \
    --checkpoint_path=${checkpoint_path} \
    ${online_learning} \
    ${online_cache}

python -u src/eval/make_topk_predictions.py \
    --image-feats=${image_feats} \
    --text-feats=${text_feats} \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output=${output}/test_predictions_3.jsonl \
    --npy_save=${npy_path}/matrix-3.npy

# fourth model
python -u src/eval/extract_features.py \
    ${extract_image_feats} \
    ${extract_text_feats} \
    --image-data=${image_data} \
    --text-data=${text_data} \
    --img-batch-size=${batch_size} \
    --text-batch-size=${batch_size} \
    --context-length=${context_length} \
    --resume=${resume_path}/${model_4} \
    --lr=${lr} \
    --wd=${wd} \
    --max-epochs=${max_epochs} \
    --vision-model=${vision_model} \
    --text-model=${text_model} \
    --checkpoint_path=${checkpoint_path} \
    ${online_learning} \
    ${online_cache}

python -u src/eval/make_topk_predictions.py \
    --image-feats=${image_feats} \
    --text-feats=${text_feats} \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output=${output}/test_predictions_4.jsonl \
    --npy_save=${npy_path}/matrix-4.npy

# fifth model
python -u src/eval/extract_features.py \
    ${extract_image_feats} \
    ${extract_text_feats} \
    --image-data=${image_data} \
    --text-data=${text_data} \
    --img-batch-size=${batch_size} \
    --text-batch-size=${batch_size} \
    --context-length=${context_length} \
    --resume=${resume_path}/${model_5} \
    --lr=${lr} \
    --wd=${wd} \
    --max-epochs=${max_epochs} \
    --vision-model=${vision_model} \
    --text-model=${text_model} \
    --checkpoint_path=${checkpoint_path} \
    ${online_learning} \
    ${online_cache}

python -u src/eval/make_topk_predictions.py \
    --image-feats=${image_feats} \
    --text-feats=${text_feats} \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output=${output}/test_predictions_5.jsonl \
    --npy_save=${npy_path}/matrix-5.npy

# sixth model
python -u src/eval/extract_features.py \
    ${extract_image_feats} \
    ${extract_text_feats} \
    --image-data=${image_data} \
    --text-data=${text_data} \
    --img-batch-size=${batch_size} \
    --text-batch-size=${batch_size} \
    --context-length=${context_length} \
    --resume=${resume_path}/${model_6} \
    --lr=${lr} \
    --wd=${wd} \
    --max-epochs=${max_epochs} \
    --vision-model=${vision_model} \
    --text-model=${text_model} \
    --checkpoint_path=${checkpoint_path} \
    ${online_learning} \
    ${online_cache}

python -u src/eval/make_topk_predictions.py \
    --image-feats=${image_feats} \
    --text-feats=${text_feats} \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output=${output}/test_predictions_6.jsonl \
    --npy_save=${npy_path}/matrix-6.npy

# seventh model
python -u src/eval/extract_features.py \
    ${extract_image_feats} \
    ${extract_text_feats} \
    --image-data=${image_data} \
    --text-data=${text_data} \
    --img-batch-size=${batch_size} \
    --text-batch-size=${batch_size} \
    --context-length=${context_length} \
    --resume=${resume_path}/${model_7} \
    --lr=${lr} \
    --wd=${wd} \
    --max-epochs=${max_epochs} \
    --vision-model=${vision_model} \
    --text-model=${text_model} \
    --checkpoint_path=${checkpoint_path} \
    ${online_learning} \
    ${online_cache}

python -u src/eval/make_topk_predictions.py \
    --image-feats=${image_feats} \
    --text-feats=${text_feats} \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output=${output}/test_predictions_7.jsonl \
    --npy_save=${npy_path}/matrix-7.npy

# eighth model
python -u src/eval/extract_features.py \
    ${extract_image_feats} \
    ${extract_text_feats} \
    --image-data=${image_data} \
    --text-data=${text_data} \
    --img-batch-size=${batch_size} \
    --text-batch-size=${batch_size} \
    --context-length=${context_length} \
    --resume=${resume_path}/${model_8} \
    --lr=${lr} \
    --wd=${wd} \
    --max-epochs=${max_epochs} \
    --vision-model=${vision_model} \
    --text-model=${text_model} \
    --checkpoint_path=${checkpoint_path} \
    ${online_learning} \
    ${online_cache}

python -u src/eval/make_topk_predictions.py \
    --image-feats=${image_feats} \
    --text-feats=${text_feats} \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output=${output}/test_predictions_8.jsonl \
    --npy_save=${npy_path}/matrix-8.npy

# ensemble
python -u src/eval/make_topk_predictions_ensemble.py \
    --image-feats=${image_feats} \
    --text-feats=${text_feats} \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output=${output}/test_predictions_ensemble.jsonl \
    --npy_path=${npy_path} \
    --npy_save=${npy_path}/matrix-ensemble.npy