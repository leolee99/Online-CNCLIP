#!/usr/bin/env

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training). 
# Please set the options below according to the comments. 
# For multi-gpu workers training, these options should be manually set for each worker. 
# After setting the options, please run the script on each worker.
# Command: bash run_scripts/muge_finetune_vit-b-16_rbt-base.sh ${DATAPATH}

# Number of GPUs per GPU worker
GPUS_PER_NODE=8 
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=localhost
# The port for communication
export MASTER_PORT=8520
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0 

export PYTHONPATH=${PYTHONPATH}:`pwd`/src/

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATAPATH=${1}
online_learning=${2}
online_cache=${3}

# data options
train_data=${DATAPATH}/datasets/MUGE/lmdb/train
val_data=${DATAPATH}/datasets/MUGE/lmdb/valid # if val_data is not specified, the validation will be automatically disabled

# restore options
resume=${DATAPATH}/pretrained_weights/clip_cn_vit-b-16.pt # or specify your customed ckpt path to resume
reset_data_offset="--reset-data-offset"
reset_optimizer="--reset-optimizer"
# reset_optimizer=""

# output options
output_base_dir=${DATAPATH}/experiments/
name=train-0
save_step_frequency=999999 # disable it
save_epoch_frequency=1
log_interval=1
report_training_batch_acc="--report-training-batch-acc"
# report_training_batch_acc=""

# training hyper-params
context_length=24
warmup=100
batch_size=128
valid_batch_size=128
lr=5e-5
wd=0.001
decay_step=20000
max_epochs=30
valid_step_interval=500
valid_epoch_interval=1
vision_model=ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese
use_augment="--use-augment"

# use_augment=""

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} \
          --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} src/training/main.py \
          --train-data=${train_data} \
          --val-data=${val_data} \
          --resume=${resume} \
          ${reset_data_offset} \
          ${reset_optimizer} \
          --logs=${output_base_dir} \
          --name=${name} \
          --save-step-frequency=${save_step_frequency} \
          --save-epoch-frequency=${save_epoch_frequency} \
          --log-interval=${log_interval} \
          ${report_training_batch_acc} \
          --context-length=${context_length} \
          --warmup=${warmup} \
          --batch-size=${batch_size} \
          --valid-batch-size=${valid_batch_size} \
          --valid-step-interval=${valid_step_interval} \
          --valid-epoch-interval=${valid_epoch_interval} \
          --lr=${lr} \
          --wd=${wd} \
          --max-epochs=${max_epochs} \
          --vision-model=${vision_model} \
          ${use_augment} \
          ${online_learning} \
          ${online_cache} \
          --text-model=${text_model}