#!/usr/bin/env

# Guide:
# This script supports test and ensemble model. 

export PYTHONPATH=${PYTHONPATH}:`pwd`/src/

export CUDA_VISIBLE_DEVICES=0

DATAPATH=${1}
ensemble=${2}
online_learning=${3}
online_cache=${4}
max_epochs=${5}
model_1=${6}
model_2=${7}
model_3=${8}
model_4=${9}
model_5=${10}
model_6=${11}
model_7=${12}
model_8=${13}

if [ ${ensemble} = "ensemble" ];
then
    bash run_scripts/test-offline.sh \
    ${DATAPATH} \
    ${online_learning} \
    ${online_cache} \
    ${max_epochs} \
    ${model_1} \
    ${model_2} \
    ${model_3} \
    ${model_4} \
    ${model_5} \
    ${model_6} \
    ${model_7} \
    ${model_8}

elif [ ${ensemble} = "ensemble-last" ];
then
    bash run_scripts/test-offline-last.sh ${DATAPATH}

else
    bash run_scripts/test-online.sh \
    ${DATAPATH} \
    ${online_learning} \
    ${online_cache} \
    ${max_epochs} \
    ${model_1}
fi

