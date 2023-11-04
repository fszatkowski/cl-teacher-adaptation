#!/bin/bash

set -e

gpu=$1
seed=$2
tag=$3
dataset=$4
num_tasks=$5
nc_first_task=$6
network=$7
num_epochs=$8
lamb=$9
wu_epochs=${10:-0}

if [ "${dataset}" = "imagenet_subset_kaggle" ]; then
  clip=1.0
else
  clip=100.0
fi

if [ ${wu_epochs} -gt 0 ]; then
  exp_name="${tag}:lamb_${lamb}:ta:wu"
  result_path="results/${tag}/lwf_ta_wu_${lamb}_${seed}"
  python3 src/main_incremental.py \
    --exp-name ${exp_name} \
    --gpu ${gpu} \
    --datasets ${dataset} \
    --num-tasks ${num_tasks} \
    --nc-first-task ${nc_first_task} \
    --network ${network} \
    --use-test-as-val \
    --lr 0.1 \
    --clipping ${clip} \
    --nepochs ${num_epochs} \
    --batch-size 128 \
    --seed ${seed} \
    --log disk wandb \
    --cache-first-task-model \
    --results-path ${result_path} \
    --tags ${tag} \
    --approach lwf \
    --ta \
    --lamb ${lamb} \
    --wu-nepochs ${wu_epochs} \
    --wu-lr 0.1 \
    --wu-fix-bn \
    --wu-scheduler onecycle \
    --wu-patience 50
else
  exp_name="${tag}:lamb_${lamb}:ta"
  result_path="results/${tag}/lwf_ta_${lamb}_${seed}"
  python3 src/main_incremental.py \
    --exp-name ${exp_name} \
    --gpu ${gpu} \
    --datasets ${dataset} \
    --num-tasks ${num_tasks} \
    --nc-first-task ${nc_first_task} \
    --network ${network} \
    --use-test-as-val \
    --lr 0.1 \
    --clipping ${clip} \
    --nepochs ${num_epochs} \
    --batch-size 128 \
    --seed ${seed} \
    --log disk wandb \
    --cache-first-task-model \
    --results-path ${result_path} \
    --tags ${tag} \
    --approach lwf \
    --ta \
    --lamb ${lamb}
fi
