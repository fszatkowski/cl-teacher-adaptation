#!/bin/bash

set -e

gpu=$1
seed=$2
tag=$3
dataset=$4
classes_per_dataset=$5
network=$6
num_epochs=$7
lamb=$8
wu_epochs=${9:-0}
wu_wd=${10:-0.0}

clip=1.0

if [ ${classes_per_dataset} -gt 0 ]; then
  if [ ${wu_epochs} -gt 0 ]; then
    exp_name="${tag}:lamb_${lamb}:ta:tw:wu:wu_wd_${wu_wd}"
    result_path="results/${tag}/lwf_ta_tw_wu_${lamb}_${wu_wd}_${seed}"
    python3 src/main_incremental.py \
      --exp-name ${exp_name} \
      --gpu ${gpu} \
      --datasets ${dataset} \
      --num-tasks 1 \
      --max-classes-per-dataset ${classes_per_dataset} \
      --network ${network} \
      --pretrained \
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
      --taskwise-kd \
      --ta \
      --lamb ${lamb} \
      --wu-nepochs ${wu_epochs} \
      --wu-lr 0.1 \
      --wu-fix-bn \
      --wu-scheduler onecycle \
      --wu-patience 50 \
      --wu-wd ${wu_wd}
  else
    exp_name="${tag}:lamb_${lamb}:ta:tw"
    result_path="results/${tag}/lwf_ta_tw_${lamb}_${seed}"
    python3 src/main_incremental.py \
      --exp-name ${exp_name} \
      --gpu ${gpu} \
      --datasets ${dataset} \
      --num-tasks 1 \
      --max-classes-per-dataset ${classes_per_dataset} \
      --network ${network} \
      --pretrained \
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
      --taskwise-kd \
      --ta \
      --lamb ${lamb}
  fi
else
  if [ ${wu_epochs} -gt 0 ]; then
    exp_name="${tag}:lamb_${lamb}:ta:tw:wu:wu_wd_${wu_wd}}"
    result_path="results/${tag}/lwf_ta_tw_wu_${lamb}_${wu_wd}_${seed}"
    python3 src/main_incremental.py \
      --exp-name ${exp_name} \
      --gpu ${gpu} \
      --datasets ${dataset} \
      --num-tasks 1 \
      --network ${network} \
      --pretrained \
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
      --taskwise-kd \
      --ta \
      --lamb ${lamb} \
      --wu-nepochs ${wu_epochs} \
      --wu-lr 0.1 \
      --wu-fix-bn \
      --wu-scheduler onecycle \
      --wu-patience 50 \
      --wu-wd ${wu_wd}
  else
    exp_name="${tag}:lamb_${lamb}:ta:tw"
    result_path="results/${tag}/lwf_ta_tw_${lamb}_${seed}"
    python3 src/main_incremental.py \
      --exp-name ${exp_name} \
      --gpu ${gpu} \
      --datasets ${dataset} \
      --num-tasks 1 \
      --network ${network} \
      --pretrained \
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
      --taskwise-kd \
      --ta \
      --lamb ${lamb}
  fi
fi
