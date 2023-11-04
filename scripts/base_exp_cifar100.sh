#!/bin/bash

if [ "$1" != "" ]; then
  echo "Running on gpu: $1"
else
  echo "No gpu has been assigned."
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && cd .. && pwd)"
SRC_DIR="$PROJECT_DIR/src"
echo "Project dir: $PROJECT_DIR"
echo "Sources dir: $SRC_DIR"

APPR=$2
NC_PER_TASK=$3
SCENARIO=$4
NUM_EXEMPLARS=$5
GRIDSEARCH_TASKS=${6:-10}
NEPOCHS=${7:-200}

for SEED in 0 1 2; do
  result_path="results/cifar100_icarl/${NC_PER_TASK}/${APPR}_${SCENARIO}_${NUM_EXEMPLARS}_${SEED}"
  exp_name="cifar100_icarl_${APPR}_${NC_PER_TASK}_${SCENARIO}_${NUM_EXEMPLARS}_${SEED}"
  if [ "${SCENARIO}" = "base" ]; then
    PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name ${exp_name} \
      --datasets cifar100_icarl --nc-per-task ${NC_PER_TASK} \
      --network resnet32 --seed ${SEED} \
      --nepochs ${NEPOCHS} --batch-size 128 --results-path ${result_path} \
      --gridsearch-tasks ${GRIDSEARCH_TASKS} --gridsearch-config gridsearch_config \
      --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
      --approach ${APPR} --gpu $1
  elif [ "${SCENARIO}" = "fixd" ]; then
    PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name ${exp_name} \
      --datasets cifar100_icarl --nc-per-task ${NC_PER_TASK} \
      --network resnet32 --seed ${SEED} \
      --nepochs ${NEPOCHS} --batch-size 128 --results-path ${result_path} \
      --gridsearch-tasks ${GRIDSEARCH_TASKS} --gridsearch-config gridsearch_config \
      --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
      --approach ${APPR} --gpu $1 \
      --num-exemplars ${NUM_EXEMPLARS} --exemplar-selection herding
  elif [ "${SCENARIO}" = "grow" ]; then
    PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name ${exp_name} \
      --datasets cifar100_icarl --nc-per-task ${NC_PER_TASK} \
      --network resnet32 --seed ${SEED} \
      --nepochs ${NEPOCHS} --batch-size 128 --results-path ${result_path} \
      --gridsearch-tasks ${GRIDSEARCH_TASKS} --gridsearch-config gridsearch_config \
      --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
      --approach ${APPR} --gpu $1 \
      --num-exemplars-per-class ${NUM_EXEMPLARS} --exemplar-selection herding
  else
    echo "No scenario provided."
  fi
done