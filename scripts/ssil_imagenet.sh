#!/bin/bash

if [ "$1" != "" ]; then
    echo "Running on gpu: $1"
else
    echo "No gpu has been assigned."
fi

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"
SRC_DIR="$PROJECT_DIR/src"
echo "Project dir: $PROJECT_DIR"
echo "Sources dir: $SRC_DIR"

RESULTS_DIR='/data2/users/btwardow/ss-il'
echo "Results dir: $RESULTS_DIR"
SEED=1
M=20000

PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name M:${M}_seed:${SEED} \
        --datasets imagenet_256 --num-tasks 10 --network resnet18 --seed $SEED \
        --nepochs 100 --batch-size 128 --results-path $RESULTS_DIR \
        --gridsearch-tasks 0 \
        --approach ssil --gpu $1 \
        --lr 0.1 --lr-factor 3 --lr-patience 10 --momentum 0.9 --weight-decay 0.0002 \
        --num-exemplars $M --exemplar-selection herding --replay-batch-size 32