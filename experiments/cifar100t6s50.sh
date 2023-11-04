#!/bin/bash

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=6
nc_first_task=50
num_epochs=200
dataset=cifar100_icarl
network=resnet32
tag=cifar100t${num_tasks}s${nc_first_task}

lamb=5
lamb_mc=0.5
beta=5
gamma=1e-3

for wu_nepochs in 0 200; do
  for seed in 0 1 2; do
    ./experiments/lwf.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb} ${wu_nepochs} &
  done
  wait

  for seed in 0 1 2; do
    ./experiments/lwf_ta.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb} ${wu_nepochs} &
  done
  wait

  for seed in 0 1 2; do
    ./experiments/lwf_mc.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb_mc} ${wu_nepochs} &
  done
  wait

  for seed in 0 1 2; do
    ./experiments/lwf_mc_ta.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb_mc} ${wu_nepochs} &
  done
  wait

done
