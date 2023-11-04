#!/bin/bash

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=26
nc_first_task=100
num_epochs=200
dataset=tiny_scaled_imnet
network=resnet32
tag=tin_t${num_tasks}s${nc_first_task}

lamb_tw=1.0
lamb_mc=0.5
lamb=10

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

  for seed in 0 1 2; do
    ./experiments/lwf_tw.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb_tw} ${wu_nepochs} &
  done
  wait

  for seed in 0 1 2; do
    ./experiments/lwf_tw_ta.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb_tw} ${wu_nepochs} &
  done
  wait

  for seed in 0 1 2; do
    ./experiments/lwfa.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb} ${wu_nepochs} &
  done
  wait

  for seed in 0 1 2; do
    ./experiments/lwfa_ta.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb} ${wu_nepochs} &
  done
  wait
done
