#!/usr/bin/bash

#SBATCH -J IterativePruning
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --gpus-per-node=1

python3 $HOME/experiments.py run $1
