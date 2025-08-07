#!/bin/bash
#SBATCH --job-name=simple
#SBATCH --partition=all6000
#SBATCH --account=all6000users
#SBATCH -t 165:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=./jobs/slurm_output_%A.out
#SBATCH --exclude=ivi-cn005,ivi-cn009,ivi-cn010,ivi-cn011,ivi-cn012,ivi-cn001
#SBATCH --export=ALL,WANDB_API_KEY=dfcd2574507b9ebe69ca13ab6f6925d864e82ee0

# Activate Conda
eval "$(conda shell.bash hook)"
conda activate prune_llm

# Create jobs directory if needed
mkdir -p ./jobs

nvidia-smi

echo "FlexViT Experiments"
echo | date
echo "Node name: $(hostname)"
echo -n memory=; ulimit -m
echo -n nproc=; nproc

srun python3 run_experiment.py run flexvit,imagenet
# srun python calc_switch_time.py

echo "Job Complete"
echo | date