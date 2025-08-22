#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=simple
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --constraint="3090|A5000|titan_rtx"         # Request Specific GPUs (modify if needed)
#SBATCH --output=./jobs/slurm_output_%A.out
#SBATCH --exclude=ivi-cn005,ivi-cn009,ivi-cn010,ivi-cn011,ivi-cn012,ivi-cn001
#SBATCH --export=ALL,WANDB_API_KEY=dfcd2574507b9ebe69ca13ab6f6925d864e82ee0

echo "FlexViT Experiments"
echo | date
echo "Node name: $(hostname)"
echo -n memory=; ulimit -m
echo -n nproc=; nproc

# Initialize Conda and activate environment
eval "$(conda shell.bash hook)"
conda activate prune_llm

# Create jobs directory if needed
mkdir -p ./jobs

nvidia-smi

# srun python latency.py
# srun python plot.py
srun python calc_switch_time.py
# srun python head_size.py

echo "Job Complete"
echo | date