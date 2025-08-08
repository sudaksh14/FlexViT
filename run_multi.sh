#!/bin/bash
#SBATCH --job-name=simple
#SBATCH -t 165:00:00
#SBATCH --partition=all           # Change based on cluster
#SBATCH --nodes=2                 # Number of nodes
#SBATCH --ntasks-per-node=4       # One task per node
#SBATCH --gres=gpu:4              # GPUs per node (adjust if needed)
#SBATCH --mem=248G                # Memory per node
#SBATCH --cpus-per-task=12        # CPU cores per task
#SBATCH --constraint="3090|A5000|titan_rtx"         # Request Specific GPUs (modify if needed)
#SBATCH --output=./jobs/slurm_output_%A.out
#SBATCH --exclude=ivi-cn005,ivi-cn009,ivi-cn010,ivi-cn011,ivi-cn012,ivi-cn001,ivi-cn002
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

# Required for Lightning to launch multi-node
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export MASTER_PORT=12345

srun python3 run_experiment.py run flexvit,imagenet

echo "Job Complete"
echo | date