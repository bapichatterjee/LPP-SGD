#!/bin/bash
# SLURM Managed Multi-node Training Script
#
# This script configures and runs distributed training on multiple nodes using SLURM.
# It executes four different training methods sequentially on ImageNet dataset:
# 1. LPPSGD (Layer-wise Periodic Parallel SGD)
# 2. LAPSGD (Layer-wise Asynchronous Parallel SGD)
# 3. MBSGD (Mini-Batch SGD)
# 4. PLSGD (Periodic Local SGD)
#
# SLURM Configuration:
# - 8 nodes with 1 GPU per node
# - 24 CPU cores per process
# - 200GB memory per node
# - Maximum runtime of 20 hours
#
# Environment Setup:
# - Uses fosscuda/2019a modules
# - Requires Anaconda/Miniconda Python
# - Assumes data is in ~/data and ImageNet in ~/ILSVRC

#!/bin/bash
#SBATCH --nodes=8               # number of nodes
#SBATCH --ntasks-per-node=1     # processes per node
#SBATCH --cpus-per-task=24      # CPU cores per process
#SBATCH --gres=gpu:1            # GPUs per node
#SBATCH --hint=compute_bound
#SBATCH --hint=multithread
#SBATCH --partition=gpu         # GPU partition/queue
#SBATCH --output=job_%j.out     # stdout file
#SBATCH --error=job_%j.err      # stderr file
#SBATCH --mem=200G             # memory per node
#SBATCH --time=20:00:00        # maximum runtime
#SBATCH --job-name=my_test     # job name

ml fosscuda/2019a

PYTHON=~/anaconda3/bin/python
PROGRAM=~/submission_code/main.py
DATADIR=~/data
IMAGENETDIR=~/ILSVRC
MASTER=$(/bin/hostname -s)

mkdir -p rnet50Imagenet_Slurm
cd rnet50Imagenet_Slurm

# The rest of the script contains the training commands for each method
# Each section is clearly marked with comments