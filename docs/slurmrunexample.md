#!/bin/bash
# Example SLURM Script for MNIST Training
#
# This script provides a basic template for running MNIST training on SLURM.
# It demonstrates the minimal configuration needed for distributed training.
#
# SLURM Configuration:
# - 2 nodes with 1 GPU per node
# - 24 CPU cores per process
# - 200GB memory per node
# - Maximum runtime of 20 hours
#
# Training Configuration:
# - Uses MNIST dataset
# - Small model architecture
# - MBSGD (Mini-Batch SGD) training
# - Multi-step learning rate schedule
#
# Environment Setup:
# - Requires Python environment with necessary packages
# - Assumes data directory is configured
#
# Usage:
# sbatch slurmrunexample.sh
#
# This script is intended as a starting point for customization.
# Users should modify parameters based on their specific needs.

#!/bin/bash
#SBATCH --nodes=2               # number of nodes
#SBATCH --ntasks-per-node=1     # processes per node
#SBATCH --cpus-per-task=24      # CPU cores per process
#SBATCH --gres=gpu:1           # GPUs per node
#SBATCH --hint=compute_bound
#SBATCH --hint=multithread
#SBATCH --partition=gpu        # GPU partition/queue
#SBATCH --output=job_%j.out    # stdout file
#SBATCH --error=job_%j.err     # stderr file
#SBATCH --mem=200G            # memory per node
#SBATCH --time=20:00:00       # maximum runtime
#SBATCH --job-name=my_test    # job name

# The rest of the script contains the training command
# with detailed parameter explanations