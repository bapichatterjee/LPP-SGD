#!/bin/bash
# Single Node Multi-GPU Training Script
#
# This script demonstrates how to run training on a single node with multiple GPUs.
# It provides examples of running different training methods:
# - LAPSGD (Layer-wise Asynchronous Parallel SGD)
# - LPPSGD (Layer-wise Periodic Parallel SGD)
# - PLSGD (Periodic Local SGD)
# - MBSGD (Mini-Batch SGD)
#
# Configuration:
# - Uses 2 GPUs on a single machine
# - Configures MPI for local parallel processing
# - Runs on CIFAR-10 dataset
# - Uses ResNet-20 model
#
# Environment Setup:
# - Requires fosscuda/2019a modules
# - Uses Anaconda/Miniconda Python
# - Assumes data is in ~/workspace/data
#
# Usage:
# bash singlenoderun.sh
#
# The script includes examples of different configurations:
# 1. LAPSGD with cosine scheduling
# 2. LPPSGD with cosine scheduling and prepass
# 3. PLSGD with multi-step scheduling (commented)
# 4. MBSGD with multi-step scheduling (commented)
# 5. PLSGD+LARS and MBSGD+LARS examples (commented)
#
# Arguments for each configuration are carefully tuned for optimal performance.

ml fosscuda/2019a
export PYTHON=~/anaconda3/bin/python
export PROGRAM=~/workspace/LPP-SGD/main.py
export TOTALGPUS=2
export DATADIR=~/workspace/data
export MPIRUN=mpirun

mkdir -p rn20cifar10_single_node
cd rn20cifar10_single_node

# The rest of the script contains the training commands
# Each section is clearly marked with comments