# LPP-SGD Documentation

This documentation provides detailed information about each component of the LPP-SGD (Layer-wise Periodic Parallel SGD) project.

## Core Files

### main.py
The main entry point for the training process. It:
- Parses command line arguments through `prepare_experiment()`
- Routes to different training methods based on `training_type`:
  - LPPSGD (Layer-wise Periodic Parallel SGD)
  - LAPSGD (Layer-wise Asynchronous Parallel SGD)
  - MBSGD (Mini-Batch SGD)
  - PLSGD (Periodic Local SGD)

### train/MBSGD.py
Mini-Batch SGD implementation that:
- Implements distributed training using PyTorch's DistributedDataParallel
- Supports LARS (Layer-wise Adaptive Rate Scaling)
- Includes custom optimizer with momentum and nesterov options
- Handles learning rate scheduling (Cosine and MultiStep)
- Provides real-time training metrics and progress visualization

### utilities/args.py
Command line argument handling that:
- Configures system parameters (data directories, host settings)
- Sets up distributed training parameters
- Manages learning rate scheduling options
- Handles averaging and periodic synchronization parameters
- Prepares directory structure for results and snapshots

## Sample Scripts

### samplescripts/slurmrun.sh
A SLURM job script for multi-node training that:
- Configures 8 nodes with 1 GPU per node
- Sets memory and time limits
- Runs different training types sequentially:
  - LPPSGD with cosine scheduling
  - LAPSGD with cosine scheduling
  - MBSGD with multi-step scheduling
  - PLSGD with multi-step scheduling

### samplescripts/singlenoderun.sh
Single node training script that:
- Runs on 2 GPUs on a single machine
- Configures MPI for local parallel processing
- Includes examples for LAPSGD and LPPSGD training
- Contains commented examples for PLSGD and MBSGD

### samplescripts/slurmrunexample.sh
An example SLURM script that:
- Demonstrates basic SLURM configuration
- Shows how to run on 2 nodes
- Includes example for MNIST dataset training
- Provides template for custom configurations

## Project Structure

```
.
├── dataloaders/       # Data loading and preprocessing
├── models/           # Neural network model implementations
├── pyhessian/        # Hessian computation utilities
├── train/           # Training implementations
├── utilities/       # Helper functions and utilities
└── samplescripts/   # Example running scripts
```

## Running the Code

1. Single Node Training:
```bash
bash samplescripts/singlenoderun.sh
```

2. Multi-Node Training with SLURM:
```bash
sbatch samplescripts/slurmrun.sh
```

3. Example MNIST Training:
```bash
sbatch samplescripts/slurmrunexample.sh
```