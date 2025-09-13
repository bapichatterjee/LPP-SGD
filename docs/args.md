"""
Command Line Argument Handling Module

This module manages all command-line arguments and configuration for the training process.
It provides comprehensive argument parsing and validation for various aspects of training.

Key Components:

1. Directory Management:
    - prepare_dir: Creates and manages directory structure
    - prepare_run_files: Sets up tensorboard and flask visualization

2. Argument Categories:
    - System Arguments: Data directories, host settings
    - Distribution Arguments: Backend, URL, node configuration
    - Learning Rate Arguments: Scheduler, decay, warmup
    - Averaging Arguments: Frequency, methods
    - Common Arguments: Model, dataset, batch size, etc.

Functions:
    prepare_args: Main function for argument processing
        - Filters and validates arguments
        - Sets up CUDA environment
        - Configures distributed training

    add_*_args: Family of functions adding specific argument groups
        - add_sys_args: System configuration
        - add_dist_args: Distribution settings
        - add_lr_args: Learning rate configuration
        - add_averaging_args: Averaging parameters
        - add_common_args: General training parameters

Usage:
    Called by main.py to set up all training parameters
    Provides validated arguments to training implementations

Configuration:
    Supports multiple training types:
    - LPPSGD: Layer-wise Periodic Parallel SGD
    - LAPSGD: Layer-wise Asynchronous Parallel SGD
    - MBSGD: Mini-Batch SGD
    - PLSGD: Periodic Local SGD
"""