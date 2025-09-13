"""
Mini-Batch SGD Training Implementation

This module implements distributed training using Mini-Batch SGD with the following features:
- Distributed training using PyTorch's DistributedDataParallel
- Custom optimizer with momentum and Nesterov momentum options
- Support for LARS (Layer-wise Adaptive Rate Scaling)
- Learning rate scheduling (Cosine and MultiStep)
- Real-time training metrics and progress visualization

Classes:
    opt: Custom optimizer implementation
        - Supports momentum and Nesterov momentum
        - Optional LARS implementation
        - Handles weight decay

Functions:
    test_train: Main training function
        - Initializes distributed training
        - Sets up model, optimizer, and scheduler
        - Manages training loop and testing

    train_epoch: Single epoch training implementation
        - Handles mini-batch processing
        - Updates learning rate
        - Tracks and reports metrics
        - Performs periodic testing

    run: Entry point function
        - Sets up environment (GPU, seeds)
        - Initializes data loaders
        - Manages training process
        - Handles results saving

Usage:
    Called from main.py when training_type is "MBSGD"
    Requires proper argument configuration via utilities/args.py
"""