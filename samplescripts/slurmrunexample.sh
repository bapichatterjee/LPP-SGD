#!/bin/bash
#This is a slurm managed runscript to run on 8 GPUs, one on each node

#!/bin/bash
#SBATCH --nodes=2               # number of nodes
#SBATCH --ntasks-per-node=1     # processes per node
#SBATCH --cpus-per-task=24       # number of CPU cores per process
#SBATCH --gres=gpu:1            # GPUs per node
#SBATCH --hint=compute_bound
#SBATCH --hint=multithread
#SBATCH --partition=gpu         # put the job into the gpu partition/queue
#SBATCH --output=job_%j.out     # file name for stdout/stderr
#SBATCH --error=job_%j.err
#SBATCH --mem=200G              # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=20:00:00         # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=my_test        # job name (default is the name of this file)

# ml fosscuda/2019a #Load the modules for mpirun, etc.

PYTHON=~/miniconda3/bin/python
PROGRAM=~/directory-master/LPP-SGD/main.py #Assuming that the program is unzipped in the /home/$USER folder
DATADIR=~/directory-master/LPP-SGD/data
MASTER=$(/bin/hostname -s)

mkdir -p test_mnist_Slurm
cd test_mnist_Slurm

# Print job information
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Job ID: $SLURM_JOB_ID"
echo "Starting job at: $(date)"
srun \
	$PYTHON $PROGRAM --data-dir $DATADIR \
	--dataset mnist --num-classes 10 --momentum 0.9 --model small \
	--cuda --train_processing_bs 128 --test_processing_bs 128 --lr 0.4 \
	--baseline_lr 0.1 --weight-decay 0.0005 --seed 0 --pm \
	--nesterov --workers 8 --num-threads 8 --test-freq 1 --partition \
	--scheduler-type mstep --lrmilestone 30 60 80 \
	--training-type MBSGD --numnodes $SLURM_JOB_NUM_NODES \
	--bs_multiple 1 --test_bs_multiple 1 --epochs 90 \
	--warm_up_epochs 5 --dist-url tcp://$HOSTNAME:23456 --storeresults

# Change the current directory to the parent directory, one level up from the current location
# This command is often used when the script needs to operate from the parent directory's context
cd ..
