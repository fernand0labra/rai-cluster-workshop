#!/bin/bash

#SBATCH --account NAISS2024-5-442       # The name of the account you are running in, mandatory.
#SBATCH --job-name multi_gpu_test       # Give a sensible name for the job
#SBATCH --time=00:05:00                 # Request runtime for the job (HHH:MM:SS) where 168 hours is the maximum

#SBATCH --ntasks 1                      # Number of workers (preferable to be same as number of GPUs i.e. 'nodes' x 'gpus-per-node')
#SBATCH --nodes 1                       # Number of nodes
#SBATCH --gpus-per-node=V100:2          # Number of GPU cards needed. Here asking for 2 V100 cards

#SBATCH --error=job.standalone.err      # Set the names for the error and output files 
#SBATCH --output=job.standalone.out


## MODULE SYSTEM VERSION
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1                                  # Load modules
# torchrun --standalone --nnodes=1 --nproc_per_node=2 src/standalone.py             # Spawn processes according to SBATCH specification


## CONTAINER VERSION
apptainer exec --nv apptainer/apptainer.torch.sif \
    torchrun --standalone --nnodes=1 --nproc_per_node=2 src/standalone.py