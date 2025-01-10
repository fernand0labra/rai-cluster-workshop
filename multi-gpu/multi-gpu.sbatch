#!/bin/bash

#SBATCH --account hpc2nXXXX-YYY         # The name of the account you are running in, mandatory.
#SBATCH --job-name multi_gpu_test       # Give a sensible name for the job
#SBATCH --time=00:15:00                 # Request runtime for the job (HHH:MM:SS) where 168 hours is the maximum

#SBATCH --ntasks 2                      # Number of workers (preferable to be same as number of GPUs i.e. 'nodes' x 'gpus-per-node')
#SBATCH --nodes 1                       # Number of nodes
#SBATCH --gpus-per-node=v100:2          # Number of GPU cards needed. Here asking for 2 V100 cards

#SBATCH --error=job.%J.err              # Set the names for the error and output files 
#SBATCH --output=job.%J.out


# Activate your virtual environment if needed
source ~/env/bin/activate

# Spawn processes according to SBATCH specification
srun python multi-gpu-torch.py
# srun python multi-gpu-tensor.py