#!/bin/bash

#SBATCH --account hpc2nXXXX-YYY         # The name of the account you are running in, mandatory.
#SBATCH --job-name multi_gpu_test       # Give a sensible name for the job
#SBATCH --time=00:15:00                 # Request runtime for the job (HHH:MM:SS) where 168 hours is the maximum

#SBATCH --ntasks 4                      # Number of workers (preferable to be same as number of GPUs i.e. 'nodes' x 'gpus-per-node')
#SBATCH --nodes 4                       # Number of nodes
#SBATCH --gpus-per-node=v100:1          # Number of GPU cards needed. Here asking for 1 V100 cards

#SBATCH --error=job.%J.err              # Set the names for the error and output files 
#SBATCH --output=job.%J.out


# Activate your virtual environment if needed
source ~/env/bin/activate

# Define SLURM environment variables
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

# Spawn processes according to SBATCH specification
srun python multi-node-torch.py
# srun python multi-node-tensor.py