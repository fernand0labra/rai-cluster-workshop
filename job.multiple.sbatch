#!/bin/bash

#SBATCH --account NAISS2024-22-1107     # The name of the account you are running in, mandatory.
#SBATCH --job-name multi_node_test      # Give a sensible name for the job
#SBATCH --time=00:15:00                 # Request runtime for the job (HHH:MM:SS) where 168 hours is the maximum

#SBATCH --ntasks 4                      # Number of workers (preferable to be same as number of GPUs i.e. 'nodes' x 'gpus-per-node')
#SBATCH --nodes 4                       # Number of nodes
#SBATCH --gpus-per-node=v100:1          # Number of GPU cards needed. Here asking for 1 V100 cards

#SBATCH --error=job.multiple.err        # Set the names for the error and output files 
#SBATCH --output=job.multiple.out


## MODULE SYSTEM VERSION

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1        # Load modules
mpirun -np 4 bash src/deploy.sh                         # Spawn processes according to SBATCH specification


## CONTAINER VERSION

mpirun -np 4 \
    apptainer exec --nv apptainer/apptainer.torch.sif \
        bash src/deploy.sh