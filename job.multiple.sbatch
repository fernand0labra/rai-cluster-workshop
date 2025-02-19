#!/bin/bash

#SBATCH --account NAISS2024-5-442       # The name of the account you are running in, mandatory.
#SBATCH --job-name multi_node_test      # Give a sensible name for the job
#SBATCH --time=00:05:00                 # Request runtime for the job (HHH:MM:SS) where 168 hours is the maximum

#SBATCH --ntasks 2                      # Number of workers (preferable to be same as number of GPUs i.e. 'nodes' x 'gpus-per-node')
#SBATCH --nodes 2                       # Number of nodes
#SBATCH --gpus-per-node=V100:4          # Number of GPU cards needed. Here asking for 4 V100 cards

#SBATCH --error=job.multiple.err        # Set the names for the error and output files 
#SBATCH --output=job.multiple.out


## MODULE SYSTEM VERSION
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1    # Load modules
# mpirun -np 2 bash job.deploy.sh src/standalone.py   # Spawn processes according to MPI specification


## CONTAINER VERSION
module load OpenMPI/4.1.5-GCC-12.3.0
mpirun -np 2 \
   apptainer exec --nv apptainer/apptainer.torch.sif \
       bash job.deploy.sh