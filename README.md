# HPC Cluster Workshop

This repository contains code and job scripts with examples on **(1)** single-machine multi-gpu and **(2)** cluster deployment, showcasing the necessary code changes in Pytorch and Tensorflow frameworks for distributed NN training. Moreover, the repository contains guidelines for the deployment of SLURM job scripts.

More examples and information on distributed execution can be found on: https://github.com/c3se/alvis-intro

## Table of Contents
1. [Introduction to HPC Clusters](#introduction-to-hpc-clusters)
   * [About HPC Clusters](#about-hpc-clusters)
   * [HPC2N & Alvis](#hpc2n--alvis)
   * [Cluster Resource Allocation](#cluster-resource-allocation)
2. [Code Migration & Dependencies](#code-migration--dependencies)
   * [About Proprietary Code](#about-proprietary-code)
   * [Modules System](#modules-system)
   * [Singularity/Apptainer](#singularityapptainer)
3. [Multi-GPU/Multi-Node Training](#multi-gpumulti-node-training)
   * [SLURM Workload Manager](#slurm-workload-manager)
   * [Multi-GPU Code Adaptation](#multi-gpu-code-adaptation)
   * [Multi-Node Code Adaptation](#multi-node-code-adaptation)
4. [Visual Applications with ComputeNode Desktop OnDemand](#visual-applications-with-computenode-desktop-ondemand)
   * [Desktop OnDemand Platform](#desktop-ondemand-platform)
   * [Visual Applications](#visual-applications)
5. [References](#references)



## Introduction to HPC Clusters

### About HPC Clusters

A computer cluster consists of multiple computers (nodes) connected via high-speed networks and working together as a single system. Clusters are cost-effective alternatives to large single computers, offering improved performance and availability.

A node is an individual computer within a cluster, typically containing one or more CPUs (with multiple cores) and possibly GPUs. While memory is shared between cores within the same CPU, it is not shared across different nodes.

Jobs on a cluster are managed through a batch system. Users log in to a "login node" and submit job scripts, which specify requirements like the number of nodes, CPUs, GPUs, memory, runtime, and input data. These scripts enable non-interactive job execution, ideal for resource-intensive tasks that run without user interaction.

### HPC2N & Alvis

There are two clusters available to the RAI group.

1. High Performance Computing Center North (HPC2N) is a national center for Scientific and Parallel Computing. This collaboration and coordination between universities and research institutes form a competence network for high performance computing (HPC), scientific visualization, and virtual reality (VR) in Northern Sweden.

2. The Alvis cluster is a national NAISS resource dedicated for Artificial Intelligence and Machine Learning research. The system is built around Graphical Processing Units (GPUs) accelerator cards, and consists of several types of compute nodes with multiple NVIDIA GPUs.

### Cluster Resource Allocation

Batch or scheduling systems are essential for managing multi-user jobs on clusters or supercomputers. These systems track available resources, enforce usage policies, and schedule jobs efficiently by organizing them into priority queues. Jobs are submitted using job scripts, which specify resource requirements (e.g., nodes, cores, GPUs, memory) and include commands to execute tasks. Outputs and error logs are generated after job completion.

**salloc** is a scheduler command used to allocate a job, which is a set of resources (nodes), possibly with some set of constraints (e.g. number of processors per node). If no command is specified, then by default salloc starts the user’s default shell on the same machine.

```
# Allocate 1 node with 4 workers for 1 hour and 30 minutes
salloc  --account <your project> --nodes=1 --ntasks-per-node=4 --time=1:30:00

# You must use srun to run your job on the allocated resources
srun --ntasks 2 python program.py <ARGS>
```



## Code Migration & Dependencies

### About Proprietary Code
In order for your personal code to be run in a compute node, it is necessary to allocate the computing resources as well as locate the terminal in the same folder as your main file. Dependencies can be then loaded through the modules system by using the job allocation script format or by bundling them in a container and running the main file.

If modules that are not installed in the system or need to be modified have to be included, then the modules need to be located in the same folder as the main file. However, the dependencies of the modules (e.g. requirements.txt) have to be imported from one of the previously mentioned options.

```
├── module_1
├── module_2
├── ...
└── main.py   -> import module_1 as m1; import module_2 as m2
```

ABOUT STORAGE (e.g. WinSCP)

### Modules System

In high-performance computation, a module system functions as an organized toolbox for software and tools. It enables us to easily access, load, and manage different software packages, compilers, and libraries needed for specific computing tasks. By segregating software environments, we can prevent conflicts and customize setups according to task requirements.

```
module spider MODULE
module list
```

### Singularity/Apptainer

Apptainer is a container platform. It allows you to create and run containers that package up pieces of software in a way that is portable and reproducible. You can build a container using Apptainer on your laptop, and then run it on many of the largest HPC clusters in the world, local university or company clusters, a single server, in the cloud, or on a workstation down the hall. Your container is a single file, and you don’t have to worry about how to install all the software you need on each different operating system.

```
apptainer build image.sif image.def
apptainer exec --nv image.sif COMMAND

# Include different options
```

https://github.com/c3se/containers
https://catalog.ngc.nvidia.com
https://hub.docker.com

```
EXAMPLE FILE
```

## Multi-GPU/Multi-Node Training

### SLURM Workload Manager

SLURM (Simple Linux Utility for Resource Management) is a widely used open-source job scheduling system for Linux and Unix-like environments. It plays a crucial role in managing resources and scheduling on clusters, including many supercomputers.

SLURM provides three main functions:

1. Resource Allocation: Grants users exclusive or shared access to nodes for a set duration.
2. Job Management: Offers a framework for starting, executing, and monitoring parallel job.
3. Queue Management: Manages job queues, resolving contention for resources.

Jobs and the resources associated are requested and controlled through the following commands. The job.sh file contains all the specifications for the allocation and the programs to be executed.

```
sbatch job.sh                  # Submit a job
squeue -u USERNAME -j JOBID    # Job status
scontrol show job JOBID        # Job information
scancel JOBID                  # Cancel a job
```

SLURM directives in job scripts are prefixed with **#SBATCH**, while general comments are prefixed with **#**. This system enables efficient resource utilization and streamlined job execution on high-performance computing systems.

```
''' Job Script Example file: job.sh '''

#SBATCH --account hpc2nXXXX-YYY         # The name of the account you are running in, mandatory.
#SBATCH --job-name my_job_name          # Give a sensible name for the job
#SBATCH --time=00:15:00                 # Request runtime for the job (HHH:MM:SS)

#SBATCH --error=job.%J.err              # Set the names for the error and output files 
#SBATCH --output=job.%J.out             # %J is equivalent to the specified job name

# The following directives set up two nodes with 1 V100 GPU each
# ********************************************************************************************* #
#SBATCH --ntasks 2                      # Number of workers (recommended 1 worker per GPU)
#SBATCH --nodes 2                       # Number of nodes
#SBATCH --gpus-per-node=v100:1          # Number of GPU cards needed per node
# ********************************************************************************************* #

srun python program.py <ARGS>           # Run program on allocated resources
```

GPU types include **v100**, **a40**, **a6000**, **l40s** and **h100**. When no GPU type is specified the scheduler will allocate any free GPU in the cluster. For more information read the clusters' documentation indicated in the references.

The following directive allocates a node exclusively for a job even if there is enough resources for another job.
```
#SBATCH --exclusive
```

The following directive allows the selection of the type of instance of the nodes allocated on the cluster.
```
#SBATCH --constraint=skylake  # HPC2N example
```

### Multi-GPU Code Adaptation

### Multi-Node Code Adaptation



## Visual Applications with ComputeNode Desktop OnDemand

### Desktop OnDemand Platform

There are two desktop apps "Desktop (Compute)" and "Desktop (Login)". Both will give you an interactive desktop session, the difference if it will be on a compute node where you can do some actual computations or if it is on a shared login node where you can need to refrain from heavy usage.

### Visual Applications



## References

### About HPC2N

|                           |                                                   | 
| :-                        | :-                                                |
| HP2CN Information         | https://www.hpc2n.umu.se/about                    |
| HP2CN HP2CN Documentation | https://docs.hpc2n.umu.se/tutorials/clusterguide/ |


### About Alvis


|                           |                                                                           | 
| :-                        | :-                                                                        |
| Alvis Information         | https://www.c3se.chalmers.se/about/Alvis/                                 |
| Alvis Documentation       | https://www.c3se.chalmers.se/documentation/for_users/intro-alvis/slides/  |


### SLURM Documentation (HPC2N)


|                           |                                                                           | 
| :-                        | :-                                                                        |
| Basic Commands            | https://docs.hpc2n.umu.se/documentation/batchsystem/basic_commands/       |
| Basic Examples            | https://docs.hpc2n.umu.se/documentation/batchsystem/basic_examples/       |
| Submit File Design        | https://docs.hpc2n.umu.se/documentation/batchsystem/submit_file_design/   |
| Job Submission            | https://docs.hpc2n.umu.se/documentation/batchsystem/job_submission/       |
| Batch Scripts             | https://docs.hpc2n.umu.se/documentation/batchsystem/batch_scripts/        |
