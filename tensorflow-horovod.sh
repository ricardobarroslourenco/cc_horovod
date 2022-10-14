#!/bin/bash
#SBATCH --nodes 2              # Request 2 nodes so all resources are in two nodes.
#SBATCH --gres=gpu:2          # Request 2 GPU "generic resources”. You will get 2 per node.

#SBATCH --tasks-per-node=2   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter if your input pipeline can handle parallel data-loading/data-transforms

#SBATCH --mem=8G
#SBATCH --time=0-00:30
#SBATCH --output=%N-%j.out


module load python/3.9 tensorflow/2.9.0 horovod/0.25.0
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
##### --- #### pip install --no-index tensorflow==2.5.0 horovod

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.

srun python tensorflow-horovod.py