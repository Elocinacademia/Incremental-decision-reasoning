#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --job-name=gpu
#SBATCH --gres=gpu
source /users/${USER}/.bashrc
source activate nnlearn
which python
python support.py
echo "Hello, World! From $HOSTNAME"
nvidia-debugdump -l
echo "Goodbye, World! From $HOSTNAME"
