#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexey.bokhovkin@skoltech.ru
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4
#SBATCH --partition=submit

export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate torch

pip uninstall -y numpy
pip install numpy
pip install numpy-quaternion
pip install numba

/rhome/abokhovkin/miniconda3/envs/torch/bin/python prepare_partial_parts_uniform.py
