#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexey.bokhovkin@skoltech.ru
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4

export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate torch

/rhome/abokhovkin/miniconda3/envs/torch/bin/python compute_shapenetv1_to_partnet_transformations.py
