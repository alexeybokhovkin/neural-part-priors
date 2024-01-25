#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=submit
#SBATCH --constraint="rtx_2080"

export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate torch

python3 train_deep_sdf.py -e /cluster/daidalos/abokhovkin/DeepSDF_v2/full_experiments_v3/chair_full_surface_pe --batch_split 4 --cat_name chair --pe 1 --continue latest
