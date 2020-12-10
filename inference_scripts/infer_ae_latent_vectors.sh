#!/bin/bash

CUDA_VISIBLE_DEVICES=0 /rhome/abokhovkin/miniconda3/envs/torch/bin/python infer_ae_latent_vectors.py \
  --exp ae_class_allshapes_32_filled_v1 --ckpt epoch=34-val_loss=0.1129.ckpt --poolsize 4 --savedir "../data/latents"