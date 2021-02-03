#!/bin/bash

CUDA_VISIBLE_DEVICES=0 /rhome/abokhovkin/miniconda3/envs/torch/bin/python infer_ae_latent_vectors.py \
  --exp class_allshapes_32_filled_wd4_gnnlike --ckpt epoch=5-val_loss=0.0767.ckpt --poolsize 1 --savedir "../data/latents"