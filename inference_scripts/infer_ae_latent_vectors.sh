#!/bin/bash

CUDA_VISIBLE_DEVICES=0 /rhome/abokhovkin/miniconda3/envs/torch/bin/python infer_ae_latent_vectors.py \
  --exp ae_skip_allshapes_32_filled_wd5 --ckpt epoch=192-val_loss=0.0002.ckpt --poolsize 2 --savedir "../data/latents"