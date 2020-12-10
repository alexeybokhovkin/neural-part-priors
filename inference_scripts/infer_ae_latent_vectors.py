import os
import sys
import argparse
import json
from tqdm import tqdm
import pickle

import numpy as np
import torch
from torch.nn import MaxPool3d

sys.path.append('..')
from src.datasets.partnet import VoxelPartnetAllShapesDataset
from src.lightning_models.ae import AELightning

LOG_DIR = '../logs/AEPartnet'


def main(args: argparse.Namespace) -> None:
    experiment_dir = os.path.join(LOG_DIR, args.exp)
    with open(os.path.join(experiment_dir, 'config.json'), 'r') as fin:
        config = json.load(fin)
    device = torch.device('cuda:0')

    full_dataset = VoxelPartnetAllShapesDataset(config['data'], config['dataset'],
                                                config['partnet_to_dirs_path'], 'full.txt')

    model = AELightning(config, eval_mode=True)
    pretrained_model = model.load_from_checkpoint(
        checkpoint_path=os.path.join(experiment_dir, 'checkpoints', args.ckpt)
    )
    pretrained_model.to(device)
    pretrained_model.eval()
    pretrained_model.freeze()

    ps = args.poolsize
    pool_layer = MaxPool3d((ps, ps, ps))

    latents = {}
    for i in tqdm(range(len(full_dataset))):
        batch = list(full_dataset[i])
        partnet_geos = batch[0].to(device)[None, ...]
        batch[0] = partnet_geos
        partnet_id = batch[2]

        output = pretrained_model.infer(batch)

        latent_vector = output[0]
        latent_vector = pool_layer(latent_vector)
        latent_vector = np.squeeze(latent_vector.cpu().numpy())
        if i == 0:
            print(latent_vector)
        latents[partnet_id] = latent_vector

    save_path = os.path.join(args.savedir, args.exp, '.'.join(args.ckpt.split('.')[:-1]))
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'latents.pkl'), 'wb') as fout:
        pickle.dump(latents, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Infer and save latent codes for partnet shapes")
    parser.add_argument("--exp", type=str, help="which exp to infer")
    parser.add_argument("--ckpt", type=str, help="which ckpt to use")
    parser.add_argument("--poolsize", type=int, help="last pooling size")
    parser.add_argument("--savedir", type=str, help="where to save latent vectors")

    args = parser.parse_args()

    main(args)