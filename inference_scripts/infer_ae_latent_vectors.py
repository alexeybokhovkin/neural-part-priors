import os
import argparse
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

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
    dataloader = DataLoader(full_dataset, batch_size=1,
                            shuffle=False, num_workers=8, drop_last=False)

    model = AELightning(config)
    pretrained_model = model.load_from_checkpoint(
        checkpoint_path=os.path.join(experiment_dir, 'checkpoints', args.ckpt)
    )
    pretrained_model.to(device)
    pretrained_model.eval()
    pretrained_model.freeze()

    for i in tqdm(range(len(full_dataset))):
        batch = list(full_dataset[i])
        partnet_geos = batch[0].to(device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Infer and save latent codes for partnet shapes")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument("--exp", type=str, help="which exp to infer")
    parser.add_argument("--ckpt", type=str, help="which ckpt to use")
    parser.add_argument("__savedir", type=str, help="where to save latent vectors")

    args = parser.parse_args()

    main(args)