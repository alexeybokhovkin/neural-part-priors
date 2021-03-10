import os
import sys
import json
import pickle
import gc

import torch
from tqdm import tqdm
import numpy as np

from src.utils.config import load_config
from src.lightning_models.gnn_scannet_contrastive import GNNPartnetLightning
from src.datasets.partnet import generate_scannet_allshapes_contrastive_datasets


def main(args):
    config = load_config(args)
    config_dict = config.__dict__

    datasets = generate_scannet_allshapes_contrastive_datasets(**config_dict)

    PREDICTED_TREES_SAVE_DIR = os.path.join('/'.join(config.resume_from_checkpoint.split('/')[:-2]), config.tto_savedir)
    os.makedirs(PREDICTED_TREES_SAVE_DIR, exist_ok=True)

    for i in tqdm(range(len(datasets['val']))[:50]):

        device = torch.device('cuda:0')
        model = GNNPartnetLightning(config)
        pretrained_model = model.load_from_checkpoint(checkpoint_path=config.resume_from_checkpoint, **config_dict)
        pretrained_model.to(device)
        pretrained_model.train()

        batch = list(datasets['val'][i])
        tokens = batch[6]
        partnet_id = tokens[0]
        scan_id = '_'.join([tokens[1], tokens[2]])
        instance_id = int(tokens[3])

        scan_geo = batch[0].to(device)[None, ...]
        batch[0] = (scan_geo,)
        shape = batch[2][0].to(device)
        batch[2] = (shape,)
        batch[3] = (batch[3],)
        batch[5] = (batch[5],)
        output = pretrained_model.tto_root(batch, config.tto_iterations)

        SAVE_LOCAL_DIR = os.path.join(PREDICTED_TREES_SAVE_DIR, '_'.join(tokens))
        os.makedirs(SAVE_LOCAL_DIR, exist_ok=True)
        predicted_trees = output[0]
        all_losses_detached = output[1]
        all_child_feats = output[2]

        losses_aggregated = {}
        for losses in all_losses_detached:
            for key in losses:
                if key not in losses_aggregated:
                    losses_aggregated[key] = [losses[key]]
                else:
                    losses_aggregated[key] += [losses[key]]
        for j, pd_tree in enumerate(predicted_trees):
            flag = False
            if j < 20:
                flag = True
            elif 20 <= j < 100 and j % 5 == 0:
                flag = True
            elif j >= 100 and j % 20 == 0:
                flag = True
            if flag:
                with open(os.path.join(SAVE_LOCAL_DIR, f'tree_{j}.tr'), 'wb') as fout:
                    pickle.dump(pd_tree, fout)
                with open(os.path.join(SAVE_LOCAL_DIR, f'children_feats_{j}'), 'wb') as fout:
                    np.save(fout, all_child_feats[j])
        with open(os.path.join(SAVE_LOCAL_DIR, 'losses.json'), 'w') as fout:
            json.dump(losses_aggregated, fout)

        gc.collect()


if __name__ == '__main__':
    main(sys.argv[1:])
