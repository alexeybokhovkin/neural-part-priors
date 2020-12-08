import os
import random
from argparse import Namespace
import json

import numpy as np
import torch
import pytorch_lightning as pl

from ..models.ae import AE_Encoder, AE_Decoder
from ..datasets.partnet import generate_partnet_allshapes_datasets

class AELightning(pl.LightningModule):
    def __init__(self, hparams):
        super(AELightning, self).__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        self.hparams = hparams
        config = hparams.__dict__
        self.config = config
        for _k, _v in self.config.items():
            if _v is None:
                if _k == "gpus":
                    self.config[_k] = "cpu"
                else:
                    self.config[_k] = "null"

        datasets = generate_partnet_allshapes_datasets(**config)
        self.train_dataset = datasets['train']
        self.val_dataset = datasets['val']

        self.encoder = AE_Encoder(**config)
        self.decoder = AE_Decoder(**config)

        random.seed(config['manual_seed'])
        np.random.seed(config['manual_seed'])
        torch.manual_seed(config['manual_seed'])
        torch.cuda.manual_seed(config['manual_seed'])
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

        with open(os.path.join(config['checkpoint_dir'], config['model'], config['version'], 'config.json'), 'w') as f:
            json.dump(self.config, f)