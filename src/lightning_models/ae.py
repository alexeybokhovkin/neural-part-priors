import os
import random
from argparse import Namespace
import json

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        scheduler = StepLR(optimizer, gamma=self.config['gamma'],
                           step_size=self.config['decay_every'])
        return [optimizer], [scheduler]

    def forward(self, batch):
        partnet_geos = batch[0]

        encoder_features = []
        fmap, features = self.encoder(partnet_geos)
        encoder_features += [features]
        output = self.decoder(fmap)

        self.decoder.


        losses = {'geo': 0}

        for i, object_losses in enumerate(all_losses):
            for loss_name, loss in object_losses.items():
                losses[loss_name] = losses[loss_name] + loss
        losses['kldiv'] = kldiv_loss
        losses['part_centers'] = part_center_loss

        del all_losses

        return losses