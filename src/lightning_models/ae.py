import os
import random
from argparse import Namespace
import json

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
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

        geo_loss = self.decoder.loss(output, partnet_geos)

        losses = {'geo': geo_loss.mean()}

        return losses

    def training_step(self, batch, batch_idx):

        batch[0] = [x[None, ...] for x in batch[0]]
        partnet_geos = torch.cat(batch[0]).unsqueeze(dim=1)

        input_batch = tuple([partnet_geos])

        total_loss = self.forward(input_batch)

        return {'loss': total_loss}

    def training_epoch_end(self, outputs):

        log = {}
        train_loss = torch.zeros(1)

        for output in outputs:
            train_loss += output['loss']
        train_loss /= len(outputs)

        log.update({'loss': train_loss})
        results = {'log': log}
        return results

    def validation_step(self, batch, batch_idx):

        batch[0] = [x[None, ...] for x in batch[0]]
        partnet_geos = torch.cat(batch[0]).unsqueeze(dim=1)

        input_batch = tuple([partnet_geos])

        total_loss = self.forward(input_batch)

        return {'val_loss': total_loss}

    def validation_epoch_end(self, outputs):

        log = {}
        val_loss = torch.zeros(1)

        for output in outputs:
            val_loss += output['loss']
        val_loss /= len(outputs)

        log.update({'val_loss': val_loss})
        results = {'log': log}
        return results

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'],
                          shuffle=self.config['shuffle'], num_workers=self.config['num_workers'], drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['batch_size'],
                          shuffle=False, num_workers=self.config['num_workers'], drop_last=True)