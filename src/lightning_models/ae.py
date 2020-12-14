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

from ..models.ae import AE_Encoder, AE_Decoder, AE_Skip_Decoder, Class_Head
from ..datasets.partnet import generate_partnet_allshapes_datasets

class AELightning(pl.LightningModule):

    def __init__(self, hparams, eval_mode=False):
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

        self.use_classification = config['use_classification']
        self.use_reconstruction = config['use_reconstruction']
        self.use_skip = config['use_skip']

        if not eval_mode:
            datasets = generate_partnet_allshapes_datasets(config['data'], config['dataset'], config['partnet_to_dirs_path'])
            self.train_dataset = datasets['train']
            self.val_dataset = datasets['val']

        self.encoder = AE_Encoder(**config)
        if self.use_reconstruction:
            if self.use_skip:
                self.decoder = AE_Skip_Decoder(**config)
            else:
                self.decoder = AE_Decoder(**config)
        if self.use_classification:
            self.classifier = Class_Head(**config)

        random.seed(config['manual_seed'])
        np.random.seed(config['manual_seed'])
        torch.manual_seed(config['manual_seed'])
        torch.cuda.manual_seed(config['manual_seed'])
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

        if not eval_mode:
            with open(os.path.join(config['base'], config['checkpoint_dir'], config['model'], config['version'], 'config.json'), 'w') as f:
                json.dump(self.config, f)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        scheduler = StepLR(optimizer, gamma=self.config['gamma'],
                           step_size=self.config['decay_every'])
        return [optimizer], [scheduler]

    def forward(self, batch):
        partnet_geos = batch[0]
        class_ids = batch[1]

        losses = {}

        encoder_features = []
        fmap, features = self.encoder(partnet_geos)
        encoder_features += [features]

        if self.use_reconstruction:
            output = self.decoder(fmap, features)
            geo_loss = self.decoder.loss(output, partnet_geos)
            losses['geo'] = geo_loss.mean()

        if self.use_classification:
            pred_classes = self.classifier(fmap)
            class_loss = self.classifier.loss(pred_classes, class_ids)
            losses['class'] = class_loss.mean()

        return losses

    def infer(self, batch):
        partnet_geos = batch[0]
        class_ids = batch[1]

        output = []
        encoder_features = []
        fmap, features = self.encoder(partnet_geos)
        output.append(fmap)
        encoder_features += [features]

        if self.use_reconstruction:
            pred_masks = self.decoder(fmap, features)
            output.append(pred_masks)

        if self.use_classification:
            pred_classes = self.classifier(fmap)
            output.append(pred_classes)

        return output

    def training_step(self, batch, batch_idx):
        partnet_geos = batch[0]
        class_ids = batch[1]
        input_batch = tuple([partnet_geos, class_ids])

        losses = self.forward(input_batch)
        losses_unweighted = losses.copy()

        for key in losses:
            losses[key] *= self.config['loss_weight_' + key]

        total_loss = 0
        for loss_name, loss in losses.items():
            total_loss += loss

        return {'loss': total_loss,
                'loss_components': losses,
                'loss_components_unweighted': losses_unweighted}

    def training_epoch_end(self, outputs):
        log = {}
        losses = {}
        losses_unweighted = {}
        train_loss = torch.tensor(0).type_as(outputs[0]['loss'])
        for key in outputs[0]['loss_components']:
            losses[key] = 0
            losses_unweighted[key] = 0

        for output in outputs:
            train_loss += output['loss']
            for key in losses:
                losses[key] += output['loss_components'][key]
                losses_unweighted[key] += output['loss_components_unweighted'][key]
        train_loss /= len(outputs)
        for key in losses:
            losses[key] /= len(outputs)
            losses_unweighted[key] /= len(outputs)

        log.update(losses)
        log.update({'loss': train_loss})

        self.log('loss', log['loss'])
        for key in losses:
            self.log(key, log[key])

    def validation_step(self, batch, batch_idx):
        partnet_geos = batch[0]
        class_ids = batch[1]
        input_batch = tuple([partnet_geos, class_ids])

        losses = self.forward(input_batch)
        losses_unweighted = losses.copy()

        for key in losses:
            losses[key] *= self.config['loss_weight_' + key]

        total_loss = 0
        for loss_name, loss in losses.items():
            total_loss += loss

        return {'val_loss': total_loss,
                'val_loss_components': losses,
                'val_loss_components_unweighted': losses_unweighted}

    def validation_epoch_end(self, outputs):
        log = {}
        losses = {}
        losses_unweighted = {}
        val_loss = torch.tensor(0).type_as(outputs[0]['val_loss'])
        for key in outputs[0]['val_loss_components']:
            losses['val_' + key] = 0
            losses_unweighted['val_' + key] = 0

        for output in outputs:
            val_loss += output['val_loss']
            for key in losses:
                losses[key] += output['val_loss_components'][key[4:]]
                losses_unweighted[key] += output['val_loss_components_unweighted'][key[4:]]
        val_loss /= len(outputs)
        for key in losses:
            losses[key] /= len(outputs)
            losses_unweighted[key] /= len(outputs)

        log.update(losses)
        log.update({'val_loss': val_loss})

        self.log('val_loss', log['val_loss'])
        for key in losses:
            self.log(key, log[key])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'],
                          shuffle=True, num_workers=self.config['num_workers'], drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['batch_size'],
                          shuffle=False, num_workers=self.config['num_workers'], drop_last=True)