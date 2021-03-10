import os
import json
from argparse import Namespace
import random
import numpy as np
import gc

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from src.models.gnn_models import GeoEncoder, HierarchicalDecoder
from src.data_utils.hierarchy import Tree
from src.datasets.partnet import generate_scannet_allshapes_datasets, generate_scannet_allshapes_contrastive_datasets
from src.utils.gnn import collate_feats, sym_reflect_tree


class LatentLearner(pl.LightningModule):
    def __init__(self, hparams):
        super(LatentLearner, self).__init__()

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

        datasets = generate_scannet_allshapes_contrastive_datasets(**config)
        self.train_dataset = datasets['train']
        self.val_dataset = datasets['val']

        self.config['parts_to_ids_path'] = os.path.join(config['datadir'], config['dataset'],
                                                        config['parts_to_ids_path'])

        Tree.load_category_info(os.path.join(config['datadir'], config['dataset']))
        self.encoder = GeoEncoder(**config)

        random.seed(config['manual_seed'])
        np.random.seed(config['manual_seed'])
        torch.manual_seed(config['manual_seed'])
        torch.cuda.manual_seed(config['manual_seed'])
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

        # with open(os.path.join(config['base'], config['checkpoint_dir'], config['model'], config['version'],
        #                        'config.json'), 'w') as f:
        #     json.dump(self.config, f)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'], momentum=0.9)
        scheduler = StepLR(optimizer, gamma=self.config['gamma'],
                           step_size=self.config['decay_every'])
        return [optimizer], [scheduler]

    def forward(self, batch):
        scannet_geos = batch[0]
        sdfs = batch[1]
        masks = batch[2]
        gt_trees = batch[3]
        partnet_ids = batch[4]

        rotations = batch[5]
        latents = batch[6]

        x_roots = []
        encoder_features = []

        all_losses = []

        for i, mask in enumerate(masks):
            x_root, feature = self.encoder.root_latents(scannet_geos[i][None, ...])
            encoder_features += [feature]

            latent_loss = self.encoder.encode_loss(x_root, latents[i])
            all_losses += [{'latent': latent_loss}]

            x_roots += [x_root]

        losses = {'latent': 0}

        for i, object_losses in enumerate(all_losses):
            for loss_name, loss in object_losses.items():
                losses[loss_name] = losses[loss_name] + loss
        for loss_name in losses:
            losses[loss_name] /= len(all_losses)

        gc.collect()

        return losses

    def inference(self, batch):
        scannet_geos = batch[0]
        sdfs = batch[1]
        masks = batch[2]
        gt_trees = batch[3]
        partnet_ids = batch[4]

        rotations = batch[5]
        latents = list(batch[6])

        x_roots = []
        encoder_features = []

        for i, mask in enumerate(masks):
            cuda_device = mask.get_device()
            x_root, feature = self.encoder.root_latents(scannet_geos[i][None, ...])
            encoder_features += [feature]

            x_roots += [x_root]

        all_losses = []

        losses = {'geo': 0,
                  'geo_prior': 0,
                  'leaf': 0,
                  'exists': 0,
                  'semantic': 0,
                  'latent': 0,
                  'edge_exists': 0,
                  'root_cls': 0,
                  'rotation': 0}

        for object_losses in all_losses:
            for loss_name, loss in object_losses.items():
                losses[loss_name] = losses[loss_name] + loss

        output = [x_roots]
        output = tuple(output)

        del all_losses

        return output

    def training_step(self, batch, batch_idx):
        Tree.load_category_info(os.path.join(self.config['datadir'], self.config['dataset']))

        batch[0] = [x[None, ...] for x in batch[0]]
        scannet_geos = torch.cat(batch[0]).unsqueeze(dim=1)
        shape_sdfs = torch.cat(batch[1]).unsqueeze(dim=1)
        shape_mask = torch.cat(batch[2]).unsqueeze(dim=1)
        gt_trees = batch[3]
        partnet_ids = batch[4]
        rotations = batch[5]
        latents = batch[7]

        input_batch = tuple([scannet_geos, shape_sdfs, shape_mask, gt_trees, partnet_ids, rotations, latents])

        losses = self.forward(input_batch)

        for key in losses:
            losses[key] *= self.config['loss_weight_' + key]

        loss_components = {}
        for key in losses:
            if isinstance(losses[key], float):
                loss_components[key] = losses[key]
            else:
                loss_components[key] = losses[key].detach()

        total_loss = 0
        for loss in losses.values():
            total_loss += loss

        gc.collect()

        return {'loss': total_loss,
                'train_loss_components': loss_components}

    def training_epoch_end(self, outputs):
        log = {}
        losses = {}
        train_loss = torch.tensor(0).type_as(outputs[0]['loss'])
        for key in outputs[0]['train_loss_components']:
            losses[key] = 0

        for output in outputs:
            train_loss += output['loss'].detach().item()
            for key in losses:
                if isinstance(output['train_loss_components'][key], float):
                    losses[key] += output['train_loss_components'][key]
                else:
                    losses[key] += output['train_loss_components'][key].detach().item()
        train_loss /= len(outputs)
        for key in losses:
            losses[key] /= len(outputs)

        log.update(losses)
        log.update({'loss': train_loss})

        self.log('loss', log['loss'])
        for key in losses:
            self.log(key, log[key])

        del outputs, log
        gc.collect()

    def validation_step(self, batch, batch_idx):
        Tree.load_category_info(os.path.join(self.config['datadir'], self.config['dataset']))

        batch[0] = [x[None, ...] for x in batch[0]]
        scannet_geos = torch.cat(batch[0]).unsqueeze(dim=1)
        shape_sdfs = torch.cat(batch[1]).unsqueeze(dim=1)
        shape_mask = torch.cat(batch[2]).unsqueeze(dim=1)
        gt_trees = batch[3]
        partnet_ids = batch[4]
        rotations = batch[5]
        latents = batch[7]

        input_batch = tuple([scannet_geos, shape_sdfs, shape_mask, gt_trees, partnet_ids, rotations, latents])

        losses = self.forward(input_batch)

        for key in losses:
            losses[key] *= self.config['loss_weight_' + key]

        loss_components = {}
        for key in losses:
            if isinstance(losses[key], float):
                loss_components[key] = losses[key]
            else:
                loss_components[key] = losses[key].detach()

        total_loss = 0
        for loss_name, loss in losses.items():
            total_loss += loss

        gc.collect()

        return {'val_loss': total_loss,
                'val_loss_components': loss_components}

    def validation_epoch_end(self, outputs):
        log = {}
        losses = {}
        val_loss = torch.tensor(0).type_as(outputs[0]['val_loss'])
        for key in outputs[0]['val_loss_components']:
            losses['val_' + key] = 0

        for output in outputs:
            val_loss += output['val_loss'].detach().item()
            for key in losses:
                if isinstance(output['val_loss_components'][key[4:]], float):
                    losses[key] += output['val_loss_components'][key[4:]]
                else:
                    losses[key] += output['val_loss_components'][key[4:]].detach().item()
        val_loss /= len(outputs)
        for key in losses:
            losses[key] /= len(outputs)

        log.update(losses)
        log.update({'val_loss': val_loss})

        self.log('val_loss', log['val_loss'])
        for key in losses:
            self.log(key, log[key])

        del outputs, log
        torch.cuda.empty_cache()
        gc.collect()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'],
                          shuffle=True, num_workers=self.config['num_workers'], drop_last=True,
                          collate_fn=collate_feats)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['batch_size'],
                          shuffle=False, num_workers=self.config['num_workers'], drop_last=True,
                          collate_fn=collate_feats)
