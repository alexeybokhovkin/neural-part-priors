import os
import json
from argparse import Namespace
import random
import gc

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl

from src.models.gnn_models import GeoEncoder, HierarchicalDecoder
from src.data_utils.hierarchy import Tree
from src.utils.gnn import collate_feats, sym_reflect_tree
from src.datasets.partnet import generate_scannet_allshapes_datasets, generate_scannet_allshapes_rot_datasets


class GNNPartnetLightning(pl.LightningModule):
    def __init__(self, hparams):
        super(GNNPartnetLightning, self).__init__()

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

        datasets = generate_scannet_allshapes_rot_datasets(**config)
        # datasets = generate_scannet_allshapes_datasets(**config)
        self.train_dataset = datasets['train']
        self.val_dataset = datasets['val']

        self.config['parts_to_ids_path'] = os.path.join(config['datadir'], config['dataset'], config['parts_to_ids_path'])

        Tree.load_category_info(os.path.join(config['datadir'], config['dataset']))
        self.encoder = GeoEncoder(**config)
        self.decoder = HierarchicalDecoder(**config)
        if self.config['encode_mask']:
            self.mask_encoder = GeoEncoder(**config)

        random.seed(config['manual_seed'])
        np.random.seed(config['manual_seed'])
        torch.manual_seed(config['manual_seed'])
        torch.cuda.manual_seed(config['manual_seed'])
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

        with open(os.path.join(config['base'], config['checkpoint_dir'], config['model'], config['version'], 'config.json'), 'w') as f:
            json.dump(self.config, f)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
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
        all_losses = []
        mask_codes, mask_features = [], []
        encoder_features = []

        if self.config['use_latent_constraint']:
            loss_latent_constraint = 0

        for i, mask in enumerate(masks):
            x_root, feature = self.encoder.root_latents(scannet_geos[i][None, ...])
            encoder_features += [feature]
            if self.config['encode_mask']:
                mask_code, mask_feature = self.mask_encoder.root_latents(scannet_geos[i][None, ...])
                mask_codes += [mask_code]
                mask_features += [mask_feature]
            else:
                mask_codes += [None]
                mask_features += [None]

            x_roots += [x_root]

        for i, x_root in enumerate(x_roots):
            cuda_device = x_root.get_device()
            gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))
            output = self.decoder.structure_recon_loss(x_root, gt_tree,
                                                       mask_code=mask_codes[i],
                                                       mask_feature=mask_features[i],
                                                       scan_geo=scannet_geos[i][None, ...],
                                                       encoder_features=encoder_features[i],
                                                       rotation=rotations[i])
            object_losses = output[0]
            all_losses += [object_losses]

        losses = {'geo': 0,
                  'geo_prior': 0,
                  'leaf': 0,
                  'exists': 0,
                  'semantic': 0,
                  'edge_exists': 0,
                  'root_cls': 0,
                  'rotation': 0}

        for i, object_losses in enumerate(all_losses):
            for loss_name, loss in object_losses.items():
                losses[loss_name] = losses[loss_name] + loss

        del all_losses

        return losses

    def inference(self, batch):
        scannet_geos = batch[0]
        sdfs = batch[1]
        masks = batch[2]
        gt_trees = batch[3]
        partnet_ids = batch[4]
        rotations = batch[5]

        x_roots = []
        mask_codes, mask_features = [], []
        encoder_features = []
        all_losses = []
        predicted_trees = []
        all_priors = []
        all_leaves_geos = []
        pred_rotations = []

        for i, mask in enumerate(masks):
            x_root, feature = self.encoder.root_latents(scannet_geos[i][None, ...])
            encoder_features += [feature]
            if self.config['encode_mask']:
                mask_code, mask_feature = self.mask_encoder.root_latents(scannet_geos[i][None, ...])
                mask_codes += [mask_code]
                mask_features += [mask_feature]
            else:
                mask_codes += [None]
                mask_features += [None]
            x_roots += [x_root]

        for i, x_root in enumerate(x_roots):
            cuda_device = x_root.get_device()
            gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))
            object_losses, all_geos, all_leaf_geos, all_S_priors = self.decoder.structure_recon_loss(x_root, gt_tree,
                                                                                                     mask_code=mask_codes[i],
                                                                                                     mask_feature=mask_features[i],
                                                                                                     scan_geo=scannet_geos[i][None, ...],
                                                                                                     encoder_features=encoder_features[i],
                                                                                                     rotation=rotations[i])

            predicted_tree, S_priors, pred_rotation = self.decoder(x_root,
                                                                   mask_code=mask_codes[i],
                                                                   mask_feature=mask_features[i],
                                                                   scan_geo=scannet_geos[i][None, ...],
                                                                   full_label=gt_tree.root.label,
                                                                   encoder_features=encoder_features[i],
                                                                   rotation=rotations[i])

            predicted_trees += [predicted_tree]
            pred_rotations += [pred_rotation]
            all_losses += [object_losses]
            all_leaves_geos += [all_leaf_geos]
            all_priors += [S_priors]

        losses = {'geo': [],
                  'geo_prior': [],
                  'leaf': [],
                  'exists': [],
                  'semantic': [],
                  'latent': [],
                  'edge_exists': [],
                  'num_children': [],
                  'root_cls': [],
                  'rotation': []}

        for i, object_losses in enumerate(all_losses):
            for loss_name, loss in object_losses.items():
                losses[loss_name].append(loss)

        output = [predicted_trees, x_roots, losses, all_leaves_geos, all_priors, pred_rotations]
        output = tuple(output)

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
        latents = batch[6]

        input_batch = tuple([scannet_geos, shape_sdfs, shape_mask, gt_trees, partnet_ids, rotations, latents])

        losses = self.forward(input_batch)

        for key in losses:
            losses[key] *= self.config['loss_weight_' + key]
        total_loss = 0
        for loss in losses.values():
            total_loss += loss

        loss_components = {}
        for key in losses:
            loss_components[key] = losses[key].detach()
        print('rotation:', losses['rotation'])
        print('root_cls:', losses['root_cls'])
        print('semantic:', losses['semantic'])
        print('geo:', losses['geo'])
        print('geo_prior:', losses['geo_prior'])

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
        latents = batch[6]

        input_batch = tuple([scannet_geos, shape_sdfs, shape_mask, gt_trees, partnet_ids, rotations, latents])

        losses = self.forward(input_batch)

        for key in losses:
            losses[key] *= self.config['loss_weight_' + key]

        loss_components = {}
        for key in losses:
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