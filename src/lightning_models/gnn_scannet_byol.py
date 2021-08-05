import os
import json
from argparse import Namespace
import random
import numpy as np
import gc
from copy import deepcopy
from scipy.ndimage import rotate

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from src.models.gnn_models import GeoEncoder, HierarchicalDecoder
from src.data_utils.hierarchy import Tree
from src.datasets.partnet import generate_scannet_allshapes_datasets, generate_scannet_allshapes_contrastive_datasets
from src.utils.gnn import collate_feats, sym_reflect_tree, rotate_tree_geos

from src.utils.losses_byol import set_requires_grad, byol_constraint


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

        datasets = generate_scannet_allshapes_contrastive_datasets(**config)
        self.train_dataset = datasets['train']
        self.val_dataset = datasets['val']

        self.config['parts_to_ids_path'] = os.path.join(config['datadir'], config['dataset'],
                                                        config['parts_to_ids_path'])

        Tree.load_category_info(os.path.join(config['datadir'], config['dataset']))
        self.encoder_online = GeoEncoder(**config)
        self.encoder_target = GeoEncoder(**config)
        self.decoder_online = HierarchicalDecoder(**config)
        self.decoder_target = HierarchicalDecoder(**config)
        set_requires_grad(self.encoder_target, False)
        set_requires_grad(self.decoder_target, False)
        if self.config['encode_mask']:
            self.mask_encoder_online = GeoEncoder(**config)
            self.mask_encoder_target = GeoEncoder(**config)
            set_requires_grad(self.mask_encoder_target, False)

        random.seed(config['manual_seed'])
        np.random.seed(config['manual_seed'])
        torch.manual_seed(config['manual_seed'])
        torch.cuda.manual_seed(config['manual_seed'])
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

        self.byol_constraint = byol_constraint(device=torch.device('cuda'), moving_average_decay=config['moving_average_decay_byol'])

        # with open(os.path.join(config['base'], config['checkpoint_dir'], config['model'], config['version'],
        #                        'config.json'), 'w') as f:
        #     json.dump(self.config, f)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        # optimizer = optim.SGD(self.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'], momentum=0.9)
        scheduler = StepLR(optimizer, gamma=self.config['gamma'],
                           step_size=self.config['decay_every'])
        return [optimizer], [scheduler]

    def augment_rotate(self, trees, scan_geos, scan_sdfs):
        angles = [27.5 * i for i in range(16)]
        num_trees = len(trees)
        device = scan_geos.get_device()

        new_trees = []
        new_scan_geos = []
        new_scan_sdfs = []
        for i in range(num_trees):
            random_angle = angles[np.random.randint(0, 16)]
            rotated_tree = rotate_tree_geos(deepcopy(trees[i][0]), random_angle)
            scan_geo_rotated = torch.FloatTensor(rotate(scan_geos[i][0].cpu().numpy().astype('uint8'), random_angle, axes=[0, 2], reshape=False).astype('float32'))[None, ...]
            scan_sdf_rotated = torch.FloatTensor(rotate(scan_sdfs[i][0].cpu().numpy(), random_angle, axes=[0, 2], reshape=False).astype('float32'))[None, ...]
            new_scan_geos += [scan_geo_rotated]
            new_scan_sdfs += [scan_sdf_rotated]
            new_trees += [(rotated_tree,)]
        new_scan_geos = torch.cat(new_scan_geos, dim=0).unsqueeze(dim=1).to(device)
        new_scan_sdfs = torch.cat(new_scan_sdfs, dim=0).unsqueeze(dim=1).to(device)

        return new_trees, new_scan_geos, new_scan_sdfs

    def forward(self, batch):
        scannet_geos = batch[0]
        sdfs = batch[1]
        masks = batch[2]
        gt_trees = batch[3]
        partnet_ids = batch[4]

        # gt_trees, scannet_geos, sdfs = self.augment_rotate(gt_trees, scannet_geos, sdfs)

        rotations = batch[5]

        scannet_geos_pos = batch[7]
        sdfs_pos = batch[8]
        shape_mask_pos = batch[9]
        gt_trees_pos = batch[10]

        # gt_trees_pos, scannet_geos_pos, sdfs_pos = self.augment_rotate(gt_trees_pos, scannet_geos_pos, sdfs_pos)

        x_roots_online, x_roots_pos_online = [], []
        mask_codes_online, mask_features_online, mask_codes_pos_online, mask_features_pos_online = [], [], [], []
        encoder_features_online, encoder_features_pos_online = [], []

        x_roots_target, x_roots_pos_target = [], []
        mask_codes_target, mask_features_target, mask_codes_pos_target, mask_features_pos_target = [], [], [], []
        encoder_features_target, encoder_features_pos_target = [], []

        for i, mask in enumerate(masks):
            x_root_online, feature_online = self.encoder_online.root_latents(scannet_geos[i][None, ...])
            encoder_features_online += [feature_online]
            x_root_pos_online, feature_pos_online = self.encoder_online.root_latents(scannet_geos_pos[i][None, ...])
            encoder_features_pos_online += [feature_pos_online]

            with torch.no_grad():
                x_root_target, feature_target = self.encoder_target.root_latents(scannet_geos[i][None, ...])
                encoder_features_target += [feature_target]
                x_root_pos_target, feature_pos_target = self.encoder_target.root_latents(scannet_geos_pos[i][None, ...])
                encoder_features_pos_target += [feature_pos_target]

            if self.config['encode_mask']:
                mask_code_online, mask_feature_online = self.mask_encoder_online.root_latents(scannet_geos[i][None, ...])
                mask_codes_online += [mask_code_online]
                mask_features_online += [mask_feature_online]
                mask_code_pos_online, mask_feature_pos_online = self.mask_encoder_online.root_latents(scannet_geos_pos[i][None, ...])
                mask_codes_pos_online += [mask_code_pos_online]
                mask_features_pos_online += [mask_feature_pos_online]

                with torch.no_grad():
                    mask_code_target, mask_feature_target = self.mask_encoder_target.root_latents(scannet_geos[i][None, ...])
                    mask_codes_target += [mask_code_target]
                    mask_features_target += [mask_feature_target]
                    mask_code_pos_target, mask_feature_pos_target = self.mask_encoder_target.root_latents(scannet_geos_pos[i][None, ...])
                    mask_codes_pos_target += [mask_code_pos_target]
                    mask_features_pos_target += [mask_feature_pos_target]

            x_roots_online += [x_root_online]
            x_roots_pos_online += [x_root_pos_online]
            x_roots_target += [x_root_target]
            x_roots_pos_target += [x_root_pos_target]

        all_losses = []

        all_child_feats = []
        all_child_sem_logits = []
        all_child_decoder_features = []

        all_child_feats_pos = []
        all_child_decoder_features_pos = []

        all_child_feats_target = []
        all_child_feats_pos_target = []

        for i, x_root_online in enumerate(x_roots_online):
            cuda_device = x_root_online.get_device()
            gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))
            gt_tree_pos = gt_trees_pos[i][0].to("cuda:{}".format(cuda_device))

            output_online = self.decoder_online.structure_recon_loss(x_roots_online[i], gt_tree,
                                                       mask_code=mask_codes_online[i],
                                                       mask_feature=mask_features_online[i],
                                                       encoder_features=encoder_features_online[i],
                                                       rotation=rotations[i]
                                                       )

            output_pos_online = self.decoder_online.structure_recon_loss(x_roots_pos_online[i], gt_tree_pos,
                                                       mask_code=mask_codes_pos_online[i],
                                                       mask_feature=mask_features_pos_online[i],
                                                       encoder_features=encoder_features_pos_online[i],
                                                       rotation=rotations[i]
                                                       )
            object_losses = output_online[0]
            all_losses += [object_losses]

            child_feats = output_online[4][0]             # Tensor[n_parts, 128]
            child_sem_logits = output_online[5]           # list<Tensor[50]>
            child_decoder_features = output_online[6]     # list<list<Tensor[fmap]>>
            all_child_feats.append(child_feats)
            all_child_decoder_features.append(child_decoder_features)

            child_feats_pos = output_pos_online[4][0]  # Tensor[n_parts, 128]
            child_decoder_features_pos = output_pos_online[6]  # list<list<Tensor[fmap]>>
            all_child_feats_pos.append(child_feats_pos)
            all_child_decoder_features_pos.append(child_decoder_features_pos)

            with torch.no_grad():
                output_target = self.decoder_target.structure_recon_loss(x_roots_target[i], gt_tree,
                                                                  mask_code=mask_codes_target[i],
                                                                  mask_feature=mask_features_target[i],
                                                                  encoder_features=encoder_features_target[i],
                                                                  rotation=rotations[i]
                                                                  )

                output_pos_target = self.decoder_target.structure_recon_loss(x_roots_pos_target[i], gt_tree_pos,
                                                                      mask_code=mask_codes_pos_target[i],
                                                                      mask_feature=mask_features_pos_target[i],
                                                                      encoder_features=encoder_features_pos_target[i],
                                                                      rotation=rotations[i]
                                                                      )
                child_feats_target = output_target[4][0]
                child_feats_pos_target = output_pos_target[4][0]
                all_child_feats_target.append(child_feats_target)
                all_child_feats_pos_target.append(child_feats_pos_target)

        loss_byol_total = 0
        for i in range(len(x_roots_online)):
            loss_byol = self.byol_constraint(x_roots_online[i], mask_codes_online[i], all_child_feats[i], x_roots_pos_online[i], mask_codes_pos_online[i], all_child_feats_pos[i],
                                             x_roots_target[i], mask_codes_target[i], all_child_feats_target[i], x_roots_pos_target[i], mask_codes_pos_target[i], all_child_feats_pos_target[i])
            loss_byol_total = loss_byol_total + loss_byol

        losses = {'geo': 0,
                  'geo_prior': 0,
                  'leaf': 0,
                  'exists': 0,
                  'semantic': 0,
                  'edge_exists': 0,
                  'root_cls': 0,
                  'rotation': 0,
                  'byol': loss_byol_total}

        for i, object_losses in enumerate(all_losses):
            for loss_name, loss in object_losses.items():
                losses[loss_name] = losses[loss_name] + loss
        for loss_name in losses:
            losses[loss_name] /= len(all_losses)

        del all_losses, output_pos_online, output_online, all_child_feats, all_child_decoder_features, all_child_feats_pos, \
            all_child_decoder_features_pos, \
            gt_tree, gt_tree_pos

        gc.collect()

        self.byol_constraint.update_moving_average(self.encoder_target, self.encoder_online)
        self.byol_constraint.update_moving_average(self.decoder_target, self.decoder_online)

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
        children_roots_all = []
        mask_codes, mask_features = [], []
        encoder_features = []
        for i, mask in enumerate(masks):
            cuda_device = mask.get_device()
            x_root, feature = self.encoder_online.root_latents(scannet_geos[i][None, ...])
            encoder_features += [feature]
            if self.config['encode_mask']:
                mask_code, mask_feature = self.mask_encoder_online.root_latents(scannet_geos[i][None, ...])
                mask_codes += [mask_code]
                mask_features += [mask_feature]
            else:
                mask_codes += [None]
                mask_features += [None]
            x_roots += [x_root]

        all_losses = []
        predicted_trees = []
        all_priors = []
        all_leaves_geos = []
        pred_rotations = []
        all_child_nodes = []

        all_child_feats = []
        all_child_decoder_features = []

        for i, x_root in enumerate(x_roots):
            cuda_device = x_root.get_device()
            gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))
            mask = masks[i]
            object_losses, all_geos, all_leaf_geos, all_S_priors, child_feats, _, child_decoder_features, _ = self.decoder_online.structure_recon_loss(x_root, gt_tree,
                                                                                                    mask_code=mask_codes[i],
                                                                                                    mask_feature=mask_features[i],
                                                                                                    scan_geo=scannet_geos[i][None, ...],
                                                                                                    encoder_features=encoder_features[i],
                                                                                                    rotation=rotations[i]
                                                                                                    )

            predicted_tree, S_priors, pred_rotation, child_nodes, child_feats_exist = self.decoder_online(x_root,
                                          mask_code=mask_codes[i],
                                          mask_feature=mask_features[i],
                                          scan_geo=scannet_geos[i][None, ...],
                                          full_label=gt_tree.root.label,
                                          encoder_features=encoder_features[i],
                                          rotation=rotations[i],
                                          gt_tree=gt_tree
                                          )

            predicted_trees += [predicted_tree]
            pred_rotations += [pred_rotation]

            all_losses += [object_losses]
            all_leaves_geos += [all_leaf_geos]
            all_priors += [S_priors]
            all_child_nodes += [child_nodes]

            all_child_feats.append(child_feats[0])
            all_child_decoder_features.append(child_decoder_features)

        losses = {'geo': 0,
                  'geo_prior': 0,
                  'leaf': 0,
                  'exists': 0,
                  'semantic': 0,
                  'latent': 0,
                  'edge_exists': 0,
                  'num_children': 0,
                  'split_enc_children': 0,
                  'root_cls': 0,
                  'geo_refined': 0,
                  'geo_full_shape': 0,
                  'rotation': 0}

        for object_losses in all_losses:
            for loss_name, loss in object_losses.items():
                losses[loss_name] = losses[loss_name] + loss

        output = [predicted_trees, x_roots, losses, all_leaves_geos, all_priors, pred_rotations, latents,
                  all_child_nodes, all_child_feats, all_child_decoder_features]
        output = tuple(output)

        del all_losses

        return output

    def training_step(self, batch, batch_idx):
        Tree.load_category_info(os.path.join(self.config['datadir'], self.config['dataset']))

        batch[0] = [x[None, ...] for x in batch[0]]
        scannet_geos = torch.cat(batch[0]).unsqueeze(dim=1)
        batch[1] = [x[None, ...] for x in batch[1]]
        shape_sdfs = torch.cat(batch[1]).unsqueeze(dim=1)
        shape_mask = torch.cat(batch[2]).unsqueeze(dim=1)
        gt_trees = batch[3]
        partnet_ids = batch[4]
        rotations = batch[5]
        latents = batch[7]

        batch[8] = [x[None, ...] for x in batch[8]]
        scannet_geos_pos = torch.cat(batch[8]).unsqueeze(dim=1)
        batch[9] = [x[None, ...] for x in batch[9]]
        shape_sdfs_pos = torch.cat(batch[9]).unsqueeze(dim=1)
        shape_mask_pos = torch.cat(batch[10]).unsqueeze(dim=1)
        gt_trees_pos = batch[11]

        input_batch = tuple([scannet_geos, shape_sdfs, shape_mask, gt_trees, partnet_ids, rotations, latents,
                             scannet_geos_pos, shape_sdfs_pos, shape_mask_pos, gt_trees_pos,
                             ])

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

        print('rotation:', losses['rotation'])
        print('root_cls:', losses['root_cls'])
        print('geo:', losses['geo'])
        print('geo_prior:', losses['geo_prior'])
        print()
        print('byol_constraint:', losses['byol'])
        print()

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
        batch[1] = [x[None, ...] for x in batch[1]]
        shape_sdfs = torch.cat(batch[1]).unsqueeze(dim=1)
        shape_mask = torch.cat(batch[2]).unsqueeze(dim=1)
        gt_trees = batch[3]
        partnet_ids = batch[4]
        rotations = batch[5]
        latents = batch[7]

        batch[8] = [x[None, ...] for x in batch[8]]
        scannet_geos_pos = torch.cat(batch[8]).unsqueeze(dim=1)
        batch[9] = [x[None, ...] for x in batch[9]]
        shape_sdfs_pos = torch.cat(batch[9]).unsqueeze(dim=1)
        shape_mask_pos = torch.cat(batch[10]).unsqueeze(dim=1)
        gt_trees_pos = batch[11]

        input_batch = tuple([scannet_geos, shape_sdfs, shape_mask, gt_trees, partnet_ids, rotations, latents,
                             scannet_geos_pos, shape_sdfs_pos, shape_mask_pos, gt_trees_pos,
                             ])

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

    def interpolate(self, batch_1, batch_2):
        scannet_geos_1 = batch_1[0]
        masks_1 = batch_1[2]
        gt_trees_1 = batch_1[3]
        rotations_1 = batch_1[5]

        scannet_geos_2 = batch_2[0]
        masks_2 = batch_2[2]
        gt_trees_2 = batch_2[3]

        x_roots_1 = []
        x_roots_2 = []
        mask_codes_1, mask_features_1 = [], []
        mask_codes_2, mask_features_2 = [], []
        encoder_features_1 = []
        for i, mask in enumerate(masks_1):
            cuda_device = mask.get_device()
            x_root_1, feature_1 = self.encoder_online.root_latents(scannet_geos_1[i][None, ...])
            encoder_features_1 += [feature_1]
            x_root_2, feature_2 = self.encoder_online.root_latents(scannet_geos_2[i][None, ...])
            # print(torch.sqrt(torch.sum((x_root_1 - x_root_2) ** 2)))
            # print(torch.sqrt(torch.sum((scannet_geos_1[0] - scannet_geos_2[0]) ** 2)))
            # print(torch.sqrt(torch.sum((feature_1[0] - feature_2[0]) ** 2)))
            # print(torch.sqrt(torch.sum((feature_1[1] - feature_2[1]) ** 2)))
            # print(torch.sqrt(torch.sum((feature_1[2] - feature_2[2]) ** 2)))
            # print(torch.sqrt(torch.sum((feature_1[3] - feature_2[3]) ** 2)))

            if self.config['encode_mask']:
                mask_code_1, mask_feature_1 = self.mask_encoder_online.root_latents(scannet_geos_1[i][None, ...])
                mask_codes_1 += [mask_code_1]
                mask_features_1 += [mask_feature_1]
                mask_code_2, mask_feature_2 = self.mask_encoder_online.root_latents(scannet_geos_2[i][None, ...])
                mask_codes_2 += [mask_code_2]
                mask_features_2 += [mask_feature_2]
            else:
                mask_codes_1 += [None]
                mask_features_1 += [None]
            x_roots_1 += [x_root_1]
            x_roots_2 += [x_root_2]

        predicted_trees = []

        for i, x_root_1 in enumerate(x_roots_1):
            for step in range(11):
                x_root_interp = x_root_1 + (x_root_2 - x_root_1) * (step / 10)
                mask_code_interp = mask_codes_1[i] + (mask_codes_2[i] - mask_codes_1[i]) * (step / 10)
                print('Step:', torch.sqrt(torch.sum((x_root_interp - x_root_1) ** 2)))
                print('Step:', torch.sqrt(torch.sum((x_root_interp - x_root_2) ** 2)))
                print()

                cuda_device = x_root_1.get_device()
                gt_tree_1 = gt_trees_1[i][0].to("cuda:{}".format(cuda_device))
                object_losses, all_geos, all_leaf_geos, all_S_priors, child_feats, _, child_decoder_features, _ = self.decoder_online.structure_recon_loss(x_root_interp, gt_tree_1,
                                                                                                        mask_code=mask_code_interp,
                                                                                                        mask_feature=mask_features_1[i],
                                                                                                        encoder_features=encoder_features_1[i],
                                                                                                        rotation=rotations_1[i]
                                                                                                        )

                predicted_tree, S_priors, pred_rotation, child_nodes, _ = self.decoder_online(x_root_interp,
                                              mask_code=mask_code_interp,
                                              mask_feature=mask_features_1[i],
                                              scan_geo=scannet_geos_1[i][None, ...],
                                              full_label=gt_tree_1.root.label,
                                              encoder_features=encoder_features_1[i],
                                              rotation=rotations_1[i],
                                              gt_tree=gt_tree_1
                                              )

                predicted_trees += [predicted_tree]

        output = [predicted_trees]
        output = tuple(output)

        return output

    def interpolate_all_latents(self, batch_1, batch_2):

        scannet_geos_1 = batch_1[0]
        masks_1 = batch_1[2]
        gt_trees_1 = batch_1[3]
        rotations_1 = batch_1[5]

        scannet_geos_2 = batch_2[0]
        masks_2 = batch_2[2]
        gt_trees_2 = batch_2[3]

        x_roots_1 = []
        x_roots_2 = []
        mask_codes_1, mask_features_1 = [], []
        encoder_features_1 = []
        mask_codes_2, mask_features_2 = [], []
        encoder_features_2 = []
        for i, mask in enumerate(masks_1):
            cuda_device = mask.get_device()
            x_root_1, feature_1 = self.encoder_online.root_latents(scannet_geos_1[i][None, ...])
            encoder_features_1 += [feature_1]
            x_root_2, feature_2 = self.encoder_online.root_latents(scannet_geos_2[i][None, ...])
            encoder_features_2 += [feature_2]

            if self.config['encode_mask']:
                mask_code_1, mask_feature_1 = self.mask_encoder_online.root_latents(scannet_geos_1[i][None, ...])
                mask_codes_1 += [mask_code_1]
                mask_features_1 += [mask_feature_1]
                mask_code_2, mask_feature_2 = self.mask_encoder_online.root_latents(scannet_geos_2[i][None, ...])
                mask_codes_2 += [mask_code_2]
                mask_features_2 += [mask_feature_2]
            else:
                mask_codes_1 += [None]
                mask_features_1 += [None]
            x_roots_1 += [x_root_1]
            x_roots_2 += [x_root_2]

        for i, x_root in enumerate(x_roots_1):
            cuda_device = x_root.get_device()
            gt_tree_1 = gt_trees_1[i][0].to("cuda:{}".format(cuda_device))
            gt_tree_2 = gt_trees_2[i][0].to("cuda:{}".format(cuda_device))

            with torch.no_grad():
                output_1 = self.decoder_online.structure_recon_loss(x_root_1, gt_tree_1,
                                                           mask_code=mask_codes_1[i],
                                                           mask_feature=mask_features_1[i],
                                                           scan_geo=scannet_geos_1[i][None, ...],
                                                           encoder_features=encoder_features_1[i],
                                                           rotation=rotations_1[i]
                                                           )

                output_2 = self.decoder_online.structure_recon_loss(x_root_2, gt_tree_2,
                                                           mask_code=mask_codes_2[i],
                                                           mask_feature=mask_features_2[i],
                                                           scan_geo=scannet_geos_2[i][None, ...],
                                                           encoder_features=encoder_features_2[i],
                                                           rotation=rotations_1[i]
                                                           )

            child_feats_1 = output_1[4][0]
            child_feats_2 = output_2[4][0]

        predicted_trees = []

        for step in range(11):
            x_root_interp = x_root_1 + (x_root_2 - x_root_1) * (step / 10)
            child_feats_interp = []
            for k in range(len(child_feats_1)):
                child_feats_interp += [child_feats_1[k] + (child_feats_2[k] - child_feats_1[k]) * (step / 10)]
            predicted_tree = self.decoder_online.inference_from_latents(x_root_interp, child_feats_interp, mask_codes_1[0], mask_features_1[0])
            predicted_trees += [predicted_tree]

        output = (predicted_trees,)

        return output

    def tto_latent_root(self, batch, iterations=1000, batch_init=None):

        Tree.load_category_info(os.path.join(self.config['datadir'], self.config['dataset']))
        # sets decoder to trainable or not
        # set_requires_grad(self.decoder_online, False)

        scan_geos = batch[0]
        scan_sdfs = batch[1]
        masks = batch[2]
        gt_trees = batch[3]
        partnet_ids = batch[4]
        rotations = batch[5]
        latents = list(batch[6])

        if batch_init is not None:
            scan_geos_init = batch_init[0]
            scan_sdfs_init = batch_init[1]
            masks_init = batch_init[2]
            gt_trees_init = batch_init[3]
            partnet_ids_init = batch_init[4]

        x_roots = []
        mask_codes, mask_features = [], []
        encoder_features = []
        with torch.no_grad():
            for i, mask in enumerate(masks):
                if batch_init is None:
                    x_root, feature = self.encoder_online.root_latents(scan_geos[i][None, ...])
                else:
                    x_root, feature = self.encoder_online.root_latents(scan_geos_init[i][None, ...])
                feature_detached = [x.clone().detach() for x in feature]
                encoder_features += [feature_detached]
                if self.config['encode_mask']:
                    if batch_init is None:
                        mask_code, mask_feature = self.mask_encoder_online.root_latents(scan_geos[i][None, ...])
                    else:
                        mask_code, mask_feature = self.mask_encoder_online.root_latents(scan_geos_init[i][None, ...])
                    mask_code_detached = mask_code.clone().detach()
                    mask_feature_detached = [x.clone().detach() for x in mask_feature]
                    mask_codes += [mask_code_detached]
                    mask_features += [mask_feature_detached]
                else:
                    mask_codes += [None]
                    mask_features += [None]
                x_root_detached = x_root.clone().detach()
                x_roots += [x_root_detached]

        x_root_learnable = x_root_detached.clone()
        mask_code_learnable = mask_code_detached.clone()
        cuda_device = x_root_detached.get_device()

        predicted_trees = []
        all_losses_detached = []
        all_child_feats_detached = []
        all_x_roots = []

        for j in range(iterations):

            children_latent_loss_full = 0
            children_geo_sum_loss_full = 0
            all_losses = []
            for i, x_root in enumerate(x_roots):

                cuda_device = x_root.get_device()
                gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))

                with torch.no_grad():
                    predicted_tree, S_priors, pred_rotation, child_nodes, child_feats_exist = self.decoder_online(x_root_learnable,
                                                                                        mask_code=mask_code_learnable,
                                                                                        mask_feature=mask_features[i],
                                                                                        scan_geo=scan_geos[i][None, ...],
                                                                                        full_label=gt_tree.root.label,
                                                                                        encoder_features=
                                                                                        encoder_features[i],
                                                                                        rotation=rotations[i],
                                                                                        gt_tree=gt_tree
                                                                                        )
                    predicted_tree.to('cpu').detach()
                    predicted_trees += [predicted_tree]

                    if j == 0:
                        initial_predicted_tree = predicted_tree.to(cuda_device)
                        object_losses, all_geos, all_leaf_geos_init, all_S_priors, child_feats_init, _, child_decoder_features, _ = self.decoder_online.tto_loss(
                            x_root_learnable, gt_tree,
                            mask_code=mask_code_learnable,
                            mask_feature=mask_features[i],
                            scan_geo=scan_geos[i][None, ...],
                            encoder_features=encoder_features[i],
                            rotation=rotations[i],
                            scan_sdf=scan_sdfs[i],
                            predicted_tree=initial_predicted_tree
                        )
                        child_feats_init = [x.detach() for x in child_feats_init]
                        all_leaf_geos_init = [x.detach() for x in all_leaf_geos_init]

                        # x_root_learnable = x_root_learnable + 20.0 * torch.rand(x_root_learnable.shape).to(cuda_device)
                        x_root_learnable.detach().requires_grad_(True)
                        x_root_learnable.requires_grad = True
                        mask_code_learnable.detach().requires_grad_(True)
                        mask_code_learnable.requires_grad = True
                        opt = optim.Adam([x_root_learnable, mask_code_learnable], lr=self.config['learning_rate'],
                                         weight_decay=self.config['weight_decay'])
                        # opt = optim.Adam([x_root_learnable, mask_code_learnable] + list(self.decoder_online.parameters()), lr=self.config['learning_rate'],
                        #                  weight_decay=self.config['weight_decay'])

                object_losses, all_geos, all_leaf_geos, all_S_priors, child_feats, _, child_decoder_features, _ = self.decoder_online.tto_loss(
                    x_root_learnable, gt_tree,
                    mask_code=mask_code_learnable,
                    mask_feature=mask_features[i],
                    scan_geo=scan_geos[i][None, ...],
                    encoder_features=encoder_features[i],
                    rotation=rotations[i],
                    scan_sdf=scan_sdfs[i],
                    predicted_tree=initial_predicted_tree
                    )
                all_losses += [object_losses]
                child_feats_detached = np.array([x.detach().cpu().numpy() for x in child_feats])
                all_child_feats_detached += [child_feats_detached]
                all_x_roots += [x_root_learnable.detach().cpu().numpy()]

                children_latent_loss = self.decoder_online.children_mse_loss(child_feats, child_feats_init)
                children_latent_loss_full += children_latent_loss

                children_geo_sum_loss = self.decoder_online.children_geo_sum_loss(all_leaf_geos, all_leaf_geos_init)
                children_geo_sum_loss_full += children_geo_sum_loss

                gt_tree.to('cpu').detach()

            losses = {'geo': 0,
                      'geo_prior': 0,
                      'leaf': 0,
                      'exists': 0,
                      'semantic': 0,
                      'edge_exists': 0,
                      'root_cls': 0,
                      'rotation': 0,
                      'geo_children': 0,
                      'children_latent': children_latent_loss_full,
                      'children_geo_sum': children_geo_sum_loss_full,
                      'tv': 0}

            for i, object_losses in enumerate(all_losses):
                for loss_name, loss in object_losses.items():
                    losses[loss_name] = losses[loss_name] + loss
            for loss_name in losses:
                losses[loss_name] /= len(all_losses)

            losses_detached = {}
            for key in losses:
                if not isinstance(losses[key], float):
                    losses_detached[key] = float(np.squeeze(losses[key].detach().cpu().numpy()))
                else:
                    losses_detached[key] = losses[key]
            all_losses_detached += [losses_detached]
            if j % 20 == 0:
                print('Geo loss:', losses_detached['geo'])
                print('Geo children loss:', losses_detached['geo_children'])
                print()

            for key in losses:
                losses[key] *= self.config['loss_weight_' + key]
            total_loss = 0
            for loss_name, loss in losses.items():
                total_loss += loss

            total_loss.backward()
            opt.step()
            opt.zero_grad()

        initial_predicted_tree.to('cpu')
        del initial_predicted_tree
        output = (predicted_trees, all_losses_detached, all_child_feats_detached, all_x_roots)

        gc.collect()

        return output

    def tto_latent_leaves(self, batch, iterations=1000):

        Tree.load_category_info(os.path.join(self.config['datadir'], self.config['dataset']))

        scan_geos = batch[0]
        scan_sdfs = batch[1]
        masks = batch[2]
        gt_trees = batch[3]
        partnet_ids = batch[4]
        rotations = batch[5]
        latents = list(batch[6])

        x_roots = []
        mask_codes, mask_features = [], []
        encoder_features = []
        with torch.no_grad():
            for i, mask in enumerate(masks):
                x_root, feature = self.encoder_online.root_latents(scan_geos[i][None, ...])
                feature_detached = [x.clone().detach() for x in feature]
                encoder_features += [feature_detached]
                if self.config['encode_mask']:
                    mask_code, mask_feature = self.mask_encoder_online.root_latents(scan_geos[i][None, ...])
                    mask_code_detached = mask_code.clone().detach()
                    mask_feature_detached = [x.clone().detach() for x in mask_feature]
                    mask_codes += [mask_code_detached]
                    mask_features += [mask_feature_detached]
                else:
                    mask_codes += [None]
                    mask_features += [None]
                x_root_detached = x_root.clone().detach()
                x_roots += [x_root_detached]

        x_root_learnable = x_root_detached.clone()
        mask_code_learnable = mask_code_detached.clone()
        cuda_device = x_root_detached.get_device()

        predicted_trees = []
        all_losses_detached = []
        all_child_feats_detached = []
        all_x_roots = []
        children_names = []

        for j in range(iterations):

            children_latent_loss_full = 0
            children_geo_sum_loss_full = 0
            all_losses = []
            for i, x_root in enumerate(x_roots):

                cuda_device = x_root.get_device()
                gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))

                with torch.no_grad():
                    if j == 0:
                        predicted_tree, S_priors, pred_rotation, child_feats, child_feats_exist = self.decoder_online(x_root_learnable,
                                                                                                   mask_code=mask_code_learnable,
                                                                                                   mask_feature=mask_features[i],
                                                                                                   scan_geo=scan_geos[i][None, ...],
                                                                                                   full_label=gt_tree.root.label,
                                                                                                   encoder_features=encoder_features[i],
                                                                                                   rotation=rotations[i],
                                                                                                   gt_tree=gt_tree
                                                                                                   )
                        predicted_tree.to('cpu').detach()
                        children_names = [x.label for x in predicted_tree.root.children]
                        predicted_trees += [predicted_tree]

                        initial_predicted_tree = predicted_tree.to(cuda_device)
                        object_losses, all_geos, all_leaf_geos_init, all_S_priors, child_feats_init, _, child_decoder_features, _ = self.decoder_online.tto_loss(
                            x_root_learnable, gt_tree,
                            mask_code=mask_code_learnable,
                            mask_feature=mask_features[i],
                            scan_geo=scan_geos[i][None, ...],
                            encoder_features=encoder_features[i],
                            rotation=rotations[i],
                            scan_sdf=scan_sdfs[i],
                            predicted_tree=initial_predicted_tree
                        )
                        child_feats_init = child_feats_exist[None, ...].detach()

                        x_root_learnable.detach().requires_grad_(True)
                        x_root_learnable.requires_grad = True
                        mask_code_learnable.detach().requires_grad_(True)
                        mask_code_learnable.requires_grad = True
                        child_feats_learnable = child_feats_init[0].detach().requires_grad_(True)
                        child_feats_learnable.requires_grad = True
                        opt = optim.Adam([x_root_learnable, mask_code_learnable, child_feats_learnable], lr=self.config['learning_rate'],
                                         weight_decay=self.config['weight_decay'])
                    else:
                        predicted_tree = self.decoder_online.inference_from_latents(x_root_learnable,
                                                                                    child_feats_learnable,
                                                                                    mask_code=mask_code_learnable,
                                                                                    mask_feature=mask_features[i],
                                                                                    children_names=children_names)
                        predicted_trees += [predicted_tree]

                object_losses = self.decoder_online.tto_leaves_loss(x_root_learnable, child_feats_learnable,
                                                                    mask_code=mask_code_learnable,
                                                                    mask_feature=mask_features[i],
                                                                    scan_geo=scan_geos[i][None, ...],
                                                                    scan_sdf=scan_sdfs[i],
                                                                    predicted_tree=initial_predicted_tree
                                                                    )

                all_losses += [object_losses]
                child_feats_detached = np.array([x.detach().cpu().numpy() for x in child_feats_learnable])
                all_child_feats_detached += [child_feats_detached]
                all_x_roots += [x_root_learnable.detach().cpu().numpy()]

                gt_tree.to('cpu').detach()

            losses = {'geo': 0,
                      'geo_children': 0,
                      'tv': 0}

            for i, object_losses in enumerate(all_losses):
                for loss_name, loss in object_losses.items():
                    losses[loss_name] = losses[loss_name] + loss
            for loss_name in losses:
                losses[loss_name] /= len(all_losses)

            losses_detached = {}
            for key in losses:
                if not isinstance(losses[key], float):
                    losses_detached[key] = float(np.squeeze(losses[key].detach().cpu().numpy()))
                else:
                    losses_detached[key] = losses[key]
            all_losses_detached += [losses_detached]
            if j % 20 == 0:
                print('Geo loss:', losses_detached['geo'])
                print('Geo children loss:', losses_detached['geo_children'])
                print()

            for key in losses:
                losses[key] *= self.config['loss_weight_' + key]
            total_loss = 0
            for loss_name, loss in losses.items():
                total_loss += loss

            total_loss.backward()
            opt.step()
            opt.zero_grad()

        initial_predicted_tree.to('cpu')
        del initial_predicted_tree
        output = (predicted_trees, all_losses_detached, all_child_feats_detached, all_x_roots)

        gc.collect()

        return output

