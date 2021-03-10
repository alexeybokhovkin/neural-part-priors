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

from src.utils.losses_simclr import NTXentLoss, NTXentLossChildren, NTXentLossChildrenFmaps
from src.utils.losses_simclr import NTXentLossClean, NTXentLossChildrenClean, NTXentLossChildrenFmapsClean
from src.utils.losses_simclr import NTXentLossChildrenPartwise, NTXentLossChildrenFmapsPartwise
from src.utils.losses_simclr import NTXentLossCleanProj, NTXentLossChildrenCleanProj


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

        with open(os.path.join(config['base'], config['checkpoint_dir'], config['model'], config['version'],
                               'config.json'), 'w') as f:
            json.dump(self.config, f)

        self.root_contrastive_loss = NTXentLossCleanProj(torch.device("cuda:0"), config['batch_size'], 0.5, True)
        self.children_contrastive_loss = NTXentLossChildrenCleanProj(torch.device("cuda:0"), config['batch_size'], 0.5)
        self.children_contrastive_fmaps_loss = NTXentLossChildrenFmapsClean(torch.device("cuda:0"), config['batch_size'], 0.5)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        # optimizer = optim.SGD(self.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'], momentum=0.9)
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

        scannet_geos_pos = batch[7]
        shape_mask_pos = batch[8]
        gt_trees_pos = batch[9]

        x_roots = []
        mask_codes, mask_features = [], []
        encoder_features = []

        x_roots_pos = []
        mask_codes_pos, mask_features_pos = [], []
        encoder_features_pos = []

        for i, mask in enumerate(masks):
            x_root, feature = self.encoder.root_latents(scannet_geos[i][None, ...])
            encoder_features += [feature]

            x_root_pos, feature_pos = self.encoder.root_latents(scannet_geos_pos[i][None, ...])
            encoder_features_pos += [feature_pos]
            if self.config['encode_mask']:
                mask_code, mask_feature = self.mask_encoder.root_latents(scannet_geos[i][None, ...])
                mask_codes += [mask_code]
                mask_features += [mask_feature]

                mask_code_pos, mask_feature_pos = self.mask_encoder.root_latents(scannet_geos_pos[i][None, ...])
                mask_codes_pos += [mask_code_pos]
                mask_features_pos += [mask_feature_pos]
            else:
                mask_codes += [None]
                mask_features += [None]
                mask_codes_pos += [None]
                mask_features_pos += [None]

            x_roots += [x_root]
            x_roots_pos += [x_root_pos]

        all_losses = []

        all_child_feats = []
        all_child_sem_logits = []
        all_child_decoder_features = []

        all_child_feats_pos = []
        all_child_decoder_features_pos = []

        for i, x_root in enumerate(x_roots):
            cuda_device = x_root.get_device()
            gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))
            gt_tree_pos = gt_trees_pos[i][0].to("cuda:{}".format(cuda_device))

            output = self.decoder.structure_recon_loss(x_root, gt_tree,
                                                       mask_code=mask_codes[i],
                                                       mask_feature=mask_features[i],
                                                       scan_geo=scannet_geos[i][None, ...],
                                                       encoder_features=encoder_features[i],
                                                       rotation=rotations[i]
                                                       )

            output_pos = self.decoder.structure_recon_loss(x_root_pos, gt_tree_pos,
                                                       mask_code=mask_codes_pos[i],
                                                       mask_feature=mask_features_pos[i],
                                                       scan_geo=scannet_geos_pos[i][None, ...],
                                                       encoder_features=encoder_features_pos[i],
                                                       rotation=rotations[i]
                                                       )
            object_losses = output[0]
            all_losses += [object_losses]

            child_feats = output[4][0]             # Tensor[n_parts, 128]
            child_sem_logits = output[5]           # list<Tensor[50]>
            child_decoder_features = output[6]     # list<list<Tensor[fmap]>>
            all_child_feats.append(child_feats)
            all_child_decoder_features.append(child_decoder_features)

            child_feats_pos = output_pos[4][0]  # Tensor[n_parts, 128]
            child_decoder_features_pos = output_pos[6]  # list<list<Tensor[fmap]>>
            all_child_feats_pos.append(child_feats_pos)
            all_child_decoder_features_pos.append(child_decoder_features_pos)

        all_child_decoder_features_reshaped = [[[] for i in range(len(all_child_decoder_features))] for k in range(3)]
        for i in range(len(all_child_decoder_features)):
            for j in range(len(all_child_decoder_features[i])):
                for k in range(len(all_child_decoder_features[i][j]) - 1):
                    all_child_decoder_features_reshaped[k][i].append(all_child_decoder_features[i][j][k])
        all_child_decoder_features_pos_reshaped = [[[] for i in range(len(all_child_decoder_features_pos))] for k in range(3)]
        for i in range(len(all_child_decoder_features_pos)):
            for j in range(len(all_child_decoder_features_pos[i])):
                for k in range(len(all_child_decoder_features_pos[i][j]) - 1):
                    all_child_decoder_features_pos_reshaped[k][i].append(all_child_decoder_features_pos[i][j][k])

        loss_contrastive_root = self.root_contrastive_loss(torch.cat(x_roots), torch.cat(x_roots_pos), partnet_ids) # torch.Tensor[bsize, 128]
        loss_contrastive_children = self.children_contrastive_loss(all_child_feats, all_child_feats_pos, partnet_ids)
        loss_contrastive_children_fmaps = self.children_contrastive_fmaps_loss(all_child_decoder_features_reshaped, all_child_decoder_features_pos_reshaped, partnet_ids)

        losses = {'geo': 0,
                  'geo_prior': 0,
                  'leaf': 0,
                  'exists': 0,
                  'semantic': 0,
                  'edge_exists': 0,
                  'root_cls': 0,
                  'rotation': 0}

        if self.config['use_contrastive_constraint']:
            losses['contrastive_constraint'] = loss_contrastive_root
            losses['contrastive_children'] = loss_contrastive_children
            losses['contrastive_children_fmaps'] = loss_contrastive_children_fmaps
        else:
            losses['contrastive_constraint'] = 0
            losses['contrastive_children'] = 0
            losses['contrastive_children_fmaps'] = 0

        for i, object_losses in enumerate(all_losses):
            for loss_name, loss in object_losses.items():
                losses[loss_name] = losses[loss_name] + loss
        for loss_name in losses:
            losses[loss_name] /= len(all_losses)

        del all_losses, output_pos, output, all_child_feats, all_child_decoder_features, all_child_feats_pos, \
            all_child_decoder_features_pos, all_child_decoder_features_reshaped, all_child_decoder_features_pos_reshaped, \
            gt_tree, gt_tree_pos, x_roots, mask_codes, mask_features, encoder_features

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
        children_roots_all = []
        mask_codes, mask_features = [], []
        encoder_features = []
        for i, mask in enumerate(masks):
            cuda_device = mask.get_device()
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
            object_losses, all_geos, all_leaf_geos, all_S_priors, child_feats, _, child_decoder_features, _ = self.decoder.structure_recon_loss(x_root, gt_tree,
                                                                                                    mask_code=mask_codes[i],
                                                                                                    mask_feature=mask_features[i],
                                                                                                    scan_geo=scannet_geos[i][None, ...],
                                                                                                    encoder_features=encoder_features[i],
                                                                                                    rotation=rotations[i]
                                                                                                    )

            predicted_tree, S_priors, pred_rotation, child_nodes = self.decoder(x_root,
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
        shape_sdfs = torch.cat(batch[1]).unsqueeze(dim=1)
        shape_mask = torch.cat(batch[2]).unsqueeze(dim=1)
        gt_trees = batch[3]
        partnet_ids = batch[4]
        rotations = batch[5]
        latents = batch[7]

        batch[8] = [x[None, ...] for x in batch[8]]
        scannet_geos_pos = torch.cat(batch[8]).unsqueeze(dim=1)
        shape_mask_pos = torch.cat(batch[9]).unsqueeze(dim=1)
        gt_trees_pos = batch[10]

        input_batch = tuple([scannet_geos, shape_sdfs, shape_mask, gt_trees, partnet_ids, rotations, latents,
                             scannet_geos_pos, shape_mask_pos, gt_trees_pos,
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
        print('contrastive_constraint:', losses['contrastive_constraint'])
        print('contrastive_children:', losses['contrastive_children'])
        print('contrastive_children_fmaps:', losses['contrastive_children_fmaps'])
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
        shape_sdfs = torch.cat(batch[1]).unsqueeze(dim=1)
        shape_mask = torch.cat(batch[2]).unsqueeze(dim=1)
        gt_trees = batch[3]
        partnet_ids = batch[4]
        rotations = batch[5]
        latents = batch[7]

        batch[8] = [x[None, ...] for x in batch[8]]
        scannet_geos_pos = torch.cat(batch[8]).unsqueeze(dim=1)
        shape_mask_pos = torch.cat(batch[9]).unsqueeze(dim=1)
        gt_trees_pos = batch[10]

        input_batch = tuple([scannet_geos, shape_sdfs, shape_mask, gt_trees, partnet_ids, rotations, latents,
                       scannet_geos_pos, shape_mask_pos, gt_trees_pos])

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

    def inference_from_root(self, batch, x_root_):
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

        all_losses = []
        predicted_trees = []
        all_priors = []
        all_leaves_geos = []
        pred_rotations = []
        all_child_nodes = []

        all_child_feats = []
        all_child_decoder_features = []

        x_root_ = torch.FloatTensor(x_root_).to(cuda_device)

        for i, x_root in enumerate(x_roots):
            cuda_device = x_root.get_device()
            gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))
            mask = masks[i]
            object_losses, all_geos, all_leaf_geos, all_S_priors, child_feats, _, child_decoder_features, _ = self.decoder.structure_recon_loss(x_root_, gt_tree,
                                                                                                    mask_code=mask_codes[i],
                                                                                                    mask_feature=mask_features[i],
                                                                                                    scan_geo=scannet_geos[i][None, ...],
                                                                                                    encoder_features=encoder_features[i],
                                                                                                    rotation=rotations[i]
                                                                                                    )

            predicted_tree, S_priors, pred_rotation, child_nodes = self.decoder(x_root_,
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

    def tto_root(self, batch, iterations=1000):

        opt = optim.Adam(self.decoder.parameters(), lr=self.config['learning_rate'],
                         weight_decay=self.config['weight_decay'])

        Tree.load_category_info(os.path.join(self.config['datadir'], self.config['dataset']))

        scannet_geos = batch[0]
        sdfs = batch[1]
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
                x_root, feature = self.encoder.root_latents(scannet_geos[i][None, ...])
                feature_detached = [x.clone().detach() for x in feature]
                encoder_features += [feature_detached]
                if self.config['encode_mask']:
                    mask_code, mask_feature = self.mask_encoder.root_latents(scannet_geos[i][None, ...])
                    mask_code_detached = mask_code.clone().detach()
                    mask_feature_detached = [x.clone().detach() for x in mask_feature]
                    mask_codes += [mask_code_detached]
                    mask_features += [mask_feature_detached]
                else:
                    mask_codes += [None]
                    mask_features += [None]
                x_root_detached = x_root.clone().detach()
                x_roots += [x_root_detached]

        predicted_trees = []
        all_losses_detached = []
        all_child_feats_detached = []

        for j in range(iterations):

            children_latent_loss_full = 0
            children_geo_sum_loss_full = 0
            all_losses = []
            for i, x_root in enumerate(x_roots):

                cuda_device = x_root.get_device()
                gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))

                with torch.no_grad():
                    predicted_tree, S_priors, pred_rotation, child_nodes = self.decoder(x_root,
                                                                                        mask_code=mask_codes[i],
                                                                                        mask_feature=mask_features[i],
                                                                                        scan_geo=scannet_geos[i][None, ...],
                                                                                        full_label=gt_tree.root.label,
                                                                                        encoder_features=
                                                                                        encoder_features[i],
                                                                                        rotation=rotations[i],
                                                                                        gt_tree=gt_tree
                                                                                        )
                    predicted_tree.to('cpu').detach()
                    predicted_trees += [predicted_tree]

                    if j == 0:
                        object_losses, all_geos, all_leaf_geos_init, all_S_priors, child_feats_init, _, child_decoder_features, _ = self.decoder.tto_recon_loss(
                            x_root, gt_tree,
                            mask_code=mask_codes[i],
                            mask_feature=mask_features[i],
                            scan_geo=scannet_geos[i][None, ...],
                            encoder_features=encoder_features[i],
                            rotation=rotations[i]
                        )
                        child_feats_init = [x.detach() for x in child_feats_init]
                        all_leaf_geos_init = [x.detach() for x in all_leaf_geos_init]

                object_losses, all_geos, all_leaf_geos, all_S_priors, child_feats, _, child_decoder_features, _ = self.decoder.tto_recon_loss(
                    x_root, gt_tree,
                    mask_code=mask_codes[i],
                    mask_feature=mask_features[i],
                    scan_geo=scannet_geos[i][None, ...],
                    encoder_features=encoder_features[i],
                    rotation=rotations[i]
                    )
                all_losses += [object_losses]
                child_feats_detached = np.array([x.detach().cpu().numpy() for x in child_feats])
                all_child_feats_detached += [child_feats_detached]

                children_latent_loss = self.decoder.children_mse_loss(child_feats, child_feats_init)
                children_latent_loss_full += children_latent_loss

                children_geo_sum_loss = self.decoder.children_geo_sum_loss(all_leaf_geos, all_leaf_geos_init)
                children_geo_sum_loss_full += children_geo_sum_loss

                gt_tree.to('cpu').detach()
                x_root.to('cpu')

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
                      'children_geo_sum': children_geo_sum_loss_full}

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
                print(losses_detached['geo'])

            for key in losses:
                losses[key] *= self.config['loss_weight_' + key]
            total_loss = 0
            for loss_name, loss in losses.items():
                total_loss += loss

            total_loss.backward()
            opt.step()
            opt.zero_grad()

        output = (predicted_trees, all_losses_detached, all_child_feats_detached)

        gc.collect()

        return output
