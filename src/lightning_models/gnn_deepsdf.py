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

from src.models.gnn_models import GeoEncoder, HierarchicalDeepSDFDecoder
from src.data_utils.hierarchy import Tree
from src.datasets.partnet import generate_gnn_deepsdf_datasets, generate_gnn_deepsdf_scannet_datasets
from src.utils.gnn import collate_feats
from deep_sdf_utils.optimizer import StepLearningRateSchedule


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

        # datasets = generate_gnn_deepsdf_datasets(**config)
        datasets = generate_gnn_deepsdf_scannet_datasets(**config)
        self.train_dataset = datasets['train']
        self.val_dataset = datasets['val']
        self.num_parts = self.train_dataset.num_parts
        self.num_shapes = self.train_dataset.num_shapes

        self.config['parts_to_ids_path'] = os.path.join(config['datadir'], config['dataset'],
                                                        config['parts_to_ids_path'])

        Tree.load_category_info(os.path.join(config['datadir'], config['dataset']))
        self.encoder = GeoEncoder(**config)
        self.decoder = HierarchicalDeepSDFDecoder(**config,
                                                  num_parts=self.num_parts,
                                                  num_shapes=self.num_shapes)

        # ===================================================== #

        # Make filtered model loading
        # pretrained_dict = torch.load(config['resume_from_checkpoint'])
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs/GNNPartnet/deepsdf_parts_geoscan_pointcls_1/checkpoints/42.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs/GNNPartnet/deepsdf_parts_geoscan_pointcls_df_1/checkpoints/36.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs/GNNPartnet/deepsdf_parts_fullshape_geoscan_pointcls_1/checkpoints/39.ckpt')

        # pretrained_dict = torch.load(config['resume_from_checkpoint'])
        pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf/GNNPartnet/deepsdf_parts_fullshape_geoscan_2_2_overfit100_fixed_1_l2_latent_1/checkpoints/75.ckpt')
        encoder_dict = self.encoder.state_dict()
        encoder_dict_update = {k[8:]: v for k, v in pretrained_dict['state_dict'].items()
                               if k[8:] in encoder_dict and
                               pretrained_dict['state_dict'][k].shape == encoder_dict[k[8:]].shape}
        encoder_dict.update(encoder_dict_update)
        print('Updated keys (encoder):', len(encoder_dict_update))
        print('Not loaded keys:', list(set(encoder_dict.keys()) - set(encoder_dict_update.keys())))
        self.encoder.load_state_dict(encoder_dict)

        # pretrained_dict = torch.load(config['resume_from_checkpoint'])
        pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf/GNNPartnet/deepsdf_parts_fullshape_geoscan_2_2_overfit100_fixed_1_l2_latent_1/checkpoints/75.ckpt')
        decoder_dict = self.decoder.state_dict()
        decoder_dict_update = {k[8:]: v for k, v in pretrained_dict['state_dict'].items()
                               if k[8:] in decoder_dict and
                               pretrained_dict['state_dict'][k].shape == decoder_dict[k[8:]].shape}
        decoder_dict.update(decoder_dict_update)
        print('Updated keys (decoder):', len(decoder_dict_update))
        print('Not loaded keys:', list(set(decoder_dict.keys()) - set(decoder_dict_update.keys())))
        self.decoder.load_state_dict(decoder_dict)

        self.load_deepsdf_outside(
            '/rhome/abokhovkin/projects/DeepSDF/experiments/full_experiments/chair_full',
            '/rhome/abokhovkin/projects/DeepSDF/experiments/full_experiments/chair_parts'
        )

        # ===================================================== #

        # Load full shape decoder optionally
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs/GNNPartnet/deepsdf_parts_fullshape_geoscan_pointcls_2/checkpoints/27.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs/GNNPartnet/deepsdf_parts_fullshape_geoscan_pointcls_2/checkpoints/27.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs/GNNPartnet/deepsdf_parts_fullshape_geoscan_pointcls_1/checkpoints/39.ckpt')
        # decoder_dict = self.decoder.state_dict()
        # decoder_dict_update_fullshape = {k[8:]: v for k, v in pretrained_dict['state_dict'].items()
        #                                  if 'decoder.recursive_decoder.deepsdf_shape_decoder' in k}
        # decoder_dict.update(decoder_dict_update_fullshape)
        # self.decoder.load_state_dict(decoder_dict)

        # self.decoder.recursive_decoder.map_part_embeddings(
        #     '/cluster/pegasus/abokhovkin/scannet-relationships/logs/GNNPartnet/deepsdf_parts_geoscan_pointcls_1/latents/42.pth',
        #     '/cluster/pegasus/abokhovkin/scannet-relationships/logs/GNNPartnet/deepsdf_parts_fullshape_geoscan_pointcls_2/latents/27_full.pth'
        # )

        # random.seed(config['manual_seed'])
        # np.random.seed(config['manual_seed'])
        # torch.manual_seed(config['manual_seed'])
        # torch.cuda.manual_seed(config['manual_seed'])
        # torch.backends.cudnn.enabled = False
        # torch.backends.cudnn.deterministic = True

        with open(os.path.join(config['base'], config['checkpoint_dir'], config['model'], config['version'],
                               'config.json'), 'w') as f:
            json.dump(self.config, f)

    def load_deepsdf_outside(self, shape_exp_path=None, parts_exp_path=None):
        shape_dict = self.decoder.recursive_decoder.deepsdf_shape_decoder.state_dict()
        parts_dict = self.decoder.recursive_decoder.deepsdf_decoder.state_dict()

        shape_dict_trained = torch.load(os.path.join(shape_exp_path, 'ModelParameters/latest.pth'))['model_state_dict']
        shape_latents_dict_trained = torch.load(os.path.join(shape_exp_path, 'LatentCodes/latest.pth'))['latent_codes']['weight']
        parts_dict_trained = torch.load(os.path.join(parts_exp_path, 'ModelParameters/latest.pth'))['model_state_dict']
        parts_latents_dict_trained = torch.load(os.path.join(parts_exp_path, 'LatentCodes/latest.pth'))['latent_codes']['weight']

        shape_dict_update = {k[7:]: v for k, v in shape_dict_trained.items()}
        parts_dict_update = {k[7:]: v for k, v in parts_dict_trained.items()}

        print(self.decoder.recursive_decoder.lat_shape_vecs.weight.data.shape, shape_latents_dict_trained.shape)
        self.decoder.recursive_decoder.lat_shape_vecs.weight.data = shape_latents_dict_trained
        self.decoder.recursive_decoder.lat_vecs.weight.data = parts_latents_dict_trained
        shape_dict.update(shape_dict_update)
        self.decoder.recursive_decoder.deepsdf_shape_decoder.load_state_dict(shape_dict)
        parts_dict.update(parts_dict_update)
        self.decoder.recursive_decoder.deepsdf_decoder.load_state_dict(parts_dict)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        # optimizer = optim.Adam(
        #     [
        #         {
        #             "params": list(self.decoder.recursive_decoder.deepsdf_decoder.parameters()) +
        #                       list(self.decoder.recursive_decoder.deepsdf_shape_decoder.parameters()),
        #             "lr": self.config['param_learning_rate'],
        #         },
        #         {
        #             "params": list(self.decoder.recursive_decoder.lat_vecs.parameters()) +
        #                       list(self.decoder.recursive_decoder.lat_shape_vecs.parameters()),
        #             "lr": self.config['lat_learning_rate'],
        #         },
        #     ]
        # )
        # optimizer = optim.SGD(self.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
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
        sdf_filenames = batch[6]
        sdf_parts = batch[7]
        parts_indices = batch[8]
        full_shape_indices = batch[9]
        noise_full = batch[10]

        x_roots = []
        encoder_features = []

        for i, gt_tree in enumerate(gt_trees):
            x_root, feature = self.encoder.root_latents(scannet_geos[i][None, ...])
            encoder_features += [feature]

            x_roots += [x_root]

        # x_roots, features = self.encoder.root_latents(scannet_geos)
        # encoder_features = features

        all_losses = []

        for i, x_root in enumerate(x_roots):
            cuda_device = x_root.get_device()
            gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))

            # Pretraining
            # output = self.decoder.structure_recon_loss(x_roots[i], gt_tree, sdf_parts[i],
            #                                            encoder_features=encoder_features[i],
            #                                            rotation=0, parts_indices=parts_indices[i],
            #                                            full_shape_idx=full_shape_indices[i],
            #                                            epoch=self.current_epoch,
            #                                            noise_full=noise_full[i])

            # Learning latent space
            output = self.decoder.latent_recon_loss(x_roots[i], gt_tree, sdf_parts[i],
                                                    encoder_features=encoder_features[i],
                                                    rotation=0, parts_indices=parts_indices[i],
                                                    full_shape_idx=full_shape_indices[i],
                                                    epoch=self.current_epoch,
                                                    noise_full=noise_full[i])

            object_losses = output[0]
            all_losses += [object_losses]

        # all_losses = []
        # output = self.decoder.deepsdf_recon_loss(sdf_parts[0], parts_indices=parts_indices[0],
        #                                          full_shape_idx=full_shape_indices[0],
        #                                          epoch=self.current_epoch,
        #                                          noise_full=noise_full[0])
        # object_losses = output[0]
        # all_losses += [object_losses]

        losses = {'exists': 0,
                  'semantic': 0,
                  'edge_exists': 0,
                  'root_cls': 0,
                  'rotation': 0,
                  'sdf': 0,
                  'shape_sdf': 0,
                  'point_part': 0,
                  'eikonal_part': 0,
                  'eikonal_shape': 0,
                  'mse_shape': 0,
                  'mse_parts': 0}

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
        sdf_filenames = batch[7]
        sdf_parts = batch[8]
        parts_indices = batch[9]

        x_roots = []
        encoder_features = []
        for i, mask in enumerate(masks):
            cuda_device = mask.get_device()
            x_root, feature = self.encoder.root_latents(scannet_geos[i][None, ...])
            encoder_features += [feature]
            x_roots += [x_root]

        predicted_trees = []
        all_parts_sdfs_pred = []
        all_shapes_sdfs_pred = []

        for i, x_root in enumerate(x_roots):
            cuda_device = x_root.get_device()
            gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))

            output = self.decoder.forward(x_root,
                                          sdf_parts,
                                          full_label=gt_tree.root.label,
                                          encoder_features=encoder_features[i],
                                          rotation=rotations[i],
                                          gt_tree=gt_tree
                                          )

            predicted_tree = output[0]
            parts_sdfs_pred = output[1]
            pred_output = output[2]
            shape_sdf_pred = output[3]
            shape_output = output[4]

            predicted_trees += [predicted_tree]
            all_parts_sdfs_pred += [parts_sdfs_pred]
            all_shapes_sdfs_pred += [shape_sdf_pred]

        output = [predicted_trees, all_parts_sdfs_pred, all_shapes_sdfs_pred]
        output = tuple(output)

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
        sdf_filenames = batch[7]
        sdf_parts = batch[8]
        parts_indices = batch[9]
        full_shape_indices = batch[10]
        noise_full = batch[11]
        indices = batch[12]

        print('Indices:', indices)
        input_batch = tuple([scannet_geos, shape_sdfs, shape_mask, gt_trees, partnet_ids, rotations,
                             sdf_filenames, sdf_parts, parts_indices, full_shape_indices,
                             noise_full])

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

        print()
        print('exists:', losses['exists'])
        print('semantic:', losses['semantic'])
        print('edge_exists:', losses['edge_exists'])
        print('root_cls:', losses['root_cls'])
        print('rotation:', losses['rotation'])
        print('sdf:', losses['sdf'])
        print('shape_sdf', losses['shape_sdf'])
        print('point_part:', losses['point_part'])
        print('eikonal_part:', losses['eikonal_part'])
        print('eikonal_shape:', losses['eikonal_shape'])
        print('mse_shape:', losses['mse_shape'])
        print('mse_parts:', losses['mse_parts'])
        print('total_loss:', total_loss)
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

        if self.current_epoch % self.config['save_every'] == 0:
            latent_vec = self.decoder.get_latent_vecs()
            all_latents = latent_vec.state_dict()
            torch.save(
                {"epoch": self.current_epoch, "latent_codes": all_latents},
                os.path.join(self.config["checkpoint_dir"], self.config["model"], self.config["version"], 'latents', f'{self.current_epoch}.pth'),
            )

        if self.current_epoch % self.config['save_every'] == 0:
            latent_vec = self.decoder.get_latent_shape_vecs()
            all_latents = latent_vec.state_dict()
            torch.save(
                {"epoch": self.current_epoch, "latent_codes": all_latents},
                os.path.join(self.config["checkpoint_dir"], self.config["model"], self.config["version"], 'latents', f'{self.current_epoch}_full.pth'),
            )

        del outputs, log
        gc.collect()

    def validation_step(self, batch, batch_idx):
        total_loss = 0
        return {'val_loss': total_loss}

    def validation_epoch_end(self, outputs):
        self.log('val_loss', 0)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'],
                          shuffle=True, num_workers=self.config['num_workers'], drop_last=True,
                          collate_fn=collate_feats)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['batch_size'],
                          shuffle=False, num_workers=self.config['num_workers'], drop_last=True,
                          collate_fn=collate_feats)

    def tto(self, batch, index=0):
        scannet_geos = batch[0]
        sdfs = batch[1]
        masks = batch[2]
        gt_trees = batch[3]
        partnet_ids = batch[4]
        rotations = batch[5]
        sdf_filenames = batch[7]
        sdf_parts = batch[8]
        parts_indices = batch[9]

        x_roots = []
        encoder_features = []
        for i, mask in enumerate(masks):
            cuda_device = mask.get_device()
            x_root, feature = self.encoder.root_latents(scannet_geos[i][None, ...])
            encoder_features += [feature]
            x_roots += [x_root]

        predicted_trees = []
        all_parts_sdfs_pred = []
        all_shapes_sdfs_pred = []
        all_parts_sdfs_pred_tto = []
        all_shapes_sdfs_pred_tto = []

        for i, x_root in enumerate(x_roots):
            cuda_device = x_root.get_device()
            gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))

            output = self.decoder.forward(x_root,
                                          sdf_parts,
                                          full_label=gt_tree.root.label,
                                          encoder_features=encoder_features[i],
                                          rotation=rotations[i],
                                          gt_tree=gt_tree,
                                          index=index
                                          )

            predicted_tree = output[0]
            parts_sdfs_pred = output[1]
            pred_output = output[2]
            shape_sdf_pred = output[3]
            shape_output = output[4]

            predicted_trees += [predicted_tree]
            all_parts_sdfs_pred += [parts_sdfs_pred]
            all_shapes_sdfs_pred += [shape_sdf_pred]

            parts_sdfs_pred_tto, shape_sdf_pred_tto = self.decoder.tto(pred_output, shape_output)
            all_parts_sdfs_pred_tto += [parts_sdfs_pred_tto]
            all_shapes_sdfs_pred_tto += [shape_sdf_pred_tto]

        output = [predicted_trees, all_parts_sdfs_pred, all_shapes_sdfs_pred, all_parts_sdfs_pred_tto, all_shapes_sdfs_pred_tto]
        output = tuple(output)

        return output

    def tto_two_stage(self, batch, index=0):
        scannet_geos = batch[0]
        sdfs = batch[1]
        masks = batch[2]
        gt_trees = batch[3]
        partnet_ids = batch[4]
        rotations = batch[5]
        sdf_filenames = batch[7]
        sdf_parts = batch[8]
        parts_indices = batch[9]
        full_shape_indices = batch[10]
        noise_full = batch[11]

        x_roots = []
        encoder_features = []
        for i, mask in enumerate(masks):
            cuda_device = mask.get_device()
            x_root, feature = self.encoder.root_latents(scannet_geos[i][None, ...])
            encoder_features += [feature]
            x_roots += [x_root]

        predicted_trees = []
        all_parts_sdfs_pred = []
        all_shapes_sdfs_pred = []
        all_parts_sdfs_pred_tto = []
        all_shapes_sdfs_pred_tto = []

        for i, x_root in enumerate(x_roots):
            cuda_device = x_root.get_device()
            gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))

            output = self.decoder.forward_two_stage(x_root,
                                                    sdf_parts,
                                                    full_label=gt_tree.root.label,
                                                    encoder_features=encoder_features[i],
                                                    rotation=rotations[i],
                                                    gt_tree=gt_tree,
                                                    index=index,
                                                    parts_indices=parts_indices,
                                                    full_shape_idx=full_shape_indices,
                                                    noise_full=noise_full
                                                    )

            predicted_tree = output[0]
            parts_sdfs_pred = output[1]
            pred_output = output[2]
            shape_sdf_pred = output[3]
            shape_output = output[4]

            predicted_trees += [predicted_tree]
            all_parts_sdfs_pred += [parts_sdfs_pred]
            all_shapes_sdfs_pred += [shape_sdf_pred]

            parts_sdfs_pred_tto, shape_sdf_pred_tto, \
            parts_stats_before, parts_stats_after, shape_stats_before_tto, shape_stats_after_tto \
                = self.decoder.tto(pred_output, shape_output)
            tto_stats = (parts_stats_before, parts_stats_after, shape_stats_before_tto, shape_stats_after_tto)

            all_parts_sdfs_pred_tto += [parts_sdfs_pred_tto]
            all_shapes_sdfs_pred_tto += [shape_sdf_pred_tto]

        output = [predicted_trees, all_parts_sdfs_pred, all_shapes_sdfs_pred, all_parts_sdfs_pred_tto,
                  all_shapes_sdfs_pred_tto, tto_stats]
        output = tuple(output)

        return output
