import os
import json
from argparse import Namespace
import numpy as np
from scipy.ndimage import rotate
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.models.gnn_models import GeoEncoder, HierarchicalDeepSDFDecoder
from src.utils.hierarchy import Tree
from src.datasets.partnet import generate_gnn_deepsdf_datasets, generate_gnn_deepsdf_scannet_datasets
from src.utils.gnn import collate_feats


class GNNPartnetLightning(pl.LightningModule):

    def __init__(self, hparams, cat_name='chair', mode='training'):
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

        if 'cat_name' in self.config:
            cat_name = self.config['cat_name']

        self.cat_name = cat_name
        self.mode = mode

        if cat_name == 'chair':
            class2id = {
                'chair_arm_left': 0,
                'chair_arm_right': 1,
                'chair_back': 2,
                'chair_seat': 3,
                'regular_leg_base': 4,
                'star_leg_base': 5,
                'surface_base': 6
            }
        elif cat_name == 'bed':
            class2id = {
                'bed_frame_base': 0,
                'bed_side_surface': 1,
                'bed_sleep_area': 2,
                'headboard': 3,
            }
        elif cat_name == 'storagefurniture':
            class2id = {
                'cabinet_door': 0,
                'shelf': 1,
                'cabinet_frame': 2,
                'cabinet_base': 3,
                'countertop': 4
            }
        elif cat_name == 'trashcan':
            class2id = {
                'base': 0,
                'container_bottom': 1,
                'container_box': 2,
                'cover': 3,
                'other': 4
            }
        elif cat_name == 'table':
            class2id = {
                'central_support': 0,
                'drawer': 1,
                'leg': 2,
                'pedestal': 3,
                'shelf': 4,
                'table_surface': 5,
                'vertical_side_panel': 6
            }
        self.class2id = class2id

        datasets = generate_gnn_deepsdf_scannet_datasets(**config)
        self.train_dataset = datasets['train']
        self.val_dataset = datasets['val']
        self.num_parts = self.train_dataset.num_parts
        self.num_shapes = self.train_dataset.num_shapes

        if 'scene_aware_points_path' in config:
            self.scene_aware_points_path = config['scene_aware_points_path']
        else:
            self.scene_aware_points_path = None

        Tree.load_category_info(os.path.join(config['datadir'], config['dataset']))
        self.encoder = GeoEncoder(**config)
        self.decoder = HierarchicalDeepSDFDecoder(**config,
                                                  num_parts=self.num_parts,
                                                  num_shapes=self.num_shapes,
                                                  cat_name=cat_name,
                                                  class2id=self.class2id,
                                                  scene_aware_points_path=self.scene_aware_points_path)

        # mlcvnet v2
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn_hardnoise_partialpoints/checkpoints/60.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn2_hardnoise_partialpoints/checkpoints/30.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_table_entropypn_hardnoise_partialpoints/checkpoints/80.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_table_entropypn2_hardnoise_partialpoints/checkpoints/140.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_storagefurniture_entropypn_hardnoise_partialpoints/checkpoints/80.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_storagefurniture_entropypn2_hardnoise_partialpoints/checkpoints/80.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_bed_entropypn_hardnoise_partialpoints/checkpoints/120.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_bed_entropypn2_hardnoise_partialpoints/checkpoints/120.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_trashcan_entropypn_hardnoise_partialpoints/checkpoints/160.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_trashcan_entropypn2_hardnoise_partialpoints/checkpoints/80.ckpt'
        # )

        if mode in ['finetune', 'tto']:
            # LOAD ENCODER
            self.encoder_ckpt_path = self.config['encoder_ckpt_path']
            # self.encoder_ckpt_path = '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn_hardnoise_partialpoints/checkpoints/60.ckpt'
            self.load_model_encoder(self.encoder_ckpt_path)

        # mlcvnet v2
        pretrained_dict = torch.load(
            '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn_hardnoise_partialpoints/checkpoints/60.ckpt'
        )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn2_hardnoise_partialpoints/checkpoints/30.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_table_entropypn_hardnoise_partialpoints/checkpoints/80.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_table_entropypn2_hardnoise_partialpoints/checkpoints/140.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_storagefurniture_entropypn_hardnoise_partialpoints/checkpoints/80.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_storagefurniture_entropypn2_hardnoise_partialpoints/checkpoints/80.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_bed_entropypn_hardnoise_partialpoints/checkpoints/120.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_bed_entropypn2_hardnoise_partialpoints/checkpoints/120.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_trashcan_entropypn_hardnoise_partialpoints/checkpoints/160.ckpt'
        # )
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_trashcan_entropypn2_hardnoise_partialpoints/checkpoints/80.ckpt'
        # )

        if mode in ['finetune', 'tto']:
            # LOAD DECODER
            self.decoder_ckpt_path = self.config['decoder_ckpt_path']
            # self.decoder_ckpt_path = '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_mlcvnet/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn_hardnoise_partialpoints/checkpoints/60.ckpt'
            self.load_model_decoder(self.decoder_ckpt_path)

        self.deepsdf_shape_path = self.config['deepsdf_shape_path']
        self.deepsdf_parts_path = self.config['deepsdf_parts_path']
        self.load_deepsdf_outside(
            self.deepsdf_shape_path,
            self.deepsdf_parts_path
            # f'/cluster/daidalos/abokhovkin/DeepSDF_v2/full_experiments_v2/{cat_name}_full_surface_pe',
            # f'/cluster/daidalos/abokhovkin/DeepSDF_v2/full_experiments_v2/{cat_name}_parts_onehot_pe'
        )

        if mode in ['training', 'finetune']:
            # save config file
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

        self.decoder.recursive_decoder.lat_shape_vecs.weight.data = shape_latents_dict_trained
        self.decoder.recursive_decoder.lat_vecs.weight.data = parts_latents_dict_trained
        shape_dict.update(shape_dict_update)
        self.decoder.recursive_decoder.deepsdf_shape_decoder.load_state_dict(shape_dict)
        parts_dict.update(parts_dict_update)
        self.decoder.recursive_decoder.deepsdf_decoder.load_state_dict(parts_dict)

    def load_model_encoder(self, ckpt_path):
        pretrained_dict = torch.load(ckpt_path)
        encoder_dict = self.encoder.state_dict()
        encoder_dict_update = {k[8:]: v for k, v in pretrained_dict['state_dict'].items()
                               if k[8:] in encoder_dict and
                               pretrained_dict['state_dict'][k].shape == encoder_dict[k[8:]].shape}
        encoder_dict.update(encoder_dict_update)
        print('Updated keys (encoder):', len(encoder_dict_update))
        print('Not loaded keys:', list(set(encoder_dict.keys()) - set(encoder_dict_update.keys())))
        self.encoder.load_state_dict(encoder_dict)

    def load_model_decoder(self, ckpt_path):
        pretrained_dict = torch.load(ckpt_path)
        decoder_dict = self.decoder.state_dict()
        decoder_dict_update = {k[8:]: v for k, v in pretrained_dict['state_dict'].items()
                               if k[8:] in decoder_dict and
                               pretrained_dict['state_dict'][k].shape == decoder_dict[k[8:]].shape}
        decoder_dict.update(decoder_dict_update)
        print('Updated keys (decoder):', len(decoder_dict_update))
        print('Not loaded keys:', list(set(decoder_dict.keys()) - set(decoder_dict_update.keys())))
        self.decoder.load_state_dict(decoder_dict)

    def configure_optimizers(self):
        lr = self.config['learning_rate'] if self.mode == 'training' else 0.0002
        optimizer = optim.Adam(self.parameters(),
                               lr=lr, 
                               weight_decay=self.config['weight_decay']
                               )
        step_size = self.config['decay_every'] if self.mode == 'training' else 40
        scheduler = StepLR(optimizer,
                           gamma=self.config['gamma'],
                           step_size=step_size,
                           )
        return [optimizer], [scheduler]

    def forward(self, batch):

        def rotate_voxels(geos, rots):
            rots = [x * 30 for x in rots]
            num_geos = geos.shape[0]
            rotated_geos = []
            for i in range(num_geos):
                geo_np = geos[i, 0].cpu().numpy().astype('uint8')
                rotated_scan_geo = rotate(geo_np, rots[i], axes=[0, 2], reshape=False).astype('float32')
                rotated_scan_geo = torch.FloatTensor(rotated_scan_geo)[None, ...]
                rotated_geos += [rotated_scan_geo]
            rotated_geos = torch.stack(rotated_geos)
            return rotated_geos

        def scale_points(all_points, noise_full, s):
            transformed_points = []
            transformed_noise = []
            for i, points in enumerate(all_points):
                noise = noise_full[i]
                points[:, :, :3] = points[:, :, :3] * s[i]
                noise[:, :3] = noise[:, :3] * s[i]
                transformed_points += [points]
                transformed_noise += [noise]
            return transformed_points, transformed_noise

        def rotate_points(all_points, noise_full, rots):
            rots = [x * 30 for x in rots]
            transformed_points = []
            transformed_noise = []
            for i, points in enumerate(all_points):
                noise = noise_full[i]
                rot = rots[i]
                angle = np.pi * rot / 180.

                a, b = np.cos(angle), np.sin(angle)
                matrix = np.array([[a, 0, b],
                                   [0, 1, 0],
                                   [-b, 0, a]])
                matrix = torch.FloatTensor(matrix).cuda()
                for j in range(len(points)):
                    points[j, :, :3] = points[j, :, :3] @ matrix.T
                noise[:, :3] = noise[:, :3] @ matrix.T
                transformed_points += [points]
                transformed_noise += [noise]
            return transformed_points, transformed_noise

        def jitter_points(all_points, noise_full):

            transformed_points = []
            transformed_noise = []
            for i, points in enumerate(all_points):
                noise = noise_full[i]
                rand_vector = (torch.rand((3)) - 0.5) / 4
                rand_vector = rand_vector.cuda()
                for j in range(len(points)):
                    points[j, :, :3] = points[j, :, :3] + rand_vector[None, ...]
                noise[:, :3] = noise[:, :3] + rand_vector[None, ...]
                transformed_points += [points]
                transformed_noise += [noise]
            return transformed_points, transformed_noise

        scannet_geos = batch[0]
        gt_trees = batch[1]
        sdf_parts = batch[5]
        parts_indices = batch[6]
        full_shape_indices = batch[7]
        noise_full = batch[8]
        dataset_indices = batch[9]

        x_roots = []
        encoder_features = []

        scannet_geos = scannet_geos.cpu().numpy()
        if self.mode == 'training':
            # partnetpartial
            scannet_geos = scannet_geos[:, :, ::-1, :, ::-1].copy()
        else:
            # mlcvnet
            scannet_geos = scannet_geos[:, :, ::-1, :, :].copy()
        scannet_geos = torch.FloatTensor(scannet_geos)

        if self.mode in ['training', 'finetune']:
            rand_rots = np.random.randint(12, size=(scannet_geos.shape[0],))
        else:
            rand_rots = np.zeros((scannet_geos.shape[0],))
        scannet_geos = rotate_voxels(scannet_geos, rand_rots)
        scannet_geos = scannet_geos.cuda()

        if self.mode in ['training', 'finetune']:
            rand_scales = np.random.uniform(0.8, 1.2, (self.config['batch_size'],))
            sdf_parts, noise_full = scale_points(sdf_parts, noise_full, rand_scales)
            sdf_parts, noise_full = rotate_points(sdf_parts, noise_full, rand_rots)
            sdf_parts, noise_full = jitter_points(sdf_parts, noise_full)

        x_roots, features = self.encoder.root_latents(scannet_geos)
        encoder_features = features

        all_losses = []

        for i, x_root in enumerate(x_roots):
            cuda_device = x_root.get_device()
            gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))

            # learning latent space
            output = self.decoder.latent_recon_loss(x_roots[i][None, ...], gt_tree, sdf_parts[i],
                                                    encoder_features=encoder_features[0],
                                                    rotation=0, parts_indices=parts_indices[i],
                                                    full_shape_idx=full_shape_indices[i],
                                                    epoch=self.current_epoch,
                                                    noise_full=noise_full[i],
                                                    rotations=rand_rots[i],
                                                    class2id=self.class2id
                                                    )

            object_losses = output[0]
            all_losses += [object_losses]

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

        return losses

    def training_step(self, batch, batch_idx):
        Tree.load_category_info(os.path.join(self.config['datadir'], self.config['dataset']))

        batch[0] = [x[None, ...] for x in batch[0]]
        scannet_geos = torch.cat(batch[0]).unsqueeze(dim=1)
        gt_trees = batch[1]
        partnet_ids = batch[2]
        rotations = batch[3]
        sdf_filenames = batch[5]
        sdf_parts = batch[6]
        parts_indices = batch[7]
        full_shape_indices = batch[8]
        noise_full = batch[9]
        indices = batch[10]

        input_batch = tuple([scannet_geos, gt_trees, partnet_ids, rotations,
                             sdf_filenames, sdf_parts, parts_indices, full_shape_indices,
                             noise_full, indices])

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

    def validation_step(self, batch, batch_idx):
        Tree.load_category_info(os.path.join(self.config['datadir'], self.config['dataset']))

        batch[0] = [x[None, ...] for x in batch[0]]
        scannet_geos = torch.cat(batch[0]).unsqueeze(dim=1)
        gt_trees = batch[1]
        partnet_ids = batch[2]
        rotations = batch[3]
        sdf_filenames = batch[5]
        sdf_parts = batch[6]
        parts_indices = batch[7]
        full_shape_indices = batch[8]
        noise_full = batch[9]
        indices = batch[10]

        input_batch = tuple([scannet_geos, gt_trees, partnet_ids, rotations,
                             sdf_filenames, sdf_parts, parts_indices, full_shape_indices,
                             noise_full, indices])

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

        return {'val_loss': total_loss,
                'val_loss_components': loss_components}

    def validation_epoch_end(self, outputs):
        log = {}
        losses = {}
        train_loss = torch.tensor(0).type_as(outputs[0]['val_loss'])
        for key in outputs[0]['val_loss_components']:
            losses[key] = 0

        for output in outputs:
            train_loss += output['val_loss'].detach().item()
            for key in losses:
                if isinstance(output['val_loss_components'][key], float):
                    losses[key] += output['val_loss_components'][key]
                else:
                    losses[key] += output['val_loss_components'][key].detach().item()
        train_loss /= len(outputs)
        for key in losses:
            losses[key] /= len(outputs)

        log.update(losses)
        log.update({'val_loss': train_loss})

        self.log('val_loss', log['val_loss'])
        for key in losses:
            self.log('val_' + key, log[key])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'],
                          shuffle=True, num_workers=32, drop_last=True,
                          collate_fn=collate_feats)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['batch_size'],
                          shuffle=False, num_workers=32, drop_last=True,
                          collate_fn=collate_feats)

    def tto_two_stage(self, batch, only_align=False,
                      constr_mode=0, cat_name=None,
                      num_shapes=0, k_near=0, scene_id='0', target_sample_names=None,
                      scale=1, wconf=0, w_full_noise=1, w_part_u_noise=1,
                      w_part_part_noise=1, lr_dec_full=0,
                      lr_dec_part=0, sa_mode=None,
                      parts_indices=None, shape_idx=None, store_dir=None):


        scannet_geos = batch[0]
        gt_trees = batch[1]
        rotations = batch[3]
        sdf_parts = batch[6]
        parts_indices = batch[7]

        scannet_geos = scannet_geos[0].cpu().numpy()
        scannet_geos = scannet_geos[:, ::-1, :, :].copy()
        scannet_geos = (torch.FloatTensor(scannet_geos).cuda(),)

        x_roots = []
        encoder_features = []
        for i, scannet_geo in enumerate(scannet_geos):
            cuda_device = scannet_geo.get_device()
            x_root, feature = self.encoder.root_latents(scannet_geos[i][None, ...])
            encoder_features += [feature]
            x_roots += [x_root]

        predicted_trees = []
        all_parts_sdfs_pred = []
        all_shapes_sdfs_pred = []
        all_parts_sdfs_pred_tto = []
        all_shapes_sdfs_pred_tto = []
        all_meta_data = []

        for i, x_root in enumerate(x_roots):
            cuda_device = x_root.get_device()
            gt_tree = gt_trees[i][0].to("cuda:{}".format(cuda_device))

            output = self.decoder.forward_two_stage(x_root,
                                                    sdf_parts,
                                                    full_label=gt_tree.root.label,
                                                    encoder_features=encoder_features[i],
                                                    rotation=rotations[i],
                                                    gt_tree=gt_tree,
                                                    cat_name=cat_name,
                                                    scale=scale)

            predicted_tree = output[0]
            parts_sdfs_pred = output[1]
            pred_output = output[2]
            shape_sdf_pred = output[3]
            shape_output = output[4]
            meta_data = output[5]

            predicted_trees += [predicted_tree]
            all_parts_sdfs_pred += [parts_sdfs_pred]
            all_shapes_sdfs_pred += [shape_sdf_pred]
            all_meta_data += [meta_data]

            parts_sdfs_pred_tto, all_shape_sdf_pred_tto, parts_stats_before, parts_stats_after \
                = self.decoder.tto(pred_output, shape_output, only_align=only_align, constr_mode=constr_mode, cat_name=cat_name,
                                   num_shapes=num_shapes, k_near=k_near, scene_id=scene_id, wconf=wconf,
                                   w_full_noise=w_full_noise, w_part_u_noise=w_part_u_noise,
                                   w_part_part_noise=w_part_part_noise, lr_dec_full=lr_dec_full,
                                   lr_dec_part=lr_dec_part, target_sample_names=target_sample_names,
                                   sa_mode=sa_mode, parts_indices=parts_indices, shape_idx=shape_idx,
                                   store_dir=store_dir, class2id=self.class2id)
            tto_stats = (parts_stats_before, parts_stats_after)

            all_parts_sdfs_pred_tto += [parts_sdfs_pred_tto]
            all_shapes_sdfs_pred_tto += [all_shape_sdf_pred_tto]

        output = [predicted_trees, all_parts_sdfs_pred, all_shapes_sdfs_pred, all_parts_sdfs_pred_tto,
                  all_shapes_sdfs_pred_tto, tto_stats, all_meta_data]
        output = tuple(output)

        return output
