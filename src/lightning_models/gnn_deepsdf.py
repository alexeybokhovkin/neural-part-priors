import os
import json
from argparse import Namespace
import random
import numpy as np
import gc
from scipy.ndimage import rotate
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from src.models.gnn_models import GeoEncoder, GeoEncoderProj, HierarchicalDeepSDFDecoder
from src.data_utils.hierarchy import Tree
from src.datasets.partnet import generate_gnn_deepsdf_datasets, generate_gnn_deepsdf_scannet_datasets
from src.utils.gnn import collate_feats
from ..utils.transformations import apply_transform_torch, from_tqs_to_matrix
from src.data_utils.transformations import rotationMatrixToEulerAngles
from deep_sdf_utils.optimizer import StepLearningRateSchedule


class GNNPartnetLightning(pl.LightningModule):

    def __init__(self, hparams, cat_name='chair'):
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
        print('Cat name', cat_name)

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

        self.config['parts_to_ids_path'] = os.path.join(config['datadir'], config['dataset'],
                                                        config['parts_to_ids_path'])

        Tree.load_category_info(os.path.join(config['datadir'], config['dataset']))
        self.encoder = GeoEncoder(**config)
        self.decoder = HierarchicalDeepSDFDecoder(**config,
                                                  num_parts=self.num_parts,
                                                  num_shapes=self.num_shapes,
                                                  cat_name=cat_name,
                                                  class2id=self.class2id)

        # ===================================================== #

        # Make filtered model loading

        # pretrained_dict = torch.load(config['resume_from_checkpoint'])
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf/GNNPartnet/deepsdf_parts_fullshape_geoscan_2_2_overfit100_fixed_1_l2_latent_1/checkpoints/75.ckpt')
        # partial
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf/GNNPartnet/deepsdf_parts_fullshape_partial_2_2_latent_onehot/checkpoints/20.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair/checkpoints/14.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_table/checkpoints/15.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_bed/checkpoints/300.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_trashcan/checkpoints/300.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_storagefurniture/checkpoints/60.ckpt')

        # partial v2 chairs
        # pretrained_dict = torch.load('/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair_entropypn_acc1/checkpoints/4.ckpt')
        # pretrained_dict = torch.load('/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair_entropypn2_acc1/checkpoints/4.ckpt')

        # partial v2
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair_entropypn_acc1/checkpoints/16.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair_entropypn2_acc1/checkpoints/12.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_table_entropypn_acc1/checkpoints/18.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_table_entropypn2_acc1/checkpoints/12.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_storagefurniture_entropypn_acc1/checkpoints/50.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_storagefurniture_entropypn2_acc1/checkpoints/epoch=38.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_bed_entropypn_acc1/checkpoints/400.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_bed_entropypn2_acc1/checkpoints/180.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_trashcan_entropypn_acc1/checkpoints/330.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_trashcan_entropypn2_acc1/checkpoints/200.ckpt')


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


        # ## mlcvnet ##
        # if cat_name == 'chair':
        #     # old
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_chair/checkpoints/25.ckpt')
        #
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_chair_entropy_2/checkpoints/220.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_chair_entropypn_2/checkpoints/180.ckpt')
        #
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair_contrast/checkpoints/22.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair_entropypnrotation/checkpoints/21.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair_entropyrotation/checkpoints/25.ckpt')
        #
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_chair_entropyrotation/checkpoints/100.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_chair_entropypnrotation/checkpoints/78.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_chair_contrast/checkpoints/100.ckpt')
        #
        #     ## NEW PARTS ##
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn_hardnoise/checkpoints/120.ckpt')
        #     pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn_hardnoise_partialpoints/checkpoints/80.ckpt')
        #
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn2_hardnoise/checkpoints/100.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn2_hardnoise_partialpoints/checkpoints/102.ckpt')
        #
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_contrastpn_hardnoise/checkpoints/72.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_contrastpn_hardnoise_partialpoints/checkpoints/58.ckpt')
        #
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_contrastpn2_hardnoise/checkpoints/56.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_contrastpn2_hardnoise_partialpoints/checkpoints/55.ckpt')
        #
        #     ## ABLATION ##
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_chair_entropypn_hardnoise_partialpoints_ablation/checkpoints/72.ckpt')
        #
        # elif cat_name == 'bed':
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_bed/checkpoints/70.ckpt')
        #
        #     ## NEW PARTS ##
        #     pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_bed_entropypn_hardnoise_partialpoints/checkpoints/125.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_bed_entropypn2_hardnoise_partialpoints/checkpoints/72.ckpt')
        #
        #     ## ABLATION ##
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_bed_entropypn_hardnoise_partialpoints_ablation/checkpoints/116.ckpt')
        # elif cat_name == 'storagefurniture':
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_storagefurniture/checkpoints/40.ckpt')
        #
        #     ## NEW PARTS ##
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_storagefurniture_entropypn_hardnoise_10/checkpoints/160.ckpt')
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_storagefurniture_entropypn_hardnoise_partialpoints_14/checkpoints/140.ckpt')
        #
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_storagefurniture_entropypn2_hardnoise_13/checkpoints/120.ckpt')
        #     pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_storagefurniture_entropypn2_hardnoise_partialpoints_9/checkpoints/120.ckpt')
        #
        #     ## ABLATION ##
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_storagefurniture_entropypn_hardnoise_partialpoints_ablation/checkpoints/100.ckpt')
        #
        # elif cat_name == 'trashcan':
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_trashcan/checkpoints/90.ckpt')
        #
        #     ## NEW PARTS ##
        #     pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_trashcan_entropypn_hardnoise_partialpoints/checkpoints/70.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_trashcan_entropypn2_hardnoise_partialpoints/checkpoints/67.ckpt')
        #
        #     ## ABLATION ##
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_trashcan_entropypn_hardnoise_partialpoints_ablation/checkpoints/100.ckpt')
        #
        # elif cat_name == 'table':
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_table/checkpoints/26.ckpt')
        #
        #     ## NEW PARTS ##
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_table_entropypn_hardnoise_2/checkpoints/104.ckpt')
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_table_entropypn_hardnoise_partialpoints_5/checkpoints/88.ckpt')
        #
        #     pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_table_entropypn2_hardnoise_2/checkpoints/80.ckpt')
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_table_entropypn2_hardnoise_partialpoints_5/checkpoints/88.ckpt')
        #
        #     ## ABLATION ##
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_table_entropypn_hardnoise_partialpoints_ablation/checkpoints/76.ckpt')
        #
        # else:
        #     raise ValueError(f'Category name {cat_name} is unknown')

        # pretrain for all categories
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn_hardnoise_partialpoints/checkpoints/80.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn2_hardnoise_partialpoints/checkpoints/80.ckpt')


        # LOAD ENCODER
        encoder_dict = self.encoder.state_dict()
        encoder_dict_update = {k[8:]: v for k, v in pretrained_dict['state_dict'].items()
                               if k[8:] in encoder_dict and
                               pretrained_dict['state_dict'][k].shape == encoder_dict[k[8:]].shape}
        encoder_dict.update(encoder_dict_update)
        print('Updated keys (encoder):', len(encoder_dict_update))
        print('Not loaded keys:', list(set(encoder_dict.keys()) - set(encoder_dict_update.keys())))
        self.encoder.load_state_dict(encoder_dict)

        # pretrained_dict = torch.load(config['resume_from_checkpoint'])
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf/GNNPartnet/deepsdf_parts_fullshape_geoscan_2_2_overfit100_fixed_1_l2_latent_1/checkpoints/75.ckpt')
        # partial
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf/GNNPartnet/deepsdf_parts_fullshape_partial_2_2_latent_onehot/checkpoints/20.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair/checkpoints/14.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_table/checkpoints/15.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_bed/checkpoints/300.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_trashcan/checkpoints/300.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_storagefurniture/checkpoints/60.ckpt')

        # partial v2 chairs
        # pretrained_dict = torch.load('/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair_entropypn_acc1/checkpoints/4.ckpt')
        # pretrained_dict = torch.load('/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair_entropypn2_acc1/checkpoints/4.ckpt')

        # partial v2
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair_entropypn_acc1/checkpoints/16.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair_entropypn2_acc1/checkpoints/12.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_table_entropypn_acc1/checkpoints/18.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_table_entropypn2_acc1/checkpoints/12.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_storagefurniture_entropypn_acc1/checkpoints/50.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_storagefurniture_entropypn2_acc1/checkpoints/epoch=38.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_bed_entropypn_acc1/checkpoints/400.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_bed_entropypn2_acc1/checkpoints/180.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_trashcan_entropypn_acc1/checkpoints/330.ckpt')
        # pretrained_dict = torch.load(
        #     '/cluster/valinor/abokhovkin/scannet-relationships-v2/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_trashcan_entropypn2_acc1/checkpoints/200.ckpt')

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


        # ## mlcvnet ##
        # if cat_name == 'chair':
        #     # old
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_chair/checkpoints/25.ckpt')
        #
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_chair_entropy_2/checkpoints/220.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_chair_entropypn_2/checkpoints/180.ckpt')
        #
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair_contrast/checkpoints/22.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair_entropypnrotation/checkpoints/21.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_partial/GNNPartnet/deepsdf_parts_fullshape_partial_onehot_chair_entropyrotation/checkpoints/25.ckpt')
        #
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_chair_entropyrotation/checkpoints/100.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_chair_entropypnrotation/checkpoints/78.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_chair_contrast/checkpoints/100.ckpt')
        #
        #     ## NEW PARTS ##
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn_hardnoise/checkpoints/120.ckpt')
        #     pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn_hardnoise_partialpoints/checkpoints/80.ckpt')
        #
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn2_hardnoise/checkpoints/100.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn2_hardnoise_partialpoints/checkpoints/102.ckpt')
        #
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_contrastpn_hardnoise/checkpoints/72.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_contrastpn_hardnoise_partialpoints/checkpoints/58.ckpt')
        #
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_contrastpn2_hardnoise/checkpoints/56.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_contrastpn2_hardnoise_partialpoints/checkpoints/55.ckpt')
        #
        #     ## ABLATION ##
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_chair_entropypn_hardnoise_partialpoints_ablation/checkpoints/72.ckpt')
        #
        # elif cat_name == 'bed':
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_bed/checkpoints/70.ckpt')
        #
        #     ## NEW PARTS ##
        #     pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_bed_entropypn_hardnoise_partialpoints/checkpoints/125.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_bed_entropypn2_hardnoise_partialpoints/checkpoints/72.ckpt')
        #
        #     ## ABLATION ##
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_bed_entropypn_hardnoise_partialpoints_ablation/checkpoints/116.ckpt')
        #
        # elif cat_name == 'storagefurniture':
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_storagefurniture/checkpoints/40.ckpt')
        #
        #     ## NEW PARTS ##
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_storagefurniture_entropypn_hardnoise_10/checkpoints/160.ckpt')
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_storagefurniture_entropypn_hardnoise_partialpoints_14/checkpoints/140.ckpt')
        #
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_storagefurniture_entropypn2_hardnoise_13/checkpoints/120.ckpt')
        #     pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_storagefurniture_entropypn2_hardnoise_partialpoints_9/checkpoints/120.ckpt')
        #
        #     ## ABLATION ##
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_storagefurniture_entropypn_hardnoise_partialpoints_ablation/checkpoints/100.ckpt')
        #
        # elif cat_name == 'trashcan':
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_trashcan/checkpoints/90.ckpt')
        #
        #     ## NEW PARTS ##
        #     pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_trashcan_entropypn_hardnoise_partialpoints/checkpoints/70.ckpt')
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_trashcan_entropypn2_hardnoise_partialpoints/checkpoints/67.ckpt')
        #
        #     ## ABLATION ##
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_trashcan_entropypn_hardnoise_partialpoints_ablation/checkpoints/100.ckpt')
        #
        # elif cat_name == 'table':
        #     # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_table/checkpoints/26.ckpt')
        #
        #     ## NEW PARTS ##
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_table_entropypn_hardnoise_2/checkpoints/104.ckpt')
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_table_entropypn_hardnoise_partialpoints_5/checkpoints/88.ckpt')
        #
        #     pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_table_entropypn2_hardnoise_2/checkpoints/80.ckpt')
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_table_entropypn2_hardnoise_partialpoints_5/checkpoints/88.ckpt')
        #
        #     ## ABLATION ##
        #     # pretrained_dict = torch.load('/cluster/daidalos/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_mlcvnet_onehot_table_entropypn_hardnoise_partialpoints_ablation/checkpoints/76.ckpt')
        #
        # else:
        #     raise ValueError(f'Category name {cat_name} is unknown')

        # pretrain for all categories
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn_hardnoise_partialpoints/checkpoints/80.ckpt')
        # pretrained_dict = torch.load('/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet/deepsdf_parts_fullshape_finetune_onehot_chair_entropypn2_hardnoise_partialpoints/checkpoints/80.ckpt')


        # LOAD DECODER
        decoder_dict = self.decoder.state_dict()
        decoder_dict_update = {k[8:]: v for k, v in pretrained_dict['state_dict'].items()
                               if k[8:] in decoder_dict and
                               pretrained_dict['state_dict'][k].shape == decoder_dict[k[8:]].shape}
        decoder_dict.update(decoder_dict_update)
        print('Updated keys (decoder):', len(decoder_dict_update))
        print('Not loaded keys:', list(set(decoder_dict.keys()) - set(decoder_dict_update.keys())))
        self.decoder.load_state_dict(decoder_dict)

        # self.load_deepsdf_outside(
        #     f'/rhome/abokhovkin/projects/DeepSDF/experiments/full_experiments/{cat_name}_full',
        #     f'/rhome/abokhovkin/projects/DeepSDF/experiments/full_experiments/{cat_name}_parts_onehot'
        # )

        self.load_deepsdf_outside(
            f'/cluster/daidalos/abokhovkin/DeepSDF_v2/full_experiments_v2/{cat_name}_full_surface_pe', # <- CHECK HERE
            f'/cluster/daidalos/abokhovkin/DeepSDF_v2/full_experiments_v2/{cat_name}_parts_onehot_pe'
        )

        # ===================================================== #

        # fix random seeds
        # random.seed(config['manual_seed'])
        # np.random.seed(config['manual_seed'])
        # torch.manual_seed(config['manual_seed'])
        # torch.cuda.manual_seed(config['manual_seed'])
        # torch.backends.cudnn.enabled = False
        # torch.backends.cudnn.deterministic = True

        # save config file
        # with open(os.path.join(config['base'], config['checkpoint_dir'], config['model'], config['version'],
        #                        'config.json'), 'w') as f:
        #     json.dump(self.config, f)

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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               # lr=self.config['learning_rate'],
                               # mlcvnet
                               lr=0.0002, # 0.0002
                               weight_decay=self.config['weight_decay']
                               )
        scheduler = StepLR(optimizer,
                           gamma=self.config['gamma'],
                           # step_size=self.config['decay_every'],
                           # mlcvnet
                           step_size=40 # 30
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

        def rotate_voxels_with_quat(geos, rots):
            rots = [x.degrees for x in rots]
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
            # print('All points:', len(all_points))
            # print('Noise full:', len(noise_full))
            for i, points in enumerate(all_points):
                # print(points.shape)
                noise = noise_full[i]
                # print(noise.shape)
                # print('Scale:', s[i])
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
        q_rot = batch[11]
        dataset_indices = batch[12]

        x_roots = []
        encoder_features = []

        t0 = time.time()

        scannet_geos = scannet_geos.cpu().numpy()
        # partnetpartial
        # scannet_geos = scannet_geos[:, :, ::-1, :, ::-1].copy()
        # mlcvnet
        scannet_geos = scannet_geos[:, :, ::-1, :, :].copy()
        scannet_geos = torch.FloatTensor(scannet_geos)
        print('scannet_geos', scannet_geos.shape)
        # train
        rand_rots = np.random.randint(12, size=(scannet_geos.shape[0],))
        # rand_rots = np.zeros((scannet_geos.shape[0],))
        scannet_geos = rotate_voxels(scannet_geos, rand_rots)
        scannet_geos = scannet_geos.cuda()

        rand_scales = np.random.uniform(0.8, 1.2, (self.config['batch_size'],))
        sdf_parts, noise_full = scale_points(sdf_parts, noise_full, rand_scales)
        sdf_parts, noise_full = rotate_points(sdf_parts, noise_full, rand_rots)
        sdf_parts, noise_full = jitter_points(sdf_parts, noise_full)

        # torch.save(scannet_geos.cpu(), 'voxels.pth')
        # torch.save(sdf_parts, 'sdf_parts.pth')
        # torch.save(noise_full, 'noise_full.pth')
        # raise NotImplementedError

        # inference
        # scannet_geos = rotate_voxels_with_quat(scannet_geos, q_rot)
        # scannet_geos = scannet_geos.cuda()

        x_roots, features = self.encoder.root_latents(scannet_geos)
        encoder_features = features

        t1 = time.time()

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
            output = self.decoder.latent_recon_loss(x_roots[i][None, ...], gt_tree, sdf_parts[i],
                                                    encoder_features=encoder_features[0],
                                                    rotation=0, parts_indices=parts_indices[i],
                                                    full_shape_idx=full_shape_indices[i],
                                                    epoch=self.current_epoch,
                                                    noise_full=noise_full[i],
                                                    rotations=rand_rots[i],
                                                    # rotations=q_rot[i],
                                                    class2id=self.class2id
                                                    )

            object_losses = output[0]
            all_losses += [object_losses]

        t2 = time.time()
        # print('(0) Encode:', t1 - t0)
        # print('(0) Decode:', t2 - t1)

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
                if loss_name == 'point_part' and loss > 200:
                    print('Loss:', loss, dataset_indices[i])
                    for child in gt_trees[i][0].root.children:
                        print(child.label)
                    for j in range(len(sdf_parts[i])):
                        print(gt_trees[i][0].root.children[j].label, sdf_parts[i][j].shape)
                        print(gt_trees[i][0].root.children[j].label, torch.max(sdf_parts[i][j]), torch.min(sdf_parts[i][j]))
                    print('background noise:', noise_full[i].shape)
                    print('background noise:', torch.max(noise_full[i]), torch.min(noise_full[i]))

                    raise NotImplementedError
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
        q_rot = batch[13]

        print('Indices:', indices)
        input_batch = tuple([scannet_geos, shape_sdfs, shape_mask, gt_trees, partnet_ids, rotations,
                             sdf_filenames, sdf_parts, parts_indices, full_shape_indices,
                             noise_full, q_rot, indices])

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

    # def validation_step(self, batch, batch_idx):
    #     total_loss = 0
    #     return {'val_loss': total_loss}
    #
    # def validation_epoch_end(self, outputs):
    #     self.log('val_loss', 0)

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
        sdf_filenames = batch[7]
        sdf_parts = batch[8]
        parts_indices = batch[9]
        full_shape_indices = batch[10]
        noise_full = batch[11]
        indices = batch[12]
        q_rot = batch[13]

        print('Indices:', indices)
        input_batch = tuple([scannet_geos, shape_sdfs, shape_mask, gt_trees, partnet_ids, rotations,
                             sdf_filenames, sdf_parts, parts_indices, full_shape_indices,
                             noise_full, q_rot, indices])

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

        del outputs, log
        gc.collect()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'],
                          shuffle=True, num_workers=32, drop_last=True,
                          collate_fn=collate_feats)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['batch_size'],
                          shuffle=False, num_workers=32, drop_last=True,
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

    def tto_two_stage(self, batch, index=0, rot_aug=0, shift_x=0, shift_y=0, only_align=False,
                      bck_thr=0.5, constr_mode=0, cat_name=None,
                      num_shapes=0, k_near=0, scene_id='0', target_sample_names=None,
                      scale=1, wconf=0, w_full_noise=1, w_part_u_noise=1,
                      w_part_part_noise=1, lr_dec_full=0,
                      lr_dec_part=0, sa_mode=None,
                      parts_indices=None, shape_idx=None, store_dir=None):

        def apply_matrix(pc, matrix):
            matrix = torch.FloatTensor(matrix).to(pc.device)
            pc[0, :, :3] = pc[0, :, :3] @ matrix.T

            return pc, matrix

        def rotate_voxels_with_quat(geos, q):
            print('Quat:', q)
            print('Quat:', q.axis, q.degrees)
            geo_np = geos[0][0].cpu().numpy().astype('uint8')
            rotated_scan_geo = rotate(geo_np, q.degrees, axes=[0, 2], reshape=False).astype('float32')
            rotated_scan_geo = torch.FloatTensor(rotated_scan_geo)[None, ...].cuda()
            return (rotated_scan_geo,)

        def rotate_points_with_quat(all_points, noise_full, q):
            transform = torch.FloatTensor(from_tqs_to_matrix([0, 0, 0], q.elements, [1, 1, 1]))
            print('Points transform', transform)
            transformed_points = []
            for points in all_points:
                points[:, :3] = apply_transform_torch(points[:, :3], transform)
                transformed_points += [points]
            transformed_points = torch.cat(transformed_points)
            noise_full[:, :3] = apply_transform_torch(noise_full[:, :3], transform)
            return transformed_points, noise_full

        def rotate_voxels(geos, rot):
            geo_np = geos[0][0].cpu().numpy().astype('uint8')
            rotated_scan_geo = rotate(geo_np, rot, axes=[0, 2], reshape=False).astype('float32')
            rotated_scan_geo = torch.FloatTensor(rotated_scan_geo)[None, ...].cuda()
            return (rotated_scan_geo,)

        if cat_name == 'chair':
            ROTATIONS_DIR = '/cluster/pegasus/abokhovkin/scannet-relationships/test_output/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_chair_0.25_0'
        elif cat_name == 'table':
            ROTATIONS_DIR = '/cluster/pegasus/abokhovkin/scannet-relationships/test_output/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_table_0.35_4'
        elif cat_name == 'storagefurniture':
            ROTATIONS_DIR = '/cluster/pegasus/abokhovkin/scannet-relationships/test_output/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_storagefurniture_0.35_4'
        elif cat_name == 'bed':
            ROTATIONS_DIR = '/cluster/pegasus/abokhovkin/scannet-relationships/test_output/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_bed_0.35_4'
        elif cat_name == 'trashcan':
            ROTATIONS_DIR = '/cluster/pegasus/abokhovkin/scannet-relationships/test_output/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_trashcan_0.35_0'
        else:
            raise ValueError(f'Category name {cat_name} is unknown')
        # angle = torch.load(os.path.join(ROTATIONS_DIR, str(index), 'rotation.pth'))
        # angle = 30 * angle
        # rot_matrix = torch.load(os.path.join(ROTATIONS_DIR, str(index), 'rot_matrix.pth'))
        # icp_matrix = torch.load(os.path.join(ROTATIONS_DIR, str(index), 'icp.pth'))

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
        q_rot = batch[13]

        # apply orientation predicted previously in 'only_align' mode (only old version)
        # scannet_geos = rotate_voxels(scannet_geos, -angle)
        # sdf_parts, _ = apply_matrix(sdf_parts, rot_matrix)

        # only for new models (mlcvnet)
        scannet_geos = scannet_geos[0].cpu().numpy()
        scannet_geos = scannet_geos[:, ::-1, :, :].copy()
        scannet_geos = (torch.FloatTensor(scannet_geos).cuda(),)

        # torch.save(scannet_geos, 'voxels.pth')
        # torch.save(sdf_parts, 'sdf_parts.pth')
        # raise NotImplementedError

        t0 = time.time()

        x_roots = []
        encoder_features = []
        for i, mask in enumerate(masks):
            cuda_device = mask.get_device()
            x_root, feature = self.encoder.root_latents(scannet_geos[i][None, ...])
            encoder_features += [feature]
            x_roots += [x_root]

        t1 = time.time()

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
                                                    index=index,
                                                    parts_indices=parts_indices,
                                                    full_shape_idx=full_shape_indices,
                                                    noise_full=noise_full,
                                                    rot_aug=rot_aug, shift_x=shift_x, shift_y=shift_y,
                                                    bck_thr=bck_thr, cat_name=cat_name,
                                                    scale=scale)

            t2 = time.time()

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

            parts_sdfs_pred_tto, all_shape_sdf_pred_tto, \
            parts_stats_before, parts_stats_after, shape_stats_before_tto, shape_stats_after_tto \
                = self.decoder.tto(pred_output, shape_output, only_align=only_align, constr_mode=constr_mode, cat_name=cat_name,
                                   num_shapes=num_shapes, k_near=k_near, scene_id=scene_id, wconf=wconf,
                                   w_full_noise=w_full_noise, w_part_u_noise=w_part_u_noise,
                                   w_part_part_noise=w_part_part_noise, lr_dec_full=lr_dec_full,
                                   lr_dec_part=lr_dec_part, target_sample_names=target_sample_names,
                                   sa_mode=sa_mode, parts_indices=parts_indices, shape_idx=shape_idx,
                                   store_dir=store_dir)
            tto_stats = (parts_stats_before, parts_stats_after, shape_stats_before_tto, shape_stats_after_tto)

            t3 = time.time()

            all_parts_sdfs_pred_tto += [parts_sdfs_pred_tto]
            all_shapes_sdfs_pred_tto += [all_shape_sdf_pred_tto]

        output = [predicted_trees, all_parts_sdfs_pred, all_shapes_sdfs_pred, all_parts_sdfs_pred_tto,
                  all_shapes_sdfs_pred_tto, tto_stats, all_meta_data]
        output = tuple(output)

        print('(1) Encode:', t1 - t0)
        print('(1) Decode tree:', t2 - t1)
        print('(1) TTO:', t3 - t2)

        return output
