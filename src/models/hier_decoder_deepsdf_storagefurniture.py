import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
from copy import deepcopy
import math
from scipy.spatial import cKDTree
import json
import open3d as o3d
import time
from scipy.spatial.transform import Rotation

from ..utils.hierarchy import Tree
from ..utils.gnn import linear_assignment
from ..utils.losses import IoULoss
from .deep_sdf_decoder import Decoder as DeepSDFDecoder
from .gnn_contrast import LatentDecoder, NumChildrenClassifier, GNNChildDecoder, LatentProjector
from ..utils.transformations import apply_transform, convert_sdf_samples_to_ply, create_grid, decode_sdf, apply_transform_torch, from_tqs_to_matrix, decompose_mat4
from ..utils.embedder import get_embedder_nerf

# sys.path.append('/home/bohovkin/cluster/abokhovkin_home/external_packages/chamferdist')
# sys.path.append('/rhome/abokhovkin/external_packages/chamferdist')
sys.path.append('path to chamferdist lib')
try:
    from chamferdist import ChamferDistance
except:
    print('ChamferDistance not imported')


GRID_RESOLUTION = 128


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def reconstruct_part_and_shape_tto(
        decoder_part,
        decoder_shape,
        num_iterations,
        sdf_data_part,
        part_uniform_noise,
        part_another_parts_noise,
        sdf_data_shape,
        clamp_dist,
        num_samples=30000,
        lr_1=5e-4,
        lr_2=5e-4,
        lr_3=0,
        w_cons=0,
        w_full_noise=1,
        w_part_u_noise=1,
        w_part_part_noise=1,
        l2reg=False,
        cal_scale=1,
        add_feat=None,
        init_emb_part=None,
        init_emb_shape=None,
        mode=0,
        class_one_hot=None,
        store_dir=None,
        child_name=None,
        iter=None
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    embedder, embedder_out_dim = get_embedder_nerf(10, input_dims=3, i=0)

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    latent_part = init_emb_part.detach().clone().cuda()
    latent_shape = init_emb_shape.detach().clone().cuda()

    latent_shape.requires_grad = True

    if class_one_hot is not None:
        class_one_hot = class_one_hot.cuda()
        latent_part = torch.hstack([latent_part, class_one_hot])

    latent_part.requires_grad = True

    optimizer_1 = torch.optim.Adam([latent_part, latent_shape], lr=lr_1)

    print('Decoder part parameters:')
    for name, param in decoder_part.named_parameters():
        if param.requires_grad is True:
            if not name.startswith('lin1') and not name.startswith('lin2'):
                param.requires_grad = False
                print(name, param.requires_grad)
    optimizer_2 = torch.optim.Adam(decoder_part.parameters(), lr=lr_2)

    print('Decoder shape parameters:')
    # for name, param in decoder_shape.named_parameters():
    #     if param.requires_grad is True:
    #         if not name.startswith('lin1') and not name.startswith('lin2'):
    #             param.requires_grad = False
    #             print(name, param.requires_grad)
    optimizer_3 = torch.optim.Adam(decoder_shape.parameters(), lr=lr_3)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss(reduction='none')
    loss_mse = torch.nn.MSELoss(reduction='none')
    loss_part_history = []
    all_latents = []
    cal_scale = float(cal_scale)
    pred_sdf_part = {}
    pred_sdf_full = {}
    j = 0

    # part points
    indices_not_noise = torch.where(sdf_data_part[:, 3] < clamp_dist)[0]
    sdf_data_part_not_noise = sdf_data_part[indices_not_noise]
    sdf_data_part_not_noise = sdf_data_part_not_noise.cuda()
    xyz_part_not_noise = sdf_data_part_not_noise[:, 0:3]
    xyz_part_not_noise_pe = embedder(xyz_part_not_noise)
    sdf_gt_part_not_noise = sdf_data_part_not_noise[:, 3].unsqueeze(1)
    sdf_gt_part_not_noise = torch.clamp(cal_scale * sdf_gt_part_not_noise, -clamp_dist, clamp_dist).cuda()

    sdf_data_part_uniform_noise = part_uniform_noise.cuda()
    xyz_part_uniform_noise = sdf_data_part_uniform_noise[:, 0:3]
    xyz_part_uniform_noise_pe = embedder(xyz_part_uniform_noise)
    sdf_gt_part_uniform_noise = sdf_data_part_uniform_noise[:, 3].unsqueeze(1)
    sdf_gt_part_uniform_noise = torch.clamp(sdf_gt_part_uniform_noise, -clamp_dist, clamp_dist)
    sdf_data_part_another_parts_noise = part_another_parts_noise.cuda()
    xyz_part_another_parts_noise = sdf_data_part_another_parts_noise[:, 0:3]
    xyz_part_another_parts_noise_pe = embedder(xyz_part_another_parts_noise)
    sdf_gt_part_another_parts_noise = sdf_data_part_another_parts_noise[:, 3].unsqueeze(1)
    sdf_gt_part_another_parts_noise = torch.clamp(sdf_gt_part_another_parts_noise, -clamp_dist, clamp_dist)

    # full shape
    sdf_data_full = sdf_data_shape.cuda()
    indices_not_noise = torch.where(sdf_data_full[:, 3] < clamp_dist)[0]
    sdf_data_full_not_noise = sdf_data_full[indices_not_noise]
    xyz_full_not_noise = sdf_data_full_not_noise[:, 0:3]
    xyz_full_not_noise_pe = embedder(xyz_full_not_noise)
    sdf_gt_full_not_noise = sdf_data_full_not_noise[:, 3].unsqueeze(1)
    sdf_gt_full_not_noise = torch.clamp(cal_scale * sdf_gt_full_not_noise, -clamp_dist, clamp_dist)

    indices_noise = torch.where(sdf_data_full[:, 3] >= clamp_dist)[0]
    sdf_data_full_noise = sdf_data_full[indices_noise]
    xyz_full_noise = sdf_data_full_noise[:, 0:3]
    xyz_full_noise_pe = embedder(xyz_full_noise)
    sdf_gt_full_noise = sdf_data_full_noise[:, 3].unsqueeze(1)
    sdf_gt_full_noise = torch.clamp(cal_scale * sdf_gt_full_noise, -clamp_dist, clamp_dist)

    for e in range(num_iterations):
        if e % 200 == 0:
            all_latents += [latent_part.detach().clone()]

        adjust_learning_rate(lr_1, optimizer_1, e, decreased_by, adjust_lr_every)
        adjust_learning_rate(lr_2, optimizer_2, e, decreased_by, adjust_lr_every)
        adjust_learning_rate(lr_3, optimizer_3, e, decreased_by, adjust_lr_every)
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        optimizer_3.zero_grad()

        ## construct inputs for TTO
        # part-based
        latent_inputs_part = latent_part.expand(len(xyz_part_not_noise), -1)
        latent_inputs_shape = latent_shape.expand(len(xyz_part_not_noise), -1)
        inputs_part = torch.cat([latent_inputs_part, xyz_part_not_noise_pe], 1).cuda()
        inputs_shape = torch.cat([latent_inputs_shape, xyz_part_not_noise_pe], 1).cuda()

        latent_inputs_part_uniform_noise = latent_part.expand(len(xyz_part_uniform_noise), -1)
        inputs_part_uniform_noise = torch.cat([latent_inputs_part_uniform_noise, xyz_part_uniform_noise_pe], 1).cuda()
        latent_inputs_part_another_parts_noise = latent_part.expand(len(xyz_part_another_parts_noise), -1)
        inputs_part_another_parts_noise = torch.cat([latent_inputs_part_another_parts_noise, xyz_part_another_parts_noise_pe], 1).cuda()

        # full shape
        latent_inputs_full_not_noise = latent_shape.expand(len(xyz_full_not_noise), -1)
        inputs_full_not_noise = torch.cat([latent_inputs_full_not_noise, xyz_full_not_noise_pe], 1).cuda()
        latent_inputs_full_noise = latent_shape.expand(len(xyz_full_noise), -1)
        inputs_full_noise = torch.cat([latent_inputs_full_noise, xyz_full_noise_pe], 1).cuda()

        ## forward pass
        # part-based
        pred_sdf_part_ = decoder_part(inputs_part)
        pred_sdf_shape = decoder_shape(inputs_shape)
        pred_sdf_part_uniform_noise = decoder_part(inputs_part_uniform_noise)
        pred_sdf_part_another_parts_noise = decoder_part(inputs_part_another_parts_noise)


        pred_sdf_part_ = torch.clamp(pred_sdf_part_, -clamp_dist, clamp_dist)
        pred_sdf_shape = torch.clamp(pred_sdf_shape, -clamp_dist, clamp_dist)
        pred_sdf_part_uniform_noise = torch.clamp(pred_sdf_part_uniform_noise, -clamp_dist, clamp_dist)
        pred_sdf_part_another_parts_noise = torch.clamp(pred_sdf_part_another_parts_noise, -clamp_dist, clamp_dist)

        # full shape
        pred_sdf_full_not_noise = decoder_shape(inputs_full_not_noise)
        pred_sdf_full_not_noise = torch.clamp(pred_sdf_full_not_noise, -clamp_dist, clamp_dist)
        pred_sdf_full_noise = decoder_shape(inputs_full_noise)
        pred_sdf_full_noise = torch.clamp(pred_sdf_full_noise, -clamp_dist, clamp_dist)

        ## compute all losses
        # part-based
        loss_part = loss_l1(pred_sdf_part_, sdf_gt_part_not_noise)
        loss_part_uniform_noise = loss_l1(pred_sdf_part_uniform_noise, sdf_gt_part_uniform_noise)
        loss_part_another_parts_noise = loss_l1(pred_sdf_part_another_parts_noise, sdf_gt_part_another_parts_noise)
        pred_sdf_shape_clone = pred_sdf_shape.clone().detach()
        pred_sdf_shape_clone.requires_grad = False
        loss_part_consistency = loss_mse(pred_sdf_part_, pred_sdf_shape_clone)

        # loss full shape
        loss_full = loss_l1(pred_sdf_full_not_noise, sdf_gt_full_not_noise).mean() + w_full_noise * loss_l1(pred_sdf_full_noise, sdf_gt_full_noise).mean()

        # loss part
        loss = (loss_part.mean() + w_part_u_noise * loss_part_uniform_noise.mean() + w_part_part_noise * loss_part_another_parts_noise.mean()) + w_cons * loss_part_consistency.mean() # 10 10 200
        loss = loss.mean()

        if e == 0:
            # (classified points + uniform noise + part noise) + part loss + (part gt sdf + uniform noise gt sdf + part noise gt sdf)
            stats_before_tto = (np.vstack([xyz_part_not_noise.detach().cpu().numpy(),
                                           xyz_part_uniform_noise.detach().cpu().numpy(),
                                           xyz_part_another_parts_noise.detach().cpu().numpy()]),
                                loss_part.detach().cpu().numpy(),
                                np.vstack([sdf_gt_part_not_noise.detach().cpu().numpy(),
                                           sdf_gt_part_uniform_noise.detach().cpu().numpy(),
                                           sdf_gt_part_another_parts_noise.detach().cpu().numpy()]),
                                xyz_full_not_noise.detach().cpu().numpy())
        if e == num_iterations - 1:
            # classified points + part loss + part gt sdf
            stats_after_tto = (xyz_part_not_noise.detach().cpu().numpy(),
                               loss_part.detach().cpu().numpy(),
                               sdf_gt_part_not_noise.detach().cpu().numpy())

        if l2reg:
            loss = loss + 1e-4 * torch.mean(latent_part.pow(2))

        loss.backward()
        loss_full.backward()

        optimizer_1.step()
        if e > 400:
            optimizer_2.step()
            optimizer_3.step()

        loss_part_history += [float(loss.detach().item())]

        # optimization in time
        if e % 200 == 0:

            decoder_part.eval()
            with torch.no_grad():
                pred_sdf_part_loc = {}

                sdf_values, voxel_origin, voxel_size, offset, scale, _ = create_grid(
                    decoder_part,
                    latent_part[:256],
                    N=GRID_RESOLUTION,
                    max_batch=int(2 ** 18),
                    input_samples=None,
                    add_feat=add_feat,
                    mode=mode,
                    class_one_hot=class_one_hot
                )
                pred_sdf_part_loc['sdf'] = sdf_values.detach().cpu().clone()
                pred_sdf_part_loc['vox_origin'] = voxel_origin
                pred_sdf_part_loc['vox_size'] = voxel_size
                pred_sdf_part_loc['offset'] = offset
                pred_sdf_part_loc['scale'] = scale
                pred_sdf_part_loc['latent'] = latent_part.detach().cpu().clone()

                os.makedirs(store_dir, exist_ok=True)
                convert_sdf_samples_to_ply(
                    pred_sdf_part_loc['sdf'],
                    pred_sdf_part_loc['vox_origin'],
                    pred_sdf_part_loc['vox_size'],
                    os.path.join(store_dir, child_name + '.' + str(e) + '.ply'),
                    offset=pred_sdf_part_loc['offset'],
                    scale=pred_sdf_part_loc['scale'],
                    level=0.0
                )

                del sdf_values
                del voxel_origin
                del voxel_size
                del offset
                del scale
                del _

            decoder_part.train()

        if e % 600 == 0:
            decoder_part.eval()
            with torch.no_grad():
                pred_sdf_part[j] = {}

                sdf_values, voxel_origin, voxel_size, offset, scale, _ = create_grid(
                    decoder_part,
                    latent_part[:256],
                    N=GRID_RESOLUTION,
                    max_batch=int(2 ** 18),
                    input_samples=None,
                    add_feat=add_feat,
                    mode=mode,
                    class_one_hot=class_one_hot
                )
                pred_sdf_part[j]['sdf'] = sdf_values.detach().cpu().clone()
                pred_sdf_part[j]['vox_origin'] = voxel_origin
                pred_sdf_part[j]['vox_size'] = voxel_size
                pred_sdf_part[j]['offset'] = offset
                pred_sdf_part[j]['scale'] = scale
                pred_sdf_part[j]['latent'] = latent_part.detach().cpu().clone()

                del sdf_values
                del voxel_origin
                del voxel_size
                del offset
                del scale
                del _

            decoder_part.train()

            decoder_shape.eval()
            with torch.no_grad():
                pred_sdf_full[j] = {}
                sdf_values, voxel_origin, voxel_size, offset, scale, _ = create_grid(
                    decoder_shape,
                    latent_shape,
                    N=GRID_RESOLUTION,
                    max_batch=int(2 ** 18),
                    input_samples=None,
                    add_feat=add_feat,
                    mode=mode
                )
                pred_sdf_full[j]['sdf'] = sdf_values.detach().cpu().clone()
                pred_sdf_full[j]['vox_origin'] = voxel_origin
                pred_sdf_full[j]['vox_size'] = voxel_size
                pred_sdf_full[j]['offset'] = offset
                pred_sdf_full[j]['scale'] = scale
                pred_sdf_full[j]['latent'] = latent_shape.detach().cpu().clone()

                del sdf_values
                del voxel_origin
                del voxel_size
                del offset
                del scale
                del _

            decoder_shape.train()

            j += 1

    return loss_num, all_latents, loss_part_history, pred_sdf_part, stats_before_tto, stats_after_tto, pred_sdf_full, latent_shape.detach()


class PointPartClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size_1, hidden_size_2, hidden_size_3):
        super(PointPartClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size_1)
        self.mlp2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.mlp3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.mlp4 = nn.Linear(hidden_size_3, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.mlp1(x))
        x = torch.relu(self.mlp2(x))
        x = torch.relu(self.mlp3(x))
        x = self.mlp4(x)
        x = self.activation(x)

        return x


class PointPartClassifierEntropy(nn.Module):

    def __init__(self, feature_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4):
        super(PointPartClassifierEntropy, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size_1)
        self.mlp2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.mlp3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.mlp4 = nn.Linear(hidden_size_3, hidden_size_4)

    def forward(self, x):
        x = torch.relu(self.mlp1(x))
        x = torch.relu(self.mlp2(x))
        x = torch.relu(self.mlp3(x))
        x = self.mlp4(x)

        return x


class PointPartClassifierEntropyPointNet(nn.Module):

    def __init__(self, feature_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, sigmoid=False):
        super(PointPartClassifierEntropyPointNet, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size_1)
        self.mlp2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.mlp3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.mlp4 = nn.Linear(2 * hidden_size_3, hidden_size_3)
        self.mlp5 = nn.Linear(hidden_size_3, hidden_size_3)
        self.mlp6 = nn.Linear(hidden_size_3, hidden_size_4)

        self.use_sigmoid = sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        feat1 = torch.relu(self.mlp1(input))
        feat2 = torch.relu(self.mlp2(feat1))
        feat3 = torch.relu(self.mlp3(feat2))
        global_feat = F.max_pool1d(feat3.T[None, ...], kernel_size=feat3.size()[0])[..., 0]
        global_feat = global_feat.repeat(len(input), 1)

        feat4 = torch.hstack([feat2, global_feat])
        feat5 = torch.relu(self.mlp4(feat4))
        feat6 = torch.relu(self.mlp5(feat5))
        feat7 = self.mlp6(feat6)

        if self.use_sigmoid:
            feat7 = self.sigmoid(feat7)

        return feat7


class PointPartClassifierEntropyPointNet2(nn.Module):

    def __init__(self, feature_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, hidden_size_5, sigmoid=False):
        super(PointPartClassifierEntropyPointNet2, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size_1)
        self.mlp2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.mlp3 = nn.Linear(hidden_size_2, hidden_size_3)

        self.mlp4 = nn.Linear(hidden_size_3, hidden_size_4)
        self.mlp5 = nn.Linear(hidden_size_4, hidden_size_4)

        self.mlp6 = nn.Linear(hidden_size_2 + hidden_size_3 + hidden_size_4, hidden_size_3)
        self.mlp7 = nn.Linear(hidden_size_3, hidden_size_4)
        self.mlp8 = nn.Linear(hidden_size_4, hidden_size_5)

        self.use_sigmoid = sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        feat1 = torch.relu(self.mlp1(input))
        feat2 = torch.relu(self.mlp2(feat1))
        feat3 = torch.relu(self.mlp3(feat2))
        global_feat1 = F.max_pool1d(feat3.T[None, ...], kernel_size=feat3.size()[0])[..., 0]
        global_feat1 = global_feat1.repeat(len(input), 1)

        feat4 = torch.relu(self.mlp4(feat3))
        feat5 = torch.relu(self.mlp5(feat4))
        global_feat2 = F.max_pool1d(feat5.T[None, ...], kernel_size=feat5.size()[0])[..., 0]
        global_feat2 = global_feat2.repeat(len(input), 1)

        feat6 = torch.hstack([feat2, global_feat1, global_feat2])
        feat7 = torch.relu(self.mlp6(feat6))
        feat8 = torch.relu(self.mlp7(feat7))
        feat9 = self.mlp8(feat8)

        if self.use_sigmoid:
            feat9 = self.sigmoid(feat9)

        return feat9


class RecursiveDeepSDFDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size, max_child_num, device,
                 edge_symmetric_type, num_iterations, edge_type_num,
                 num_parts=None, num_shapes=None, deep_sdf_specs=None, cat_name=None,
                 class2id=None, scene_aware_points_path=None):
        super(RecursiveDeepSDFDecoder, self).__init__()

        self.label_to_id = {
            'chair': 0,
            'bed': 1,
            'storage_furniture': 2,
            'table': 3,
            'trash_can': 4
        }
        self.cat_name = cat_name
        self.class2id = class2id
        self.scene_aware_points_path = scene_aware_points_path

        self.edge_types = ['ADJ', 'SYM']
        self.device = device
        self.max_child_num = max_child_num

        self.latent_decoder = LatentDecoder(feature_size, hidden_size, hidden_size)
        # self.root_classifier = NumChildrenClassifier(feature_size, hidden_size, 5)
        self.rotation_classifier = NumChildrenClassifier(feature_size, hidden_size, 12)
        self.child_decoder = GNNChildDecoder(feature_size, hidden_size,
                                             max_child_num, edge_symmetric_type,
                                             num_iterations, edge_type_num)

        # self.point_part_classifier = PointPartClassifierEntropyPointNet(332, 128, 128, 128, len(class2id) + 1)
        self.point_part_classifier = PointPartClassifierEntropyPointNet2(332, 256, 128, 128, 64, len(class2id) + 1) 

        # Part level
        deep_sdf_latent_size = feature_size
        deep_sdf_specs['dims'] = [512, 512, 512, 512, 512, 512, 512, 512]
        self.deepsdf_decoder = DeepSDFDecoder(deep_sdf_latent_size,
                                              mode=2,
                                              class_one_hot=True,
                                              cat_name=cat_name,
                                              **deep_sdf_specs).cuda()
        self.lat_vecs = torch.nn.Embedding(num_parts, deep_sdf_latent_size, max_norm=deep_sdf_specs['code_bound'])
        torch.nn.init.normal_(
            self.lat_vecs.weight.data,
            0.0,
            deep_sdf_specs['code_init_std_dev'] / math.sqrt(deep_sdf_latent_size),
        )
        self.lat_vecs.to(device)

        # Shape level
        deep_sdf_specs['dims'] = [512, 512, 512, 512, 512, 512, 512, 512]
        self.deepsdf_shape_decoder = DeepSDFDecoder(deep_sdf_latent_size,
                                                    mode=2,
                                                    **deep_sdf_specs).cuda()
        self.lat_shape_vecs = torch.nn.Embedding(num_shapes, deep_sdf_latent_size,
                                                 max_norm=deep_sdf_specs['code_bound'])
        torch.nn.init.normal_(
            self.lat_shape_vecs.weight.data,
            0.0,
            deep_sdf_specs['code_init_std_dev'] / math.sqrt(deep_sdf_latent_size),
        )
        self.lat_shape_vecs.to(device)

        self.shape_latent_projector = LatentProjector(feature_size, 512)
        self.part_latent_projector = LatentProjector(feature_size, 512)

        self.embedder, self.embedder_out_dim = get_embedder_nerf(10, input_dims=3, i=0)

        self.softmax_layer = nn.Softmax()

        self.mseLoss = nn.MSELoss(reduction='none')
        self.voxelLoss = nn.BCELoss(reduction='none')
        self.bceLoss = nn.BCELoss(reduction='none')
        self.semCELoss = nn.CrossEntropyLoss(reduction='none')
        self.childrenCELoss = nn.CrossEntropyLoss()
        self.iouLoss = IoULoss()
        self.L1Loss = nn.L1Loss()
        self.L1LossDeepSDF = torch.nn.L1Loss(reduction="sum")

        self.do_code_regularization = deep_sdf_specs['code_regularization']
        self.code_reg_lambda = deep_sdf_specs['code_regularization_lambda']

    def tto(self, children_initial_data, shape_initial_data=None, index=0, only_align=False, constr_mode=0, cat_name=None,
            num_shapes=0, k_near=0, scene_id='0', wconf=0, w_full_noise=1, w_part_u_noise=1,
            w_part_part_noise=1, lr_dec_full=0, lr_dec_part=0, target_sample_names=None, sa_mode=None,
            parts_indices=None, shape_idx=None, store_dir=None, class2id=None):

        print('Start TTO')
        # Start TTO process
        # Return all necessary outputs -- parts sdfs, shape sdfs, classified points, final losses, uniform and part noise
        parts_sdfs_pred, all_shape_sdf_pred, parts_stats_before, parts_stats_after \
            = self.adjust_embedding(children_initial_data, shape_initial_data,
                                    mode=1,
                                    only_align=only_align,
                                    constr_mode=constr_mode,
                                    cat_name=cat_name,
                                    num_shapes=num_shapes, k_near=k_near,
                                    wconf=wconf, scene_id=scene_id,
                                    w_full_noise=w_full_noise, w_part_u_noise=w_part_u_noise,
                                    w_part_part_noise=w_part_part_noise, lr_dec_full=lr_dec_full,
                                    lr_dec_part=lr_dec_part,
                                    target_sample_names=target_sample_names, sa_mode=sa_mode,
                                    parts_indices=parts_indices, shape_idx=shape_idx,
                                    store_dir=store_dir,
                                    class2id=class2id)
        print('Finish TTO')
        return parts_sdfs_pred, all_shape_sdf_pred, parts_stats_before, parts_stats_after

    def rebalance_points(self, all_points, uniform_noise, another_parts_noise, subsample=None):

        def sample_points(points, num_samples):
            if num_samples > len(points):
                all_indices = np.arange(len(points))
                indices = np.hstack([np.random.choice(len(points), num_samples - len(points)), all_indices])
            else:
                indices = np.random.choice(len(points), num_samples, replace=False)
            indices = torch.LongTensor(indices).cuda()
            sample = points[indices]
            return sample

        all_negatives = all_points[torch.where(all_points[:, 3] < 0.0)[0]]
        all_positives = all_points[torch.where((all_points[:, 3] > 0.0) & (all_points[:, 3] < 0.07))[0]]

        if subsample is None:
            subsample = 2 * max(len(all_negatives), len(all_positives))

        num_pos_samples = min(int(0.5 * subsample), 20000)
        num_neg_samples = min(int(0.5 * subsample), 20000)
        num_uniform_noise = 20000 # 20000
        num_parts_noise = min(int(0.2 * num_pos_samples), 20000) # 8000

        sample_uniform_noise = sample_points(uniform_noise, num_uniform_noise)
        sample_uniform_noise = sample_uniform_noise.cuda()
        if len(another_parts_noise) > 0:
            sample_parts_noise = sample_points(another_parts_noise, num_parts_noise)
        else:
            sample_parts_noise = sample_uniform_noise[:1]

        if len(all_positives) == 0:
            sample_pos = sample_uniform_noise
        else:
            sample_pos = sample_points(all_positives, num_pos_samples)
        if len(all_negatives) == 0:
            sample_neg = sample_uniform_noise
        else:
            sample_neg = sample_points(all_negatives, num_neg_samples)

        samples = torch.cat([sample_pos, sample_neg], 0)

        return samples, len(sample_pos), sample_uniform_noise, sample_parts_noise

    def adjust_embedding(self, children_data, shape_data, mode, only_align=False,
                         num_shapes=0, k_near=0, wconf=0, w_full_noise=1, w_part_u_noise=1,
                         w_part_part_noise=1, lr_dec_full=0, lr_dec_part=0, target_sample_names=None,
                         sa_mode=None, store_dir=None, class2id=None):

        # the main wrapper for TTO

        # only_align means to save results without optimization, only classification
        num_iterations = 1 if only_align else 601
        print('Only align without deformation:', only_align)

        # unpack predicted entities from forward pass
        children_embeddings = children_data[0]
        children_names = children_data[1]
        children_sdf_points = children_data[2]
        extra_sdf_points = children_data[4]
        if mode == 1:
            children_feats = children_data[3]
        else:
            children_feats = [0 for _ in range(len(children_names))]
        chamfer_dist_parts_before_secdir = children_data[5]
        uniform_noise = shape_data[3]
        uniform_noise = uniform_noise.cuda()

        # prepare DeepSDF before optimization and dicts to store TTO outputs
        parts_sdfs_pred = {}
        parts_stats_before = {}
        parts_stats_after = {}
        initial_parameters = deepcopy(self.deepsdf_decoder.state_dict())
        self.deepsdf_decoder.train()
        for param in self.deepsdf_decoder.parameters():
            param.requires_grad = True
        self.deepsdf_shape_decoder.train()
        for param in self.deepsdf_shape_decoder.parameters():
            param.requires_grad = True
        all_pred_sdf_full = []

        # add points before TTO to incomplete regions
        before_tto_points = {}
        for k, child_name in enumerate(children_names):
            before_tto_points[child_name] = extra_sdf_points[k].cpu().detach().numpy()
        full_shape_points = torch.vstack(children_sdf_points).cpu().detach().numpy()
        indices = np.where(np.abs(full_shape_points[:, 3]) < 0.007)[0] # 0.007
        full_shape_points = full_shape_points[indices]
        full_shape_points = full_shape_points[:, :3]
        full_shape_tree = cKDTree(full_shape_points)
        additional_points_from_pred = {}
        for child_name in before_tto_points:
            if child_name not in ['chair_arm_left', 'chair_arm_right', 'chair_back', 'chair_seat']:
                min_dist, min_idx = full_shape_tree.query(before_tto_points[child_name][:, :3])
                above_thr_indices = np.where(min_dist > 0.1)[0] # 0.07 <-- add points from before TTO to full shape and/or parts with dist > dist_thr
                additional_points_from_pred[child_name] = before_tto_points[child_name][above_thr_indices]
        all_points = []
        for child_name in additional_points_from_pred:
            if chamfer_dist_parts_before_secdir[child_name] > 90:
                all_points += [additional_points_from_pred[child_name]]
        if len(all_points) > 0:
            additional_points_from_pred['full'] = np.vstack(all_points)
        else:
            additional_points_from_pred['full'] = []

        # Main cycle for TTO per child
        for i, child_name in enumerate(children_names):

            try:
                # unpack entities for this part
                parts_sdfs_pred[child_name] = {}
                sdf_part = children_sdf_points[i]
                idx_latent = class2id[child_name]
                class_one_hot = torch.FloatTensor(torch.zeros((len(class2id))))
                class_one_hot[idx_latent] = 1

                # subsample points for target part
                try:
                    sdf_part = torch.stack([x for x in sdf_part if x[3] > -0.025])
                    indices = np.random.choice(len(sdf_part), min(80000, len(sdf_part)), replace=False)
                    sdf_part = sdf_part[indices]
                except:
                    sdf_part = torch.vstack(children_sdf_points)

                # add points from other parts as noise
                try:
                    sdf_another_parts = [torch.vstack([children_sdf_points[j], extra_sdf_points[j].cuda()]) for j in range(len(children_sdf_points)) if j != i]
                    another_parts_random_indices = [np.random.choice(len(x), min(len(x), 7500)) for x in sdf_another_parts] # 10000
                    another_parts_points = torch.vstack([sdf_another_parts[j][another_parts_random_indices[j]] for j in range(len(sdf_another_parts))])
                    another_parts_points[:, 3] = 0.07
                except:
                    another_parts_points = torch.FloatTensor(np.array([[0, 0, 0, 0.0]])).cuda()

                # filter some noise points that are close to current part
                kd_tree = cKDTree(np.vstack([sdf_part.cpu().numpy()[:, :3], extra_sdf_points[i].cpu().numpy()[:, :3]]))
                min_dist_rand, min_idx_rand = kd_tree.query(another_parts_points.cpu().numpy()[:, :3])
                above_thr_indices = torch.LongTensor(np.where(min_dist_rand > 0.02)[0]).cuda() # 0.1
                another_parts_noise = another_parts_points[above_thr_indices]

                # self.scene_aware_points_path = f'/cluster/daidalos/abokhovkin/scannet-relationships/test_output_full_eccv/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_{cat_name}_0.25_0'
                # self.scene_aware_points_path = f'/cluster/valinor/abokhovkin/scannet-relationships-v2/test_output_full_cvpr/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_{cat_name}'
                
                # global points voting
                if self.scene_aware_points_path is not None:
                    sdf_part = sdf_part.cpu()
                    all_children_extra_points = []
                    for sample_name in target_sample_names[:min(num_shapes, len(target_sample_names))]:
                        child_points_path = os.path.join(self.scene_aware_points_path, f'{sample_name}_tto', f'{child_name}.pts.npy')
                        child_sdfs_path = os.path.join(self.scene_aware_points_path, f'{sample_name}_tto', f'{child_name}.sdf.npy')
                        if os.path.exists(child_points_path):
                            child_points = np.load(child_points_path)[:, -3:]
                            child_sdfs = np.load(child_sdfs_path)
                            indices = np.random.choice(len(child_points), min(40000, len(child_points)), replace=False)
                            child_points = child_points[indices]
                            child_sdfs = child_sdfs[indices]
                            child_samples = np.hstack([child_points, child_sdfs])
                            all_children_extra_points += [child_samples]
                    if len(all_children_extra_points) > 0:
                        all_children_extra_points = np.vstack(all_children_extra_points)
                        all_children_extra_points_and_source = np.vstack([all_children_extra_points, sdf_part.cpu().numpy()])
                        indices = np.random.choice(len(all_children_extra_points_and_source), min(30000, len(all_children_extra_points_and_source)), replace=False)
                        all_points_sampled = all_children_extra_points_and_source[indices]
                        kd_tree_avg = cKDTree(all_children_extra_points_and_source)
                        min_dist_avg, min_idx_avg = kd_tree_avg.query(all_points_sampled, k=k_near)
                        if sa_mode == 'min':
                            all_points_sampled[:, 3] = all_children_extra_points_and_source[:, 3][min_idx_avg].min(axis=1)
                        elif sa_mode == 'mean':
                            all_points_sampled[:, 3] = all_children_extra_points_and_source[:, 3][min_idx_avg].mean(axis=1)
                        sdf_part = torch.FloatTensor(all_points_sampled).cuda()

                # equalize number of positive/negative/noise points
                sdf_part, num_positives, part_uniform_noise, part_another_parts_noise = self.rebalance_points(sdf_part, uniform_noise, another_parts_noise)
                sdf_part = sdf_part.cuda()

                # add additional points from prediction to parts
                if child_name in additional_points_from_pred:
                    if len(additional_points_from_pred[child_name]) > 0:
                        sdf_part = torch.vstack([sdf_part,
                                                 torch.FloatTensor(additional_points_from_pred[child_name]).cuda()])

                if i == 0:
                    shape_embedding = shape_data[0]
                    # shape_embedding = torch.rand(shape_embedding.shape).cuda() / 10.
                    # part_embedding = torch.rand(children_embeddings[i].shape).cuda() / 10.
                    sdf_data = shape_data[1]
                    sdf_data = torch.stack([x for x in sdf_data if x[3] > -0.025])
                    indices = np.random.choice(len(sdf_data), min(120000, len(sdf_data)), replace=False)
                    sdf_data = sdf_data[indices]

                    # add additional points from prediction to shape
                    if len(additional_points_from_pred['full']) > 0:
                        sdf_data = torch.vstack([sdf_data,
                                                 torch.FloatTensor(additional_points_from_pred['full']).cuda()])

                    # add uniform noise to full point cloud
                    sdf_data = torch.vstack([sdf_data, uniform_noise]) # <-- this is unnecessary as uniform noise already presented in sdf_data

                # launch TTO
                err, all_latents, loss_history, pred_sdf_part, stats_before_tto, stats_after_tto, pred_sdf_shape, shape_embedding = reconstruct_part_and_shape_tto(
                    self.deepsdf_decoder,
                    self.deepsdf_shape_decoder,
                    num_iterations,
                    sdf_part,
                    part_uniform_noise,
                    part_another_parts_noise,
                    sdf_data,
                    0.07,
                    num_samples=len(sdf_part),
                    lr_1=3e-4,
                    lr_2=lr_dec_part, 
                    lr_3=lr_dec_full, 
                    w_cons=wconf,
                    w_full_noise=w_full_noise,
                    w_part_u_noise=w_part_u_noise,
                    w_part_part_noise=w_part_part_noise,
                    l2reg=True,
                    cal_scale=1,
                    add_feat=children_feats[i],
                    init_emb_part=children_embeddings[i], 
                    init_emb_shape=shape_embedding,
                    mode=2,
                    class_one_hot=class_one_hot,
                    store_dir=store_dir,
                    child_name=child_name
                )

                parts_sdfs_pred[child_name] = pred_sdf_part
                parts_stats_before[child_name] = stats_before_tto
                parts_stats_after[child_name] = stats_after_tto

                del pred_sdf_part
                del stats_before_tto
                del stats_after_tto

                self.deepsdf_decoder.load_state_dict(initial_parameters)
                all_pred_sdf_full += [pred_sdf_shape]
            except NotImplementedError:
                continue

        return parts_sdfs_pred, all_pred_sdf_full, \
               parts_stats_before, parts_stats_after

    def latent_recon_loss(self, z, gt_tree, sdf_data, encoder_features=None, rotation=None,
                          parts_indices=None, epoch=0, full_shape_idx=None, noise_full=None,
                          rotations=None, class2id=None):
        root_latent = self.latent_decoder(z)  # torch.Tensor[bsize, 256], torch.Tensor[bsize, 256]
        output = self.node_latent_loss(root_latent, gt_tree.root, sdf_data, level=0,
                                       encoder_features=encoder_features, rotation=rotation,
                                       parts_indices=parts_indices, epoch=epoch,
                                       full_shape_idx=full_shape_idx,
                                       noise_full=noise_full,
                                       rotations=rotations,
                                       class2id=class2id)
        return output

    def node_latent_loss(self, node_latent, gt_node, sdf_data, level=0,
                         encoder_features=None, rotation=None,
                         pred_rotation=None, parts_indices=None, epoch=0,
                         full_shape_idx=None, noise_full=None,
                         rotations=None, class2id=None):

        if level == 0:
            child_names = []
            for child in gt_node.children:
                if child.label in class2id:
                    child_names += [child.label]

        cuda_device = node_latent.get_device()

        if level == 0:
            rotation_cls_pred = self.rotation_classifier(node_latent)
            rotation_cls_gt = torch.zeros(1, dtype=torch.long)
            rotation_cls_gt[0] = rotations
            rotation_cls_gt = rotation_cls_gt.to("cuda:{}".format(cuda_device))
            rotation_cls_CE_loss = self.childrenCELoss(rotation_cls_pred, rotation_cls_gt).mean()
            pred_rotation = self.softmax_layer(rotation_cls_pred)
            pred_rotation = int(torch.argmax(pred_rotation).cpu().detach().numpy())

        if level == 1:
            loss_dict = {}

            loss_dict['exists'] = torch.zeros((1, 1)).to(cuda_device)
            loss_dict['semantic'] = torch.zeros((1, 1)).to(cuda_device)
            loss_dict['edge_exists'] = torch.zeros((1, 1)).to(cuda_device)

            return loss_dict
        else:

            loss_dict = {}
            child_feats, child_sem_logits, child_exists_logits, edge_exists_logits = \
                self.child_decoder(node_latent)

            num_pred = child_feats.size(1)

            with torch.no_grad():
                child_gt_geo = torch.cat(
                    [child_node.geo for child_node in gt_node.children], dim=0)
                child_gt_geo = child_gt_geo.unsqueeze(dim=1)
                num_gt = child_gt_geo.size(0)

                child_gt_sem_vectors = []
                child_gt_sem_classes = torch.zeros(num_gt)
                for j, child_node in enumerate(gt_node.children):
                    child_gt_sem_vector = torch.zeros((1, Tree.num_sem))
                    child_gt_sem_vector[0, child_node.get_semantic_id()] = 1
                    child_gt_sem_vectors += [child_gt_sem_vector]
                    child_gt_sem_classes[j] = child_node.get_semantic_id()
                child_gt_sem_classes = child_gt_sem_classes.long()

                child_sem_logits_tiled = child_sem_logits[0].unsqueeze(dim=0).repeat(num_gt, 1, 1)
                child_gt_sem_classes_tiled = child_gt_sem_classes.unsqueeze(dim=1).repeat(1, num_pred).to(cuda_device)

                # get edge ground truth
                edge_type_list_gt, edge_indices_gt = gt_node.edge_tensors(
                    edge_types=self.edge_types, device=child_feats.device, type_onehot=False)

                dist_mat = self.semCELoss(child_sem_logits_tiled.view(-1, Tree.num_sem),
                                          child_gt_sem_classes_tiled.view(-1)).view(1, num_gt, num_pred)

                try:
                    _, matched_gt_idx, matched_pred_idx = linear_assignment(dist_mat)
                except ValueError:
                    print(dist_mat)

                gt2pred = {gt_idx: pred_idx for gt_idx, pred_idx in zip(matched_gt_idx, matched_pred_idx)}
                edge_exists_gt = torch.zeros_like(edge_exists_logits)

                adj_from = []
                adj_to = []
                sym_from = []
                sym_to = []
                for i in range(edge_indices_gt.shape[1] // 2):
                    if edge_indices_gt[0, i, 0].item() not in gt2pred or edge_indices_gt[0, i, 1].item() not in gt2pred:
                        """
                            one of the adjacent nodes of the current gt edge was not matched
                            to any node in the prediction, ignore this edge
                        """
                        continue

                    # correlate gt edges to pred edges
                    edge_from_idx = gt2pred[edge_indices_gt[0, i, 0].item()]
                    edge_to_idx = gt2pred[edge_indices_gt[0, i, 1].item()]
                    edge_exists_gt[:, edge_from_idx, edge_to_idx, edge_type_list_gt[0:1, i]] = 1
                    edge_exists_gt[:, edge_to_idx, edge_from_idx, edge_type_list_gt[0:1, i]] = 1

                    # compute binary edge parameters for each matched pred edge
                    if edge_type_list_gt[0, i].item() == 0:  # ADJ
                        adj_from.append(edge_from_idx)
                        adj_to.append(edge_to_idx)
                    else:  # SYM
                        sym_from.append(edge_from_idx)
                        sym_to.append(edge_to_idx)

            # Process the input data
            num_sdf_samples_per_shape = sdf_data.shape[1]
            sdf_data_flattened = sdf_data.reshape(-1, 4)
            sdf_data_flattened.requires_grad = False

            # Process children features
            gt_parts_indices = []
            gt_parts_indices_split = []
            corresponding_children_feats = []
            for gt_idx, pred_idx in gt2pred.items():
                gt_child_name = gt_node.children[gt_idx].label
                gt_parts_indices += [parts_indices[gt_child_name] for _ in range(num_sdf_samples_per_shape)]
                gt_parts_indices_split += [[parts_indices[gt_child_name] for _ in range(num_sdf_samples_per_shape)]]
                corresponding_children_feats += [child_feats[:, pred_idx, ...].repeat(num_sdf_samples_per_shape, 1)]


            # train point to part associations (entropy)
            all_sdf_data = []
            all_points_labels = []
            for k, n in enumerate(range(len(sdf_data))):
                _sdf_data = sdf_data[n][:, 0:4]
                all_sdf_data += [_sdf_data]
                child_name = child_names[k]
                child_id = self.class2id[child_name] + 1
                gt_labels = (torch.ones((len(_sdf_data))).long() * child_id).cuda()
                all_points_labels += [gt_labels]
            all_sdf_data += [noise_full]
            noise_labels = (torch.ones((len(noise_full))).long() * 0).cuda()
            all_points_labels += [noise_labels]
            all_sdf_data = torch.cat(all_sdf_data, dim=0)
            all_sdf_data_pe = self.embedder(all_sdf_data[:, :3])
            all_sdf_data_pe = torch.cat([all_sdf_data_pe, all_sdf_data[:, 3:4]], dim=1)
            # concat rotation vector
            node_latent_rotation = torch.hstack([node_latent, rotation_cls_pred])
            all_node_latents = node_latent_rotation.repeat(len(all_sdf_data_pe), 1)
            all_sdf_data_with_feats = torch.cat([all_sdf_data_pe, all_node_latents], dim=1) # all_sdf_data_pe / all_sdf_data
            all_points_labels = torch.cat(all_points_labels, dim=0)
            point_part_pred = self.point_part_classifier(all_sdf_data_with_feats)
            point_part_loss = self.childrenCELoss(point_part_pred, all_points_labels).mean()
            loss_dict['point_part'] = point_part_loss

            # train MSE for full shape
            full_shape_idx = torch.tensor(full_shape_idx).to(cuda_device)
            batch_shape_vecs = self.lat_shape_vecs(full_shape_idx)[None, :]
            node_latent_projected = self.shape_latent_projector(node_latent)
            shape_mse_loss = self.mseLoss(node_latent_projected, batch_shape_vecs.detach().clone())
            loss_dict['mse_shape'] = shape_mse_loss.mean()

            children_feats = []
            gt_latent_parts_indices = []
            for gt_idx, pred_idx in gt2pred.items():
                gt_child_name = gt_node.children[gt_idx].label
                gt_latent_parts_indices += [parts_indices[gt_child_name]]
                children_feats += [child_feats[:, pred_idx, ...]]
            children_feats = torch.vstack(children_feats)
            gt_latent_parts_indices = torch.tensor(gt_latent_parts_indices).to(cuda_device)
            batch_vecs = self.lat_vecs(gt_latent_parts_indices)[:, :256]
            children_feats_projected = self.part_latent_projector(children_feats)
            parts_mse_loss = self.mseLoss(children_feats_projected, batch_vecs.detach().clone())
            loss_dict['mse_parts'] = parts_mse_loss.mean()

            # gather information
            child_sem_gt_labels = []
            child_sem_pred_logits = []
            child_exists_gt = torch.zeros_like(child_exists_logits)
            for i in range(len(matched_gt_idx)):
                child_sem_gt_labels.append(gt_node.children[matched_gt_idx[i]].get_semantic_id())
                child_sem_pred_logits.append(child_sem_logits[0, matched_pred_idx[i], :].view(1, -1))
                child_exists_gt[:, matched_pred_idx[i], :] = 1

            # train semantic labels
            child_sem_pred_logits = torch.cat(child_sem_pred_logits, dim=0)
            child_sem_gt_labels = torch.tensor(child_sem_gt_labels, dtype=torch.int64,
                                               device=child_sem_pred_logits.device)
            semantic_loss = self.semCELoss(child_sem_pred_logits, child_sem_gt_labels)
            semantic_loss = semantic_loss.sum()

            # train exist scores
            child_exists_loss = F.binary_cross_entropy_with_logits(
                input=child_exists_logits,
                target=child_exists_gt,
                reduction='none')
            child_exists_loss = child_exists_loss.sum()

            # train edge exists scores
            edge_exists_loss = F.binary_cross_entropy_with_logits(
                input=edge_exists_logits,
                target=edge_exists_gt,
                reduction='none')
            edge_exists_loss = edge_exists_loss.sum()
            # rescale to make it comparable to other losses,
            # which are in the order of the number of child nodes
            edge_exists_loss = edge_exists_loss / (edge_exists_gt.shape[2] * edge_exists_gt.shape[3])

            # call children + aggregate losses
            for i in range(len(matched_gt_idx)):
                child_losses = self.node_latent_loss(
                    child_feats[:, matched_pred_idx[i], :], gt_node.children[matched_gt_idx[i]], sdf_data, level + 1,
                    encoder_features=encoder_features, rotation=rotation,
                    pred_rotation=pred_rotation, parts_indices=parts_indices, epoch=epoch
                )

                child_exists_loss = child_exists_loss + child_losses['exists']
                semantic_loss = semantic_loss + child_losses['semantic']
                edge_exists_loss = edge_exists_loss + child_losses['edge_exists']

            loss_dict['exists'] = child_exists_loss.view((1))
            loss_dict['semantic'] = semantic_loss.view((1))
            loss_dict['edge_exists'] = edge_exists_loss.view((1))
            loss_dict['rotation'] = rotation_cls_CE_loss.view((1))

            return loss_dict, pred_rotation

    def get_latent_vecs(self):
        return self.lat_vecs

    def get_latent_shape_vecs(self):
        return self.lat_shape_vecs

    # decode a root code into a tree structure
    def decode_structure_two_stage(self, z, sdf_data, full_label=None, encoder_features=None, rotation=None,
                                   gt_tree=None, index=0, parts_indices=None, full_shape_idx=None, noise_full=None,
                                   rot_aug=0, shift_x=0, shift_y=0, bck_thr=0.5, cat_name=None, scale=1):
        root_latent = self.latent_decoder(z)  # torch.Tensor[bsize, 256], torch.Tensor[bsize, 256]
        if full_label is None:
            full_label = Tree.root_sem
        output = self.decode_node_two_stage(root_latent, sdf_data, full_label=full_label, level=0,
                                            encoder_features=encoder_features, rotation=rotation,
                                            gt_tree=gt_tree, index=index, parts_indices=parts_indices,
                                            full_shape_idx=full_shape_idx, noise_full=noise_full,
                                            rot_aug=rot_aug, shift_x=shift_x, shift_y=shift_y,
                                            bck_thr=bck_thr, cat_name=cat_name, scale=scale)

        obj = Tree(root=output[0])
        new_output = [obj, ]
        for i in range(1, len(output)):
            new_output += [output[i]]
        new_output = tuple(new_output)
        return new_output

    def evaluate_confidence(self, parts_sdfs_pred, splitted_sdf_points, children_names):

        chamfer_module = ChamferDistance()
        chamfer_dist_parts_before_onedir = {}
        chamfer_dist_parts_before_secdir = {}

        for child_name in children_names:
            if child_name in parts_sdfs_pred and child_name in splitted_sdf_points:
                predicted_samples = parts_sdfs_pred[child_name]['samples']
                inside_shape_indices = torch.where(predicted_samples[:, 3] < 0.0)[0]
                random_indices = np.random.choice(inside_shape_indices, min(5000, len(inside_shape_indices)), replace=False)
                predicted_samples = predicted_samples[random_indices][:, :3]

                scannet_samples = splitted_sdf_points[child_name].cpu()
                inside_shape_indices = torch.where((scannet_samples[:, 3] < 0.0) & (scannet_samples[:, 3] > -0.005))[0]
                random_indices = np.random.choice(inside_shape_indices, min(5000, len(inside_shape_indices)), replace=False)
                scannet_samples = scannet_samples[random_indices][:, :3]

                pc_1 = scannet_samples[None, ...].cpu()
                pc_2 = predicted_samples[None, ...].cpu()
                dist1 = chamfer_module(pc_1, pc_2, reduction="mean")
                dist2 = chamfer_module(pc_2, pc_1, reduction="mean")
                chamfer_dist_parts_before_onedir[child_name] = dist1.cpu().numpy()
                chamfer_dist_parts_before_secdir[child_name] = dist2.cpu().numpy()

        return chamfer_dist_parts_before_onedir, chamfer_dist_parts_before_secdir

    # decode a part node
    def decode_node_two_stage(self, node_latent, sdf_data, full_label, level=0, encoder_features=None, rotation=None,
                              pred_rotation=None, gt_tree=None, index=0, parts_indices=None,
                              full_shape_idx=None, noise_full=None,
                              rot_aug=0, shift_x=0, shift_y=0,
                              bck_thr=0.5, cat_name=None, scale=1):

        def perform_30_rot(pc, rot):
            angle = 30 * rot
            angle = np.pi * angle / 180.

            a, b = np.cos(angle), np.sin(angle)
            matrix = np.array([[a, 0, b],
                               [0, 1, 0],
                               [-b, 0, a]])
            matrix = torch.FloatTensor(matrix).to(pc.device)
            pc[:, :3] = pc[:, :3] @ matrix.T

            return pc, matrix

        # inference method to predict all entities that are necessary for TTO
        if level == 1:
            return Tree.Node(is_leaf=True, full_label=full_label, label=full_label.split('/')[-1], geo=0)
        else:
            # Predict rotation for the input object
            rotation_cls_pred = self.rotation_classifier(node_latent)
            pred_rotation = self.softmax_layer(rotation_cls_pred)
            pred_rotation = int(torch.argmax(pred_rotation).cpu().detach().numpy())

            # decode all latents for children (features, semantics, existance, edge existance)
            child_feats, child_sem_logits, child_exists_logit, edge_exists_logits = \
                self.child_decoder(node_latent)
            # torch.Tensor[bsize, max_child_num, 256]
            # torch.Tensor[bsize, max_child_num, tree.num_sem]
            # torch.Tensor[bsize, max_child_num, 1]
            # torch.Tensor[bsize, max_child_num, max_child_num, 4]

            child_sem_logits_numpy = child_sem_logits.cpu().detach().numpy().squeeze()
            # torch.Tensor[max_child_num, tree.num_sem]

            # recursively predict children nodes using the same 'self.decode_node_two_stage' method
            child_nodes = []  # list[exist_child_num]<Tree.Node>
            child_idx = {}
            child_feats_exist = []
            for ci in range(child_feats.shape[1]):
                if torch.sigmoid(child_exists_logit[:, ci, :]).item() > 0.5:
                    idx = np.argmax(child_sem_logits_numpy[ci, Tree.part_name2cids[full_label]])  # int
                    idx = Tree.part_name2cids[full_label][idx]  # int (1 <= idx <= tree.num_sem)
                    child_full_label = Tree.part_id2name[idx]
                    child_node = self.decode_node_two_stage(
                        child_feats[:, ci, :], sdf_data, child_full_label, level=level + 1,
                        encoder_features=encoder_features, rotation=rotation,
                        pred_rotation=pred_rotation, gt_tree=gt_tree)
                    child_nodes.append(child_node)
                    child_idx[ci] = len(child_nodes) - 1
                    child_feats_exist.append(child_feats[:, ci, :])

            # predict edges between parts
            child_edges = []  # list[<=nnz_num]<dict['part_a', 'part_b', 'type']>
            nz_inds = torch.nonzero(torch.sigmoid(edge_exists_logits) > 0.5)  # torch.Tensor[nnz_num, 4]
            edge_from = nz_inds[:, 1]  # torch.Tensor[nnz_num]
            edge_to = nz_inds[:, 2]  # torch.Tensor[nnz_num]
            edge_type = nz_inds[:, 3]  # torch.Tensor[nnz_num]

            for i in range(edge_from.numel()):
                cur_edge_from = edge_from[i].item()
                cur_edge_to = edge_to[i].item()
                cur_edge_type = edge_type[i].item()

                if cur_edge_from in child_idx and cur_edge_to in child_idx:
                    child_edges.append({
                        'part_a': child_idx[cur_edge_from],
                        'part_b': child_idx[cur_edge_to],
                        'type': self.edge_types[cur_edge_type]})

            sdf_data_flattened = sdf_data.reshape(-1, 4)

            # input point cloud
            sdf_flat = deepcopy(sdf_data_flattened.cpu().detach())

            # rotated point cloud
            sdf_flat_rot = deepcopy(sdf_data_flattened.cpu().detach())
            sdf_not_noise = torch.stack([x for x in sdf_data_flattened if x[3] < 0.07]).cuda()

            # use PointNet for points classification
            sdf_data_flattened = sdf_not_noise.cuda()
            num_samples = len(sdf_data_flattened)
            node_latent_rotation = torch.hstack([node_latent, rotation_cls_pred])
            node_feat_expanded = node_latent_rotation.expand(num_samples, -1)
            all_sdf_data_pe = self.embedder(sdf_data_flattened[:, :3])
            all_sdf_data_pe = torch.cat([all_sdf_data_pe, sdf_data_flattened[:, 3:4]], dim=1)
            inputs = torch.cat([all_sdf_data_pe, node_feat_expanded], 1).cuda()
            pred_accociations = self.point_part_classifier(inputs)
            pred_accociations = torch.max(pred_accociations, dim=1)[1]
            splitted_sdf_points = {}
            for i in range(len(child_feats_exist)):
                child_name = child_nodes[i].label
                part_point_indices = torch.where(pred_accociations == self.class2id[child_name] + 1)[0]
                valid_points = [x for x in sdf_data_flattened[part_point_indices] if x[3] < 0.07]
                if len(valid_points) == 0:
                    valid_points = (torch.FloatTensor([0, 0, 0, 0.7]),)
                sdf_not_noise = torch.stack(valid_points).cuda()
                splitted_sdf_points[child_name] = torch.vstack([sdf_not_noise])

            # perform rotation using predicted 30 grad sector
            # test - pred_rotation, train/val - -pred_rotation
            # (new models) test - -pred_rotation
            for child_name in splitted_sdf_points:
                splitted_sdf_points[child_name][:, :3], rot_matrix = perform_30_rot(splitted_sdf_points[child_name][:, :3], -pred_rotation)
            # test - pred_rotation, train/val - -pred_rotation (new training approaches)
            sdf_data_flattened[:, :3], rot_matrix = perform_30_rot(sdf_data_flattened[:, :3], pred_rotation)

            # retrieve GT part embeddings (only for testing or ablation)
            children_names = [x.label for x in gt_tree.root.children]

            # predict projection of children features onto DeepSDF embedding space
            child_feats_exist_projection = self.part_latent_projector(torch.vstack(child_feats_exist))
            # reconstruct parts
            parts_sdfs_pred, children_embeddings, children_names, children_sdf_points, children_feats, extra_sdf_points = self.reconstruct_children_from_latents(
                child_nodes,
                child_feats_exist_projection,
                splitted_sdf_points,
                cat_name,
                children_names,
                mode=2
            )
            for k in range(len(children_sdf_points)):
                children_sdf_points[k] = children_sdf_points[k].cuda()

            # compute CD between parts and corresponding part point clouds
            chamfer_dist_parts_before_onedir, chamfer_dist_parts_before_secdir = self.evaluate_confidence(parts_sdfs_pred, splitted_sdf_points, children_names)

            # ICP SECTION #
            # perform ICP from points to predicted meshes
            # take points from ScanNet (ICP)
            all_points_icp = torch.vstack(children_sdf_points)
            indices = torch.where(all_points_icp[:, 3] < 0.01)[0]
            all_points_icp = all_points_icp[indices].cpu().detach().numpy()
            random_indices = np.random.choice(len(all_points_icp), min(20000, len(all_points_icp)), replace=False)
            all_points_icp = all_points_icp[random_indices]

            # take points from predicted mesh (ICP)
            all_points_mesh = []
            for child_name in parts_sdfs_pred:
                samples = parts_sdfs_pred[child_name]['samples']
                indices = torch.where(samples[:, 3] < 0.01)[0]
                samples = samples[indices].numpy()
                all_points_mesh += [samples]
            all_points_mesh = np.vstack(all_points_mesh)
            random_indices = np.random.choice(len(all_points_mesh), min(5000, len(all_points_mesh)), replace=False)
            all_points_mesh = all_points_mesh[random_indices]

            # apply ICP
            sdf_data_pcd = o3d.geometry.PointCloud()
            sdf_data_pcd.points = o3d.utility.Vector3dVector(all_points_icp[:, :3])
            mesh_pcd = o3d.geometry.PointCloud()
            mesh_pcd.points = o3d.utility.Vector3dVector(all_points_mesh[:, :3])
            scan_points_icp = deepcopy(all_points_icp[:, :3])
            mesh_points_icp = deepcopy(all_points_mesh[:, :3])
            threshold = 0.1
            reg_p2p = o3d.pipelines.registration.registration_icp(
                sdf_data_pcd, mesh_pcd, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

            # apply ICP corrections
            t, q, s = decompose_mat4(np.array(reg_p2p.transformation))
            q = np.array([q.w, q.x, q.y, q.z])
            r = Rotation.from_quat(q)
            roteuler = r.as_euler('zxy', degrees=True)
            roteuler[1] = 0
            roteuler[0] = 180
            s = s * scale
            corrected_r = Rotation.from_euler('zxy', [roteuler[0], roteuler[1], roteuler[2]], degrees=True)
            corrected_icp = from_tqs_to_matrix(t, corrected_r.as_quat(), s)
            transform = torch.FloatTensor(corrected_icp).cuda()
            icp = deepcopy(transform.cpu().detach())

            # apply ICP transform to points
            for k in range(len(children_sdf_points)):
                children_sdf_points[k][:, :3] = apply_transform_torch(children_sdf_points[k][:, :3], transform)
            for k in range(len(extra_sdf_points)):
                extra_sdf_points[k] = extra_sdf_points[k].cuda()

            # change table legs with synthetics
            for k in range(len(children_sdf_points)):
                if children_names[k] == 'leg':
                    children_sdf_points[k] = children_sdf_points[k][:2]
                    splitted_sdf_points['leg'] = splitted_sdf_points['leg'][:2]

            filtered_sdf_points = []
            for k in range(len(children_sdf_points)):
                if children_names[k] in ['table_surface', 'central_support', 'pedestal']:
                    filtered_sdf_points += [children_sdf_points[k]]
            all_parts_points = torch.vstack(extra_sdf_points + filtered_sdf_points).cpu().numpy()[:, :3] # extra_sdf_points + children_sdf_points
            num_samples_random = 50000
            random_samples = np.random.uniform(-1.0, 1.0, (num_samples_random, 3))
            kd_tree = cKDTree(all_parts_points)
            min_dist_rand, min_idx_rand = kd_tree.query(random_samples)
            above_thr_indices = np.where(min_dist_rand > 0.08)[0]  # 0.1 <--- noise_thr, noise from before TTO
            above_thr_points = random_samples[above_thr_indices]
            above_thr_points = np.hstack([above_thr_points, 0.07 * np.ones((len(above_thr_points), 1))]).astype('float32')
            above_thr_points = torch.FloatTensor(above_thr_points)

            # form outputs for the following TTO stage
            pred_output = (children_embeddings, children_names, children_sdf_points, children_feats, extra_sdf_points, chamfer_dist_parts_before_secdir)
            # predict full shape projection onto DeepSDF full shape embedding space
            node_latent_projection = self.shape_latent_projector(node_latent)
            # reconstruct full shape
            shape_sdf_pred, shape_embedding, sdf_data, shape_feat, samples = self.reconstruct_full_shape_from_latents(
                node_latent_projection,
                torch.vstack(list(splitted_sdf_points.values())),
                mode=2)
            sdf_data = sdf_data.cuda()

            # add noise to full shape points
            random_indices = np.random.choice(len(above_thr_points), min(len(above_thr_points), 25000), replace=False)
            above_thr_points_sampled = above_thr_points[random_indices]
            sdf_data = torch.vstack([sdf_data.cuda(), above_thr_points_sampled.cuda()])

            # apply ICP transform to full shape points + uniform noise
            sdf_data[:, :3] = apply_transform_torch(sdf_data[:, :3], transform)

            # form outputs for the following TTO stage
            shape_output = (shape_embedding, sdf_data, shape_feat, above_thr_points)

            # form auxiliary data for testing
            sdf_full_unrot = deepcopy(sdf_data.cpu().detach())
            meta_data = (sdf_flat, sdf_flat_rot, sdf_full_unrot, scan_points_icp, mesh_points_icp, icp, rot_matrix, pred_rotation)

            return (Tree.Node(is_leaf=False, children=child_nodes, edges=child_edges,
                              full_label=full_label, label=full_label.split('/')[-1], geo=0),
                    parts_sdfs_pred,
                    pred_output,
                    shape_sdf_pred,
                    shape_output,
                    meta_data)

    def reconstruct_children_from_latents(self, child_nodes, child_feats, splitted_sdf_points, cat_name, child_names, mode):

        parts_sdfs_pred = {}
        children_embeddings = []
        children_names = []
        children_sdf_points = []
        children_feats = []
        extra_sdf_points = []
        for i in range(len(child_feats)):

            # unpack input entities
            child_node = child_nodes[i]
            child_name = child_node.label
            if child_name == 'surface_base' and child_name not in splitted_sdf_points:
                child_name = 'regular_leg_base'
            parts_sdfs_pred[child_name] = {}
            sdf_part = splitted_sdf_points[child_name]
            children_names += [child_name]
            children_feats += [child_feats[i].detach().cpu().clone()]

            children_sdf_points += [sdf_part.detach().cpu().clone()]

            idx_latent = self.class2id[child_name]
            class_one_hot = torch.FloatTensor(torch.zeros((len(self.class2id))))
            class_one_hot[idx_latent] = 1

            # perform initial DeepSDF reconstruction -- grid prediction
            self.deepsdf_decoder.eval()
            latent = child_feats[i].detach().clone()
            children_embeddings += [latent.detach().cpu().clone()]
            with torch.no_grad():
                sdf_values, voxel_origin, voxel_size, offset, scale, samples = create_grid(
                    self.deepsdf_decoder,
                    latent,
                    N=GRID_RESOLUTION,
                    max_batch=int(2 ** 18),
                    input_samples=None,
                    add_feat=child_feats[i],
                    mode=mode,
                    class_one_hot=class_one_hot
                )
                parts_sdfs_pred[child_name]['sdf'] = sdf_values.detach().cpu().clone()
                parts_sdfs_pred[child_name]['vox_origin'] = voxel_origin
                parts_sdfs_pred[child_name]['vox_size'] = voxel_size
                parts_sdfs_pred[child_name]['offset'] = offset
                parts_sdfs_pred[child_name]['scale'] = scale
                parts_sdfs_pred[child_name]['samples'] = samples.detach().cpu().clone()

                del sdf_values
                del voxel_origin
                del voxel_size
                del offset
                del scale

            # add points from grid to TTO (equally positive and negative)
            # helps to avoid uniform noise sampling in incomplete regions
            # and add points for full shape to enhance shape consistency during TTO
            inside_shape_indices = torch.where(samples[:, 3] < 0.0)[0]
            random_indices = np.random.choice(inside_shape_indices, min(25000, len(inside_shape_indices)),
                                              replace=False)
            extra_points_negative = samples[random_indices]
            inside_shape_indices = torch.where((0.0 < samples[:, 3]) & (samples[:, 3] < 0.02))[0]
            random_indices = np.random.choice(inside_shape_indices, min(25000, len(inside_shape_indices)),
                                              replace=False)
            extra_points_positive = samples[random_indices]
            extra_sdf_points += [torch.vstack([extra_points_negative.detach().cpu().clone(), extra_points_positive.detach().cpu().clone()])]

            del samples

        return parts_sdfs_pred, children_embeddings, children_names, children_sdf_points, children_feats, extra_sdf_points


    def reconstruct_full_shape_from_latents(self, shape_feat, sdf_data, mode):
        sdf_data = sdf_data.cuda()

        shape_sdf_pred = {}
        self.deepsdf_decoder.eval()
        latent = shape_feat.detach().clone()
        with torch.no_grad():
            sdf_values, voxel_origin, voxel_size, offset, scale, samples = create_grid(
                self.deepsdf_shape_decoder,
                latent,
                N=GRID_RESOLUTION,
                max_batch=int(2 ** 18),
                input_samples=None,
                add_feat=shape_feat,
                mode=mode
            )
            shape_sdf_pred['sdf'] = sdf_values
            shape_sdf_pred['vox_origin'] = voxel_origin
            shape_sdf_pred['vox_size'] = voxel_size
            shape_sdf_pred['offset'] = offset
            shape_sdf_pred['scale'] = scale

            del sdf_values
            del voxel_origin
            del voxel_size
            del offset
            del scale

        samples = samples.detach().cpu()
        shape_feat = shape_feat.detach().cpu()
        sdf_data = sdf_data.detach().cpu()
        latent = latent.detach().cpu()

        return shape_sdf_pred, latent, sdf_data, shape_feat, samples