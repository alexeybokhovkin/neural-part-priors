import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
from copy import deepcopy
import math
from scipy.spatial import cKDTree
import json
from pyquaternion import Quaternion
import open3d as o3d
import time
import plyfile
import skimage
from scipy.spatial.transform import Rotation

from ..data_utils.hierarchy import Tree
from ..utils.gnn import linear_assignment
from ..utils.losses import IoULoss
from .deep_sdf_decoder import Decoder as DeepSDFDecoder
from .gnn_contrast import LatentDecoder, NumChildrenClassifier, GNNChildDecoder, LatentProjector
from deep_sdf_utils.data import unpack_sdf_samples_from_ram
from deep_sdf_utils.utils import decode_sdf
from ..utils.losses import gradient
from ..data_utils.transformations import perform_rot, perform_translate_x, perform_translate_y, apply_transform
from ..utils.transformations import apply_transform_torch, from_tqs_to_matrix, decompose_mat4
from ..utils.embedder import get_embedder_nerf

# sys.path.append('/home/bohovkin/cluster/abokhovkin_home/external_packages/chamferdist')
# sys.path.append('/rhome/abokhovkin/external_packages/chamferdist')
try:
    from chamferdist import ChamferDistance
except:
    print('ChamferDistance not imported')


GRID_RESOLUTION = 128


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def reconstruct(
        decoder,
        num_iterations,
        latent_size,
        sdf_data,
        stat,
        clamp_dist,
        num_samples=30000,
        lr=5e-4,
        l2reg=False,
        cal_scale=1,
        add_feat=None,
        mode=0
):
    def adjust_learning_rate(
            initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True
    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()
    loss_history = []
    cal_scale = float(cal_scale)

    for e in range(num_iterations):
        decoder.eval()
        sdf_data = sdf_data.cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)
        sdf_gt = torch.clamp(cal_scale * sdf_gt, -clamp_dist, clamp_dist)

        new_lr = adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)
        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        if mode == 2:
            inputs = torch.cat([latent_inputs, xyz], 1).cuda()
        elif mode == 1:
            add_feat = add_feat.expand(num_samples, -1)
            inputs = torch.cat([add_feat, latent_inputs, xyz], 1).cuda()
        elif mode == 3:
            add_feat = add_feat.expand(num_samples, -1)
            inputs = torch.cat([add_feat, xyz], 1).cuda()

        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs)
        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()
        loss_num = loss.cpu().data.numpy()

        loss_history += [float(loss.cpu().data.numpy())]

    return loss_num, latent, loss_history


def reconstruct_tto(
        decoder,
        num_iterations,
        sdf_data,
        clamp_dist,
        num_samples=30000,
        lr_1=5e-4,
        lr_2=5e-4,
        l2reg=False,
        cal_scale=1,
        add_feat=None,
        init_emb=None,
        mode=0
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)
    # adjust_lr_every = 5000

    latent = init_emb.detach().clone().cuda()
    # latent = torch.ones(1, 256).normal_(mean=0, std=0.01).cuda()

    latent.requires_grad = True
    optimizer_1 = torch.optim.Adam([latent], lr=lr_1)
    optimizer_2 = torch.optim.Adam(decoder.parameters(), lr=lr_2)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss(reduction='none')
    # loss_l1 = torch.nn.MSELoss(reduction='none')
    loss_history = []
    all_latents = []
    cal_scale = float(cal_scale)
    pred_sdf_part = {}
    j = 0

    for e in range(num_iterations):
        if e % 200 == 0:
            # all_latents += [latent.detach().clone()]
            all_latents += [0]

        # decoder.eval()
        sdf_data = sdf_data.cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)
        sdf_gt = torch.clamp(cal_scale * sdf_gt, -clamp_dist, clamp_dist)

        new_lr = adjust_learning_rate(lr_1, optimizer_1, e, decreased_by, adjust_lr_every)
        adjust_learning_rate(lr_2, optimizer_2, e, decreased_by, adjust_lr_every)
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        if mode == 2:
            inputs = torch.cat([latent_inputs, xyz], 1).cuda()
        elif mode == 1:
            add_feat_exp = add_feat.expand(num_samples, -1)
            inputs = torch.cat([add_feat_exp, latent_inputs, xyz], 1).cuda()
        elif mode == 3:
            add_feat_exp = add_feat.expand(num_samples, -1)
            inputs = torch.cat([add_feat_exp, xyz], 1).cuda()

        pred_sdf = decoder(inputs)
        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)
        loss = loss.mean()

        if e == 0:
            stats_before_tto = 0
            # stats_before_tto = (inputs.detach().cpu().numpy(),
            #                     loss.detach().item(),
            #                     sdf_gt.detach().cpu().numpy())
        if e == num_iterations - 1:
            stats_after_tto = 0
            # stats_after_tto = (inputs.detach().cpu().numpy(),
            #                    loss.detach().item(),
            #                    sdf_gt.detach().cpu().numpy())

        if l2reg:
            loss = loss + 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer_1.step()
        optimizer_2.step()
        # loss_num = loss.detach().item()
        loss_num = 0

        # loss_history += [float(loss.detach().item())]
        loss_history += [0]

        if e % 200 == 0:
            print('loss:', loss.detach().item())
            decoder.eval()
            with torch.no_grad():
                pred_sdf_part[j] = {}
                sdf_values, voxel_origin, voxel_size, offset, scale, _ = create_grid(
                    decoder,
                    latent,
                    N=GRID_RESOLUTION,
                    max_batch=int(2 ** 18),
                    input_samples=None,
                    add_feat=add_feat,
                    mode=mode
                )
                pred_sdf_part[j]['sdf'] = sdf_values.detach().cpu().clone()
                pred_sdf_part[j]['vox_origin'] = voxel_origin
                pred_sdf_part[j]['vox_size'] = voxel_size
                pred_sdf_part[j]['offset'] = offset
                pred_sdf_part[j]['scale'] = scale

                del sdf_values
                del voxel_origin
                del voxel_size
                del offset
                del scale
                del _

            decoder.train()
            j += 1

    del latent

    return loss_num, all_latents, loss_history, pred_sdf_part, stats_before_tto, stats_after_tto


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

    t00 = time.time()

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
    # loss_l1 = torch.nn.MSELoss(reduction='none')
    loss_mse = torch.nn.MSELoss(reduction='none')
    loss_part_history = []
    loss_shape_history = []
    all_latents = []
    cal_scale = float(cal_scale)
    pred_sdf_part = {}
    pred_sdf_full = {}
    j = 0

    # part points
    indices_not_noise = torch.where(sdf_data_part[:, 3] < clamp_dist)[0]
    sdf_data_part_not_noise = sdf_data_part[indices_not_noise]
    # indices_noise = torch.where(sdf_data_part[:, 3] >= clamp_dist)[0]
    # sdf_data_part_noise = sdf_data_part[indices_noise]
    sdf_data_part_not_noise = sdf_data_part_not_noise.cuda()
    # sdf_data_part_noise = sdf_data_part_noise.cuda()
    xyz_part_not_noise = sdf_data_part_not_noise[:, 0:3]
    xyz_part_not_noise_pe = embedder(xyz_part_not_noise)
    # xyz_part_noise = sdf_data_part_noise[:, 0:3]
    sdf_gt_part_not_noise = sdf_data_part_not_noise[:, 3].unsqueeze(1)
    # sdf_gt_part_noise = sdf_data_part_noise[:, 3].unsqueeze(1)
    sdf_gt_part_not_noise = torch.clamp(cal_scale * sdf_gt_part_not_noise, -clamp_dist, clamp_dist).cuda()
    # sdf_gt_part_noise = torch.clamp(cal_scale * sdf_gt_part_noise, -clamp_dist, clamp_dist).cuda()

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

    t01 = time.time()

    all_t_init = []
    all_t_forward = []
    all_t_loss = []
    all_t_back = []
    all_t_opt = []
    all_t_full = []

    for e in range(num_iterations):
        if e % 200 == 0:
            # all_latents += [latent_part.detach().clone()]
            all_latents += [0]

        t0 = time.time()

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

        # latent_inputs_part_noise = latent_part.expand(len(xyz_part_noise), -1)
        # inputs_part_noise = torch.cat([latent_inputs_part_noise, xyz_part_noise], 1).cuda()

        latent_inputs_part_uniform_noise = latent_part.expand(len(xyz_part_uniform_noise), -1)
        inputs_part_uniform_noise = torch.cat([latent_inputs_part_uniform_noise, xyz_part_uniform_noise_pe], 1).cuda()
        latent_inputs_part_another_parts_noise = latent_part.expand(len(xyz_part_another_parts_noise), -1)
        inputs_part_another_parts_noise = torch.cat([latent_inputs_part_another_parts_noise, xyz_part_another_parts_noise_pe], 1).cuda()

        # full shape
        latent_inputs_full_not_noise = latent_shape.expand(len(xyz_full_not_noise), -1)
        inputs_full_not_noise = torch.cat([latent_inputs_full_not_noise, xyz_full_not_noise_pe], 1).cuda()
        latent_inputs_full_noise = latent_shape.expand(len(xyz_full_noise), -1)
        inputs_full_noise = torch.cat([latent_inputs_full_noise, xyz_full_noise_pe], 1).cuda()

        t1 = time.time()

        ## forward pass
        # part-based
        pred_sdf_part_ = decoder_part(inputs_part)
        pred_sdf_shape = decoder_shape(inputs_shape)
        # pred_sdf_part_noise = decoder_part(inputs_part_noise)
        pred_sdf_part_uniform_noise = decoder_part(inputs_part_uniform_noise)
        pred_sdf_part_another_parts_noise = decoder_part(inputs_part_another_parts_noise)


        pred_sdf_part_ = torch.clamp(pred_sdf_part_, -clamp_dist, clamp_dist)
        pred_sdf_shape = torch.clamp(pred_sdf_shape, -clamp_dist, clamp_dist)
        # pred_sdf_part_noise = torch.clamp(pred_sdf_part_noise, -clamp_dist, clamp_dist)
        pred_sdf_part_uniform_noise = torch.clamp(pred_sdf_part_uniform_noise, -clamp_dist, clamp_dist)
        pred_sdf_part_another_parts_noise = torch.clamp(pred_sdf_part_another_parts_noise, -clamp_dist, clamp_dist)

        # full shape
        if e == 0:
            print('Part shape input:', inputs_part.shape)
            print('Part shape input:', inputs_shape.shape)
            # print('Noise shape input:', inputs_part_noise.shape)
            # print('Full shape input:', inputs_full.shape)
            print('Full shape (not noise):', inputs_full_not_noise.shape)
            print('Full shape (noise):', inputs_full_noise.shape)
        pred_sdf_full_not_noise = decoder_shape(inputs_full_not_noise)
        pred_sdf_full_not_noise = torch.clamp(pred_sdf_full_not_noise, -clamp_dist, clamp_dist)
        pred_sdf_full_noise = decoder_shape(inputs_full_noise)
        pred_sdf_full_noise = torch.clamp(pred_sdf_full_noise, -clamp_dist, clamp_dist)

        t2 = time.time()

        ## compute all losses
        # part-based
        loss_part = loss_l1(pred_sdf_part_, sdf_gt_part_not_noise)
        # loss_part_noise = loss_l1(pred_sdf_part_noise, sdf_gt_part_noise)
        loss_part_uniform_noise = loss_l1(pred_sdf_part_uniform_noise, sdf_gt_part_uniform_noise)
        loss_part_another_parts_noise = loss_l1(pred_sdf_part_another_parts_noise, sdf_gt_part_another_parts_noise)
        pred_sdf_shape_clone = pred_sdf_shape.clone().detach()
        pred_sdf_shape_clone.requires_grad = False
        loss_part_consistency = loss_mse(pred_sdf_part_, pred_sdf_shape_clone)

        # full shape
        loss_full = loss_l1(pred_sdf_full_not_noise, sdf_gt_full_not_noise).mean() + w_full_noise * loss_l1(pred_sdf_full_noise, sdf_gt_full_noise).mean()

        # loss = (loss_part.mean() + 200.0 * loss_part_noise.mean()) + 200.0 * loss_part_consistency.mean()
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

        t3 = time.time()
        loss.backward()
        loss_full.backward()

        t4 = time.time()
        optimizer_1.step()
        if e > 400:
            optimizer_2.step()
            optimizer_3.step()

        loss_part_history += [float(loss.detach().item())]

        t5 = time.time()

        # optimization in time
        if e % 200 == 0:

            decoder_part.eval()
            with torch.no_grad():
                pred_sdf_part_loc = {}

                t51 = time.time()
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
                t52 = time.time()
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

        # interpolation
        # if e == 0:
        #
        #     decoder_part.eval()
        #     with torch.no_grad():
        #         pred_sdf_part_loc = {}
        #
        #         sdf_values, voxel_origin, voxel_size, offset, scale, _ = create_grid(
        #             decoder_part,
        #             latent_part[:256],
        #             N=256,
        #             max_batch=int(2 ** 18),
        #             input_samples=None,
        #             add_feat=add_feat,
        #             mode=mode,
        #             class_one_hot=class_one_hot
        #         )
        #         pred_sdf_part_loc['sdf'] = sdf_values.detach().cpu().clone()
        #         pred_sdf_part_loc['vox_origin'] = voxel_origin
        #         pred_sdf_part_loc['vox_size'] = voxel_size
        #         pred_sdf_part_loc['offset'] = offset
        #         pred_sdf_part_loc['scale'] = scale
        #         pred_sdf_part_loc['latent'] = latent_part.detach().cpu().clone()
        #
        #         os.makedirs(store_dir + '_interpolation', exist_ok=True)
        #         convert_sdf_samples_to_ply(
        #             pred_sdf_part_loc['sdf'],
        #             pred_sdf_part_loc['vox_origin'],
        #             pred_sdf_part_loc['vox_size'],
        #             os.path.join(store_dir + '_interpolation', child_name + '.' + str(iter) + '.ply'),
        #             offset=pred_sdf_part_loc['offset'],
        #             scale=pred_sdf_part_loc['scale'],
        #             level=0.005
        #         )
        #
        #         del sdf_values
        #         del voxel_origin
        #         del voxel_size
        #         del offset
        #         del scale
        #         del _
        #
        #     decoder_part.train()

        if e % 600 == 0:
            decoder_part.eval()
            with torch.no_grad():
                pred_sdf_part[j] = {}

                t51 = time.time()
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
                t52 = time.time()
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

            t53 = time.time()
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
                t54 = time.time()
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
            t55 = time.time()

            j += 1

        t6 = time.time()

        all_t_init += [t1 - t0]
        all_t_forward += [t2 - t1]
        all_t_loss += [t3 - t2]
        all_t_back += [t4 - t3]
        all_t_opt += [t5 - t4]
        all_t_full += [t5 - t0]

    del latent_part

    return loss_num, all_latents, loss_part_history, pred_sdf_part, stats_before_tto, stats_after_tto, pred_sdf_full, latent_shape.detach()



def create_grid(
        decoder, latent_vec, N=256, max_batch=32 ** 3, offset=None, scale=None,
        input_samples=None, add_feat=None, mode=0, class_one_hot=None
):
    embedder, embedder_out_dim = get_embedder_nerf(10, input_dims=3, i=0)
    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    if input_samples is None:
        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() / N) % N
        samples[:, 0] = ((overall_index.long() / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    else:
        samples = input_samples
        samples = torch.cat([samples, torch.zeros((len(samples), 1)).cuda()], dim=1)

    num_samples = N ** 3
    samples.requires_grad = False
    head = 0

    if class_one_hot is not None:
        class_one_hot = class_one_hot.cuda()
        latent_vec = torch.hstack([latent_vec, class_one_hot])

    while head < num_samples:
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:3]
        sample_subset = embedder(sample_subset).cuda()

        samples[head: min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset, add_feat=add_feat, mode=mode)
                .squeeze(1)
                .detach()
                .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    sdf_values = sdf_values.detach().cpu().clone()
    samples = samples.detach().cpu().clone()

    return sdf_values, voxel_origin, voxel_size, offset, scale, samples


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
                 class2id=None):
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

        self.edge_types = ['ADJ', 'SYM']
        self.device = device
        self.max_child_num = max_child_num

        self.latent_decoder = LatentDecoder(feature_size, hidden_size, hidden_size)
        self.root_classifier = NumChildrenClassifier(feature_size, hidden_size, 5)
        self.rotation_classifier = NumChildrenClassifier(feature_size, hidden_size, 12)
        self.child_decoder = GNNChildDecoder(feature_size, hidden_size,
                                             max_child_num, edge_symmetric_type,
                                             num_iterations, edge_type_num)

        # self.point_part_classifier = PointPartClassifier(260, 128, 128, 64)
        # self.point_part_classifier = PointPartClassifierEntropyPointNet(260, 128, 128, 128, len(class2id) + 1)

        self.point_part_classifier = PointPartClassifierEntropyPointNet(332, 128, 128, 128, len(class2id) + 1) # 272 / 332
        # self.point_part_classifier = PointPartClassifierEntropyPointNet2(332, 256, 128, 128, 64, len(class2id) + 1) # 272 / 332

        # self.point_part_classifier = PointPartClassifierEntropyPointNet(260, 128, 128, 128, 1, sigmoid=True)
        # self.point_part_classifier = PointPartClassifierEntropyPointNet2(260, 256, 128, 128, 64, 1, sigmoid=True)

        # Part level
        deep_sdf_latent_size = feature_size
        # deep_sdf_specs['dims'] = [512, 512, 512, 768, 512, 512, 512, 512]
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

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # set_requires_grad(self.deepsdf_decoder, False)

        # Shape level
        # deep_sdf_specs['dims'] = [512, 512, 512, 768, 512, 512, 512, 512]
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

    def map_part_embeddings(self, source_emb_path, source_shape_emb_path):
        with open('/rhome/abokhovkin/projects/scannet-relationships/dicts/partnet_to_parts.json', 'rb') as fin:
            partnet_to_parts = json.load(fin)

        with open('/cluster/pegasus/abokhovkin/part-segmentation/hierarchies_32_1lvl_v3/all_chair_scannet/train.txt',
                  'r') as f:
            object_names = [item.rstrip() for item in f.readlines()]
        with open(
                '/cluster/sorona/abokhovkin/part-segmentation/hierarchies_32_1lvl_filled_v2/all_chair_geoscan/train_deepsdf.txt',
                'r') as f:
            source_object_names = [item.rstrip() for item in f.readlines()]

        source_parts_to_indices = {}
        idx = 0
        for object_name in source_object_names:
            source_partnet_id = object_name.split('_')[0]
            source_parts = partnet_to_parts[source_partnet_id]
            if source_partnet_id not in source_parts_to_indices:
                source_parts_to_indices[source_partnet_id] = {}
                for source_part in source_parts:
                    source_parts_to_indices[source_partnet_id][source_part] = idx
                    idx += 1

        existing_partnet_ids = []
        parts_to_indices = {}
        idx = 0
        for object_name in object_names:
            partnet_id = object_name.split('_')[0]
            existing_partnet_ids += [partnet_id]
            parts = partnet_to_parts[partnet_id]
            if partnet_id not in parts_to_indices:
                parts_to_indices[partnet_id] = {}
                for part in parts:
                    parts_to_indices[partnet_id][part] = idx
                    idx += 1
        existing_partnet_ids = sorted(list(set(existing_partnet_ids)))

        source_embedding = torch.load(source_emb_path)['latent_codes']['weight']
        init_weights = torch.clone(self.lat_vecs.weight.data.cpu().detach())

        for partnet_id in parts_to_indices:
            for part in parts_to_indices[partnet_id]:
                if partnet_id in source_parts_to_indices:
                    init_weights[parts_to_indices[partnet_id][part]] = source_embedding[
                        source_parts_to_indices[partnet_id][part]]
        self.lat_vecs.weight.data = init_weights

        all_partnet_ids = sorted(list(partnet_to_parts.keys()))

        source_shape_embedding = torch.load(source_shape_emb_path)['latent_codes']['weight']
        init_shape_weights = torch.clone(self.lat_shape_vecs.weight.data.cpu().detach())
        for partnet_id in parts_to_indices:
            if partnet_id in all_partnet_ids:
                try:
                    init_shape_weights[existing_partnet_ids.index(partnet_id)] = source_shape_embedding[
                        all_partnet_ids.index(partnet_id)]
                except IndexError:
                    continue
        self.lat_shape_vecs.weight.data = init_shape_weights

    def isLeafLossEstimator(self, is_leaf_logit, gt_is_leaf):
        return self.bceLoss(is_leaf_logit, gt_is_leaf).view(-1)

    def tto(self, children_initial_data, shape_initial_data=None, index=0, only_align=False, constr_mode=0, cat_name=None,
            num_shapes=0, k_near=0, scene_id='0', wconf=0, w_full_noise=1, w_part_u_noise=1,
            w_part_part_noise=1, lr_dec_full=0, lr_dec_part=0, target_sample_names=None, sa_mode=None,
            parts_indices=None, shape_idx=None, store_dir=None):

        print('Start TTO')
        # Start TTO process
        # Return all necessary outputs -- parts sdfs, shape sdfs, classified points, final losses, uniform and part noise
        parts_sdfs_pred, all_shape_sdf_pred, \
        parts_stats_before, parts_stats_after, shape_stats_before_tto, shape_stats_after_tto \
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
                                    store_dir=store_dir)
        print('Finish TTO')
        return parts_sdfs_pred, all_shape_sdf_pred, parts_stats_before, parts_stats_after, shape_stats_before_tto, shape_stats_after_tto

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

        # num_uniform_noise = 0
        # num_parts_noise = 0

        print('Num samples:', num_pos_samples, num_neg_samples, num_uniform_noise, num_parts_noise)

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

        print('After rebalance:', len(sample_pos), len(sample_neg), len(sample_uniform_noise), len(sample_parts_noise))

        return samples, len(sample_pos), sample_uniform_noise, sample_parts_noise

    def adjust_embedding(self, children_data, shape_data, mode, only_align=False, constr_mode=0, cat_name=None,
                         num_shapes=0, k_near=0, scene_id='0', wconf=0, w_full_noise=1, w_part_u_noise=1,
                         w_part_part_noise=1, lr_dec_full=0, lr_dec_part=0, target_sample_names=None,
                         sa_mode=None, parts_indices=None, shape_idx=None, store_dir=None):

        # the main wrapper for TTO

        # only_align means to save results without optimization, only classification
        num_iterations = 3 if only_align else 601
        print('Only align:', only_align)

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

        # constr_mode defines the level of constraints harshness, defined by user
        if constr_mode == 0:
            # lr weakened 0
            def chamfer_to_confidence(chamfer_value):
                if chamfer_value < 100:
                    return 1e-3, 25
                elif chamfer_value > 200:
                    return 5e-5, 175
                else:
                    return 3e-4, 75

        elif constr_mode == 1:
            # lr weakened mlcvnet 1
            def chamfer_to_confidence(chamfer_value):
                if chamfer_value < 100:
                    return 1e-3, 50
                elif chamfer_value > 200:
                    return 5e-5, 175
                else:
                    return 3e-4, 100

        elif constr_mode == 2:
            # lr weakened mlcvnet 2
            def chamfer_to_confidence(chamfer_value):
                if chamfer_value < 100:
                    return 1e-3, 75
                elif chamfer_value > 200:
                    return 5e-5, 175
                else:
                    return 3e-4, 125

        elif constr_mode == 3:
            # lr weakened -1
            def chamfer_to_confidence(chamfer_value):
                if chamfer_value < 200:
                    return 1e-3, 25
                elif chamfer_value > 400:
                    return 5e-5, 175
                else:
                    return 3e-4, 75

        elif constr_mode == 4:
            # lr weakened -1
            def chamfer_to_confidence(chamfer_value):
                if chamfer_value < 250:
                    return 1e-3, 25
                elif chamfer_value > 500:
                    return 5e-5, 125
                else:
                    return 3e-4, 60

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
        all_added_extra_points = []
        all_pred_sdf_full = []

        # scenes for scene optimization
        if scene_id == 'scene0621_00':
            EXTRA_POINTS_DIR = '/cluster/daidalos/abokhovkin/scannet-relationships/test_output_full_eccv/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_chair_0.25_0'
            sample_names = ['42021_scene0621_00_18',
                            '42021_scene0621_00_13',
                            '42021_scene0621_00_12',
                            '42021_scene0621_00_6',
                            '42021_scene0621_00_9',
                            '42021_scene0621_00_8',
                            '42021_scene0621_00_11',
                            '42021_scene0621_00_15',
                            '42021_scene0621_00_0']
        if scene_id == 'scene0342_00':
            EXTRA_POINTS_DIR = '/cluster/daidalos/abokhovkin/scannet-relationships/test_output_full_eccv/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_chair_0.25_0'
            sample_names = ['44887_scene0342_00_1',
                            '44887_scene0342_00_10',
                            '44887_scene0342_00_11',
                            '44887_scene0342_00_12',
                            '44887_scene0342_00_13',
                            '44887_scene0342_00_14',
                            '44887_scene0342_00_15',
                            '44887_scene0342_00_16',
                            '44887_scene0342_00_17']
        if scene_id == 'scene0011_01':
            EXTRA_POINTS_DIR = '/cluster/daidalos/abokhovkin/scannet-relationships/test_output_full_eccv/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_chair_0.25_0'
            sample_names = ['43297_scene0011_01_1',
                            '43297_scene0011_01_2',
                            '43297_scene0011_01_3',
                            '43297_scene0011_01_4',
                            '43297_scene0011_01_5',
                            '43297_scene0011_01_6',
                            '43297_scene0011_01_7',
                            '43297_scene0011_01_8',
                            '43297_scene0011_01_9']
        if scene_id == 'scene0081_02':
            EXTRA_POINTS_DIR = '/cluster/daidalos/abokhovkin/scannet-relationships/test_output_full_eccv/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_chair_0.25_0'
            sample_names = ['43310_scene0081_02_6',
                            '43310_scene0081_02_7',
                            '43310_scene0081_02_8',
                            '43310_scene0081_02_9',
                            '43310_scene0081_02_10',
                            '43310_scene0081_02_11',
                            '43310_scene0081_02_12',
                            '43310_scene0081_02_13',
                            '43310_scene0081_02_14']
        if scene_id == 'scene0088_00':
            EXTRA_POINTS_DIR = '/cluster/daidalos/abokhovkin/scannet-relationships/test_output_full_eccv/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_chair_0.25_0'
            sample_names = ['39276_scene0088_00_1',
                            '39276_scene0088_00_2',
                            '39276_scene0088_00_3',
                            '39276_scene0088_00_4',
                            '39276_scene0088_00_6',
                            '39276_scene0088_00_7',
                            '39276_scene0088_00_8',
                            '39276_scene0088_00_9',
                            '39276_scene0088_00_10']
        if scene_id == 'scene0015_00':
            EXTRA_POINTS_DIR = '/cluster/daidalos/abokhovkin/scannet-relationships/test_output_full_eccv/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_chair_0.25_0'
            sample_names = ['2298_scene0015_00_14',
                            '2298_scene0015_00_15',
                            '2298_scene0015_00_17',
                            '2298_scene0015_00_19',
                            '2298_scene0015_00_20',
                            '2298_scene0015_00_21',
                            '2298_scene0015_00_22',
                            '2298_scene0015_00_23',
                            '2298_scene0015_00_24']
        if scene_id == 'scene0095_01':
            EXTRA_POINTS_DIR = '/cluster/daidalos/abokhovkin/scannet-relationships/test_output_full_eccv/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_chair_0.25_0'
            sample_names = ['38879_scene0095_01_2',
                            '38879_scene0095_01_3',
                            '38879_scene0095_01_4',
                            '38879_scene0095_01_5',
                            '38879_scene0095_01_6',
                            '38879_scene0095_01_7',
                            '38879_scene0095_01_8',
                            '38879_scene0095_01_9',
                            '38879_scene0095_01_10']
        if scene_id == 'scene0500_00':
            EXTRA_POINTS_DIR = '/cluster/daidalos/abokhovkin/scannet-relationships/test_output_full_eccv/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_chair_0.25_0'
            sample_names = ['43131_scene0500_00_0',
                            '43131_scene0500_00_1',
                            '43131_scene0500_00_2',
                            '43131_scene0500_00_4',
                            '43131_scene0500_00_5',
                            '43131_scene0500_00_7',
                            '43131_scene0500_00_12',
                            '43131_scene0500_00_17',
                            '43131_scene0500_00_18']
        if scene_id == 'scene0088_03':
            EXTRA_POINTS_DIR = '/cluster/daidalos/abokhovkin/scannet-relationships/test_output_full_eccv/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_chair_0.25_0'
            sample_names = ['39276_scene0088_03_9',
                            '39276_scene0088_03_4',
                            '39276_scene0088_03_2',
                            '39276_scene0088_03_6',
                            '39276_scene0088_03_10',
                            '39276_scene0088_03_1',
                            '39276_scene0088_03_12',
                            '39276_scene0088_03_13',
                            '39276_scene0088_03_7',
                            '39276_scene0088_03_0']
        if scene_id == 'scene0430_00':
            EXTRA_POINTS_DIR = '/cluster/daidalos/abokhovkin/scannet-relationships/test_output_full_eccv/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_chair_0.25_0'
            sample_names = ['42480_scene0430_00_21',
                            '42480_scene0430_00_17',
                            '42480_scene0430_00_11',
                            '42480_scene0430_00_10',
                            '42480_scene0430_00_18',
                            '42480_scene0430_00_14',
                            '42480_scene0430_00_15',
                            '42480_scene0430_00_12',
                            '42480_scene0430_00_22',
                            '42480_scene0430_00_9',
                            '42480_scene0430_00_6',
                            '42480_scene0430_00_16']

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
            if chamfer_dist_parts_before_secdir[child_name] > 90: # 90 <-- to parts if CD > CD_thr
                all_points += [additional_points_from_pred[child_name]]
        if len(all_points) > 0:
            additional_points_from_pred['full'] = np.vstack(all_points)
        else:
            additional_points_from_pred['full'] = []

        # Main cycle for TTO per child
        for i, child_name in enumerate(children_names):

            # (DEPRECATED)
            # Compute modified lr and w_cons, see constr_mode
            # print('Modified lr and w_cons')
            # if child_name in chamfer_dist_parts_before_secdir:
            #     alt_lr, alt_w_cons = chamfer_to_confidence(chamfer_dist_parts_before_secdir[child_name])
            #     print(child_name, chamfer_dist_parts_before_secdir[child_name], alt_lr, alt_w_cons)
            # else:
            #     alt_lr, alt_w_cons = 1e-3, 25
            #     print(child_name, alt_lr, alt_w_cons)
            # # debug mode (w/o constraint)
            # alt_lr = 1e-3
            # alt_w_cons = 0

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

                # EXTRA_POINTS_DIR = f'/cluster/daidalos/abokhovkin/scannet-relationships/test_output_full_eccv/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_{cat_name}_0.25_0'
                EXTRA_POINTS_DIR = f'/cluster/valinor/abokhovkin/scannet-relationships-v2/test_output_full_cvpr/meshes_scannet/deepsdf_parts_fullshape_mlcvnet_finetune_predlat_predpoints_{cat_name}'
                # global points voting
                sdf_part = sdf_part.cpu()
                all_children_extra_points = []
                for sample_name in target_sample_names[:min(num_shapes, len(target_sample_names))]:
                    child_points_path = os.path.join(EXTRA_POINTS_DIR, f'{sample_name}_tto', f'{child_name}.pts.npy')
                    child_sdfs_path = os.path.join(EXTRA_POINTS_DIR, f'{sample_name}_tto', f'{child_name}.sdf.npy')
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

                sdf_part = sdf_part.cuda()
                # equalize number of positive/negative/noise points
                sdf_part, num_positives, part_uniform_noise, part_another_parts_noise = self.rebalance_points(sdf_part, uniform_noise, another_parts_noise)
                sdf_part = sdf_part.cuda()

                # add additional points from prediction to parts
                if child_name in additional_points_from_pred:
                    if len(additional_points_from_pred[child_name]) > 0 and chamfer_dist_parts_before_secdir[child_name] > 50:
                        sdf_part = torch.vstack([sdf_part,
                                                 torch.FloatTensor(additional_points_from_pred[child_name]).cuda()])

                print('Reconstructed part and number of points', child_name, len(sdf_part))
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

                    # add extra points from another scene samples
                    # all_children_extra_points = torch.FloatTensor(all_children_extra_points).cuda()
                    # sdf_data = torch.vstack([sdf_data, all_children_extra_points])
                    # all_points_numpy = sdf_data[:, :3].cpu().detach().numpy()
                    # kd_tree_avg = cKDTree(all_points_numpy)
                    # min_dist_avg, min_idx_avg = kd_tree_avg.query(all_points_numpy, k=k_near)
                    # sdf_data[:, 3] = sdf_data[:, 3][min_idx_avg].mean(axis=1)

                # part & shape geometry interpolation
                # chair 4 <-> 20
                # parts_indices_1 = {'chair_back': 13890,
                #                      'chair_arm_right': 7892,
                #                      'chair_arm_left': 18094,
                #                      'chair_seat': 3686,
                #                      'regular_leg_base': 22169}
                # shape_idx_1 = 3403
                # parts_indices_2 = {'chair_back': 13282,
                #                      'chair_arm_right': 7626,
                #                      'chair_arm_left': 17827,
                #                      'chair_seat': 3071,
                #                      'regular_leg_base': 21685}
                # shape_idx_2 = 940
                # table 1 <-> 4
                # parts_indices_1 = {'table_surface': 16669, 'leg': 9301}
                # shape_idx_1 = 4726
                # parts_indices_2 = {'table_surface': 17431, 'leg': 9943}
                # shape_idx_2 = 5498
                # storage 0 <-> 3
                # parts_indices_1 = {'cabinet_frame': 3001, 'cabinet_door': 352, 'cabinet_base': 4750}
                # shape_idx_1 = 467
                # parts_indices_2 = {'cabinet_frame': 3027, 'cabinet_base': 4762, 'cabinet_door': 369}
                # shape_idx_2 = 493
                # bed 11 <-> 1
                # parts_indices_1 = {'bed_sleep_area': 223, 'headboard': 64, 'bed_frame_base': 422}
                # shape_idx_1 = 181
                # parts_indices_2 = {'bed_sleep_area': 208, 'headboard': 54, 'bed_frame_base': 407}
                # shape_idx_2 = 166
                # trashcan 0 <-> 13
                # parts_indices_1 = {'container_box': 559, 'container_bottom': 296}
                # shape_idx_1 = 79
                # parts_indices_2 = {'container_box': 492, 'container_bottom': 283}
                # shape_idx_2 = 9

                # shape_idx_1 = torch.tensor(shape_idx_1).cuda()
                # shape_idx_2 = torch.tensor(shape_idx_2).cuda()
                # part_idx_1 = torch.tensor(parts_indices_1[child_name]).cuda()
                # part_idx_2 = torch.tensor(parts_indices_2[child_name]).cuda()
                # shape_embedding_1 = self.lat_shape_vecs(shape_idx_1)[None, :]
                # part_embedding_1 = self.lat_vecs(part_idx_1)
                # shape_embedding_2 = self.lat_shape_vecs(shape_idx_2)[None, :]
                # part_embedding_2 = self.lat_vecs(part_idx_2)

                # t = 0.0, 0.166, 0.333, 0.5, 0.667, 0.833, 1.0
                # t = 1.0
                # num_interpolations = 300
                # step = 1. / num_interpolations
                # for m in range(num_interpolations):
                #     t = m * step
                #     shape_embedding = t * shape_embedding_1 + (1 - t) * shape_embedding_2
                #     part_embedding = t * part_embedding_1 + (1 - t) * part_embedding_2

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
                    lr_2=lr_dec_part,    # 3e-4
                    lr_3=lr_dec_full,    # 0
                    w_cons=wconf,
                    w_full_noise=w_full_noise,
                    w_part_u_noise=w_part_u_noise,
                    w_part_part_noise=w_part_part_noise,
                    l2reg=True,
                    cal_scale=1,
                    add_feat=children_feats[i],
                    init_emb_part=children_embeddings[i], # children_embeddings[i] / part_embedding
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

        shape_stats_before_tto = 0
        shape_stats_after_tto = 0

        return parts_sdfs_pred, all_pred_sdf_full, \
               parts_stats_before, parts_stats_after, shape_stats_before_tto, shape_stats_after_tto

    # decode a root code into a tree structure
    def decode_structure(self, z, sdf_data, full_label=None, encoder_features=None, rotation=None, gt_tree=None,
                         index=0):
        root_latent = self.latent_decoder(z)  # torch.Tensor[bsize, 256], torch.Tensor[bsize, 256]
        if full_label is None:
            full_label = Tree.root_sem
        output = self.decode_node(root_latent, sdf_data, full_label=full_label, level=0,
                                  encoder_features=encoder_features, rotation=rotation,
                                  gt_tree=gt_tree, index=index)

        obj = Tree(root=output[0])
        new_output = [obj, ]
        for i in range(1, len(output)):
            new_output += [output[i]]
        new_output = tuple(new_output)
        return new_output

    # decode a part node
    def decode_node(self, node_latent, sdf_data, full_label, level=0, encoder_features=None, rotation=None,
                    pred_rotation=None, gt_tree=None, index=0):
        cuda_device = node_latent.get_device()
        gt_tree_children_names = [x.label for x in gt_tree.root.children]

        if level == 1:
            return Tree.Node(is_leaf=True, full_label=full_label, label=full_label.split('/')[-1], geo=0)
        else:
            child_feats, child_sem_logits, child_exists_logit, edge_exists_logits = \
                self.child_decoder(node_latent)
            # torch.Tensor[bsize, max_child_num, 256]
            # torch.Tensor[bsize, max_child_num, tree.num_sem]
            # torch.Tensor[bsize, max_child_num, 1]
            # torch.Tensor[bsize, max_child_num, max_child_num, 4]

            child_sem_logits_numpy = child_sem_logits.cpu().detach().numpy().squeeze()
            # torch.Tensor[max_child_num, tree.num_sem]

            # children
            child_nodes = []  # list[exist_child_num]<Tree.Node>
            child_idx = {}
            child_feats_exist = []
            for ci in range(child_feats.shape[1]):
                if torch.sigmoid(child_exists_logit[:, ci, :]).item() > 0.5:
                    idx = np.argmax(child_sem_logits_numpy[ci, Tree.part_name2cids[full_label]])  # int
                    idx = Tree.part_name2cids[full_label][idx]  # int (1 <= idx <= tree.num_sem)
                    child_full_label = Tree.part_id2name[idx]
                    child_node = self.decode_node(
                        child_feats[:, ci, :], sdf_data, child_full_label, level=level + 1,
                        encoder_features=encoder_features, rotation=rotation,
                        pred_rotation=pred_rotation, gt_tree=gt_tree)
                    child_nodes.append(child_node)
                    child_idx[ci] = len(child_nodes) - 1
                    child_feats_exist.append(child_feats[:, ci, :])

            # edges
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

            # reconstruct meshes
            # parts_sdfs_pred, children_embeddings, children_names, children_sdf_points = self.reconstruct_children(
            #                                                                             child_nodes,
            #                                                                             child_feats_exist,
            #                                                                             gt_tree_children_names,
            #                                                                             sdf_data,
            #                                                                             mode=1
            #                                                                             )
            # children_feats = 0

            # reconstruct meshes by predicting points to parts associations
            sdf_data_flattened = sdf_data.reshape(-1, 4)
            # sdf_data_flattened = torch.stack([x for x in sdf_data_flattened if x[3] < 0.07]).cuda()
            sdf_data_flattened = sdf_data_flattened.cuda()
            all_pred_associations = []
            num_samples = len(sdf_data_flattened)
            for child_feat in child_feats_exist:
                child_feat_expanded = child_feat.expand(num_samples, -1)
                inputs = torch.cat([sdf_data_flattened[:, :3], child_feat_expanded], 1).cuda()
                pred_accociations = self.point_part_classifier(inputs)
                all_pred_associations += [pred_accociations]
            all_pred_associations = torch.cat(all_pred_associations, dim=1)
            all_pred_associations = torch.argmax(all_pred_associations, dim=1)
            splitted_sdf_points = []
            for i in range(len(child_feats_exist)):
                part_point_indices = torch.where(all_pred_associations == i)[0]
                splitted_sdf_points += [sdf_data_flattened[part_point_indices]]

            # num_samples_random = 10000
            # random_samples = np.random.uniform(-1.0, 1.0, (num_samples_random, 3))
            # new_splitted_sdf_points = []
            # for points in splitted_sdf_points:
            #     kd_tree = cKDTree(points.cpu().numpy()[:, :3])
            #     min_dist_rand, min_idx_rand = kd_tree.query(random_samples)
            #     above_thr_indices = np.where(min_dist_rand > 0.2)[0]
            #     above_thr_points = random_samples[above_thr_indices]
            #     above_thr_points = np.hstack([above_thr_points, 0.07 * np.ones((len(above_thr_points), 1))]).astype('float32')
            #     grid_points_pos_sampled = torch.vstack([torch.FloatTensor(above_thr_points).cpu(), points.cpu()])
            #     new_splitted_sdf_points += [grid_points_pos_sampled]

            parts_sdfs_pred, children_embeddings, children_names, children_sdf_points, children_feats = self.reconstruct_children_with_associations(
                child_nodes,
                child_feats_exist,
                splitted_sdf_points,
                mode=1)

            pred_output = (children_embeddings, children_names, children_sdf_points, children_feats)

            # reconstruct full shape
            shape_sdf_pred, shape_embedding, sdf_data, shape_feat, samples = self.reconstruct_full_shape(node_latent,
                                                                                                         sdf_data,
                                                                                                         mode=2)

            # classify all points in the grid
            all_grid_points = []
            # samples_flattened = samples[:, :3]
            # for k, part_points in enumerate(children_sdf_points):
            #     points = torch.stack([x for x in part_points if x[3] < 0.0])
            #     grid_tree = cKDTree(points[:, :3].cpu().numpy())
            #     min_dist, min_idx = grid_tree.query(samples_flattened.numpy())
            #     grid_indices = np.where(np.abs(min_dist) < 0.1)[0]
            #     grid_sampled_points = samples_flattened[grid_indices]
            #     sampled_sdfs = shape_sdf_pred['sdf'].reshape(-1)[grid_indices]
            #     sampled_points = torch.hstack([grid_sampled_points, sampled_sdfs[:, None]])
            #     all_grid_points += [sampled_points]

            print('Full shape is processed')
            shape_output = (shape_embedding, sdf_data, shape_feat, all_grid_points)

            # SAVE_DIR = os.path.join('latents', str(index))
            # os.makedirs(SAVE_DIR, exist_ok=True)
            # dict_to_save = {}
            # for k, child_name in enumerate(children_names):
            #     dict_to_save[child_name] = children_embeddings[k]
            # dict_to_save["full"] = shape_embedding
            # torch.save(dict_to_save, os.path.join(SAVE_DIR, 'emb.pth'))
            # dict_to_save = {}
            # for k, child_name in enumerate(children_names):
            #     dict_to_save[child_name] = children_feats[k]
            # dict_to_save["full"] = shape_feat
            # torch.save(dict_to_save, os.path.join(SAVE_DIR, 'feat.pth'))

            # SAVE_DIR = os.path.join('points', str(index))
            # os.makedirs(SAVE_DIR, exist_ok=True)
            # dict_to_save = {}
            # for k, child_name in enumerate(children_names):
            #     dict_to_save[child_name] = all_grid_points[k]
            # torch.save(dict_to_save, os.path.join(SAVE_DIR, 'points.pth'))

            return (Tree.Node(is_leaf=False, children=child_nodes, edges=child_edges,
                              full_label=full_label, label=full_label.split('/')[-1], geo=0),
                    parts_sdfs_pred,
                    pred_output,
                    shape_sdf_pred,
                    shape_output)

    def structure_recon_loss(self, z, gt_tree, sdf_data, encoder_features=None, rotation=None,
                             parts_indices=None, epoch=0, full_shape_idx=None, noise_full=None):
        root_latent = self.latent_decoder(z)  # torch.Tensor[bsize, 256], torch.Tensor[bsize, 256]
        output = self.node_recon_loss_latentless(root_latent, gt_tree.root, sdf_data, level=0,
                                                 encoder_features=encoder_features, rotation=rotation,
                                                 parts_indices=parts_indices, epoch=epoch,
                                                 full_shape_idx=full_shape_idx,
                                                 noise_full=noise_full)
        return output

    def node_recon_loss_latentless(self, node_latent, gt_node, sdf_data, level=0,
                                   encoder_features=None, rotation=None,
                                   pred_rotation=None, parts_indices=None, epoch=0,
                                   full_shape_idx=None, noise_full=None):

        cuda_device = node_latent.get_device()

        if level == 0:
            root_cls_pred = self.root_classifier(node_latent)
            root_cls_gt = torch.zeros(1, dtype=torch.long)
            root_cls_gt[0] = self.label_to_id[gt_node.label]
            root_cls_gt = root_cls_gt.to("cuda:{}".format(cuda_device))
            root_cls_loss = self.childrenCELoss(root_cls_pred, root_cls_gt)
        else:
            root_cls_loss = 0
            rotation_cls_loss = 0

        if level == 1:
            loss_dict = {}

            loss_dict['exists'] = torch.zeros((1, 1)).to(cuda_device)
            loss_dict['semantic'] = torch.zeros((1, 1)).to(cuda_device)
            loss_dict['edge_exists'] = torch.zeros((1, 1)).to(cuda_device)
            loss_dict['root_cls'] = root_cls_loss

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

            # some deep sdf hyperparams
            clamp_dist = 0.07
            minT = -clamp_dist
            maxT = clamp_dist
            enforce_minmax = True

            # Process the input data
            num_sdf_samples_per_shape = sdf_data.shape[1]
            sdf_data_flattened = sdf_data.reshape(-1, 4)
            num_sdf_samples = sdf_data_flattened.shape[0]
            sdf_data_flattened.requires_grad = False
            # sdf_data_flattened.requires_grad_()
            xyz = sdf_data_flattened[:, 0:3]
            sdf_gt = sdf_data_flattened[:, 3].unsqueeze(1)

            # learn DF instead of SDF
            # sdf_gt = torch.abs(sdf_gt)

            # Perform part-based deepsdf forward pass
            gt_parts_indices = []
            gt_parts_indices_split = []
            corresponding_children_feats = []
            for gt_idx, pred_idx in gt2pred.items():
                gt_child_name = gt_node.children[gt_idx].label
                gt_parts_indices += [parts_indices[gt_child_name] for _ in range(num_sdf_samples_per_shape)]
                gt_parts_indices_split += [[parts_indices[gt_child_name] for _ in range(num_sdf_samples_per_shape)]]
                corresponding_children_feats += [child_feats[:, pred_idx, ...].repeat(num_sdf_samples_per_shape, 1)]

            # train point to part associations
            all_sdf_data_with_feats = []
            all_sdf_data_with_feats_labels = []
            for m, _feat in enumerate(corresponding_children_feats):
                for n in range(len(sdf_data)):
                    _sdf_data = sdf_data[n][:, 0:4]

                    indices_not_noise = torch.where(_sdf_data[:, 3] < clamp_dist)[0]
                    _sdf_data_not_noise = _sdf_data[indices_not_noise]
                    _sdf_data_with_feats = torch.cat([_sdf_data_not_noise, _feat[:len(_sdf_data_not_noise)]], dim=1)
                    all_sdf_data_with_feats += [_sdf_data_with_feats]
                    if m == n:
                        _sdf_data_with_feats_labels = torch.ones((len(_sdf_data_not_noise), 1), device=cuda_device)
                    else:
                        _sdf_data_with_feats_labels = torch.zeros((len(_sdf_data_not_noise), 1), device=cuda_device)
                    all_sdf_data_with_feats_labels += [_sdf_data_with_feats_labels]

                    indices_noise = torch.where(_sdf_data[:, 3] >= clamp_dist)[0]
                    _sdf_data_noise = _sdf_data[indices_noise]
                    _sdf_data_with_feats = torch.cat([_sdf_data_noise, _feat[:len(_sdf_data_noise)]], dim=1)
                    _sdf_data_with_feats_labels = torch.zeros((len(_sdf_data_noise), 1), device=cuda_device)
                    all_sdf_data_with_feats += [_sdf_data_with_feats]
                    all_sdf_data_with_feats_labels += [_sdf_data_with_feats_labels]

            all_sdf_data_with_feats = torch.cat(all_sdf_data_with_feats, dim=0)
            all_sdf_data_with_feats_labels = torch.cat(all_sdf_data_with_feats_labels, dim=0)
            point_part_pred = self.point_part_classifier(all_sdf_data_with_feats)

            # if all_sdf_data_with_feats_labels.max() > 1 or all_sdf_data_with_feats_labels.min() < 0:
            #     print(all_sdf_data_with_feats_labels.min(), all_sdf_data_with_feats_labels.max())
            # if point_part_pred.max() >= 1 or point_part_pred.min() <= 0 or torch.isnan(point_part_pred.max()) or torch.isnan(point_part_pred.min()):
            #     print()
            #     print(all_sdf_data_with_feats.min(), all_sdf_data_with_feats.max())
            #     print(all_sdf_data_with_feats[:, :3].min(), all_sdf_data_with_feats[:, :3].max())
            #     print(all_sdf_data_with_feats[:, 3].min(), all_sdf_data_with_feats[:, 3].max())
            #     print(all_sdf_data_with_feats[:, 4:].min(), all_sdf_data_with_feats[:, 4:].max())
            #     print(sdf_data_flattened[:, :3].min(), sdf_data_flattened[:, :3].max())
            #     print(sdf_data_flattened[:, 3].min(), sdf_data_flattened[:, 3].max())
            #     print()
            # print()
            # print(point_part_pred.min(), point_part_pred.max())
            # print(all_sdf_data_with_feats_labels.min(), all_sdf_data_with_feats_labels.max())
            # print()
            point_part_loss = self.bceLoss(point_part_pred, all_sdf_data_with_feats_labels)
            loss_dict['point_part'] = point_part_loss.mean()
            # loss_dict['point_part'] = 0

            gt_parts_indices = torch.tensor(gt_parts_indices).to(cuda_device)
            corresponding_children_feats = torch.cat(corresponding_children_feats, dim=0)
            batch_vecs = self.lat_vecs(gt_parts_indices)
            # print('Parts norm:', batch_vecs.norm(2), 'Parts sum:', torch.sum(batch_vecs))
            deep_sdf_input = torch.cat([batch_vecs, xyz], dim=1)  # 2
            # deep_sdf_input = torch.cat([corresponding_children_feats, batch_vecs, xyz], dim=1)  # 1
            # deep_sdf_input = torch.cat([corresponding_children_feats, xyz], dim=1)              # 3
            pred_sdf = self.deepsdf_decoder(deep_sdf_input)

            if enforce_minmax:
                pred_sdf = torch.clamp(pred_sdf, minT, maxT)
            chunk_loss = self.L1LossDeepSDF(pred_sdf, sdf_gt.cuda()) / num_sdf_samples

            if self.do_code_regularization:
                l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                reg_loss = (self.code_reg_lambda * min(1, epoch / 100) * l2_size_loss) / num_sdf_samples
                chunk_loss = chunk_loss + reg_loss.cuda()
            loss_dict['sdf'] = chunk_loss

            loss_dict['eikonal_part'] = 0
            # Part Eikonal term
            # point_close_surface_indices = [p for p in range(len(sdf_gt)) if sdf_gt[p, 0] <= 0.06]
            # gradient_close_surface = gradient(xyz, pred_sdf)
            # rand_indices = np.random.choice(len(point_close_surface_indices),
            #                                 min(15000, len(point_close_surface_indices)), False)
            # grad_loss = ((gradient_close_surface[point_close_surface_indices][rand_indices].norm(2, dim=-1) - 1) ** 2).mean()
            # loss_dict['eikonal_part'] = grad_loss

            # Perform full shape deepsdf forward pass
            indices_not_noise = torch.where(sdf_data_flattened[:, 3] < clamp_dist)[0]
            xyz_not_noise = sdf_data_flattened[indices_not_noise]
            indices_noise = torch.where(sdf_data_flattened[:, 3] >= clamp_dist)[0]
            xyz_noise = sdf_data_flattened[indices_noise]
            xyz_random_indices = np.random.choice(len(xyz_not_noise), min(len(xyz_not_noise), 18000))
            # noise_random_indices = np.random.choice(len(noise_full[0, ...]), min(len(noise_full[0, ...]), 2000))
            noise_random_indices = np.random.choice(len(xyz_noise), min(len(xyz_noise), 2000))
            xyz_sampled = xyz_not_noise[torch.LongTensor(xyz_random_indices)]
            # noise_sampled = noise_full[0][torch.LongTensor(noise_random_indices)]
            noise_sampled = xyz_noise[torch.LongTensor(noise_random_indices)]
            xyz_with_noise = torch.vstack([xyz_sampled, noise_sampled])
            full_shape_idx = torch.tensor(full_shape_idx).to(cuda_device)
            batch_shape_vecs = self.lat_shape_vecs(full_shape_idx)[None, :]
            batch_shape_vecs = batch_shape_vecs.repeat(len(xyz_with_noise), 1)
            node_latent_extended = node_latent.repeat(len(xyz_with_noise), 1)
            deep_sdf_shape_input = torch.cat([batch_shape_vecs, xyz_with_noise[:, :3]], dim=1)  # 2
            # deep_sdf_shape_input = torch.cat([node_latent_extended, batch_shape_vecs, xyz_with_noise], dim=1)      # 1
            # deep_sdf_shape_input = torch.cat([node_latent_extended, xyz_with_noise], dim=1)                        # 3
            pred_sdf_shape = self.deepsdf_shape_decoder(deep_sdf_shape_input)

            if enforce_minmax:
                pred_sdf_shape = torch.clamp(pred_sdf_shape, minT, maxT)
            chunk_loss = self.L1LossDeepSDF(pred_sdf_shape, xyz_with_noise[:, 3:].cuda()) / num_sdf_samples

            if self.do_code_regularization:
                l2_size_loss = torch.sum(torch.norm(batch_shape_vecs, dim=1))
                reg_loss = (self.code_reg_lambda * min(1, epoch / 100) * l2_size_loss) / num_sdf_samples
                chunk_loss = chunk_loss + reg_loss.cuda()
            loss_dict['shape_sdf'] = chunk_loss

            loss_dict['eikonal_shape'] = 0
            # Shape Eikonal term
            # gradient_close_surface = gradient(xyz, pred_sdf_shape)
            # rand_indices = np.random.choice(len(point_close_surface_indices),
            #                                 min(15000, len(point_close_surface_indices)), False)
            # grad_loss = ((gradient_close_surface[point_close_surface_indices][rand_indices].norm(2, dim=-1) - 1) ** 2).mean()
            # loss_dict['eikonal_shape'] = grad_loss

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
                input=child_exists_logits, target=child_exists_gt, reduction='none')
            child_exists_loss = child_exists_loss.sum()

            # train edge exists scores
            edge_exists_loss = F.binary_cross_entropy_with_logits(
                input=edge_exists_logits, target=edge_exists_gt, reduction='none')
            edge_exists_loss = edge_exists_loss.sum()
            # rescale to make it comparable to other losses,
            # which are in the order of the number of child nodes
            edge_exists_loss = edge_exists_loss / (edge_exists_gt.shape[2] * edge_exists_gt.shape[3])

            # call children + aggregate losses
            for i in range(len(matched_gt_idx)):
                child_losses = self.node_recon_loss_latentless(
                    child_feats[:, matched_pred_idx[i], :], gt_node.children[matched_gt_idx[i]], sdf_data, level + 1,
                    encoder_features=encoder_features, rotation=rotation,
                    pred_rotation=pred_rotation, parts_indices=parts_indices, epoch=epoch)

                root_cls_loss = root_cls_loss + child_losses['root_cls']
                child_exists_loss = child_exists_loss + child_losses['exists']
                semantic_loss = semantic_loss + child_losses['semantic']
                edge_exists_loss = edge_exists_loss + child_losses['edge_exists']

            loss_dict['root_cls'] = root_cls_loss.view((1))
            loss_dict['exists'] = child_exists_loss.view((1))
            loss_dict['semantic'] = semantic_loss.view((1))
            loss_dict['edge_exists'] = edge_exists_loss.view((1))
            loss_dict['mse_shape'] = 0
            loss_dict['mse_parts'] = 0

            return loss_dict, 0

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

        t0 = time.time()

        if level == 0:
            child_names = []
            for child in gt_node.children:
                if child.label in class2id:
                    child_names += [child.label]

        def perform_rot(geos, rot):
            angle = 30 * rot
            angle = np.pi * angle / 180.

            a, b = np.cos(angle), np.sin(angle)
            matrix = np.array([[a, 0, b],
                               [0, 1, 0],
                               [-b, 0, a]])
            matrix = torch.FloatTensor(matrix).cuda()
            for i in range(geos.shape[0]):
                geos[i][:, :3] = geos[i][:, :3] @ matrix.T
            return geos

        def perform_rot_with_matrix(geos, mat):
            matrix = torch.FloatTensor(mat).cuda()
            for i in range(geos.shape[0]):
                geos[i][:, :3] = apply_transform(geos[i][:, :3], matrix)
            return geos

        cuda_device = node_latent.get_device()

        if level == 0:
            root_cls_pred = self.root_classifier(node_latent)
            root_cls_gt = torch.zeros(1, dtype=torch.long)
            root_cls_gt[0] = self.label_to_id[gt_node.label]
            root_cls_gt = root_cls_gt.to("cuda:{}".format(cuda_device))
            root_cls_loss = self.childrenCELoss(root_cls_pred, root_cls_gt)

            rotation_cls_pred = self.rotation_classifier(node_latent)
            rotation_cls_gt = torch.zeros(1, dtype=torch.long)
            rotation_cls_gt[0] = rotations
            rotation_cls_gt = rotation_cls_gt.to("cuda:{}".format(cuda_device))
            rotation_cls_CE_loss = self.childrenCELoss(rotation_cls_pred, rotation_cls_gt).mean()
            pred_rotation = self.softmax_layer(rotation_cls_pred)
            pred_rotation = int(torch.argmax(pred_rotation).cpu().detach().numpy())
        else:
            root_cls_loss = 0
            rotation_cls_loss = 0

        if level == 1:
            loss_dict = {}

            loss_dict['exists'] = torch.zeros((1, 1)).to(cuda_device)
            loss_dict['semantic'] = torch.zeros((1, 1)).to(cuda_device)
            loss_dict['edge_exists'] = torch.zeros((1, 1)).to(cuda_device)
            loss_dict['root_cls'] = root_cls_loss

            return loss_dict
        else:

            loss_dict = {}
            child_feats, child_sem_logits, child_exists_logits, edge_exists_logits = \
                self.child_decoder(node_latent)

            t1 = time.time()

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

            t2 = time.time()

            # some deep sdf hyperparams
            clamp_dist = 0.07

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

            t3 = time.time()

            # train point to part associations (contrastive)
            # all_sdf_data_with_feats = []
            # all_sdf_data_with_feats_labels = []
            # for m, _feat in enumerate(corresponding_children_feats):
            #     for n in range(len(sdf_data)):
            #         _sdf_data = sdf_data[n][:, 0:4]
            #
            #         indices_not_noise = torch.where(_sdf_data[:, 3] < clamp_dist)[0]
            #         _sdf_data_not_noise = _sdf_data[indices_not_noise]
            #         _sdf_data_with_feats = torch.cat([_sdf_data_not_noise, _feat[:len(_sdf_data_not_noise)]], dim=1)
            #         all_sdf_data_with_feats += [_sdf_data_with_feats]
            #         if m == n:
            #             _sdf_data_with_feats_labels = torch.ones((len(_sdf_data_not_noise), 1), device=cuda_device)
            #         else:
            #             _sdf_data_with_feats_labels = torch.zeros((len(_sdf_data_not_noise), 1), device=cuda_device)
            #         all_sdf_data_with_feats_labels += [_sdf_data_with_feats_labels]
            #
            #         indices_noise = torch.where(_sdf_data[:, 3] >= clamp_dist)[0]
            #         _sdf_data_noise = _sdf_data[indices_noise]
            #         _sdf_data_with_feats = torch.cat([_sdf_data_noise, _feat[:len(_sdf_data_noise)]], dim=1)
            #         _sdf_data_with_feats_labels = torch.zeros((len(_sdf_data_noise), 1), device=cuda_device)
            #         all_sdf_data_with_feats += [_sdf_data_with_feats]
            #         all_sdf_data_with_feats_labels += [_sdf_data_with_feats_labels]
            #
            #     _sdf_data_nb = noise_full
            #     _sdf_data_nb_with_feats = torch.cat([_sdf_data_nb, _feat[:len(_sdf_data_nb)]], dim=1)
            #     _sdf_data_nb_with_feats_labels = torch.zeros((len(_sdf_data_nb), 1), device=cuda_device)
            #     all_sdf_data_with_feats += [_sdf_data_nb_with_feats]
            #     all_sdf_data_with_feats_labels += [_sdf_data_nb_with_feats_labels]
            #
            # all_sdf_data_with_feats = torch.cat(all_sdf_data_with_feats, dim=0)
            # all_sdf_data_with_feats_labels = torch.cat(all_sdf_data_with_feats_labels, dim=0)
            # point_part_pred = self.point_part_classifier(all_sdf_data_with_feats)
            # point_part_loss = self.bceLoss(point_part_pred, all_sdf_data_with_feats_labels)
            # loss_dict['point_part'] = point_part_loss.mean()

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

            t4 = time.time()

            # train MSE for full shape
            full_shape_idx = torch.tensor(full_shape_idx).to(cuda_device)
            batch_shape_vecs = self.lat_shape_vecs(full_shape_idx)[None, :]
            node_latent_projected = self.shape_latent_projector(node_latent)
            shape_mse_loss = self.mseLoss(node_latent_projected, batch_shape_vecs.detach().clone())
            # shape_mse_loss = self.L1Loss(node_latent_projected, batch_shape_vecs.detach().clone())
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
            # parts_mse_loss = self.L1Loss(children_feats_projected, batch_vecs.detach().clone())
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

            loss_dict['root_cls'] = 0
            loss_dict['exists'] = child_exists_loss.view((1))
            loss_dict['semantic'] = semantic_loss.view((1))
            loss_dict['edge_exists'] = edge_exists_loss.view((1))
            loss_dict['rotation'] = rotation_cls_CE_loss.view((1))

            t5 = time.time()

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
                print(child_name, scannet_samples.shape)
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

        print('Start decoding')
        cuda_device = node_latent.get_device()
        gt_tree_children_names = [x.label for x in gt_tree.root.children]

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

            # DEPRECATED
            # reconstruct meshes by predicting points to parts associations
            # partnet / mlcvnet val
            # sdf_data_flattened = sdf_data.reshape(-1, 4)
            # sdf_data_flattened = torch.vstack([sdf_data_flattened, noise_full])

            # mlcvnet test
            sdf_data_flattened = sdf_data.reshape(-1, 4)

            # optionally lift points to increase min y value
            indices = torch.where(sdf_data_flattened[:, 3] < 0.07)[0]
            min_y_point = sdf_data_flattened[indices].amin(dim=0)[1]
            print('Min y:', min_y_point)
            indices = torch.where(sdf_data_flattened[:, 1] > min_y_point + 0.0)[0]
            sdf_data_flattened = sdf_data_flattened[indices]
            min_y_point = sdf_data_flattened.amin(dim=0)[1]
            print('Min y:', min_y_point)

            # input point cloud
            sdf_flat = deepcopy(sdf_data_flattened.cpu().detach())

            # DEPRECATED
            # test - pred_rotation, train/val - -pred_rotation
            # pred_rotation = 0
            # sdf_data_flattened[:, :3], rot_matrix = perform_30_rot(sdf_data_flattened[:, :3], pred_rotation)

            # rotated point cloud
            sdf_flat_rot = deepcopy(sdf_data_flattened.cpu().detach())
            sdf_not_noise = torch.stack([x for x in sdf_data_flattened if x[3] < 0.07]).cuda()
            # sdf_data_with_noise = torch.vstack([sdf_not_noise, sdf_noise])
            sdf_data_with_noise = torch.vstack([sdf_not_noise])

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
            # sdf_data_with_noise = torch.vstack([sdf_data_flattened, sdf_noise])
            # test - pred_rotation, train/val - -pred_rotation (new training approaches)
            # pred_rotation = 0
            sdf_data_flattened[:, :3], rot_matrix = perform_30_rot(sdf_data_flattened[:, :3], pred_rotation)

            # retrieve GT part embeddings (only for testing or ablation)
            # gt_latent_parts_indices = []
            children_names = [x.label for x in gt_tree.root.children]
            # for child_name in children_names:
            #     gt_latent_parts_indices += [parts_indices[child_name]]
            # gt_latent_parts_indices = torch.tensor(gt_latent_parts_indices).to(cuda_device)
            # batch_vecs = self.lat_vecs(gt_latent_parts_indices)
            # child_feats_exist_projection = batch_vecs.detach().clone()

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
            print('Covering evaluation (onedir):', chamfer_dist_parts_before_onedir)
            print('Covering evaluation (secdir):', chamfer_dist_parts_before_secdir)

            # (DEPRECATED)
            # add noise points to part points
            # for k in range(len(children_sdf_points)):
            #     children_sdf_points[k] = torch.vstack([children_sdf_points[k].cuda(), above_thr_points])

            # ICP SECTION #
            # perform ICP from points to predicted meshes
            # take points from ScanNet (ICP)
            all_points_icp = torch.vstack(children_sdf_points)
            indices = torch.where(all_points_icp[:, 3] < 0.01)[0]
            all_points_icp = all_points_icp[indices].cpu().detach().numpy()
            # optionally remove points that are close to floor
            # min_y_point = all_points_icp.min(0)[1]
            # indices = np.where(all_points_icp[:, 1] > min_y_point + 0.1)[0]
            # all_points_icp = all_points_icp[indices]
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
                extra_sdf_points[k] = extra_sdf_points[k].cuda() # <-- check if it affects
                # extra_sdf_points[k][:, :3] = apply_transform_torch(extra_sdf_points[k][:, :3], transform) # only for old models

            # sample uniform noise points inside (-1.0, 1.0) cube
            # and filter them if they are too close to classified or predicted points
            # cls_points = []
            # for k in range(len(children_sdf_points)):
            #     random_indices = np.random.choice(len(children_sdf_points[k]), min(30000, len(children_sdf_points[k])), replace=False)
            #     cls_points += [children_sdf_points[k][random_indices]]
            all_parts_points = torch.vstack(extra_sdf_points + children_sdf_points).cpu().numpy()[:, :3]
            num_samples_random = 50000
            random_samples = np.random.uniform(-1.0, 1.0, (num_samples_random, 3))
            kd_tree = cKDTree(all_parts_points)
            min_dist_rand, min_idx_rand = kd_tree.query(random_samples)
            above_thr_indices = np.where(min_dist_rand > 0.08)[0]  # 0.1 <--- noise_thr, noise from before TTO
            above_thr_points = random_samples[above_thr_indices]
            above_thr_points = np.hstack([above_thr_points, 0.07 * np.ones((len(above_thr_points), 1))]).astype('float32')
            above_thr_points = torch.FloatTensor(above_thr_points) # probably need to revise uniform noise formulation wrt CD parts (no adding extra_sdf_points)

            # classify points before TTO
            # before_tto_points = {}
            # for k, child_name in enumerate(children_names):
            #     before_tto_points[child_name] = extra_sdf_points[k].cpu().detach().numpy()
            # full_shape_points = torch.vstack(children_sdf_points).cpu().detach().numpy()
            # indices = np.where(np.abs(full_shape_points[:, 3]) < 0.05)[0]
            # full_shape_points = full_shape_points[indices]
            # full_shape_points = full_shape_points[:, :3]
            # full_shape_tree = cKDTree(full_shape_points)
            # additional_points_from_pred = {}
            # for child_name in before_tto_points:
            #     min_dist, min_idx = full_shape_tree.query(before_tto_points[child_name][:, :3])
            #     above_thr_indices = np.where(min_dist > 0.15)[0]
            #     additional_points_from_pred[child_name] = before_tto_points[child_name][above_thr_indices]
            # all_points = []
            # for child_name in additional_points_from_pred:
            #     all_points += [additional_points_from_pred[child_name]]
            # if len(all_points) > 0:
            #     additional_points_from_pred['full'] = torch.FloatTensor(np.vstack(all_points)).cuda()
            # else:
            #     additional_points_from_pred['full'] = []
            # perform classification again
            # points_to_classify = torch.vstack(children_sdf_points + [additional_points_from_pred['full']])
            # points_to_classify = torch.vstack(children_sdf_points)
            # num_samples = len(points_to_classify)
            # rotation_vector = torch.FloatTensor(torch.zeros((1, 12))).cuda()
            # rotation_vector[0][0] = 1
            # node_latent_rotation = torch.hstack([node_latent, rotation_cls_pred])
            # node_feat_expanded = node_latent_rotation.expand(num_samples, -1)
            # inputs = torch.cat([points_to_classify[:, :4], node_feat_expanded], 1).cuda()
            # torch.save(points_to_classify[:, :4],
            #            'points.pth')
            # pred_accociations = self.point_part_classifier(inputs)
            # print(pred_accociations)
            # pred_accociations = torch.max(pred_accociations, dim=1)[1]
            # print('CLS:', inputs.shape)
            # print(pred_accociations)
            # splitted_sdf_points = {}
            # for i in range(len(child_feats_exist)):
            #     child_name = child_nodes[i].label
            #     part_point_indices = torch.where(pred_accociations == self.class2id[child_name] + 1)[0]
            #     valid_points = [x for x in points_to_classify[part_point_indices] if x[3] < 0.07]
            #     print('CLS:', child_name, part_point_indices.shape)
            #     if len(valid_points) == 0:
            #         valid_points = (torch.FloatTensor([0, 0, 0, 0.7]),)
            #     sdf_not_noise = torch.stack(valid_points).cuda()
            #     splitted_sdf_points[child_name] = torch.vstack([sdf_not_noise])
            #     print('CLS:', child_name, splitted_sdf_points[child_name].shape)
            # children_sdf_points = []
            # for child_name in children_names:
            #     children_sdf_points += [splitted_sdf_points[child_name]]

            # points_to_classify = torch.vstack(children_sdf_points + [additional_points_from_pred['full']]).cpu().numpy()
            # source_points = []
            # source_labels = []
            # for k, child_name in enumerate(children_names):
            #     source_points += [before_tto_points[child_name]]
            #     source_labels += [k for _ in range(len(before_tto_points[child_name]))]
            # source_points = np.vstack(source_points)
            # source_labels = np.array(source_labels)
            # kd_tree = cKDTree(source_points)
            # min_dist, min_idx = kd_tree.query(points_to_classify)
            # projected_labels = source_labels[min_idx]
            # children_sdf_points = []
            # splitted_sdf_points = {}
            # for k, child_name in enumerate(children_names):
            #     part_indices = np.where(projected_labels == k)[0]
            #     print('CLS:', child_name, len(part_indices))
            #     children_sdf_points += [torch.FloatTensor(points_to_classify[part_indices]).cuda()]
            #     splitted_sdf_points[child_name] = torch.FloatTensor(points_to_classify[part_indices]).cuda()

            # form outputs for the following TTO stage
            pred_output = (children_embeddings, children_names, children_sdf_points, children_feats, extra_sdf_points, chamfer_dist_parts_before_secdir)

            # only for testing or ablation
            # full_shape_idx = torch.tensor(full_shape_idx).to(cuda_device)
            # batch_shape_vecs = self.lat_shape_vecs(full_shape_idx)[None, :]
            # node_latent_projection = batch_shape_vecs.detach().clone()

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

        parts_sdfs_pred = {}
        children_embeddings = []
        children_names = []
        children_sdf_points = []
        children_feats = []
        extra_sdf_points = []
        for i in range(len(child_feats)):
        # for i, child_name in enumerate(child_names):

            # unpack input entities
            child_node = child_nodes[i]
            child_name = child_node.label
            if child_name == 'surface_base' and child_name not in splitted_sdf_points:
                child_name = 'regular_leg_base' # <-- manually replace this part
            # child_name = child_names[i]
            parts_sdfs_pred[child_name] = {}
            # parts_sdfs_pred[i] = {}
            sdf_part = splitted_sdf_points[child_name]
            children_names += [child_name]
            # children_names += [i]
            children_feats += [child_feats[i].detach().cpu().clone()]

            children_sdf_points += [sdf_part.detach().cpu().clone()]

            idx_latent = class2id[child_name]
            class_one_hot = torch.FloatTensor(torch.zeros((len(class2id))))
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

                # parts_sdfs_pred[i]['sdf'] = sdf_values
                # parts_sdfs_pred[i]['vox_origin'] = voxel_origin
                # parts_sdfs_pred[i]['vox_size'] = voxel_size
                # parts_sdfs_pred[i]['offset'] = offset
                # parts_sdfs_pred[i]['scale'] = scale

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

        print('Full shape is processed')

        samples = samples.detach().cpu()
        shape_feat = shape_feat.detach().cpu()
        sdf_data = sdf_data.detach().cpu()
        latent = latent.detach().cpu()

        return shape_sdf_pred, latent, sdf_data, shape_feat, samples

    def reconstruct_children_with_associations(self, child_nodes, child_feats, splitted_sdf_points, mode):
        parts_sdfs_pred = {}
        children_embeddings = []
        children_names = []
        children_sdf_points = []
        children_feats = []
        for i, child_node in enumerate(child_nodes):
            child_name = child_node.label
            parts_sdfs_pred[child_name] = {}
            sdf_part = splitted_sdf_points[i]
            try:
                # sdf_part = torch.stack([x for x in sdf_part if x[3] > -0.015])
                # sdf_part[:, 3] = sdf_part[:, 3] - 0.00

                err, latent, loss_history = reconstruct(
                    self.deepsdf_decoder,
                    800,
                    256,
                    sdf_part,
                    0.01,  # [emp_mean, emp_var],
                    0.1,
                    num_samples=len(sdf_part),
                    lr=5e-3,
                    l2reg=True,
                    cal_scale=1,
                    add_feat=child_feats[i],
                    mode=mode
                )
            except RuntimeError:
                latent = torch.zeros_like(child_feats[i]).cuda()
            # latent = source_embedding[i]
            children_embeddings += [latent.detach().clone()]
            children_names += [child_name]
            children_sdf_points += [sdf_part]
            children_feats += [child_feats[i].detach().clone()]
            self.deepsdf_decoder.eval()
            with torch.no_grad():
                sdf_values, voxel_origin, voxel_size, offset, scale, samples = create_grid(
                    self.deepsdf_decoder,
                    latent,
                    N=GRID_RESOLUTION,
                    max_batch=int(2 ** 18),
                    input_samples=None,
                    add_feat=child_feats[i],
                    mode=mode
                )
                parts_sdfs_pred[child_name]['sdf'] = sdf_values
                parts_sdfs_pred[child_name]['vox_origin'] = voxel_origin
                parts_sdfs_pred[child_name]['vox_size'] = voxel_size
                parts_sdfs_pred[child_name]['offset'] = offset
                parts_sdfs_pred[child_name]['scale'] = scale

            print('Part', child_name, 'is processed')

        return parts_sdfs_pred, children_embeddings, children_names, children_sdf_points, children_feats

    def reconstruct_full_shape(self, shape_feat, sdf_data, mode):
        sdf_data = sdf_data.reshape(-1, 4).cuda()

        # sdf_data = torch.stack([x for x in sdf_data if x[3] > -0.015])
        # sdf_data[:, 3] = sdf_data[:, 3] - 0.01

        shape_sdf_pred = {}
        err, latent, loss_history = reconstruct(
            self.deepsdf_shape_decoder,
            800,
            256,
            sdf_data,
            0.01,  # [emp_mean, emp_var],
            0.1,
            num_samples=len(sdf_data),
            lr=5e-3,
            l2reg=True,
            cal_scale=1,
            add_feat=shape_feat,
            mode=mode
        )
        # latent = source_embedding[0]
        self.deepsdf_decoder.eval()
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

        return shape_sdf_pred, latent, sdf_data, shape_feat, samples

    def reconstruct_children(self, child_nodes, child_feats, gt_tree_children_names, sdf_data, mode):
        parts_sdfs_pred = {}
        children_embeddings = []
        children_names = []
        children_sdf_points = []
        for i, child_node in enumerate(child_nodes):
            child_name = child_node.label
            if child_name in gt_tree_children_names:
                parts_sdfs_pred[child_name] = {}
                part_id = gt_tree_children_names.index(child_name)
                sdf_part = sdf_data[part_id]
                err, latent, loss_history = reconstruct(
                    self.deepsdf_decoder,
                    800,
                    256,
                    sdf_part,
                    0.01,  # [emp_mean, emp_var],
                    0.1,
                    num_samples=30000,
                    lr=5e-3,
                    l2reg=True,
                    cal_scale=1,
                    add_feat=child_feats[i],
                    mode=mode
                )
                children_embeddings += [latent]
                children_names += [child_name]
                children_sdf_points += [sdf_part]
                self.deepsdf_decoder.eval()
                with torch.no_grad():
                    sdf_values, voxel_origin, voxel_size, offset, scale, samples = create_grid(
                        self.deepsdf_decoder,
                        latent,
                        N=GRID_RESOLUTION,
                        max_batch=int(2 ** 18),
                        input_samples=None,
                        add_feat=child_feats[i],
                        mode=mode
                    )
                    parts_sdfs_pred[child_name]['sdf'] = sdf_values
                    parts_sdfs_pred[child_name]['vox_origin'] = voxel_origin
                    parts_sdfs_pred[child_name]['vox_size'] = voxel_size
                    parts_sdfs_pred[child_name]['offset'] = offset
                    parts_sdfs_pred[child_name]['scale'] = scale

                print('Part', child_name, 'is processed')
            else:
                parts_sdfs_pred[child_name] = -1
                print('Part', child_name, 'not predicted')

        return parts_sdfs_pred, children_embeddings, children_names, children_sdf_points
