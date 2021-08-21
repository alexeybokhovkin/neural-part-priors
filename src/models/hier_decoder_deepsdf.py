import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from copy import deepcopy
import math
from scipy.spatial import cKDTree
import json

from ..data_utils.hierarchy import Tree
from ..utils.gnn import linear_assignment
from ..utils.losses import IoULoss
from .deep_sdf_decoder import Decoder as DeepSDFDecoder
from .gnn_contrast import LatentDecoder, NumChildrenClassifier, GNNChildDecoder, LatentProjector
from deep_sdf_utils.data import unpack_sdf_samples_from_ram
from deep_sdf_utils.utils import decode_sdf
from ..utils.losses import gradient


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

    new_lr = lr_1
    for e in range(num_iterations):
        if e % 200 == 0:
            all_latents += [latent.detach().clone()]

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

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs)
        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)

        if e == 0:
            stats_before_tto = (inputs.detach().cpu().numpy(),
                                loss.detach().cpu().numpy(),
                                sdf_gt.detach().cpu().numpy())
        if e == num_iterations - 1:
            stats_after_tto = (inputs.detach().cpu().numpy(),
                               loss.detach().cpu().numpy(),
                               sdf_gt.detach().cpu().numpy())

        loss = loss.mean()
        if l2reg:
            loss = loss + 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer_1.step()
        optimizer_2.step()
        loss_num = loss.cpu().data.numpy()

        loss_history += [float(loss.cpu().data.numpy())]

        if e % 200 == 0:
            print('loss:', loss.cpu().data.numpy())
            print('lat sum:', latent.detach().cpu().sum())
            print('loss norm:', torch.mean(latent.detach().cpu().pow(2)))
            decoder.eval()
            with torch.no_grad():
                pred_sdf_part[j] = {}
                sdf_values, voxel_origin, voxel_size, offset, scale, _ = create_grid(
                    decoder,
                    latent,
                    N=256,
                    max_batch=int(2 ** 18),
                    input_samples=None,
                    add_feat=add_feat,
                    mode=mode
                )
                pred_sdf_part[j]['sdf'] = sdf_values
                pred_sdf_part[j]['vox_origin'] = voxel_origin
                pred_sdf_part[j]['vox_size'] = voxel_size
                pred_sdf_part[j]['offset'] = offset
                pred_sdf_part[j]['scale'] = scale
            decoder.train()
            j += 1

    return loss_num, all_latents, loss_history, pred_sdf_part, stats_before_tto, stats_after_tto


def create_grid(
        decoder, latent_vec, N=256, max_batch=32 ** 3, offset=None, scale=None,
        input_samples=None, add_feat=None, mode=0
):
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

    while head < num_samples:
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:3].cuda()

        samples[head: min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset, add_feat=add_feat, mode=mode)
                .squeeze(1)
                .detach()
                .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    return sdf_values.data.cpu(), voxel_origin, voxel_size, offset, scale, samples


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


class RecursiveDeepSDFDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size, max_child_num, device,
                 edge_symmetric_type, num_iterations, edge_type_num,
                 num_parts=None, num_shapes=None, deep_sdf_specs=None):
        super(RecursiveDeepSDFDecoder, self).__init__()

        self.label_to_id = {
            'chair': 0,
            'bed': 1,
            'storage_furniture': 2,
            'table': 3,
            'trash_can': 4
        }

        self.edge_types = ['ADJ', 'SYM']
        self.device = device
        self.max_child_num = max_child_num

        self.latent_decoder = LatentDecoder(feature_size, hidden_size, hidden_size)
        self.root_classifier = NumChildrenClassifier(feature_size, hidden_size, 5)
        self.rotation_classifier = NumChildrenClassifier(feature_size, hidden_size, 8)
        self.child_decoder = GNNChildDecoder(feature_size, hidden_size,
                                             max_child_num, edge_symmetric_type,
                                             num_iterations, edge_type_num)

        self.point_part_classifier = PointPartClassifier(260, 128, 128, 64)

        # Part level
        deep_sdf_latent_size = feature_size
        # deep_sdf_specs['dims'] = [512, 512, 512, 768, 512, 512, 512, 512]
        deep_sdf_specs['dims'] = [512, 512, 512, 512, 512, 512, 512, 512]
        self.deepsdf_decoder = DeepSDFDecoder(deep_sdf_latent_size, mode=2, **deep_sdf_specs).cuda()
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
        self.deepsdf_shape_decoder = DeepSDFDecoder(deep_sdf_latent_size, mode=2, **deep_sdf_specs).cuda()
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

    def tto(self, children_initial_data, shape_initial_data=None, index=0):

        print('Start TTO')
        parts_sdfs_pred, shape_sdf_pred, \
        parts_stats_before, parts_stats_after, shape_stats_before_tto, shape_stats_after_tto \
            = self.adjust_embedding(children_initial_data,
                                    shape_initial_data,
                                    mode=1)
        print('Finish TTO')
        return parts_sdfs_pred, shape_sdf_pred, parts_stats_before, parts_stats_after, shape_stats_before_tto, shape_stats_after_tto

    def adjust_embedding(self, children_data, shape_data, mode):

        children_embeddings = children_data[0]
        children_names = children_data[1]
        children_sdf_points = children_data[2]
        if mode == 1:
            children_feats = children_data[3]
        else:
            children_feats = [0 for _ in range(len(children_names))]

        parts_sdfs_pred = {}
        parts_stats_before = {}
        parts_stats_after = {}

        initial_parameters = deepcopy(self.deepsdf_decoder.state_dict())

        self.deepsdf_decoder.train()
        for param in self.deepsdf_decoder.parameters():
            param.requires_grad = True

        all_grid_points = shape_data[3]

        # emb = torch.load(os.path.join('latents', '14134', 'emb.pth'))
        # feat = torch.load(os.path.join('latents', '14134', 'feat.pth'))

        for i, child_name in enumerate(children_names):
            try:
                parts_sdfs_pred[child_name] = {}
                sdf_part = children_sdf_points[i]

                # sdf_part_noise = torch.stack([x for x in sdf_part if x[3] >= 0.07])

                sdf_part = torch.stack([x for x in sdf_part if x[3] > -0.015])
                indices = np.random.choice(len(sdf_part), min(20000, len(sdf_part)), replace=False)
                sdf_part = sdf_part[indices]
                # sdf_part[:, 3] = sdf_part[:, 3] - 0.00

                # all_grid_points[i] = [x for x in all_grid_points[i] if x[3] < 0.00]
                # all_grid_points[i] = torch.stack(all_grid_points[i])
                # sdf_part_neg = torch.stack([x for x in sdf_part if x[3] < 0.0])
                # tree = cKDTree(sdf_part_neg.cpu().numpy()[:, :3])
                # min_dist, min_idx = tree.query(all_grid_points[i].cpu().numpy()[:, :3])
                # grid_indices = np.where(np.abs(min_dist) > 0.03)[0]
                # all_grid_points[i] = all_grid_points[i][grid_indices]
                # random_grid_indices = np.random.choice(len(all_grid_points[i]), min(6000, len(all_grid_points[i])), replace=False)
                # extra_grid_points = all_grid_points[i][random_grid_indices]
                # sdf_part = torch.vstack([sdf_part.cuda(), extra_grid_points.cuda()])

                # tree = cKDTree(sdf_part.cpu().numpy()[:, :3])
                # min_dist, min_idx = tree.query(sdf_part_noise.cpu().numpy()[:, :3])
                # indices = np.where(np.abs(min_dist) > 0.07)[0]
                # new_sdf_part_noise = sdf_part_noise[indices]
                # sdf_part = torch.vstack([sdf_part, new_sdf_part_noise.cuda()])

                # loaded_emb = emb[child_name]
                # loaded_feat = feat[child_name]
                # loaded_emb = loaded_emb.cuda()
                # loaded_feat = loaded_feat.cuda()
                print(child_name, len(sdf_part))
                err, all_latents, loss_history, pred_sdf_part, stats_before_tto, stats_after_tto = reconstruct_tto(
                    self.deepsdf_decoder,
                    801,
                    sdf_part,
                    0.1,
                    num_samples=len(sdf_part),
                    lr_1=0, # 1e-2
                    lr_2=5e-4, # 5e-4
                    l2reg=True,
                    cal_scale=1,
                    add_feat=children_feats[i],
                    init_emb=children_embeddings[i],
                    mode=2
                )
                parts_sdfs_pred[child_name] = pred_sdf_part
                parts_stats_before[child_name] = stats_before_tto
                parts_stats_after[child_name] = stats_after_tto

                self.deepsdf_decoder.load_state_dict(initial_parameters)

                print('processed part', child_name)
            except FileNotFoundError:
                continue

        if shape_data is not None:
            print()
            print('Process full shape')
            shape_embedding = shape_data[0]
            sdf_data = shape_data[1]
            if mode == 1:
                shape_feat = shape_data[2]
            else:
                shape_feat = 0

            self.deepsdf_shape_decoder.train()
            for param in self.deepsdf_shape_decoder.parameters():
                param.requires_grad = True

            sdf_data = torch.stack([x for x in sdf_data if x[3] > -0.015])
            # sdf_data[:, 3] = sdf_data[:, 3] - 0.00
            indices = np.random.choice(len(sdf_data), min(80000, len(sdf_data)), replace=False)
            sdf_data = sdf_data[indices]

            # all_grid_points = torch.vstack(all_grid_points)
            # random_grid_indices = np.random.choice(len(all_grid_points), min(30000, len(all_grid_points)), replace=False)
            # extra_grid_points = all_grid_points[random_grid_indices]
            # sdf_data = torch.vstack([sdf_data.cuda(), extra_grid_points.cuda()])

            # loaded_emb = emb['full']
            # loaded_feat = feat['full']
            # loaded_emb = loaded_emb.cuda()
            # loaded_feat = loaded_feat.cuda()

            print('full', len(sdf_data))
            err, all_latents, loss_history, shape_sdf_pred, shape_stats_before_tto, shape_stats_after_tto = reconstruct_tto(
                self.deepsdf_shape_decoder,
                801,
                sdf_data,
                0.1,
                num_samples=len(sdf_data),
                lr_1=0, # 1e-2
                lr_2=5e-4, # 5e-4
                l2reg=True,
                cal_scale=1,
                add_feat=shape_feat,
                init_emb=shape_embedding,
                mode=2
            )

            self.deepsdf_decoder.load_state_dict(initial_parameters)

        return parts_sdfs_pred, shape_sdf_pred, \
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

    def deepsdf_recon_loss(self, sdf_data, parts_indices=None, epoch=0, full_shape_idx=None, noise_full=None):
        output = self.deepsdf_loss(sdf_data, parts_indices=parts_indices, epoch=epoch,
                                   full_shape_idx=full_shape_idx, noise_full=noise_full)
        return output

    def deepsdf_loss(self, sdf_data,
                     parts_indices=None, epoch=0,
                     full_shape_idx=None, noise_full=None):

        loss_dict = {}

        # some deep sdf hyperparams
        clamp_dist = 0.07
        minT = -clamp_dist
        maxT = clamp_dist
        enforce_minmax = True

        # Process the input data
        num_sdf_samples_per_shape = sdf_data.shape[-2]
        sdf_data_flattened = sdf_data.reshape(-1, 4)
        num_sdf_samples = sdf_data_flattened.shape[0]
        sdf_data_flattened.requires_grad = False
        # sdf_data_flattened.requires_grad_()
        xyz = sdf_data_flattened[:, 0:3]
        sdf_gt = sdf_data_flattened[:, 3].unsqueeze(1)

        # learn DF instead of SDF
        # sdf_gt = torch.abs(sdf_gt)

        # Perform part-based deepsdf forward pass
        # gt_parts_indices = []
        # for child_name in parts_indices:
        #     gt_parts_indices += [parts_indices[child_name] for _ in range(num_sdf_samples_per_shape)]
        #
        # gt_parts_indices = torch.tensor(gt_parts_indices).cuda()
        # batch_vecs = self.lat_vecs(gt_parts_indices)
        # deep_sdf_input = torch.cat([batch_vecs, xyz], dim=1)
        # pred_sdf = self.deepsdf_decoder(deep_sdf_input)
        #
        # if enforce_minmax:
        #     pred_sdf = torch.clamp(pred_sdf, minT, maxT)
        # chunk_loss = self.L1LossDeepSDF(pred_sdf, sdf_gt.cuda()) / num_sdf_samples
        #
        # if self.do_code_regularization:
        #     l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
        #     reg_loss = (self.code_reg_lambda * min(1, epoch / 100) * l2_size_loss) / num_sdf_samples
        #     chunk_loss = chunk_loss + reg_loss.cuda()
        # loss_dict['sdf'] = chunk_loss

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
        full_shape_idx = torch.tensor(full_shape_idx).cuda()
        batch_shape_vecs = self.lat_shape_vecs(full_shape_idx)[None, :]
        batch_shape_vecs = batch_shape_vecs.repeat(len(xyz_with_noise), 1)
        deep_sdf_shape_input = torch.cat([batch_shape_vecs, xyz_with_noise[:, :3]], dim=1)
        pred_sdf_shape = self.deepsdf_shape_decoder(deep_sdf_shape_input)

        if enforce_minmax:
            pred_sdf_shape = torch.clamp(pred_sdf_shape, minT, maxT)
        chunk_loss = self.L1LossDeepSDF(pred_sdf_shape, xyz_with_noise[:, 3:].cuda()) / num_sdf_samples

        if self.do_code_regularization:
            l2_size_loss = torch.sum(torch.norm(batch_shape_vecs, dim=1))
            reg_loss = (self.code_reg_lambda * min(1, epoch / 100) * l2_size_loss) / num_sdf_samples
            chunk_loss = chunk_loss + reg_loss.cuda()
        loss_dict['shape_sdf'] = chunk_loss

        return loss_dict, 0

    def latent_recon_loss(self, z, gt_tree, sdf_data, encoder_features=None, rotation=None,
                          parts_indices=None, epoch=0, full_shape_idx=None, noise_full=None):
        root_latent = self.latent_decoder(z)  # torch.Tensor[bsize, 256], torch.Tensor[bsize, 256]
        output = self.node_latent_loss(root_latent, gt_tree.root, sdf_data, level=0,
                                       encoder_features=encoder_features, rotation=rotation,
                                       parts_indices=parts_indices, epoch=epoch,
                                       full_shape_idx=full_shape_idx,
                                       noise_full=noise_full)
        return output

    def node_latent_loss(self, node_latent, gt_node, sdf_data, level=0,
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

            # Process the input data
            num_sdf_samples_per_shape = sdf_data.shape[1]
            sdf_data_flattened = sdf_data.reshape(-1, 4)
            sdf_data_flattened.requires_grad = False

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

            point_part_loss = self.bceLoss(point_part_pred, all_sdf_data_with_feats_labels)
            loss_dict['point_part'] = point_part_loss.mean()
            # loss_dict['point_part'] = 0

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
            batch_vecs = self.lat_vecs(gt_latent_parts_indices)
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
                child_losses = self.node_recon_loss_latentless(
                    child_feats[:, matched_pred_idx[i], :], gt_node.children[matched_gt_idx[i]], sdf_data, level + 1,
                    encoder_features=encoder_features, rotation=rotation,
                    pred_rotation=pred_rotation, parts_indices=parts_indices, epoch=epoch
                )

                root_cls_loss = root_cls_loss + child_losses['root_cls']
                child_exists_loss = child_exists_loss + child_losses['exists']
                semantic_loss = semantic_loss + child_losses['semantic']
                edge_exists_loss = edge_exists_loss + child_losses['edge_exists']

            loss_dict['root_cls'] = root_cls_loss.view((1))
            loss_dict['exists'] = child_exists_loss.view((1))
            loss_dict['semantic'] = semantic_loss.view((1))
            loss_dict['edge_exists'] = edge_exists_loss.view((1))

            # loss_dict['exists'] = 0
            # loss_dict['semantic'] = 0
            # loss_dict['root_cls'] = 0
            # loss_dict['edge_exists'] = 0

            return loss_dict, 0

    def get_latent_vecs(self):
        return self.lat_vecs

    def get_latent_shape_vecs(self):
        return self.lat_shape_vecs

    # decode a root code into a tree structure
    def decode_structure_two_stage(self, z, sdf_data, full_label=None, encoder_features=None, rotation=None,
                                   gt_tree=None,
                                   index=0, parts_indices=None, full_shape_idx=None, noise_full=None):
        root_latent = self.latent_decoder(z)  # torch.Tensor[bsize, 256], torch.Tensor[bsize, 256]
        if full_label is None:
            full_label = Tree.root_sem
        output = self.decode_node_two_stage(root_latent, sdf_data, full_label=full_label, level=0,
                                            encoder_features=encoder_features, rotation=rotation,
                                            gt_tree=gt_tree, index=index, parts_indices=parts_indices,
                                            full_shape_idx=full_shape_idx, noise_full=noise_full)

        obj = Tree(root=output[0])
        new_output = [obj, ]
        for i in range(1, len(output)):
            new_output += [output[i]]
        new_output = tuple(new_output)
        return new_output

    # decode a part node
    def decode_node_two_stage(self, node_latent, sdf_data, full_label, level=0, encoder_features=None, rotation=None,
                              pred_rotation=None, gt_tree=None, index=0, parts_indices=None,
                              full_shape_idx=None, noise_full=None):
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

            # reconstruct meshes by predicting points to parts associations
            sdf_data_flattened = sdf_data.reshape(-1, 4)

            # sdf_not_noise = torch.stack([x for x in sdf_data_flattened if x[3] < 0.07]).cuda()
            noise_indices = np.random.choice(len(noise_full), 4000, replace=False)
            sdf_noise = noise_full[noise_indices].cuda()

            children_names = [x.label for x in gt_tree.root.children]
            splitted_sdf_points = []
            for k, child_name in enumerate(children_names):
                sdf_not_noise = torch.stack([x for x in sdf_data[k] if x[3] < 0.07]).cuda()
                splitted_sdf_points += [torch.vstack([sdf_not_noise, sdf_noise])]
            sdf_not_noise = torch.stack([x for x in sdf_data_flattened if x[3] < 0.07]).cuda()
            sdf_data_with_noise = torch.vstack([sdf_not_noise, sdf_noise])


            # sdf_data_flattened = sdf_not_noise.cuda()
            # all_pred_associations = []
            # num_samples = len(sdf_data_flattened)
            # for child_feat in child_feats_exist:
            #     child_feat_expanded = child_feat.expand(num_samples, -1)
            #     inputs = torch.cat([sdf_data_flattened[:, :4], child_feat_expanded], 1).cuda()
            #     pred_accociations = self.point_part_classifier(inputs)
            #     all_pred_associations += [pred_accociations]
            # all_pred_associations = torch.cat(all_pred_associations, dim=1)
            # all_pred_associations = torch.argmax(all_pred_associations, dim=1)
            # splitted_sdf_points = []
            # for i in range(len(child_feats_exist)):
            #     part_point_indices = torch.where(all_pred_associations == i)[0]
            #     splitted_sdf_points += [torch.vstack([sdf_data_flattened[part_point_indices], sdf_noise])]

            # sdf_data_with_noise = torch.vstack([sdf_data_flattened, sdf_noise])

            gt_latent_parts_indices = []
            # for child_node in child_nodes:
            #     child_name = child_node.label
            #     gt_latent_parts_indices += [parts_indices[child_name]]
            children_names = [x.label for x in gt_tree.root.children]
            for child_name in children_names:
                gt_latent_parts_indices += [parts_indices[child_name]]
            gt_latent_parts_indices = torch.tensor(gt_latent_parts_indices).to(cuda_device)
            batch_vecs = self.lat_vecs(gt_latent_parts_indices)
            child_feats_exist_projection = batch_vecs.detach().clone()

            # child_feats_exist_projection = self.part_latent_projector(torch.vstack(child_feats_exist))
            # reconstruct parts
            parts_sdfs_pred, children_embeddings, children_names, children_sdf_points, children_feats = self.reconstruct_children_from_latents(
                child_nodes,
                child_feats_exist_projection,
                splitted_sdf_points,
                children_names,
                mode=2)

            pred_output = (children_embeddings, children_names, children_sdf_points, children_feats)

            full_shape_idx = torch.tensor(full_shape_idx).to(cuda_device)
            batch_shape_vecs = self.lat_shape_vecs(full_shape_idx)[None, :]
            node_latent_projection = batch_shape_vecs.detach().clone()

            # node_latent_projection = self.shape_latent_projector(node_latent)
            # reconstruct full shape
            shape_sdf_pred, shape_embedding, sdf_data, shape_feat, samples = self.reconstruct_full_shape_from_latents(
                node_latent_projection,
                sdf_data_with_noise,
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

            return (Tree.Node(is_leaf=False, children=child_nodes, edges=child_edges,
                              full_label=full_label, label=full_label.split('/')[-1], geo=0),
                    parts_sdfs_pred,
                    pred_output,
                    shape_sdf_pred,
                    shape_output)

    def reconstruct_children_from_latents(self, child_nodes, child_feats, splitted_sdf_points, child_names, mode):
        parts_sdfs_pred = {}
        children_embeddings = []
        children_names = []
        children_sdf_points = []
        children_feats = []
        # for i in range(len(child_feats)):
        for i, child_name in enumerate(child_names):
            # child_name = child_node.label
            parts_sdfs_pred[child_name] = {}
            # parts_sdfs_pred[i] = {}
            sdf_part = splitted_sdf_points[i]
            children_names += [child_name]
            # children_names += [i]
            children_sdf_points += [sdf_part]
            children_feats += [child_feats[i].detach().clone()]
            self.deepsdf_decoder.eval()
            latent = child_feats[i].detach().clone()
            children_embeddings += [latent.detach().clone()]
            with torch.no_grad():
                sdf_values, voxel_origin, voxel_size, offset, scale, samples = create_grid(
                    self.deepsdf_decoder,
                    latent,
                    N=256,
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

                # parts_sdfs_pred[i]['sdf'] = sdf_values
                # parts_sdfs_pred[i]['vox_origin'] = voxel_origin
                # parts_sdfs_pred[i]['vox_size'] = voxel_size
                # parts_sdfs_pred[i]['offset'] = offset
                # parts_sdfs_pred[i]['scale'] = scale

            print('Part', child_name, 'is processed')
            # print('Part', i, 'is processed')

        return parts_sdfs_pred, children_embeddings, children_names, children_sdf_points, children_feats

    def reconstruct_full_shape_from_latents(self, shape_feat, sdf_data, mode):
        sdf_data = sdf_data.cuda()

        shape_sdf_pred = {}
        self.deepsdf_decoder.eval()
        latent = shape_feat.detach().clone()
        with torch.no_grad():
            sdf_values, voxel_origin, voxel_size, offset, scale, samples = create_grid(
                self.deepsdf_shape_decoder,
                latent,
                N=256,
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
                    N=256,
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
                N=256,
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
                        N=256,
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
