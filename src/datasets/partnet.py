import os, sys
import pickle
import json
from collections import namedtuple
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from pyquaternion import Quaternion

from ..data_utils.hierarchy import Tree
from deep_sdf_utils.data import unpack_sdf_samples, get_noise_points
from ..utils.transformations import from_tqs_to_matrix

class VoxelPartnetAllShapesDataset(Dataset):

    def __init__(self, datadir, dataset, partnet_to_dirs_path, object_list):

        self.datadir = datadir
        self.dataset = dataset
        self.partnet_to_dirs_path = os.path.join(datadir, dataset, partnet_to_dirs_path)

        with open(self.partnet_to_dirs_path, 'rb') as f:
            self.partnet_to_dirs = pickle.load(f)

        if isinstance(object_list, str):
            with open(os.path.join(self.datadir, self.dataset, object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
        else:
            self.object_names = object_list

        self.class2id = {'chair': 0,
                         'table': 1,
                         'storagefurniture': 2,
                         'bed': 3,
                         'trashcan': 4}

    def __getitem__(self, index):
        partnet_id = self.object_names[index]
        common_path = self.partnet_to_dirs[partnet_id]
        class_id = self.class2id[common_path]

        geo_fn = os.path.join(self.datadir, common_path + '_geo', f'{partnet_id}_full.npy')
        shape_mask = torch.FloatTensor(np.load(geo_fn))

        output = (shape_mask, class_id, partnet_id)

        return output

    def __len__(self):
        return len(self.object_names)


def generate_partnet_allshapes_datasets(datadir=None, dataset=None, partnet_to_dirs_path=None,
                                        train_samples='train.txt', val_samples='val.txt'):

    Dataset = VoxelPartnetAllShapesDataset

    train_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, train_samples)
    val_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, val_samples)

    return {
        'train': train_dataset,
        'val': val_dataset
    }


class VoxelisedScanNetAllShapesGNNDataset(Dataset):

    def __init__(self, datadir, dataset, partnet_to_dirs_path, object_list, data_features, load_geo=False,
                 shapenet_voxelized_path=None, latent_constraint_path=None):
        self.datadir = datadir
        self.dataset = dataset
        self.data_features = data_features
        self.load_geo = load_geo
        self.shapenet_voxelized_path = shapenet_voxelized_path
        # self.partnet_to_dirs_path = os.path.join(datadir, dataset, partnet_to_dirs_path)
        self.latent_constraint_path = latent_constraint_path
        self.latents = None

        # with open(self.partnet_to_dirs_path, 'rb') as f:
        #     self.partnet_to_dirs = pickle.load(f)

        print('Load dataset from:', os.path.join(self.datadir, self.dataset, object_list))
        if isinstance(object_list, str):
            with open(os.path.join(self.datadir, self.dataset, object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
        else:
            self.object_names = object_list
        print('Number of samples:', len(self.object_names))

    def __getitem__(self, index):
        partnet_scannet_id = self.object_names[index]
        tokens = partnet_scannet_id.split('_')
        partnet_id = tokens[0]
        common_path = self.partnet_to_dirs[partnet_id].split('/')[-1]
        if 'object' in self.data_features:
            geo_fn = os.path.join(self.datadir, common_path + '_geo', f'{partnet_id}.npy')
            obj = self.load_object(os.path.join(self.datadir, common_path + '_hier', partnet_id + '.json'),
                                   load_geo=self.load_geo, geo_fn=geo_fn)

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'name':
                data_feats = data_feats + (partnet_id,)
            else:
                assert False, 'ERROR: unknown feat type %s!' % feat

        voxel_path = os.path.join(self.shapenet_voxelized_path, partnet_id, 'full_vox.colored.pkl')
        # shape_sdf = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geoscan', f'{partnet_scannet_id}_sdf.npy')))
        shape_sdf = 0
        shape_mask = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geo', f'{partnet_id}_full.npy')))
        # scannet_geo = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geoscan', f'{partnet_scannet_id}.npy')))
        scannet_geo = 0

        output = (scannet_geo, shape_sdf, shape_mask, data_feats, partnet_id, 0, tokens)

        return output

    def __len__(self):
        return len(self.object_names)

    @staticmethod
    def load_object(fn, load_geo=False, geo_fn=None):
        if load_geo:
            geo_data = np.load(geo_fn)

        with open(fn, 'r') as f:
            root_json = json.load(f)

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node_json', 'parent', 'parent_child_idx'])
        stack = [StackElement(node_json=root_json, parent=None, parent_child_idx=None)]

        root = None
        # traverse the tree, converting each node json to a Node instance
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent = stack_elm.parent
            parent_child_idx = stack_elm.parent_child_idx
            node_json = stack_elm.node_json

            if 'complete_dfs_id' in node_json:
                node = Tree.Node(
                    part_id=node_json['dfs_id'],
                    complete_part_id=node_json['complete_dfs_id'],
                    is_leaf=('children' not in node_json),
                    label=node_json['name'])
            else:
                node = Tree.Node(
                    part_id=node_json['dfs_id'],
                    is_leaf=('children' not in node_json),
                    label=node_json['name'])

            if 'geo' in node_json.keys():
                node.geo = torch.tensor(np.array(node_json['geo']), dtype=torch.float32)

            if load_geo:
                node.geo = torch.tensor(geo_data[node_json['dfs_id']], dtype=torch.float32)

            if 'children' in node_json:
                for ci, child in enumerate(node_json['children']):
                    stack.append(StackElement(node_json=node_json['children'][ci], parent=node, parent_child_idx=ci))

            if 'edges' in node_json:
                for edge in node_json['edges']:
                    if 'params' in edge:
                        edge['params'] = torch.from_numpy(np.array(edge['params'])).to(dtype=torch.float32)
                    node.edges.append(edge)

            if parent is None:
                root = node
                root.full_label = root.label
            else:
                if len(parent.children) <= parent_child_idx:
                    parent.children.extend([None] * (parent_child_idx + 1 - len(parent.children)))
                parent.children[parent_child_idx] = node
                node.full_label = parent.full_label + '/' + node.label

        obj = Tree(root=root)

        return obj


def generate_scannet_allshapes_datasets(datadir=None, dataset=None, partnet_to_dirs_path=None,
                              train_samples='train.txt', val_samples='val.txt',
                              data_features=('object',), load_geo=True,
                              shapenet_voxelized_path=None,
                              latent_constraint_path=None,
                              **kwargs):
    if isinstance(data_features, str):
        data_features = [data_features]

    Dataset = VoxelisedScanNetAllShapesGNNDataset

    train_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, train_samples, data_features, load_geo,
                            shapenet_voxelized_path=shapenet_voxelized_path, latent_constraint_path=latent_constraint_path)
    val_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, val_samples, data_features, load_geo,
                          shapenet_voxelized_path=shapenet_voxelized_path, latent_constraint_path=latent_constraint_path)

    return {
        'train': train_dataset,
        'val': val_dataset
    }


class VoxelisedScanNetAllShapesRotGNNDataset(VoxelisedScanNetAllShapesGNNDataset):

    def __init__(self, datadir, dataset, partnet_to_dirs_path, object_list, data_features, load_geo=False,
                 shapenet_voxelized_path=None):
        self.datadir = datadir
        self.dataset = dataset
        self.data_features = data_features
        self.load_geo = load_geo
        self.shapenet_voxelized_path = shapenet_voxelized_path
        self.partnet_to_dirs_path = os.path.join(datadir, dataset, partnet_to_dirs_path)

        with open(self.partnet_to_dirs_path, 'rb') as f:
            self.partnet_to_dirs = pickle.load(f)

        if isinstance(object_list, str):
            with open(os.path.join(self.datadir, self.dataset, object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
        else:
            self.object_names = object_list

    def __getitem__(self, index):
        partnet_scannet_id = self.object_names[index]
        tokens = partnet_scannet_id.split('_')
        partnet_id = tokens[0]
        rotation = tokens[4]
        common_path = self.partnet_to_dirs[partnet_id].split('/')[-1]
        if 'object' in self.data_features:
            geo_fn = os.path.join(self.datadir, common_path + '_geo_8rot', partnet_id + f'_{rotation}.npy')
            obj = self.load_object(os.path.join(self.datadir, common_path + '_hier', partnet_id + '.json'),
                                   load_geo=self.load_geo, geo_fn=geo_fn)

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'name':
                data_feats = data_feats + (partnet_id,)
            else:
                assert False, 'ERROR: unknown feat type %s!' % feat

        voxel_path = os.path.join(self.shapenet_voxelized_path, partnet_id, 'full_vox.colored.pkl')
        shape_sdf = torch.zeros((32, 32, 32))
        shape_mask = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geo_8rot', f'{partnet_id}_full_{rotation}.npy')))
        scannet_geo = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_scannet_geo_8rot', f'{partnet_scannet_id}.npy')))

        output = (scannet_geo, shape_sdf, shape_mask, data_feats, partnet_id, int(rotation), tokens)

        return output


def generate_scannet_allshapes_rot_datasets(datadir=None, dataset=None, partnet_to_dirs_path=None,
                              train_samples='train.txt', val_samples='val.txt',
                              data_features=('object',), load_geo=True,
                              shapenet_voxelized_path=None,
                              **kwargs):
    if isinstance(data_features, str):
        data_features = [data_features]

    Dataset = VoxelisedScanNetAllShapesRotGNNDataset

    train_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, train_samples, data_features, load_geo,
                            shapenet_voxelized_path=shapenet_voxelized_path)
    val_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, val_samples, data_features, load_geo,
                          shapenet_voxelized_path=shapenet_voxelized_path)

    return {
        'train': train_dataset,
        'val': val_dataset
    }


class VoxelisedScanNetAllShapesGNNDatasetContrastive(VoxelisedScanNetAllShapesGNNDataset):

    def __getitem__(self, index):
        partnet_scannet_id = self.object_names[index]
        tokens = partnet_scannet_id.split('_')
        partnet_id = tokens[0]
        common_path = self.partnet_to_dirs[partnet_id].split('/')[-1]
        if 'object' in self.data_features:
            geo_fn = os.path.join(self.datadir, common_path + '_geo', f'{partnet_id}.npy')
            obj = self.load_object(os.path.join(self.datadir, common_path+'_hier', partnet_id + '.json'),
                                   load_geo=self.load_geo, geo_fn=geo_fn)

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'name':
                data_feats = data_feats + (partnet_id,)
            else:
                assert False, 'ERROR: unknown feat type %s!' % feat

        voxel_path = os.path.join(self.shapenet_voxelized_path, partnet_id, 'full_vox.colored.pkl')
        # shape_sdf = torch.zeros((32, 32, 32))
        shape_sdf = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geoscan', f'{partnet_scannet_id}_sdf.npy')))
        shape_mask = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geo', f'{partnet_id}_full.npy')))
        scannet_geo = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geoscan', f'{partnet_scannet_id}.npy')))

        pos_idxes = [i for i, x in enumerate(self.object_names) if x.split('_')[0] == partnet_id]

        pos_idx = np.random.choice(pos_idxes)
        partnet_scannet_id_pos = self.object_names[pos_idx]
        partnet_id_pos = self.object_names[pos_idx].split("_")[0]

        scannet_geo_pos = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geoscan', f'{partnet_scannet_id_pos}.npy')))
        # shape_sdf_pos = torch.zeros((32, 32, 32))
        shape_sdf_pos = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geoscan', f'{partnet_scannet_id_pos}_sdf.npy')))
        shape_mask_pos = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geo', f'{partnet_id_pos}_full.npy')))
        if 'object' in self.data_features:
            geo_fn = os.path.join(self.datadir, common_path + '_geo', f'{partnet_id_pos}.npy')
            obj = self.load_object(os.path.join(self.datadir, common_path+'_hier', partnet_id_pos + '.json'),
                                   load_geo=self.load_geo, geo_fn=geo_fn)
        data_feats_pos = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats_pos = data_feats_pos + (obj,)
            elif feat == 'name':
                data_feats_pos = data_feats_pos + (partnet_id,)
            else:
                assert False, 'ERROR: unknown feat type %s!' % feat

        if self.latents:
            latent = torch.FloatTensor(self.latents[partnet_id])[None, ...]
        else:
            latent = 0

        output = (scannet_geo, shape_sdf, shape_mask, data_feats, partnet_id, 0, tokens, latent,
                  scannet_geo_pos, shape_sdf_pos, shape_mask_pos, data_feats_pos)

        return output


def generate_scannet_allshapes_contrastive_datasets(datadir=None, dataset=None, partnet_to_dirs_path=None,
                              train_samples='train.txt', val_samples='val.txt',
                              data_features=('object',), load_geo=True,
                              shapenet_voxelized_path=None, latent_constraint_path=None,
                              **kwargs):
    if isinstance(data_features, str):
        data_features = [data_features]

    Dataset = VoxelisedScanNetAllShapesGNNDatasetContrastive

    train_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, train_samples, data_features, load_geo,
                            shapenet_voxelized_path=shapenet_voxelized_path, latent_constraint_path=latent_constraint_path)
    val_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, val_samples, data_features, load_geo,
                          shapenet_voxelized_path=shapenet_voxelized_path, latent_constraint_path=latent_constraint_path)

    return {
        'train': train_dataset,
        'val': val_dataset
    }


class VoxelisedImplicitAllShapesDataset(VoxelisedScanNetAllShapesGNNDataset):

    def __init__(self, datadir, dataset, partnet_to_dirs_path, object_list, data_features, load_geo=False,
                 shapenet_voxelized_path=None, num_subsample_points=0, sdf_data_source=None,
                 parts_to_shapes_path=None, shapes_to_cat_path=None, partnet_to_parts_path=None):
        super(VoxelisedImplicitAllShapesDataset, self).__init__(datadir, dataset, partnet_to_dirs_path,
                                                                object_list, data_features, load_geo,
                                                                shapenet_voxelized_path)

        self.num_subsample_points = num_subsample_points
        self.sdf_data_source = sdf_data_source

        # parts_to_shapes_path = '/home/bohovkin/cluster/abokhovkin_home/projects/scannet-relationships/dicts/parts_to_shapes.json'
        # shapes_to_cat_path = '/home/bohovkin/cluster/abokhovkin_home/projects/scannet-relationships/dicts/obj_to_cat.json'
        # partnet_to_parts_path = '/home/bohovkin/cluster/abokhovkin_home/projects/scannet-relationships/dicts/partnet_to_parts.json'

        # load metadata mappings
        with open(parts_to_shapes_path, 'rb') as fin:
            self.parts_to_shapes = json.load(fin)
        self.shapes_to_parts = {self.parts_to_shapes[k]: k for k in self.parts_to_shapes}
        with open(shapes_to_cat_path, 'rb') as fin:
            self.shapes_to_cats = json.load(fin)
        with open(partnet_to_parts_path, 'rb') as fin:
            self.partnet_to_parts = json.load(fin)

        self.all_partnet_ids = sorted(list(self.partnet_to_parts.keys()))

        self.parts_to_indices = {}
        idx = 0
        for object_name in self.object_names:
            cur_partnet_id = object_name.split('_')[0]
            cur_parts = self.partnet_to_parts[cur_partnet_id]
            if cur_partnet_id not in self.parts_to_indices:
                self.parts_to_indices[cur_partnet_id] = {}
                for cur_part in cur_parts:
                    self.parts_to_indices[cur_partnet_id][cur_part] = idx
                    idx += 1
        self.num_parts = idx
        self.num_shapes = len(self.all_partnet_ids)

    def __getitem__(self, index):
        partnet_scannet_id = self.object_names[index]
        tokens = partnet_scannet_id.split('_')
        partnet_id = tokens[0]
        rotation = 0
        common_path = self.partnet_to_dirs[partnet_id].split('/')[-1]
        if 'object' in self.data_features:
            geo_fn = os.path.join(self.datadir, common_path + '_geo', partnet_id + '.npy')
            obj = self.load_object(os.path.join(self.datadir, common_path + '_hier', partnet_id + '.json'),
                                   load_geo=self.load_geo, geo_fn=geo_fn)

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'name':
                data_feats = data_feats + (partnet_id,)
            else:
                assert False, 'ERROR: unknown feat type %s!' % feat

        shape_sdf = torch.zeros((32, 32, 32))
        shape_mask = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geo', f'{partnet_id}_full.npy')))
        scannet_geo = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geo', f'{partnet_id}_full.npy')))[0]

        child_names = []
        if 'object' in self.data_features:
            for child in obj.root.children:
                child_names += [child.label]
        obj_id = self.parts_to_shapes[partnet_id]
        cat_id = self.shapes_to_cats[obj_id]

        sdf_filenames = [os.path.join(self.sdf_data_source, f'{cat_id}-{child_names[i]}', f'{obj_id}.npz') for i in range(len(child_names))]
        sdf_parts = torch.cat([torch.FloatTensor(unpack_sdf_samples(filename, self.num_subsample_points))[None, ...] for filename in sdf_filenames], dim=0)
        parts_indices = {x: self.parts_to_indices[partnet_id][x] for x in child_names}

        full_shape_idx = self.all_partnet_ids.index(partnet_id)

        output = (scannet_geo, shape_sdf, shape_mask, data_feats, partnet_id, int(rotation), tokens,
                  sdf_filenames, sdf_parts, parts_indices, full_shape_idx)

        return output


def generate_gnn_deepsdf_datasets(datadir=None, dataset=None, partnet_to_dirs_path=None,
                                  train_samples='train.txt', val_samples='train.txt',
                                  data_features=('object',), load_geo=True,
                                  shapenet_voxelized_path=None, num_subsample_points=0,
                                  sdf_data_source=None, parts_to_shapes_path=None,
                                  shapes_to_cat_path=None, partnet_to_parts_path=None,
                                  **kwargs):
    if isinstance(data_features, str):
        data_features = [data_features]

    Dataset = VoxelisedImplicitAllShapesDataset

    train_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, train_samples, data_features, load_geo,
                            shapenet_voxelized_path, num_subsample_points,
                            sdf_data_source, parts_to_shapes_path,
                            shapes_to_cat_path, partnet_to_parts_path)
    val_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, val_samples, data_features, load_geo,
                          shapenet_voxelized_path, num_subsample_points,
                          sdf_data_source, parts_to_shapes_path,
                          shapes_to_cat_path, partnet_to_parts_path)

    return {
        'train': train_dataset,
        'val': val_dataset
    }


class VoxelisedImplicitScanNetDataset(VoxelisedScanNetAllShapesGNNDataset):

    def __init__(self, datadir, dataset, partnet_to_dirs_path, object_list, data_features, load_geo=False,
                 shapenet_voxelized_path=None, num_subsample_points=0, sdf_data_source=None,
                 parts_to_shapes_path=None, shapes_to_cat_path=None, partnet_to_parts_path=None,
                 cat_name=None, eval_mode=False):
        super(VoxelisedImplicitScanNetDataset, self).__init__(datadir, dataset, partnet_to_dirs_path,
                                                              object_list, data_features, load_geo,
                                                              shapenet_voxelized_path)

        self.num_subsample_points = num_subsample_points
        self.sdf_data_source = sdf_data_source
        self.mode = object_list.split('.')[0]
        self.cat_name = cat_name
        self.eval_mode = eval_mode

        print('Cat name:', self.cat_name)

        self.drop_parts = {}
        self.drop_parts['chair'] = ['other']
        self.drop_parts['table'] = []
        self.drop_parts['bed'] = []
        self.drop_parts['trashcan'] = []
        self.drop_parts['storagefurniture'] = ['object']

        self.replace_parts = {}
        self.replace_parts['chair'] = {}
        self.replace_parts['table'] = {'bar_stretcher': 'leg',
                                       'bottom_panel': 'shelf'}
        self.replace_parts['bed'] = {'other': 'bed_frame_base',
                                     'bed_frame_horizontal_surface': 'bed_frame_base'}
        self.replace_parts['trashcan'] = {'container': 'container_box',
                                          'outside_frame': 'container_box'}
        self.replace_parts['storagefurniture'] = {
            'drawer': 'cabinet_door',
            'other': 'cabinet_frame',
            'chest_box': 'cabinet_frame',
            'back_panel': 'cabinet_frame',
            'bottom_panel': 'cabinet_frame',
            'top_panel': 'cabinet_frame',
            'frame_vertical_bar': 'cabinet_frame',
            'frame_horizontal_bar': 'cabinet_frame',
            'vertical_front_panel': 'cabinet_frame',
            'vertical_divider_panel': 'cabinet_frame'
        }

        # parts_to_shapes_path = '/home/bohovkin/cluster/abokhovkin_home/projects/scannet-relationships/dicts/parts_to_shapes.json'
        # shapes_to_cat_path = '/home/bohovkin/cluster/abokhovkin_home/projects/scannet-relationships/dicts/obj_to_cat.json'
        # partnet_to_parts_path = '/home/bohovkin/cluster/abokhovkin_home/projects/scannet-relationships/dicts/partnet_to_parts.json'

        # load metadata mappings
        with open(parts_to_shapes_path, 'rb') as fin:
            self.parts_to_shapes = json.load(fin)
        self.shapes_to_parts = {self.parts_to_shapes[k]: k for k in self.parts_to_shapes}
        with open(shapes_to_cat_path, 'rb') as fin:
            self.shapes_to_cats = json.load(fin)
        with open(partnet_to_parts_path, 'rb') as fin:
            self.partnet_to_parts = json.load(fin)

        self.existing_partnet_ids = []
        self.parts_to_indices = {}
        idx = 0
        for object_name in self.object_names:
            cur_partnet_id = object_name.split('_')[0]
            self.existing_partnet_ids += [cur_partnet_id]
            cur_parts = self.partnet_to_parts[cur_partnet_id]
            if cur_partnet_id not in self.parts_to_indices:
                self.parts_to_indices[cur_partnet_id] = {}
                for cur_part in cur_parts:
                    self.parts_to_indices[cur_partnet_id][cur_part] = idx
                    idx += 1
        self.num_parts = idx
        self.existing_partnet_ids = sorted(list(set(self.existing_partnet_ids)))
        self.num_shapes = len(self.existing_partnet_ids)

        print('Num all shapes:', self.num_shapes)
        print('Num all parts:', self.num_parts)

        # try:
        #     with open(f'/rhome/abokhovkin/projects/DeepSDF/experiments/full_experiments/{self.cat_name}_full/sv2_{self.cat_name}_train.json', 'r') as f:
        #         chair_full_list = json.load(f)
        # except:
        #     with open(f'/home/bohovkin/cluster/abokhovkin_home/projects/DeepSDF/experiments/full_experiments/{self.cat_name}_full/sv2_{self.cat_name}_train.json', 'r') as f:
        #         chair_full_list = json.load(f)

        with open(f'/cluster/daidalos/abokhovkin/DeepSDF_v2/full_experiments_v2/{self.cat_name}_full_surface_pe/sv2_{self.cat_name}_train.json', 'r') as f:
            chair_full_list = json.load(f)

        cat_name_to_part_list_cat_id = {
            'chair': '03001627',
            'table': '04379243',
            'storagefurniture': '02871439',
            'bed': '02818832',
            'trashcan': '02747177'
        }

        i = 0
        self.chair_full_map = {}
        for idx in chair_full_list['ShapeNetV2'][cat_name_to_part_list_cat_id[self.cat_name]]:
            self.chair_full_map[idx] = i
            i += 1
        i = 0

        # try:
        #     with open(f'/rhome/abokhovkin/projects/DeepSDF/experiments/full_experiments/{self.cat_name}_parts_onehot/sv2_{self.cat_name}_train.json', 'r') as f:
        #         chair_parts_list = json.load(f)
        # except:
        #     with open(f'/home/bohovkin/cluster/abokhovkin_home/projects/DeepSDF/experiments/full_experiments/{self.cat_name}_parts_onehot/sv2_{self.cat_name}_train.json', 'r') as f:
        #         chair_parts_list = json.load(f)

        with open(f'/cluster/daidalos/abokhovkin/DeepSDF_v2/full_experiments_v2/{self.cat_name}_parts_onehot_pe/sv2_{self.cat_name}_train.json', 'r') as f:
            chair_parts_list = json.load(f)

        self.chair_parts_map = {}
        for part_id in chair_parts_list['ShapeNetV2']:
            part_name = part_id.split('-')[1]
            if part_name not in self.chair_parts_map:
                self.chair_parts_map[part_name] = {}
            for idx in chair_parts_list['ShapeNetV2'][part_id]:
                self.chair_parts_map[part_name][idx] = i
                i += 1

        # mlcvnet boxes to s2c correspondences
        # self.shape_to_corr = {}
        # for shape_name in ['chair', 'table', 'storagefurniture', 'bed', 'trashcan']:
        #     CORRESPONDENCES = f'/cluster_HDD/sorona/abokhovkin/ScanNetGrids/mlcvnet_corr_{shape_name}.pkl'
        #     with open(CORRESPONDENCES, 'rb') as fin:
        #         self.shape_to_corr[shape_name] = pickle.load(fin)

        SCAN2CAD_DIR = '/canis/Datasets/Scan2CAD/public'
        with open(os.path.join(SCAN2CAD_DIR, 'full_annotations.json'), 'rb') as fin:
            anno = json.load(fin)
        self.scan2cad_anno = {}
        for item in anno:
            self.scan2cad_anno[item['id_scan']] = item

        if self.cat_name != 'storagefurniture':
            MLCVNET_NOISE_DIR = f'/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_parts_mlcvnet/train/SdfSamples/ShapeNetV2/{cat_name_to_part_list_cat_id[self.cat_name]}-background'
        else:
            MLCVNET_NOISE_DIR = f'/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_parts_mlcvnet/train/SdfSamples/ShapeNetV2/02933112-background'
        self.noise_partnetid_to_filename = {}
        for filename in os.listdir(MLCVNET_NOISE_DIR):
            if filename.endswith('.npz'):
                tokens = filename.split('_')
                partnet_id = tokens[3]
                if partnet_id not in self.noise_partnetid_to_filename:
                    self.noise_partnetid_to_filename[partnet_id] = []
                self.noise_partnetid_to_filename[partnet_id] += [filename]

    def __getitem__(self, index):

        t0 = time.time()

        partnet_scannet_id = self.object_names[index]
        tokens = partnet_scannet_id.split('_')
        partnet_id = tokens[0]
        instance_id = tokens[-1]
        scan_id = '_'.join(tokens[1:3])
        rotation = 0
        common_path = self.cat_name
        if 'object' in self.data_features:
            geo_fn = os.path.join(self.datadir, common_path + '_geo', partnet_id + '.npy')
            obj = self.load_object(os.path.join(self.datadir, common_path + '_hier', partnet_id + '.json'),
                                   load_geo=self.load_geo, geo_fn=geo_fn)

            collected_parts = []
            new_children = []
            for child in obj.root.children:
                if child.label not in self.drop_parts[self.cat_name]:
                    if child.label in self.replace_parts[self.cat_name]:
                        child.label = self.replace_parts[self.cat_name][child.label]
                        child.full_label = child.full_label.split('/')[0] + '/' + child.label
                    if child.label not in collected_parts:
                        new_children += [child]
                        collected_parts += [child.label]
                else:
                    # print('Dropped', child.label)
                    pass
            obj.root.children = new_children

        t1 = time.time()

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'name':
                data_feats = data_feats + (partnet_id,)
            else:
                assert False, 'ERROR: unknown feat type %s!' % feat

        shape_sdf = torch.zeros((32, 32, 32))
        # mlcvnet
        shape_mask = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_mlcvnet_canonical', self.mode, f'{partnet_scannet_id}.npy')))
        scannet_geo = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_mlcvnet_canonical', self.mode, f'{partnet_scannet_id}.npy')))
        # partnet partial
        # shape_mask = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_partnetpartial', f'{partnet_scannet_id}.npy')))
        # scannet_geo = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_partnetpartial', f'{partnet_scannet_id}.npy')))

        t2 = time.time()

        child_names = []
        if 'object' in self.data_features:
            for child in obj.root.children:
                if child.label not in self.drop_parts[self.cat_name]:
                    child_names += [child.label]
        obj_id = self.parts_to_shapes[partnet_id]
        cat_id = self.shapes_to_cats[obj_id]
        if self.cat_name == 'storagefurniture':
            cat_id = '02933112'

        # mlcvnet
        if self.eval_mode:
            sdf_filenames = [os.path.join(self.sdf_data_source, 'val', 'SdfSamples/ShapeNetV2', f'{cat_id}-{child_names[i]}', f'{scan_id}_{obj_id}_{partnet_id}_{instance_id}.npz') for i in range(len(child_names))]
            sdf_parts = torch.cat([torch.FloatTensor(unpack_sdf_samples(filename, self.num_subsample_points)[0])[None, ...] for filename in sdf_filenames], dim=0)
        else:
            # partnet / mlcvnet val
            if self.mode in ['train', 'val']:
                if self.mode == 'train':
                    SDF_DATA_SOURCE = '/cluster/daidalos/abokhovkin/DeepSDF_v2/ShapeNetV2_dim256_partial_uniform'
                    rand_instance_id = np.random.randint(0, 10)
                    sdf_filenames = [os.path.join(SDF_DATA_SOURCE, 'SdfSamples/ShapeNetV2', f'{cat_id}-{child_names[i]}', f'{obj_id}_{partnet_id}_{rand_instance_id}.npz') for i in range(len(child_names))]
                    sdf_parts = torch.cat([torch.FloatTensor(unpack_sdf_samples(filename, 6192)[0])[None, ...] for filename in sdf_filenames], dim=0)
                else:
                    sdf_filenames = [os.path.join(self.sdf_data_source, self.mode, 'SdfSamples/ShapeNetV2', f'{cat_id}-{child_names[i]}', f'{scan_id}_{obj_id}_{partnet_id}_{instance_id}.npz') for i in range(len(child_names))]
                    sdf_parts = torch.cat([torch.FloatTensor(unpack_sdf_samples(filename, 6192)[0])[None, ...] for filename in sdf_filenames], dim=0) # self.num_subsample_points
            else:
            # mlcvnet test
                sdf_filenames = [os.path.join(self.sdf_data_source, 'test', 'SdfSamples/ShapeNetV2', f'{cat_id}', f'{scan_id}_{obj_id}_{partnet_id}_{instance_id}.npz')]
                sdf_parts = torch.cat([torch.FloatTensor(unpack_sdf_samples(filename, 150000)[0])[None, ...] for filename in sdf_filenames], dim=0) # 350000 points

        # partnet partial
        # sdf_filenames = [os.path.join(self.sdf_data_source, 'SdfSamples/ShapeNetV2', f'{cat_id}-{child_names[i]}', f'{obj_id}_{partnet_id}_{instance_id}.npz') for i in range(len(child_names))]
        # sdf_parts = torch.cat([torch.FloatTensor(unpack_sdf_samples(filename, self.num_subsample_points)[0])[None, ...] for filename in sdf_filenames], dim=0)

        t3 = time.time()

        # NOISE SECTION #

        # partnet partial
        # NOISE_DIR = f'/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_partial_uniform/SdfSamples/ShapeNetV2/{cat_id}-{child_names[0]}'
        # noise_full = np.load(os.path.join(NOISE_DIR, f'{obj_id}_{partnet_id}_{instance_id}.npz'))
        # noise_full = noise_full['random_full']
        # noise_full = np.hstack([noise_full, 0.07 * np.ones((len(noise_full), 1))])
        # indices = np.random.choice(len(noise_full), min(len(noise_full), 1024))
        # noise_full = noise_full[indices]
        # noise_full = torch.FloatTensor(noise_full)

        # partnet partial
        # try:
        #     NOISE_DIR = f'/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_partial_uniform_pickle/SdfSamples/ShapeNetV2/{cat_id}-{child_names[0]}'
        #     noise_full = torch.load(os.path.join(NOISE_DIR, f'{obj_id}_{partnet_id}_{instance_id}.pth'))
        #     noise_full = noise_full['random_full']
        #     noise_full = torch.hstack([noise_full, 0.07 * torch.ones((len(noise_full), 1))])
        #     indices = np.random.choice(len(noise_full), min(len(noise_full), 1024))
        #     noise_full = noise_full[indices]
        # except:
        #     noise_full = torch.load('/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_partial_uniform_pickle/SdfSamples/ShapeNetV2/03001627-chair_seat/7fc8b858cad8f5849df6f10c48eb6cee_40124_0.pth')
        #     noise_full = noise_full['random_full']
        #     noise_full = torch.hstack([noise_full, 0.07 * torch.ones((len(noise_full), 1))])
        #     indices = np.random.choice(len(noise_full), min(len(noise_full), 1024))
        #     noise_full = noise_full[indices]

        # mlcvnet
        if self.mode in ['train', 'val']:
            noise_mode = self.mode
        else:
            noise_mode = 'val'
        NOISE_DIR = f'/cluster/daidalos/abokhovkin/DeepSDF_v2/ShapeNetV2_dim256_parts_mlcvnet/{noise_mode}/SdfSamples/ShapeNetV2/{cat_id}-background'
        noise_full = np.load(os.path.join(NOISE_DIR, f'{scan_id}_{obj_id}_{partnet_id}_{instance_id}.npz'))
        num_noise_points_pos = len(noise_full['pos'])  # 256 for train
        num_noise_points_neg = len(noise_full['neg'])  # 256 for train
        indices = np.random.choice(len(noise_full['pos']), min(3096, len(noise_full['pos'])), replace=False)
        noise_full_pos = torch.FloatTensor(noise_full['pos'][indices])
        indices = np.random.choice(len(noise_full['neg']), min(3096, len(noise_full['neg'])), replace=False)
        noise_full_neg = torch.FloatTensor(noise_full['neg'][indices])
        noise_full = torch.cat([noise_full_pos, noise_full_neg], 0)

        # partnet partial (as mlcvnet)
        # MLCVNET_NOISE_DIR = f'/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_parts_mlcvnet/train/SdfSamples/ShapeNetV2/{cat_id}-background'
        # if partnet_id in self.noise_partnetid_to_filename:
        #     noise_filenames = self.noise_partnetid_to_filename[partnet_id]
        #     random_index = np.random.choice(len(noise_filenames), 1)[0]
        #     noise_filename = noise_filenames[random_index]
        #     try:
        #         noise_full = np.load(os.path.join(MLCVNET_NOISE_DIR, noise_filename))
        #     except ValueError:
        #         print('Failed noise', os.path.join(MLCVNET_NOISE_DIR, noise_filename))
        #         raise ValueError
        #     indices = np.random.choice(len(noise_full['pos']),
        #                                min(2048 // 2, len(noise_full['pos'])), replace=False)
        #     noise_full_pos = torch.FloatTensor(noise_full['pos'][indices])
        #     indices = np.random.choice(len(noise_full['neg']),
        #                                min(2048 // 2, len(noise_full['neg'])), replace=False)
        #     noise_full_neg = torch.FloatTensor(noise_full['neg'][indices])
        #     noise_full = torch.cat([noise_full_pos, noise_full_neg], 0)
        # else:
        #     try:
        #         NOISE_DIR = f'/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_partial_uniform_pickle/SdfSamples/ShapeNetV2/{cat_id}-{child_names[0]}'
        #         noise_full = torch.load(os.path.join(NOISE_DIR, f'{obj_id}_{partnet_id}_{instance_id}.pth'))
        #         noise_full = noise_full['random_full']
        #         noise_full = torch.hstack([noise_full, 0.07 * torch.ones((len(noise_full), 1))])
        #         indices = np.random.choice(len(noise_full), min(len(noise_full), 2048))
        #         noise_full = noise_full[indices]
        #     except:
        #         noise_full = torch.load('/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_partial_uniform_pickle/SdfSamples/ShapeNetV2/03001627-chair_seat/7fc8b858cad8f5849df6f10c48eb6cee_40124_0.pth')
        #         noise_full = noise_full['random_full']
        #         noise_full = torch.hstack([noise_full, 0.07 * torch.ones((len(noise_full), 1))])
        #         indices = np.random.choice(len(noise_full), min(len(noise_full), 2048))
        #         noise_full = noise_full[indices]

        if self.mode in ['train', 'val']:
            parts_indices = {x: self.chair_parts_map[x][obj_id] for x in child_names}
            full_shape_idx = self.chair_full_map[obj_id]
        else:
            parts_indices = {x: 0 for x in child_names}
            full_shape_idx = 0

        # find the rotation angle from correspondences (val)
        # scene_correspondences = self.shape_to_corr['chair'][scan_id]
        # flag_found = False
        # s2c_scene_annotation = self.scan2cad_anno[scan_id]
        # for corr in scene_correspondences:
        #     print('Corr:', corr[0], corr[1], instance_id)
        #     if corr[0] == int(instance_id):
        #         aligned_shape_data = s2c_scene_annotation['aligned_models'][corr[0]]
        #         t = aligned_shape_data['trs']['translation']
        #         r = aligned_shape_data['trs']['rotation']
        #         s = aligned_shape_data['trs']['scale']
        #
        #         q = Quaternion(np.array(r))
        #         flag_found = True
        #         break
        # train / test
        q = Quaternion(np.array([0, 0, 0, 0]))

        output = (scannet_geo, shape_sdf, shape_mask, data_feats, partnet_id, int(rotation), tokens,
                  sdf_filenames, sdf_parts, parts_indices, full_shape_idx,
                  noise_full, index, q)

        t4 = time.time()
        # print('(0) Read tree:', t1 - t0)
        # print('(0) Read voxels:', t2 - t1)
        # print('(0) Read SDF:', t3 - t2)
        # print('(0) Read noise:', t4 - t3)
        # print()

        return output


def generate_gnn_deepsdf_scannet_datasets(datadir=None, dataset=None, partnet_to_dirs_path=None,
                                          train_samples='train.txt', val_samples='val.txt',
                                          data_features=('object',), load_geo=True,
                                          shapenet_voxelized_path=None, num_subsample_points=0,
                                          sdf_data_source=None, parts_to_shapes_path=None,
                                          shapes_to_cat_path=None, partnet_to_parts_path=None,
                                          cat_name='chair', eval_mode=False,
                                          **kwargs):
    if isinstance(data_features, str):
        data_features = [data_features]

    Dataset = VoxelisedImplicitScanNetDataset

    train_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, train_samples, data_features, load_geo,
                            shapenet_voxelized_path, num_subsample_points,
                            sdf_data_source, parts_to_shapes_path,
                            shapes_to_cat_path, partnet_to_parts_path,
                            cat_name, eval_mode)
    val_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, val_samples, data_features, load_geo,
                          shapenet_voxelized_path, num_subsample_points,
                          sdf_data_source, parts_to_shapes_path,
                          shapes_to_cat_path, partnet_to_parts_path,
                          cat_name, eval_mode)

    return {
        'train': train_dataset,
        'val': val_dataset
    }
