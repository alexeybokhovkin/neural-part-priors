import os
import pickle
import json
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data_utils.hierarchy import Tree

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
        shape_sdf = torch.zeros((32, 32, 32))
        shape_mask = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geo', f'{partnet_id}_full.npy')))
        scannet_geo = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_scannet_geo', f'{partnet_scannet_id}.npy')))

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
                              **kwargs):
    if isinstance(data_features, str):
        data_features = [data_features]

    Dataset = VoxelisedScanNetAllShapesGNNDataset

    train_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, train_samples, data_features, load_geo,
                            shapenet_voxelized_path=shapenet_voxelized_path)
    val_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, val_samples, data_features, load_geo,
                          shapenet_voxelized_path=shapenet_voxelized_path)

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
        common_path = self.partnet_to_dirs[partnet_id]
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
        shape_sdf = torch.zeros((32, 32, 32))
        shape_mask = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geo', f'{partnet_id}_full.npy')))
        scannet_geo = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_scannet_geo', f'{partnet_scannet_id}.npy')))

        pos_idxes = [i for i, x in enumerate(self.object_names) if x.split('_')[0] == partnet_id]

        pos_idx = np.random.choice(pos_idxes)
        partnet_scannet_id_pos = self.object_names[pos_idx]
        partnet_id_pos = self.object_names[pos_idx].split("_")[0]

        scannet_geo_pos = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_scannet_geo', f'{partnet_scannet_id_pos}.npy')))
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

        output = (scannet_geo, shape_sdf, shape_mask, data_feats, partnet_id, 0, tokens, 0,
                  scannet_geo_pos, shape_mask_pos, data_feats_pos)

        return output


def generate_scannet_allshapes_contrastive_datasets(datadir=None, dataset=None, partnet_to_dirs_path=None,
                              train_samples='train.txt', val_samples='val.txt',
                              data_features=('object',), load_geo=True,
                              shapenet_voxelized_path=None,
                              **kwargs):
    if isinstance(data_features, str):
        data_features = [data_features]

    Dataset = VoxelisedScanNetAllShapesGNNDatasetContrastive

    train_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, train_samples, data_features, load_geo,
                            shapenet_voxelized_path=shapenet_voxelized_path)
    val_dataset = Dataset(datadir, dataset, partnet_to_dirs_path, val_samples, data_features, load_geo,
                          shapenet_voxelized_path=shapenet_voxelized_path)

    return {
        'train': train_dataset,
        'val': val_dataset
    }