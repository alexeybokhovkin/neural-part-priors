import os, sys
import json
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.hierarchy import Tree


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)

    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
    noise_tensor = torch.from_numpy(npz["random_part"]).float()

    # split the sample into half
    half = int(0.92 * subsample / 2)
    noise_count = int(subsample - 2 * half)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
    random_noise = (torch.rand(noise_count) * noise_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    sample_noise = torch.index_select(noise_tensor, 0, random_noise)
    
    sample_noise = torch.cat([sample_noise, 
                                 0.1 * torch.ones(len(sample_noise), 1)], dim=1).float()

    samples = torch.cat([sample_pos, sample_neg, sample_noise], 0)

    return samples


class VoxelisedScanNetAllShapesGNNDataset(Dataset):

    def __init__(self, datadir, dataset, object_list, data_features, load_geo=False):
        self.datadir = datadir
        self.dataset = dataset
        self.data_features = data_features
        self.load_geo = load_geo
        self.latents = None

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

        shape_sdf = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geoscan', f'{partnet_scannet_id}_sdf.npy')))
        shape_mask = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geo', f'{partnet_id}_full.npy')))
        scannet_geo = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_geoscan', f'{partnet_scannet_id}.npy')))

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


def generate_scannet_allshapes_datasets(datadir=None, dataset=None,
                              train_samples='train.txt', val_samples='val.txt',
                              data_features=('object',), load_geo=True,
                              latent_constraint_path=None,
                              **kwargs):
    if isinstance(data_features, str):
        data_features = [data_features]

    Dataset = VoxelisedScanNetAllShapesGNNDataset

    train_dataset = Dataset(datadir, dataset, train_samples, data_features, load_geo)
    val_dataset = Dataset(datadir, dataset, val_samples, data_features, load_geo)

    return {
        'train': train_dataset,
        'val': val_dataset
    }


class VoxelisedImplicitScanNetDataset(VoxelisedScanNetAllShapesGNNDataset):

    def __init__(self, datadir, dataset, object_list, data_features, load_geo=False,
                 num_subsample_points=0, sdf_data_source=None,
                 cat_name=None, eval_mode=False, full_shape_list_path=None, parts_list_path=None,
                 mlcvnet_noise_path=None, data_mode=None, partnet_noise_dir=None, mlcvnet_noise_dir=None,
                 partnet_noise_dummy=None):
        super(VoxelisedImplicitScanNetDataset, self).__init__(datadir, dataset,
                                                              object_list, data_features, load_geo)

        self.num_subsample_points = num_subsample_points
        self.sdf_data_source = sdf_data_source
        self.mode = object_list.split('.')[0]
        self.cat_name = cat_name
        self.eval_mode = eval_mode
        self.data_mode = data_mode
        self.partnet_noise_dir = partnet_noise_dir
        self.mlcvnet_noise_dir = mlcvnet_noise_dir
        self.partnet_noise_dummy = partnet_noise_dummy

        print('Cat name (dataset loading):', self.cat_name)

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

        # load metadata mappings
        parts_to_shapes_path = '../../dicts/parts_to_shapes.json'
        shapes_to_cat_path = '../../dicts/obj_to_cat.json'
        partnet_to_parts_path = '../../dicts/partnet_to_parts.json'
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

        # full_shape_list_path = f'/cluster/daidalos/abokhovkin/DeepSDF_v2/full_experiments_v2/{self.cat_name}_full_surface_pe/sv2_{self.cat_name}_train.json'
        with open(full_shape_list_path, 'r') as f:
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

        # parts_list_path = f'/cluster/daidalos/abokhovkin/DeepSDF_v2/full_experiments_v2/{self.cat_name}_parts_onehot_pe/sv2_{self.cat_name}_train.json'
        with open(parts_list_path, 'r') as f:
            chair_parts_list = json.load(f)

        self.chair_parts_map = {}
        for part_id in chair_parts_list['ShapeNetV2']:
            part_name = part_id.split('-')[1]
            if part_name not in self.chair_parts_map:
                self.chair_parts_map[part_name] = {}
            for idx in chair_parts_list['ShapeNetV2'][part_id]:
                self.chair_parts_map[part_name][idx] = i
                i += 1

        # mlcvnet_noise_path = f'/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_parts_mlcvnet/train/SdfSamples/ShapeNetV2/02933112-background'
        # mlcvnet_noise_path = f'/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_parts_mlcvnet/train/SdfSamples/ShapeNetV2/{cat_name_to_part_list_cat_id[self.cat_name]}-background'
        self.noise_partnetid_to_filename = {}
        for filename in os.listdir(mlcvnet_noise_path):
            if filename.endswith('.npz'):
                tokens = filename.split('_')
                partnet_id = tokens[3]
                if partnet_id not in self.noise_partnetid_to_filename:
                    self.noise_partnetid_to_filename[partnet_id] = []
                self.noise_partnetid_to_filename[partnet_id] += [filename]

    def __getitem__(self, index):

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
                    pass
            obj.root.children = new_children

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'name':
                data_feats = data_feats + (partnet_id,)
            else:
                assert False, 'ERROR: unknown feat type %s!' % feat

        # mlcvnet
        if self.data_mode == 'mlcvnet':
            scannet_geo = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_mlcvnet_canonical', self.mode, f'{partnet_scannet_id}.npy')))
        # partnet partial
        else:
            scannet_geo = torch.FloatTensor(np.load(os.path.join(self.datadir, common_path + '_partnetpartial', f'{partnet_scannet_id}.npy')))

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
                    # self.sdf_data_source = '/cluster/daidalos/abokhovkin/DeepSDF_v2/ShapeNetV2_dim256_partial_uniform'
                    rand_instance_id = np.random.randint(0, 10)
                    sdf_filenames = [os.path.join(self.sdf_data_source, 'SdfSamples/ShapeNetV2', f'{cat_id}-{child_names[i]}', f'{obj_id}_{partnet_id}_{rand_instance_id}.npz') for i in range(len(child_names))]
                    sdf_parts = torch.cat([torch.FloatTensor(unpack_sdf_samples(filename, 6192)[0])[None, ...] for filename in sdf_filenames], dim=0)
                else:
                    sdf_filenames = [os.path.join(self.sdf_data_source, self.mode, 'SdfSamples/ShapeNetV2', f'{cat_id}-{child_names[i]}', f'{scan_id}_{obj_id}_{partnet_id}_{instance_id}.npz') for i in range(len(child_names))]
                    sdf_parts = torch.cat([torch.FloatTensor(unpack_sdf_samples(filename, 6192)[0])[None, ...] for filename in sdf_filenames], dim=0) # self.num_subsample_points
            else:
                # mlcvnet test
                if self.data_mode == 'mlcvnet':
                    sdf_filenames = [os.path.join(self.sdf_data_source, 'test', 'SdfSamples/ShapeNetV2', f'{cat_id}', f'{scan_id}_{obj_id}_{partnet_id}_{instance_id}.npz')]
                    sdf_parts = torch.cat([torch.FloatTensor(unpack_sdf_samples(filename, 150000)[0])[None, ...] for filename in sdf_filenames], dim=0) # 350000 points
                # partnet partial
                else:
                    sdf_filenames = [os.path.join(self.sdf_data_source, 'SdfSamples/ShapeNetV2', f'{cat_id}-{child_names[i]}', f'{obj_id}_{partnet_id}_{instance_id}.npz') for i in range(len(child_names))]
                    sdf_parts = torch.cat([torch.FloatTensor(unpack_sdf_samples(filename, self.num_subsample_points)[0])[None, ...] for filename in sdf_filenames], dim=0)


        # NOISE SECTION #

        # mlcvnet
        if self.data_mode == 'mlcvnet':
            if self.mode in ['train', 'val']:
                noise_mode = self.mode
            else:
                noise_mode = 'val'
            # self.mlcvnet_noise_dir = f'/cluster/daidalos/abokhovkin/DeepSDF_v2/ShapeNetV2_dim256_parts_mlcvnet/{noise_mode}/SdfSamples/ShapeNetV2/{cat_id}-background'
            noise_full = np.load(os.path.join(self.mlcvnet_noise_dir, f'{scan_id}_{obj_id}_{partnet_id}_{instance_id}.npz'))
            indices = np.random.choice(len(noise_full['pos']), min(3096, len(noise_full['pos'])), replace=False)
            noise_full_pos = torch.FloatTensor(noise_full['pos'][indices])
            indices = np.random.choice(len(noise_full['neg']), min(3096, len(noise_full['neg'])), replace=False)
            noise_full_neg = torch.FloatTensor(noise_full['neg'][indices])
            noise_full = torch.cat([noise_full_pos, noise_full_neg], 0)

        # partnet partial (as mlcvnet)
        else:
            # self.mlcvnet_noise_dir = f'/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_parts_mlcvnet/train/SdfSamples/ShapeNetV2/{cat_id}-background'
            if partnet_id in self.noise_partnetid_to_filename:
                noise_filenames = self.noise_partnetid_to_filename[partnet_id]
                random_index = np.random.choice(len(noise_filenames), 1)[0]
                noise_filename = noise_filenames[random_index]
                try:
                    noise_full = np.load(os.path.join(self.mlcvnet_noise_dir, noise_filename))
                except ValueError:
                    print('Failed noise', os.path.join(self.mlcvnet_noise_dir, noise_filename))
                    raise ValueError

                indices = np.random.choice(len(noise_full['pos']),
                                        min(2048 // 2, len(noise_full['pos'])), replace=False)
                noise_full_pos = torch.FloatTensor(noise_full['pos'][indices])
                indices = np.random.choice(len(noise_full['neg']),
                                        min(2048 // 2, len(noise_full['neg'])), replace=False)
                noise_full_neg = torch.FloatTensor(noise_full['neg'][indices])
                noise_full = torch.cat([noise_full_pos, noise_full_neg], 0)
            else:
                try:
                    # self.partnet_noise_dir = f'/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_partial_uniform_pickle/SdfSamples/ShapeNetV2/{cat_id}-{child_names[0]}'
                    noise_full = np.load(os.path.join(self.partnet_noise_dir, f'{obj_id}_{partnet_id}_{instance_id}.npz'))
                    noise_full = noise_full['random_full']
                    noise_full = np.hstack([noise_full, 0.07 * np.ones((len(noise_full), 1))])
                    indices = np.random.choice(len(noise_full), min(len(noise_full), 2048))
                    noise_full = noise_full[indices]
                    noise_full = torch.FloatTensor(noise_full)
                except:
                    # self.partnet_noise_dummy = '/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_partial_uniform_pickle/SdfSamples/ShapeNetV2/03001627-chair_seat/7fc8b858cad8f5849df6f10c48eb6cee_40124_0.pth'
                    noise_full = torch.load(self.partnet_noise_dummy)
                    noise_full = noise_full['random_full']
                    noise_full = torch.hstack([noise_full, 0.07 * torch.ones((len(noise_full), 1))])
                    indices = np.random.choice(len(noise_full), min(len(noise_full), 2048))
                    noise_full = noise_full[indices]

        if self.mode in ['train', 'val']:
            parts_indices = {x: self.chair_parts_map[x][obj_id] for x in child_names}
            full_shape_idx = self.chair_full_map[obj_id]
        else:
            parts_indices = {x: 0 for x in child_names}
            full_shape_idx = 0

        output = (scannet_geo, data_feats, partnet_id, int(rotation), tokens,
                  sdf_filenames, sdf_parts, parts_indices, full_shape_idx,
                  noise_full, index)

        return output


def generate_gnn_deepsdf_scannet_datasets(datadir=None, dataset=None,
                                          train_samples='train.txt', val_samples='val.txt',
                                          data_features=('object',), load_geo=True,
                                          num_subsample_points=0,
                                          sdf_data_source=None, 
                                          cat_name='chair', eval_mode=False, full_shape_list_path=None, 
                                          parts_list_path=None, mlcvnet_noise_path=None, data_mode=None, 
                                          partnet_noise_dir=None, mlcvnet_noise_dir=None,
                                          partnet_noise_dummy=None, **kwargs):
    if isinstance(data_features, str):
        data_features = [data_features]

    Dataset = VoxelisedImplicitScanNetDataset

    train_dataset = Dataset(datadir, dataset, train_samples, data_features, load_geo,
                            num_subsample_points,
                            sdf_data_source, 
                            cat_name, eval_mode, full_shape_list_path, parts_list_path,
                            mlcvnet_noise_path, data_mode, partnet_noise_dir, mlcvnet_noise_dir,
                            partnet_noise_dummy)
    val_dataset = Dataset(datadir, dataset, val_samples, data_features, load_geo,
                          num_subsample_points,
                          sdf_data_source, 
                          cat_name, eval_mode, full_shape_list_path, parts_list_path,
                          mlcvnet_noise_path, data_mode, partnet_noise_dir, mlcvnet_noise_dir,
                          partnet_noise_dummy)

    return {
        'train': train_dataset,
        'val': val_dataset
    }
