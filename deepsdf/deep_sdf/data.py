#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import time
import pandas as pd

import deep_sdf.workspace as ws


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def remove_outliers(tensor):
    tensor_outlier = torch.where((tensor[:, 0].abs() > 10) | (tensor[:, 1].abs() > 10) | (tensor[:, 2].abs() > 10))[0]
    return tensor[~tensor_outlier, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


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


def get_noise_points(filename, trunc_value=0.07):
    npz = np.load(filename)
    noise_part = npz["random_part"]
    noise_full = npz["random_full"]

    noise_part = np.hstack([noise_part, trunc_value * np.ones((len(noise_part), 1))])
    noise_full = np.hstack([noise_full, trunc_value * np.ones((len(noise_full), 1))])

    return noise_part, noise_full


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        cat_name,
        learn_parts,
        data_source,
        split,
        subsample,
        load_ram=False
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram
        self.learn_parts = learn_parts

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

        if cat_name == 'chair':
            # chair 7
            self.class2id = {
                '03001627-chair_arm_left': 0,
                '03001627-chair_arm_right': 1,
                '03001627-chair_back': 2,
                '03001627-chair_seat': 3,
                '03001627-regular_leg_base': 4,
                '03001627-star_leg_base': 5,
                '03001627-surface_base': 6
            }
        elif cat_name == 'table':
            # table 7
            self.class2id = {
                '04379243-central_support': 0,
                '04379243-drawer': 1,
                '04379243-leg': 2,
                '04379243-pedestal': 3,
                '04379243-shelf': 4,
                '04379243-table_surface': 5,
                '04379243-vertical_side_panel': 6
            }
        elif cat_name == 'storagefurniture':
            # storagefurniture 5
            self.class2id = {
                '02871439-cabinet_door': 0,
                '02871439-shelf': 1,
                '02871439-cabinet_frame': 2,
                '02871439-cabinet_base': 3,
                '02871439-countertop': 4
            }
        elif cat_name == 'bed':
            # bed 4
            self.class2id = {
                '02818832-bed_frame_base': 0,
                '02818832-bed_side_surface': 1,
                '02818832-bed_sleep_area': 2,
                '02818832-headboard': 3,
            }
        elif cat_name == 'trashcan':
            # trashcan 5
            self.class2id = {
                '02747177-base': 0,
                '02747177-container_bottom': 1,
                '02747177-container_box': 2,
                '02747177-cover': 3,
                '02747177-other': 4
            }

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.learn_parts:
            class_name = filename.split('/')[-2]
            idx_latent = self.class2id[class_name]
            class_one_hot = torch.FloatTensor(torch.zeros((len(self.class2id))))
            class_one_hot[idx_latent] = 1
        else:
            class_one_hot = torch.FloatTensor(torch.zeros((7)))
        
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx, self.npyfiles[idx], class_one_hot
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx, self.npyfiles[idx], class_one_hot