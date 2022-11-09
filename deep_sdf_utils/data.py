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

import deep_sdf_utils.workspace as ws


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
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
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


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None, trunc_value=0.07, check_noise=True):
    t0 = time.time()

    npz = np.load(filename)
    # try:
    #     # numpy
    #     # npz = np.load(filename)
    #     # torch
    #     npz = torch.load(filename)
    # except:
    #     print(filename)
    #     # raise ValueError
    #     # npz = torch.load('/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_partial_uniform_pickle/SdfSamples/ShapeNetV2/03001627-chair_seat/7fc8b858cad8f5849df6f10c48eb6cee_40124_0.pth')

    t1 = time.time()

    num_samples = int(1.00 * subsample)
    num_pos_samples = int(0.5 * num_samples)
    num_neg_samples = int(0.5 * num_samples)
    num_random = subsample - num_samples

    if subsample is None:
        return npz

    # pos_tensor = remove_nans(torch.from_numpy(npz["pos"].astype('float32')))
    # neg_tensor = remove_nans(torch.from_numpy(npz["neg"].astype('float32')))
    # numpy
    pos_tensor = torch.from_numpy(npz["pos"].astype('float32'))
    neg_tensor = torch.from_numpy(npz["neg"].astype('float32'))
    # torch
    # pos_tensor = npz["pos"]
    # neg_tensor = npz["neg"]

    t2 = time.time()

    noise_full = None
    # if check_noise:
    #     noise_part = npz["random_part"].astype('float32')
    #     noise_full = npz["random_full"].astype('float32')
    #
    #     noise_part = torch.from_numpy(np.hstack([noise_part, trunc_value * np.ones((len(noise_part), 1))])).float()
    #     noise_full = torch.from_numpy(np.hstack([noise_full, trunc_value * np.ones((len(noise_full), 1))])).float()
    #
    #     if len(npz["pos"]) == 0:
    #         pos_tensor = noise_part
    #     if len(npz["neg"]) == 0:
    #         neg_tensor = noise_part

    if len(pos_tensor) != 0:
        if num_pos_samples > len(pos_tensor):
            all_indices = np.arange(len(pos_tensor))
            pos_indices = np.hstack([np.random.choice(len(pos_tensor), num_pos_samples - len(pos_tensor)), all_indices])
        else:
            pos_indices = np.random.choice(len(pos_tensor), num_pos_samples, replace=False)
        sample_pos = pos_tensor[pos_indices]
    else:
        sample_pos = torch.zeros((num_pos_samples, 4))
        sample_pos[:, 3] = 0.07
    if len(neg_tensor) != 0:
        if num_neg_samples > len(neg_tensor):
            all_indices = np.arange(len(neg_tensor))
            neg_indices = np.hstack([np.random.choice(len(neg_tensor), num_neg_samples - len(neg_tensor)), all_indices])
        else:
            neg_indices = np.random.choice(len(neg_tensor), num_neg_samples, replace=False)
        sample_neg = neg_tensor[neg_indices]
    else:
        sample_neg = sample_pos
    # random_indices = np.random.choice(len(noise_part), num_random)
    # sample_random = noise_part[random_indices]

    t3 = time.time()

    samples = torch.cat([sample_pos, sample_neg], 0)

    t4 = time.time()
    # print('(2) Load:', t1 - t0)
    # print('(2) Remove nans:', t2 - t1)
    # print('(2) Filter:', t3 - t2)
    # print('(2) Sum:', t4 - t0)
    # print()

    return samples, noise_full


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
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
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

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx, self.npyfiles[idx]
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx, self.npyfiles[idx]
