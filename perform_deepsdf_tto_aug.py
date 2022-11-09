import os
import sys
import json
import pickle
import gc

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import plyfile
import time
import skimage
from scipy.ndimage import rotate

from src.utils.config import load_config
from src.utils.gnn import collate_feats
from src.lightning_models.gnn_deepsdf import GNNPartnetLightning
from src.datasets.partnet import VoxelisedImplicitAllShapesDataset, VoxelisedImplicitScanNetDataset


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

    print('Min value:', numpy_3d_sdf_tensor.min())

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
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


def main(args):
    CLUSTER_DATA = '/cluster'

    # datadir = os.path.join(CLUSTER_DATA, 'sorona/abokhovkin/part-segmentation/hierarchies_32_1lvl_filled_v2')
    datadir = '/cluster/pegasus/abokhovkin/part-segmentation/hierarchies_32_1lvl_v3'

    # dataset = 'all_chair_partnet'
    # dataset = 'all_chair_geoscan'
    dataset = 'all_chair_scannet'
    # dataset = 'all_chair_partnetpartial'

    partnet_to_dirs_path = os.path.join('partnet_to_dirs.pkl')
    shapenet_voxelized_path = os.path.join(CLUSTER_DATA, 'partnet-full-voxelized-32')

    # sdf_data_source = '/cluster/sorona/abokhovkin/DeepSDF/ShapeNetV2_dim256_uniform_parts/SdfSamples/ShapeNetV2'
    # sdf_data_source = '/cluster/pegasus/abokhovkin/DeepSDF/ShapeNetV2_dim256_uniform_parts_geoscan/SdfSamples/ShapeNetV2'
    sdf_data_source = '/cluster/pegasus/abokhovkin/DeepSDF/ShapeNetV2_dim256_uniform_parts_scannet_fixed/SdfSamples/ShapeNetV2'
    # sdf_data_source = '/cluster/pegasus/abokhovkin/DeepSDF/ShapeNetV2_dim256_partial_uniform/SdfSamples/ShapeNetV2'

    parts_to_shapes_path = '/rhome/abokhovkin/projects/scannet-relationships/dicts/parts_to_shapes.json'
    shapes_to_cat_path = '/rhome/abokhovkin/projects/scannet-relationships/dicts/obj_to_cat.json'
    partnet_to_parts_path = '/rhome/abokhovkin/projects/scannet-relationships/dicts/partnet_to_parts.json'

    print('Create dataset')

    data_features = ['object']
    train_dataset = 'train.txt'
    train_dataset = VoxelisedImplicitScanNetDataset(datadir=datadir,
                                                    dataset=dataset,
                                                    partnet_to_dirs_path=partnet_to_dirs_path,
                                                    object_list=train_dataset,
                                                    data_features=data_features,
                                                    load_geo=True,
                                                    shapenet_voxelized_path=shapenet_voxelized_path,
                                                    num_subsample_points=50000,
                                                    sdf_data_source=sdf_data_source,
                                                    parts_to_shapes_path=parts_to_shapes_path,
                                                    shapes_to_cat_path=shapes_to_cat_path,
                                                    partnet_to_parts_path=partnet_to_parts_path)

    LOG_DIR = '/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf/GNNPartnet'
    exp = 'deepsdf_parts_fullshape_partial_2_2_latent_onehot'
    EXPERIMENT = os.path.join(LOG_DIR, exp)

    with open(os.path.join(EXPERIMENT, 'config.json'), 'r') as fin:
        config = json.load(fin)

    config['partnet_to_dirs_path'] = partnet_to_dirs_path
    config['datadir'] = datadir
    config['parts_to_ids_path'] = 'parts_to_ids.pkl'
    config['priors_path'] = os.path.join(config['datadir'], '../priors_geoscan/table_32_filled')
    # config['dataset'] = 'all_chair_partnet'
    # config['dataset'] = 'all_chair_geoscan'
    config['dataset'] = 'all_chair_scannet'
    # config['dataset'] = 'all_chair_partnetpartial'

    config['parts_to_shapes_path'] = parts_to_shapes_path
    config['shapes_to_cat_path'] = shapes_to_cat_path
    config['partnet_to_parts_path'] = partnet_to_parts_path

    config['train_samples'] = 'train.txt'
    config['val_samples'] = 'train.txt'

    print('Create model')

    device = torch.device('cuda:0')
    model = GNNPartnetLightning(config)

    print('Load checkpoint')

    # model = model.load_from_checkpoint(
    #     checkpoint_path=os.path.join(EXPERIMENT, 'checkpoints/25.ckpt')
    # )

    model.to(device)
    model.eval()
    model.freeze()

    level_inference = 0.000
    level_tto = 0.000

    SAVE_BASEDIR = '/cluster/pegasus/abokhovkin/scannet-relationships'
    exp = exp + '_predlat_predpoints_augtransy'
    print(exp)

    # 56 29 37 15 41
    idx = 1
    for i in range(idx, idx+1):
        print('Object', i)

        for aug_index in range(2, 6):

            batch = list(train_dataset[i])
            angle = 0

            orig_scan_geo = batch[0].cpu().numpy().astype('uint8')
            rotated_scan_geo = rotate(orig_scan_geo, -angle, axes=[0, 2], reshape=False).astype('float32')
            rotated_scan_geo_torch = torch.FloatTensor(rotated_scan_geo)[None, ...]

            scan_geo = rotated_scan_geo_torch.to(device)
            batch[0] = (scan_geo,)
            shape = batch[2][0].to(device)
            batch[2] = (shape,)
            batch[3] = (batch[3],)
            batch[5] = (batch[5],)
            output = model.tto_two_stage(batch, index=i, rot_aug=0, shift_x=0, shift_y=aug_index)

            SAVE_DIR = os.path.join(SAVE_BASEDIR, 'meshes_scannet', exp, str(i) + '_' + str(aug_index))
            os.makedirs(SAVE_DIR, exist_ok=True)
            for x in os.listdir(SAVE_DIR):
                os.remove(os.path.join(SAVE_DIR, x))

            pred_sdfs = output[1][0]
            for j, part_name in enumerate(pred_sdfs):
                try:
                    convert_sdf_samples_to_ply(
                        pred_sdfs[part_name]['sdf'],
                        pred_sdfs[part_name]['vox_origin'],
                        pred_sdfs[part_name]['vox_size'],
                        os.path.join(SAVE_DIR, str(part_name) + '.ply'),
                        offset=pred_sdfs[part_name]['offset'],
                        scale=pred_sdfs[part_name]['scale'],
                        level=level_inference
                    )
                except FileNotFoundError:
                    print(part_name)

            SAVE_DIR = os.path.join(SAVE_BASEDIR, 'meshes_scannet', exp, str(i) + '_' + str(aug_index))
            os.makedirs(SAVE_DIR, exist_ok=True)

            shape_sdf = output[2][0]
            convert_sdf_samples_to_ply(
                shape_sdf['sdf'],
                shape_sdf['vox_origin'],
                shape_sdf['vox_size'],
                os.path.join(SAVE_DIR, 'full_pred.ply'),
                offset=shape_sdf['offset'],
                scale=shape_sdf['scale'],
                level=level_inference
            )

            SAVE_DIR = os.path.join(SAVE_BASEDIR, 'meshes_scannet', exp, str(i) + '_tto' + '_' + str(aug_index))
            os.makedirs(SAVE_DIR, exist_ok=True)
            for x in os.listdir(SAVE_DIR):
                os.remove(os.path.join(SAVE_DIR, x))

            pred_sdfs = output[3][0]
            for j, part_name in enumerate(pred_sdfs):
                try:
                    for k in range(len(pred_sdfs[part_name])):
                        convert_sdf_samples_to_ply(
                            pred_sdfs[part_name][k]['sdf'],
                            pred_sdfs[part_name][k]['vox_origin'],
                            pred_sdfs[part_name][k]['vox_size'],
                            os.path.join(SAVE_DIR, part_name + '.' + str(k) + '.ply'),
                            offset=pred_sdfs[part_name][k]['offset'],
                            scale=pred_sdfs[part_name][k]['scale'],
                            level=level_tto
                        )

                    np.save(os.path.join(SAVE_DIR, part_name + '.pts.npy'), output[5][0][part_name][0])
                    np.save(os.path.join(SAVE_DIR, part_name + '.sdf.npy'), output[5][0][part_name][2])
                    np.save(os.path.join(SAVE_DIR, part_name + '.loss.' + str(0) + '.npy'), output[5][0][part_name][1])
                    np.save(os.path.join(SAVE_DIR, part_name + '.loss.' + str(1) + '.npy'), output[5][1][part_name][1])
                except KeyError:
                    print(part_name, k)

            SAVE_DIR = os.path.join(SAVE_BASEDIR, 'meshes_scannet', exp, str(i) + '_tto' + '_' + str(aug_index))
            os.makedirs(SAVE_DIR, exist_ok=True)
            shape_sdf = output[4][0]

            for j in range(len(shape_sdf)):
                for k in range(len(shape_sdf[j])):
                    convert_sdf_samples_to_ply(
                        shape_sdf[j][k]['sdf'],
                        shape_sdf[j][k]['vox_origin'],
                        shape_sdf[j][k]['vox_size'],
                        os.path.join(SAVE_DIR, f'full_pred.{j}.{k}.ply'),
                        offset=shape_sdf[j][k]['offset'],
                        scale=shape_sdf[j][k]['scale'],
                        level=level_tto
                    )

            # np.save(os.path.join(SAVE_DIR, 'full.pts.npy'), output[5][2][0])
            # np.save(os.path.join(SAVE_DIR, 'full.sdf.npy'), output[5][2][2])
            # np.save(os.path.join(SAVE_DIR, 'full.loss.' + str(0) + '.npy'), output[5][2][1])
            # np.save(os.path.join(SAVE_DIR, 'full.loss.' + str(1) + '.npy'), output[5][3][1])

            del output

            gc.collect()


if __name__ == '__main__':
    main(sys.argv[1:])
