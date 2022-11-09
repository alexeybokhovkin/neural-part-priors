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
import argparse

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
    t0 = time.time()
    CLUSTER_DATA = '/cluster'

    print(1)
    print('Cat name:', args.cat_name)

    datadir = '/cluster/pegasus/abokhovkin/part-segmentation/hierarchies_32_1lvl_v3'

    dataset = f'all_{args.cat_name}_mlcvnet'
    # dataset = 'all_chair_partnetpartial'

    partnet_to_dirs_path = os.path.join('partnet_to_dirs.pkl')
    shapenet_voxelized_path = os.path.join(CLUSTER_DATA, 'partnet-full-voxelized-32')

    # sdf_data_source = '/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_partial_uniform'
    # sdf_data_source = '/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_parts_mlcvnet'
    sdf_data_source = '/cluster/daidalos/abokhovkin/DeepSDF_v2/ShapeNetV2_dim256_parts_mlcvnet'

    parts_to_shapes_path = '/rhome/abokhovkin/projects/scannet-relationships/dicts/parts_to_shapes.json'
    shapes_to_cat_path = '/rhome/abokhovkin/projects/scannet-relationships/dicts/obj_to_cat.json'
    partnet_to_parts_path = '/rhome/abokhovkin/projects/scannet-relationships/dicts/partnet_to_parts.json'

    print('Create dataset')

    print('Cat name:', args.cat_name)

    data_features = ['object']
    dataset_list = 'test.txt'
    train_dataset = VoxelisedImplicitScanNetDataset(datadir=datadir,
                                                    dataset=dataset,
                                                    partnet_to_dirs_path=partnet_to_dirs_path,
                                                    object_list=dataset_list,
                                                    data_features=data_features,
                                                    load_geo=True,
                                                    shapenet_voxelized_path=shapenet_voxelized_path,
                                                    num_subsample_points=150000, # 150000
                                                    sdf_data_source=sdf_data_source,
                                                    parts_to_shapes_path=parts_to_shapes_path,
                                                    shapes_to_cat_path=shapes_to_cat_path,
                                                    partnet_to_parts_path=partnet_to_parts_path,
                                                    cat_name=args.cat_name,
                                                    eval_mode=False)

    with open(os.path.join(datadir, dataset, dataset_list), 'r') as fin:
        test_samples = fin.readlines()
        test_samples = [x[:-1] for x in test_samples]

    LOG_DIR = '/cluster/pegasus/abokhovkin/scannet-relationships/logs_deepsdf_finetune/GNNPartnet'
    exp = 'deepsdf_parts_fullshape_mlcvnet_finetune'
    EXPERIMENT = os.path.join(LOG_DIR, exp)

    with open(os.path.join(EXPERIMENT, 'config.json'), 'r') as fin:
        config = json.load(fin)

    config['partnet_to_dirs_path'] = partnet_to_dirs_path
    config['datadir'] = datadir
    config['parts_to_ids_path'] = 'parts_to_ids.pkl'
    config['priors_path'] = os.path.join(config['datadir'], '../priors_geoscan/table_32_filled')

    config['dataset'] = f'all_{args.cat_name}_mlcvnet'
    # config['dataset'] = 'all_chair_partnetpartial'

    config['parts_to_shapes_path'] = parts_to_shapes_path
    config['shapes_to_cat_path'] = shapes_to_cat_path
    config['partnet_to_parts_path'] = partnet_to_parts_path
    config['num_subsample_points'] = 32768

    config['train_samples'] = dataset_list
    config['val_samples'] = dataset_list

    print('Create model')

    device = torch.device('cuda:0')
    model = GNNPartnetLightning(config, cat_name=args.cat_name)

    print('Load checkpoint')

    # model = model.load_from_checkpoint(
    #     checkpoint_path=os.path.join(EXPERIMENT, 'checkpoints/25.ckpt')
    # )

    model.to(device)
    model.eval()
    model.freeze()

    level_inference = 0.000
    level_tto = 0.000

    scene0621_00_samples = [0, 120, 126, 134, 213, 214, 273, 319, 327, 343, 417, 455, 620, 727, 730, 747, 832, 866, 891, 902, 920]
    scene0342_00_samples = [1, 5, 60, 139, 144, 145, 174, 253, 312, 386, 557, 607, 643, 666, 787, 859]
    scene0011_01_samples = [6, 140, 156, 322, 375, 506, 566, 632, 815, 827]
    scene0081_02_samples = [11, 27, 118, 180, 186, 266, 336, 357, 372, 418, 427, 461, 476, 500, 551, 746, 804]
    scene0088_00_samples = [14, 79, 98, 108, 187, 398, 540, 547, 574, 685, 752, 757, 788]
    scene0015_00_samples = [15, 33, 289, 335, 359, 449, 588, 595, 664, 759]
    scene0095_01_samples = [37, 240, 301, 346, 520, 535, 640, 678, 770, 849]
    scene0500_00_samples = [38, 48, 65, 76, 104, 142, 172, 192, 216, 276, 280, 307, 369, 400, 424, 494, 516, 537, 603, 655]
    scene0088_03_samples = [71, 96, 114, 131, 183, 202, 209, 220, 235, 258, 341, 401, 597, 648]
    scene0430_00_samples = [132, 161, 164, 196, 245, 262, 294, 302, 358, 440, 604, 610, 614, 697, 800, 852, 873, 879]

    scene_samples = {}
    scene_samples['scene0621_00'] = scene0621_00_samples
    scene_samples['scene0342_00'] = scene0342_00_samples
    scene_samples['scene0011_01'] = scene0011_01_samples
    scene_samples['scene0081_02'] = scene0081_02_samples
    scene_samples['scene0088_00'] = scene0088_00_samples
    scene_samples['scene0015_00'] = scene0015_00_samples
    scene_samples['scene0095_01'] = scene0095_01_samples
    scene_samples['scene0500_00'] = scene0500_00_samples
    scene_samples['scene0088_03'] = scene0088_03_samples
    scene_samples['scene0430_00'] = scene0430_00_samples

    for scene_id in ['scene0088_03', 'scene0430_00']:
        for num_shapes in [3, 6, 9]: # 3, 6 (min)
            for k_near in [10, 30]: # 30

                SAVE_BASEDIR = f'/cluster/daidalos/abokhovkin/scannet-relationships/scene_aware_min_fullshape/val_output_{scene_id}_min_{num_shapes}shapes_k{k_near}'
                exp_full_name = exp + f'_predlat_predpoints_{args.cat_name}_{args.bck_thr}_{args.constr_mode}'
                print(exp_full_name)

                SAVE_DIR = os.path.join(SAVE_BASEDIR, 'meshes_scannet', exp_full_name)
                os.makedirs(SAVE_DIR, exist_ok=True)

                t1 = time.time()

                test_scene_1 = [0, 120, 126, 134, 213, 214, 273, 319, 327, 343, 417, 455, 491, 577, 620, 727, 730, 747, 799, 832, 866, 891, 902, 920, ]
                test_scene_2 = [2, 73, 199, 274, 511, 617, 654, 705, 857, ]
                test_scene_3 = [4, 380, 414, 790, 803, ]
                test_scene_4 = [6, 140, 156, 322, 375, 506, 566, 632, 815, 827, ]
                test_scene_5 = [10, 291, 601, 642, 652, 845, 888, ]
                test_scene_6 = [11, 27, 118, 180, 186, 266, 336, 357, 372, 418, 427, 461, 476, 500, 551, 746, 804, ]
                test_scene_7 = [13, 42, 238, 453, 748, 789, 831, 892, ]
                test_scene_8 = [15, 33, 211, 272, 289, 335, 348, 359, 449, 588, 595, 650, 661, 664, 759, 903, ]

                idx = 0
                for i in scene_samples[scene_id][0:1]:
                    print('Object', i)
                    print('Only align:', args.only_align)

                    batch = list(train_dataset[i])

                    orig_scan_geo = batch[0].cpu().numpy().astype('uint8')
                    rotated_scan_geo_torch = torch.FloatTensor(orig_scan_geo)[None, ...]

                    scan_geo = rotated_scan_geo_torch.to(device)
                    batch[0] = (scan_geo,)
                    shape = batch[2][0].to(device)
                    batch[2] = (shape,)
                    batch[3] = (batch[3],)
                    batch[5] = (batch[5],)

                    t2 = time.time()

                    output = model.tto_two_stage(batch, index=test_samples[i], only_align=args.only_align, bck_thr=0.25,
                                                 constr_mode=args.constr_mode, cat_name=args.cat_name,
                                                 num_shapes=num_shapes, k_near=k_near, scene_id=scene_id,
                                                 scale=args.scale, wconf=args.wconf,
                                                 w_full_noise=args.w_full_noise, w_part_u_noise=args.w_part_u_noise,
                                                 w_part_part_noise=args.w_part_part_noise, lr_dec_full=args.lr_dec_full,
                                                 lr_dec_part=args.lr_dec_part
                                                 )

                    t3 = time.time()

                    SAVE_DIR = os.path.join(SAVE_BASEDIR, 'meshes_scannet', exp_full_name, test_samples[i])
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    for x in os.listdir(SAVE_DIR):
                        if not x.endswith('.pth'):
                            os.remove(os.path.join(SAVE_DIR, x))

                    meta_data = output[6][0]

                    if args.only_align:
                        torch.save(meta_data[5], os.path.join(SAVE_DIR, 'icp.pth'))
                        torch.save(meta_data[6], os.path.join(SAVE_DIR, 'rot_matrix.pth'))
                        torch.save(meta_data[7], os.path.join(SAVE_DIR, 'rotation.pth'))

                    else:
                        torch.save(meta_data[6], os.path.join(SAVE_DIR, 'rot_matrix.pth'))
                        torch.save(meta_data[7], os.path.join(SAVE_DIR, 'rotation.pth'))
                        torch.save(meta_data[5], os.path.join(SAVE_DIR, 'icp_2.pth'))
                        torch.save(meta_data[0], os.path.join(SAVE_DIR, 'sdf_flat.pth'))
                        torch.save(meta_data[1], os.path.join(SAVE_DIR, 'sdf_flat_rot.pth'))
                        torch.save(meta_data[2], os.path.join(SAVE_DIR, 'sdf_full_unrot.pth'))
                        torch.save(meta_data[3], os.path.join(SAVE_DIR, 'scan_points_icp.pth'))
                        torch.save(meta_data[4], os.path.join(SAVE_DIR, 'mesh_points_icp.pth'))

                        # Save parts before TTO
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
                            except:
                                print('Bad part before TTO:', test_samples[i])
                                continue

                        SAVE_DIR = os.path.join(SAVE_BASEDIR, 'meshes_scannet', exp_full_name, test_samples[i])
                        os.makedirs(SAVE_DIR, exist_ok=True)

                        # Save shape before TTO
                        try:
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
                        except:
                            print('Bad full shape before TTO:', test_samples[i])

                        SAVE_DIR = os.path.join(SAVE_BASEDIR, 'meshes_scannet', exp_full_name, test_samples[i] + '_tto')
                        os.makedirs(SAVE_DIR, exist_ok=True)
                        for x in os.listdir(SAVE_DIR):
                            if not x.endswith('.pth'):
                                os.remove(os.path.join(SAVE_DIR, x))

                        # Save parts after TTO
                        pred_sdfs = output[3][0]
                        for j, part_name in enumerate(pred_sdfs):
                            try:
                                for k in range(len(pred_sdfs[part_name])):
                                    try:
                                        convert_sdf_samples_to_ply(
                                            pred_sdfs[part_name][k]['sdf'],
                                            pred_sdfs[part_name][k]['vox_origin'],
                                            pred_sdfs[part_name][k]['vox_size'],
                                            os.path.join(SAVE_DIR, part_name + '.' + str(k) + '.ply'),
                                            offset=pred_sdfs[part_name][k]['offset'],
                                            scale=pred_sdfs[part_name][k]['scale'],
                                            level=level_tto
                                        )
                                        torch.save(pred_sdfs[part_name][k]['latent'], os.path.join(SAVE_DIR, part_name + '.' + str(k) + '.lat_pth'))
                                    except:
                                        print('Bad part after TTO:', test_samples[i])
                                        continue

                                np.save(os.path.join(SAVE_DIR, part_name + '.pts.npy'), output[5][0][part_name][0])
                                np.save(os.path.join(SAVE_DIR, part_name + '.sdf.npy'), output[5][0][part_name][2])
                                np.save(os.path.join(SAVE_DIR, part_name + '.loss.' + str(0) + '.npy'), output[5][0][part_name][1])
                                np.save(os.path.join(SAVE_DIR, part_name + '.loss.' + str(1) + '.npy'), output[5][1][part_name][1])
                            except:
                                print(part_name, k)

                        SAVE_DIR = os.path.join(SAVE_BASEDIR, 'meshes_scannet', exp_full_name, test_samples[i] + '_tto')
                        os.makedirs(SAVE_DIR, exist_ok=True)
                        shape_sdf = output[4][0]

                        # Save shape after TTO
                        for j in range(len(shape_sdf)):
                            for k in range(len(shape_sdf[j])):
                                try:
                                    convert_sdf_samples_to_ply(
                                        shape_sdf[j][k]['sdf'],
                                        shape_sdf[j][k]['vox_origin'],
                                        shape_sdf[j][k]['vox_size'],
                                        os.path.join(SAVE_DIR, f'full_pred.{j}.{k}.ply'),
                                        offset=shape_sdf[j][k]['offset'],
                                        scale=shape_sdf[j][k]['scale'],
                                        level=level_tto
                                    )
                                    torch.save(shape_sdf[j][k]['latent'], os.path.join(SAVE_DIR, f'full_pred.{j}.{k}.lat_pth'))
                                except:
                                    print('Bad full shape after TTO:', test_samples[i])

                        t4 = time.time()

                    del output

                    gc.collect()

                    # print('(0) Init:', t1 - t0)
                    # print('(0) Data read:', t2 - t1)
                    # print('(0) Inference:', t3 - t2)
                    # print('(0) Saving:', t4 - t3)


if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('--cat_name', required=True, default='chair')
    parser.add_argument('--bck_thr', required=True, default=0.5, type=float)
    parser.add_argument('--constr_mode', required=True, default=0, type=int)
    parser.add_argument('--only_align', action='store_true')
    parser.add_argument('--wconf', required=True, default=0.0, type=float)
    parser.add_argument('--scale', required=True, default=1.0, type=float)
    parser.add_argument('--w_full_noise', required=True, default=1.0, type=float)
    parser.add_argument('--w_part_u_noise', required=True, default=1.0, type=float)
    parser.add_argument('--w_part_part_noise', required=True, default=1.0, type=float)
    parser.add_argument('--lr_dec_full', required=True, default=0.0, type=float)
    parser.add_argument('--lr_dec_part', required=True, default=0.0, type=float)
    parser.add_argument('--scene_id', required=True, default='0', type=str)

    args = parser.parse_args()

    main(args)
