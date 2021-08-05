import os, sys
from tqdm.autonotebook import tqdm
import json
import numpy as np
from scipy.ndimage import rotate
import skimage
import plyfile

import torch
from torch.utils.data import DataLoader

from src.datasets.partnet import VoxelisedImplicitAllShapesDataset
from src.utils.gnn import collate_feats
from src.lightning_models.gnn_deepsdf import GNNPartnetLightning


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
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

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
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


CLUSTER_DATA = '/cluster'

datadir = os.path.join(CLUSTER_DATA, 'sorona/abokhovkin/part-segmentation/hierarchies_32_1lvl_filled_v2')
dataset = 'all_chair_partnet'
partnet_to_dirs_path = os.path.join('partnet_to_dirs.pkl')
shapenet_voxelized_path = os.path.join(CLUSTER_DATA, 'partnet-full-voxelized-32')
sdf_data_source = '/cluster/sorona/abokhovkin/DeepSDF/ShapeNetV2_dim256_uniform_parts/SdfSamples/ShapeNetV2'
parts_to_shapes_path = '/rhome/abokhovkin/projects/scannet-relationships/dicts/parts_to_shapes.json'
shapes_to_cat_path = '/rhome/abokhovkin/projects/scannet-relationships/dicts/obj_to_cat.json'
partnet_to_parts_path = '/rhome/abokhovkin/projects/scannet-relationships/dicts/partnet_to_parts.json'

data_features = ['object']
train_dataset = 'train.txt'
train_dataset = VoxelisedImplicitAllShapesDataset(datadir=datadir,
                                                  dataset=dataset,
                                                  partnet_to_dirs_path=partnet_to_dirs_path,
                                                  object_list=train_dataset,
                                                  data_features=data_features,
                                                  load_geo=True,
                                                  shapenet_voxelized_path=shapenet_voxelized_path,
                                                  num_subsample_points=12000,
                                                  sdf_data_source=sdf_data_source,
                                                  parts_to_shapes_path=parts_to_shapes_path,
                                                  shapes_to_cat_path=shapes_to_cat_path,
                                                  partnet_to_parts_path=partnet_to_parts_path)
train_dataloader = DataLoader(train_dataset, batch_size=1,
                              shuffle=False, num_workers=8, drop_last=True,
                              collate_fn=collate_feats)

LOG_DIR = '/cluster/sorona/abokhovkin/scannet-relationships/logs/GNNPartnet'
exp = 'deepsdf_parts_partnet_1'
EXPERIMENT = os.path.join(LOG_DIR, exp)

with open(os.path.join(EXPERIMENT, 'config.json'), 'r') as fin:
    config = json.load(fin)

config['partnet_to_dirs_path'] = partnet_to_dirs_path
config['datadir'] = datadir
config['parts_to_ids_path'] = 'parts_to_ids.pkl'
config['priors_path'] = os.path.join(config['datadir'], '../priors_geoscan/table_32_filled')
config['dataset'] = 'all_chair_partnet'
config['parts_to_shapes_path'] = parts_to_shapes_path
config['shapes_to_cat_path'] = shapes_to_cat_path
config['partnet_to_parts_path'] = partnet_to_parts_path

device = torch.device('cuda:0')
model = GNNPartnetLightning(config)

pretrained_model = model.load_from_checkpoint(
    checkpoint_path=os.path.join(EXPERIMENT, 'checkpoints/500.ckpt')
)

pretrained_model.to(device)
pretrained_model.eval()
pretrained_model.freeze()

for i in tqdm(range(600)):
    batch = list(train_dataset[i])
    tokens = batch[6]
    partnet_id = tokens[0]

    angle = 0

    orig_scan_geo = batch[0].cpu().numpy().astype('uint8')
    rotated_scan_geo = rotate(orig_scan_geo, -angle, axes=[0, 2], reshape=False).astype('float32')
    rotated_scan_geo_torch = torch.FloatTensor(rotated_scan_geo)[None, ...]

    scan_geo = rotated_scan_geo_torch.to(device)
    batch[0] = (scan_geo,)
    scan_sdf = batch[1]
    shape = batch[2][0].to(device)
    batch[2] = (shape,)
    batch[3] = (batch[3],)
    batch[5] = (batch[5],)
    output = pretrained_model.inference(batch)
    predicted_tree = output[0][0]
    gt_tree = batch[3][0]

    SAVE_DIR = os.path.join('../../meshes', exp, str(i))
    os.makedirs(SAVE_DIR, exist_ok=True)

    pred_sdfs = output[1][0]
    for j, part_name in enumerate(pred_sdfs):
        try:
            convert_sdf_samples_to_ply(
                pred_sdfs[part_name]['sdf'],
                pred_sdfs[part_name]['vox_origin'],
                pred_sdfs[part_name]['vox_size'],
                os.path.join(SAVE_DIR, part_name + '.ply'),
                offset=pred_sdfs[part_name]['offset'],
                scale=pred_sdfs[part_name]['scale'],
            )
        except:
            print(j)

    part_sdfs = []
    for j, part_name in enumerate(pred_sdfs):
        part_sdf = pred_sdfs[part_name]['sdf']
        part_sdfs += [part_sdf]
    pred_sdf_final = 100 * torch.ones_like(pred_sdfs[part_name]['sdf'])
    for part_sdf in part_sdfs:
        pred_sdf_final = torch.where(torch.abs(part_sdf) < torch.abs(pred_sdf_final), part_sdf, pred_sdf_final)

    try:
        convert_sdf_samples_to_ply(
            pred_sdf_final,
            pred_sdfs[part_name]['vox_origin'],
            pred_sdfs[part_name]['vox_size'],
            os.path.join(SAVE_DIR, 'full.ply'),
            offset=pred_sdfs[part_name]['offset'],
            scale=pred_sdfs[part_name]['scale'],
        )
    except:
        print(partnet_id)

