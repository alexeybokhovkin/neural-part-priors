import os, sys
import json
import numpy as np
import trimesh
from tqdm import tqdm
from scipy.spatial import cKDTree
from copy import deepcopy, copy

sys.path.append('/rhome/abokhovkin/projects/scannet-relationships')
from src.datasets.partnet import VoxelisedScanNetAllShapesGNNDataset
from src.data_utils.transformations import apply_transform, apply_inverse_transform, from_tqs_to_matrix
from src.data_utils.vox import load_sample, load_vox


def plot_leaves(all_nodes, thr=0.5):
    leaves = [x for x in all_nodes if len(x.children) == 0]
    full_geometry = np.zeros_like(leaves[0].geo.cpu().numpy()[0])
    for i, leaf in enumerate(leaves):
        color = i + 1
        geometry = leaf.geo.cpu().numpy()[0]
        full_geometry[np.where(geometry > thr)] = color
    full_geometry = full_geometry.astype('uint8')
    return full_geometry


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                      for g in scene_or_mesh.geometry.values()))
    else:
        assert (isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def bounding_cube_normalization(vertices):
    new_vertices = vertices.copy()

    xMin = new_vertices[:, 0].min()
    xMax = new_vertices[:, 0].max()
    yMin = new_vertices[:, 1].min()
    yMax = new_vertices[:, 1].max()
    zMin = new_vertices[:, 2].min()
    zMax = new_vertices[:, 2].max()
    xCenter = (xMax + xMin) / 2.0
    yCenter = (yMax + yMin) / 2.0
    zCenter = (zMax + zMin) / 2.0
    center = np.array([xCenter, yCenter, zCenter])
    new_vertices = new_vertices - center

    max_distance = np.sqrt(np.sum(new_vertices ** 2, axis=1)).max()
    new_vertices = new_vertices / max_distance

    return new_vertices, center, max_distance


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


if __name__ == '__main__':

    target_category = 'trashcan'
    target_category_agg = 'trashcan'

    print(f'Preparing full shape SDFs for {target_category} ({target_category_agg})')

    name_category_to_id = {'chair': '03001627',
                           'table': '04379243',
                           'cabinet': '02933112',
                           'bookshelf': '02871439',
                           'filecabinet': '03337140',
                           'bed': '02818832',
                           'sofa': '04256520',
                           'trashcan': '02747177'}

    DATADIR = f'/cluster_HDD/sorona/adai/data/shapenet_core/shapenetv2_dim256/{name_category_to_id[target_category]}'
    sdf_filenames = sorted([x for x in os.listdir(DATADIR) if x.endswith('__0__.sdf')])

    DICTIONARIES = '/rhome/abokhovkin/projects/scannet-relationships/dicts'
    with open(os.path.join(DICTIONARIES, 'obj_to_cat.json'), 'r') as fin:
        obj_to_cat = json.load(fin)
    with open(os.path.join(DICTIONARIES, 'parts_to_shapes.json'), 'r') as fin:
        partnetid_to_obj = json.load(fin)
    obj_to_partnetid = {u: v for v, u in partnetid_to_obj.items()}

    partnet_to_shapenet_transforms_path = os.path.join(
        '/rhome/abokhovkin/projects/scannet-relationships/dicts/chair_table_storagefurniture_bed_shapenetv1_to_partnet_alignment.json')
    with open(partnet_to_shapenet_transforms_path, 'rb') as fin:
        p2s_transforms = json.load(fin)

    PARTNET_VOXELIZED = '/cluster/sorona/abokhovkin/partnet-full-voxelized-32'
    VOXEL_TREES = '/cluster/pegasus/abokhovkin/part-segmentation/hierarchies_32_1lvl_v3'
    PARTNET = '/cluster/pegasus/abokhovkin/datasets/PartNet'
    SHAPENET = '/canis/ShapeNet/ShapeNetCore.v2'

    SAVEDIR_PREFIX = f'/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_{target_category_agg}_surface/SdfSamples/ShapeNetV2/{name_category_to_id[target_category]}'
    os.makedirs(SAVEDIR_PREFIX, exist_ok=True)

    rotation_matrix = np.array([[0, 0, -1, 0],
                                [0, 1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1]])

    dist_thr = 0.06
    num_processed = 0

    not_processed_all = []
    not_processed_shape_ids = []
    not_processed_partnet_ids = []
    not_processed_trees = []
    not_processed_empty_sdf = []

    for sdf_filename in tqdm(sdf_filenames):

        try:
            sample = load_sample(os.path.join(DATADIR, sdf_filename))
        except:
            not_processed_empty_sdf += [sdf_filename]
            continue

        locations = sample.locations
        locations_ones = np.hstack([locations, np.ones((len(locations), 1))])
        g2w = sample.grid2world
        w2g = np.linalg.inv(g2w)
        locations_ones_world = locations_ones.dot(w2g)[:, :3] * 2
        sdfs = sample.sdfs

        indices = np.where(sdfs >= 0)[0]
        locations_ones_world_pos = locations_ones_world[indices]
        sdfs_pos = sdfs[indices]

        indices = np.where(sdfs < 0)[0]
        locations_ones_world_neg = locations_ones_world[indices]
        sdfs_neg = sdfs[indices]

        grid_points_pos = np.hstack([locations_ones_world_pos, sdfs_pos[:, None]]).astype('float32')
        grid_points_neg = np.hstack([locations_ones_world_neg, sdfs_neg[:, None]]).astype('float32')
        grid_pos_sdf = grid_points_pos[:, 3]
        grid_neg_sdf = grid_points_neg[:, 3]

        try:
            obj_id = sdf_filename.split('_')[0]

            if obj_id not in obj_to_partnetid:
                not_processed_shape_ids += [obj_id]
                continue
            partnet_id = obj_to_partnetid[obj_id]

            if partnet_id in p2s_transforms:
                p2s_transform = np.array(p2s_transforms[partnet_id]['transmat']).reshape((4, 4), order="C")
            else:
                not_processed_partnet_ids += [partnet_id]
                continue

            mesh_shapenet_path = os.path.join(SHAPENET, name_category_to_id[target_category], obj_id, 'models/model_normalized.obj')
            mesh_shapenet = as_mesh(trimesh.load(mesh_shapenet_path))

            idx_pos = np.where(np.abs(grid_points_pos[:, 3]) < 0.07)[0]
            grid_points_pos_sampled = grid_points_pos[idx_pos]
            idx_neg = np.where(np.abs(grid_points_neg[:, 3]) < 0.07)[0]
            grid_points_neg_sampled = grid_points_neg[idx_neg]

            # perform aggressive sampling near surface
            pos_gaussian = gaussian(grid_points_pos_sampled[:, 3], 0, 0.07 / 3)
            random_uniform_samples = np.random.uniform(0, 1, (len(pos_gaussian),))
            samples_mask = np.where(random_uniform_samples < pos_gaussian)[0]
            grid_points_pos_sampled = grid_points_pos_sampled[samples_mask]
            neg_gaussian = gaussian(grid_points_neg_sampled[:, 3], 0, 0.07 / 3)
            random_uniform_samples = np.random.uniform(0, 1, (len(neg_gaussian),))
            samples_mask = np.where(random_uniform_samples < neg_gaussian)[0]
            grid_points_neg_sampled = grid_points_neg_sampled[samples_mask]

            indices = np.random.choice(len(grid_points_pos_sampled), min(250000, len(grid_points_pos_sampled)), replace=False)
            grid_points_pos_sampled = grid_points_pos_sampled[indices]
            indices = np.random.choice(len(grid_points_neg_sampled), min(250000, len(grid_points_neg_sampled)), replace=False)
            grid_points_neg_sampled = grid_points_neg_sampled[indices]

            # prepare noise
            grid_points_pos_indices_anchor = np.random.choice(len(grid_points_pos_sampled),
                                                              min(20000, len(grid_points_pos_sampled)),
                                                              replace=False)
            anchor_points_pos = grid_points_pos_sampled[grid_points_pos_indices_anchor]
            grid_points_neg_indices_anchor = np.random.choice(len(grid_points_neg_sampled),
                                                              min(20000, len(grid_points_neg_sampled)),
                                                              replace=False)
            anchor_points_neg = grid_points_neg_sampled[grid_points_neg_indices_anchor]
            anchor_points = np.vstack([anchor_points_pos, anchor_points_neg])[:, :3]
            kd_tree = cKDTree(anchor_points)

            num_samples_random = 33000
            random_samples = np.random.uniform(-1.0, 1.0, (num_samples_random, 3))
            min_dist_rand, min_idx_rand = kd_tree.query(random_samples)
            above_thr_indices = np.where(min_dist_rand > 0.05)[0]
            random_part = random_samples[above_thr_indices]

            outfile = os.path.join(SAVEDIR_PREFIX, obj_id + '.npz')
            np.savez(outfile, pos=grid_points_pos_sampled, neg=grid_points_neg_sampled,
                     random_part=random_part)
            num_processed += 1
        except NotImplementedError:
            not_processed_all += [obj_id]
            continue

    print('Not processed (not in obj_to_partnetid):', len(not_processed_shape_ids))
    print('Not processed (not in p2s_transforms):', len(not_processed_partnet_ids))
    print('Not processed (not loaded tree):', len(not_processed_trees))
    print('Not processed (empty sdf):', len(not_processed_empty_sdf))
    print('Not processed (another reason):', len(not_processed_all))
