import os, sys
import json
import numpy as np
import trimesh
from tqdm import tqdm
from scipy.spatial import cKDTree
from copy import deepcopy, copy

sys.path.append('/rhome/abokhovkin/projects/scannet-relationships')
from src.datasets.partnet import VoxelisedScanNetAllShapesGNNDataset
from src.data_utils.vox import load_sample
from src.data_utils.transformations import apply_transform, apply_inverse_transform


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


if __name__ == '__main__':
    GRID_SAMPLES_DIR = '/cluster/pegasus/abokhovkin/DeepSDF/ShapeNetV2_dim256_partial/SdfSamples/ShapeNetV2/03001627'
    DICTIONARIES = '/rhome/abokhovkin/projects/scannet-relationships/dicts'
    with open(os.path.join(DICTIONARIES, 'obj_to_cat.json'), 'r') as fin:
        obj_to_cat = json.load(fin)
    with open(os.path.join(DICTIONARIES, 'parts_to_shapes.json'), 'r') as fin:
        partnetid_to_obj = json.load(fin)
    obj_to_partnetid = {u: v for v, u in partnetid_to_obj.items()}

    PARTNET_VOXELIZED = '/cluster/sorona/abokhovkin/partnet-full-voxelized-32'
    VOXEL_TREES = '/cluster/sorona/abokhovkin/part-segmentation/hierarchies_32_1lvl_filled_v2'
    PARTNET = '/cluster/pegasus/abokhovkin/datasets/PartNet'
    SHAPENET = '/canis/ShapeNet/ShapeNetCore.v2'

    SAVEDIR_PREFIX = '/cluster/pegasus/abokhovkin/DeepSDF/ShapeNetV2_dim256_partial_uniform/SdfSamples/ShapeNetV2/03001627'

    partnet_to_shapenet_transforms_path = os.path.join('/rhome/abokhovkin/projects/scannet-relationships/dicts/chair_table_storagefurniture_bed_shapenetv1_to_partnet_alignment.json')
    with open(partnet_to_shapenet_transforms_path, 'rb') as fin:
        p2s_transforms = json.load(fin)

    all_shape_ids = []
    for x in os.listdir(GRID_SAMPLES_DIR):
        shape_id = x.split('_')[0]
        all_shape_ids += [shape_id]
    all_shape_ids = sorted(list(set(all_shape_ids)))

    rotation_matrix = np.array([[0, 0, -1, 0],
                                [0, 1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1]])

    dist_thr = 0.065
    num_processed = 0

    not_processed_shape_ids = []
    not_processed_partnet_ids = []
    not_processed_all = []

    for i, shape_id in enumerate(tqdm(all_shape_ids)):
        try:
            if shape_id not in obj_to_partnetid:
                not_processed_shape_ids += [shape_id]
                continue
            partnet_id = obj_to_partnetid[shape_id]

            if partnet_id in p2s_transforms:
                p2s_transform = np.array(p2s_transforms[partnet_id]['transmat']).reshape((4, 4), order="C")
            else:
                not_processed_partnet_ids += [partnet_id]
                continue

            tree = VoxelisedScanNetAllShapesGNNDataset.load_object(
                os.path.join(VOXEL_TREES, 'chair_hier', f'{partnet_id}.json'),
                load_geo=True,
                geo_fn=os.path.join(VOXEL_TREES, 'chair_geo', f'{partnet_id}.npy'))
            full_geometry_gt = plot_leaves(tree.depth_first_traversal())
            partnet_vox = load_sample(os.path.join(PARTNET_VOXELIZED, partnet_id, 'full_vox.df'))
            grid2world = partnet_vox.grid2world

            with open(os.path.join(PARTNET, partnet_id, 'result_after_merging.json')) as f:
                parts_name = json.load(f)[0]['objs']
            mesh = trimesh.util.concatenate([
                trimesh.load(os.path.join(PARTNET, partnet_id, 'objs', part_name + '.obj')) for part_name in parts_name
            ])
            # PartNet space
            normalized_mesh = mesh.copy()
            # normalize mesh to [-0.5; 0.5]
            min_1 = deepcopy(normalized_mesh.vertices.min(0))
            normalized_mesh.vertices -= normalized_mesh.vertices.min(0)
            max_1 = deepcopy(normalized_mesh.vertices.max() / 0.95)
            normalized_mesh.vertices /= normalized_mesh.vertices.max() / 0.95
            max_2 = deepcopy(normalized_mesh.vertices.max(0) / 2)
            normalized_mesh.vertices -= normalized_mesh.vertices.max(0) / 2

            mesh_shapenet_path = os.path.join(SHAPENET, '03001627', shape_id, 'models/model_normalized.obj')
            mesh_shapenet = as_mesh(trimesh.load(mesh_shapenet_path))

            root_geo = np.squeeze(tree.root.geo.numpy())
            geo_vertices = np.where(root_geo)
            geo_vertices_stacked = np.vstack([geo_vertices[2], geo_vertices[1], geo_vertices[0]]).T
            geo_vertices_stacked = apply_transform(geo_vertices_stacked, grid2world)
            geo_vertices_stacked += max_2
            geo_vertices_stacked *= max_1
            geo_vertices_stacked += min_1
            geo_vertices_stacked = apply_inverse_transform(geo_vertices_stacked, p2s_transform)
            geo_vertices_stacked = apply_inverse_transform(geo_vertices_stacked, rotation_matrix)
            geo_vertices_stacked_full, shift, scale = bounding_cube_normalization(geo_vertices_stacked)

            num_samples_random = 33000
            random_samples = np.random.uniform(-1.0, 1.0, (num_samples_random, 3))
            kd_tree_full = cKDTree(geo_vertices_stacked_full)
            min_dist_rand, min_idx_rand = kd_tree_full.query(random_samples)
            above_thr_indices = np.where(min_dist_rand > 0.08)[0]
            random_full = random_samples[above_thr_indices]

            for child in tree.root.children:
                child_name = child.label
                child_geo = np.squeeze(child.geo.numpy())

                SAVEDIR = SAVEDIR_PREFIX + '-' + child_name
                os.makedirs(SAVEDIR, exist_ok=True)

                geo_vertices = np.where(child_geo)
                geo_vertices_stacked = np.vstack([geo_vertices[2], geo_vertices[1], geo_vertices[0]]).T
                geo_vertices_stacked = apply_transform(geo_vertices_stacked, grid2world)

                geo_vertices_stacked += max_2
                geo_vertices_stacked *= max_1
                geo_vertices_stacked += min_1

                geo_vertices_stacked = apply_inverse_transform(geo_vertices_stacked, p2s_transform)
                geo_vertices_stacked = apply_inverse_transform(geo_vertices_stacked, rotation_matrix)
                geo_vertices_stacked -= shift
                geo_vertices_stacked /= scale

                kd_tree = cKDTree(geo_vertices_stacked)
                kd_tree_part = cKDTree(geo_vertices_stacked)
                min_dist_rand, min_idx_rand = kd_tree_part.query(random_samples)
                above_thr_indices = np.where(min_dist_rand > 0.08)[0]
                random_part = random_samples[above_thr_indices]

                for k in range(10):
                    grid_points = np.load(os.path.join(GRID_SAMPLES_DIR, shape_id + '_' + str(k) + '.npz'))

                    min_dist_pos, min_idx_pos = kd_tree.query(grid_points['pos'][:, :3])
                    min_dist_neg, min_idx_neg = kd_tree.query(grid_points['neg'][:, :3])

                    idx_pos = np.where(np.abs(min_dist_pos) < dist_thr)[0]
                    grid_points_pos_sampled = grid_points['pos'][idx_pos]

                    grid_points_pos_indices_sampled = np.random.choice(len(grid_points_pos_sampled),
                                                                       min(250000, len(grid_points_pos_sampled)),
                                                                       replace=False)
                    grid_points_pos_sampled = grid_points_pos_sampled[grid_points_pos_indices_sampled]

                    idx_neg = np.where(np.abs(min_dist_neg) < dist_thr)[0]
                    grid_points_neg_sampled = grid_points['neg'][idx_neg]

                    outfile = os.path.join(SAVEDIR, f'{shape_id}_{partnet_id}_{k}.npz')
                    np.savez(outfile, pos=grid_points_pos_sampled, neg=grid_points_neg_sampled,
                             random_full=random_full, random_part=random_part)
                    num_processed += 1
        except:
            not_processed_all += [shape_id]
            continue

    print('Not processed 1:', len(not_processed_shape_ids))
    print('Not processed 2:', len(not_processed_partnet_ids))
    print('Not processed all:', len(not_processed_all))
