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

collapse_parts = {}
collapse_parts['chair'] = {
    'other': 'chair_seat'
}
collapse_parts['table'] = {
    'bar_stretcher': 'leg',
    'bottom_panel': 'shelf'
}
collapse_parts['storagefurniture'] = {
    'drawer': 'cabinet_door',
    'other': 'cabinet_frame',
    'chest_box': 'cabinet_frame'
}
collapse_parts['bed'] = {
    'other': 'bed_frame_base',
    'bed_frame_horizontal_surface': 'bed_frame_base'
}
collapse_parts['trashcan'] = {
    'container': 'container_box',
    'outside_frame': 'container_box'
}


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


if __name__ == '__main__':

    target_category = 'filecabinet'
    target_category_agg = 'storagefurniture'

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

    SAVEDIR_PREFIX = f'/cluster/daidalos/abokhovkin/DeepSDF_v2/ShapeNetV2_dim256_parts/{target_category_agg}/SdfSamples/ShapeNetV2/02871439'

    rotation_matrix = np.array([[0, 0, -1, 0],
                                [0, 1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1]])

    dist_thr = 0.05 # 0.06
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

            try:
                tree = VoxelisedScanNetAllShapesGNNDataset.load_object(
                    os.path.join(VOXEL_TREES, f'{target_category_agg}_hier', f'{partnet_id}.json'),
                    load_geo=True,
                    geo_fn=os.path.join(VOXEL_TREES, f'{target_category_agg}_geo', f'{partnet_id}.npy'))
            except:
                not_processed_trees += [partnet_id]
                continue

            full_geometry_gt = plot_leaves(tree.depth_first_traversal())
            partnet_vox = load_vox(os.path.join(PARTNET_VOXELIZED, partnet_id, 'full_vox.df'))
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

            mesh_shapenet_path = os.path.join(SHAPENET, name_category_to_id[target_category], obj_id, 'models/model_normalized.obj')
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
            geo_vertices_stacked, shift, scale = bounding_cube_normalization(geo_vertices_stacked)

            part_to_geo = {}
            for child in tree.root.children:
                child_name = child.label
                child_geo = np.squeeze(child.geo.numpy())
                collapsed_name = child_name
                if child_name in collapse_parts[target_category_agg]:
                    collapsed_name = collapse_parts[target_category_agg][child_name]
                if collapsed_name not in part_to_geo:
                    part_to_geo[collapsed_name] = child_geo
                else:
                    part_to_geo[collapsed_name] += child_geo

            if len(part_to_geo) == 1 and 'table_surface' in part_to_geo:
                print('Only one part:', partnet_id)

            for child_name in part_to_geo:
                child_geo = part_to_geo[child_name]

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

                # filter points
                kd_tree = cKDTree(geo_vertices_stacked)
                min_dist_pos, min_idx_pos = kd_tree.query(grid_points_pos[:, :3])
                min_dist_neg, min_idx_neg = kd_tree.query(grid_points_neg[:, :3])

                idx_pos = np.where(np.abs(min_dist_pos) < dist_thr)[0]
                grid_points_pos_sampled = grid_points_pos[idx_pos]

                idx_neg = np.where(np.abs(min_dist_neg) < dist_thr)[0]
                grid_points_neg_sampled = grid_points_neg[idx_neg]

                idx_pos = np.where(np.abs(grid_points_pos_sampled[:, 3]) < 0.07)[0]
                grid_points_pos_sampled = grid_points_pos_sampled[idx_pos]
                idx_neg = np.where(np.abs(grid_points_neg_sampled[:, 3]) < 0.07)[0]
                grid_points_neg_sampled = grid_points_neg_sampled[idx_neg]

                # perform aggressive sampling near surface
                pos_gaussian = gaussian(grid_points_pos_sampled[:, 3], 0, 0.07 / 2)
                random_uniform_samples = np.random.uniform(0, 1, (len(pos_gaussian),))
                samples_mask = np.where(random_uniform_samples < pos_gaussian)[0]
                grid_points_pos_sampled = grid_points_pos_sampled[samples_mask]
                neg_gaussian = gaussian(grid_points_neg_sampled[:, 3], 0, 0.07 / 2)
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
                kd_tree_part = cKDTree(anchor_points)

                num_samples_random = 33000
                random_samples = np.random.uniform(-1.0, 1.0, (num_samples_random, 3))
                min_dist_rand, min_idx_rand = kd_tree_part.query(random_samples)
                above_thr_indices = np.where(min_dist_rand > 0.05)[0]
                random_part = random_samples[above_thr_indices]

                outfile = os.path.join(SAVEDIR, obj_id + '.npz')
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
