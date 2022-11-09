import os, sys
from tqdm.autonotebook import tqdm
import json
import numpy as np
import pickle
from copy import deepcopy
import trimesh
from scipy.spatial import cKDTree

sys.path.append('/rhome/abokhovkin/projects/scannet-relationships')
from src.data_utils.transformations import apply_transform, apply_inverse_transform, from_tqs_to_matrix
from src.data_utils.vox import load_sample, load_vox
from src.datasets.partnet import VoxelisedScanNetAllShapesGNNDataset


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


def plot_leaves(all_nodes, thr=0.5):
    leaves = [x for x in all_nodes if len(x.children) == 0]
    full_geometry = np.zeros_like(leaves[0].geo.cpu().numpy()[0])
    for i, leaf in enumerate(leaves):
        color = i + 1
        geometry = leaf.geo.cpu().numpy()[0]
        full_geometry[np.where(geometry > thr)] = color
    full_geometry = full_geometry.astype('uint8')
    return full_geometry


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
    'chest_box': 'cabinet_frame',
    'back_panel': 'cabinet_frame',
    'bottom_panel': 'cabinet_frame',
    'top_panel': 'cabinet_frame',
    'frame_vertical_bar': 'cabinet_frame',
    'frame_horizontal_bar': 'cabinet_frame',
    'vertical_front_panel': 'cabinet_frame',
    'vertical_divider_panel': 'cabinet_frame'
}
collapse_parts['bed'] = {
    'other': 'bed_frame_base',
    'bed_frame_horizontal_surface': 'bed_frame_base'
}
collapse_parts['trashcan'] = {
    'container': 'container_box',
    'outside_frame': 'container_box'
}


if __name__ == '__main__':

    target_category = 'trashcan'
    target_category_agg = 'trashcan'
    target_cat_id = '02747177'

    SAVEDIR_PREFIX = '/cluster/daidalos/abokhovkin/DeepSDF_v2/ShapeNetV2_dim256_parts_mlcvnet/'

    SDF_FILES = f'/cluster_HDD/sorona/abokhovkin/ScanNetGrids/mlcvnet_sdf_256_{target_category_agg}s'
    BOXES_DIR = f'/cluster_HDD/sorona/abokhovkin/ScanNetGrids/MLCVNet_boxes_{target_category_agg}s'
    CORRESPONDENCES = f'/cluster_HDD/sorona/abokhovkin/ScanNetGrids/mlcvnet_corr_{target_category_agg}.pkl'
    SCAN2CAD_DIR = '/canis/Datasets/Scan2CAD/public'
    VOXEL_TREES = '/cluster/pegasus/abokhovkin/part-segmentation/hierarchies_32_1lvl_v3'
    PARTNET_VOXELIZED = '/cluster/sorona/abokhovkin/partnet-full-voxelized-32'
    PARTNET = '/cluster/pegasus/abokhovkin/datasets/PartNet'
    SCANNET_DIR = '/canis/Datasets/ScanNet/public/v2/scans'
    SCANNET_BENCHMARK = '/cluster/pegasus/abokhovkin/ScanNet/Tasks/Benchmark'

    with open(os.path.join(SCANNET_BENCHMARK, 'scannetv2_train.txt'), 'r') as fin:
        scannet_train = [x[:-1] for x in fin.readlines()]
    with open(os.path.join(SCANNET_BENCHMARK, 'scannetv2_val.txt'), 'r') as fin:
        scannet_val = [x[:-1] for x in fin.readlines()]

    DICTIONARIES = '/rhome/abokhovkin/projects/scannet-relationships/dicts'
    with open(os.path.join(DICTIONARIES, 'obj_to_cat.json'), 'r') as fin:
        obj_to_cat = json.load(fin)
    with open(os.path.join(DICTIONARIES, 'parts_to_shapes.json'), 'r') as fin:
        partnetid_to_obj = json.load(fin)
    obj_to_partnetid = {u: v for v, u in partnetid_to_obj.items()}
    SCANNET_RELATIONSHIPS_PATH = '/rhome/abokhovkin/projects/scannet-relationships/'
    partnet_to_shapenet_transforms_path = os.path.join(SCANNET_RELATIONSHIPS_PATH, 'dicts/partnet_to_shapenet_transforms.pkl')
    with open(partnet_to_shapenet_transforms_path, 'rb') as fin:
        p2s_transforms = pickle.load(fin)

    with open(os.path.join(SCAN2CAD_DIR, 'full_annotations.json'), 'rb') as fin:
        anno = json.load(fin)
    scan2cad_anno = {}
    for item in anno:
        scan2cad_anno[item['id_scan']] = item

    with open(CORRESPONDENCES, 'rb') as fin:
        correspondences = pickle.load(fin)

    size = 32

    voxel_transform = np.array([[ 0.03333334,  0.        ,  0.        , -0.53125   ],
                                [ 0.        ,  0.03333334,  0.        , -0.53125   ],
                                [ 0.        ,  0.        ,  0.03333334, -0.53125   ],
                                [ 0.        ,  0.        ,  0.        ,  1.        ]])
    rotation_matrix = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    dist_thr = 0.07

    not_processed_bytes = []
    not_processed_partnet = []
    not_processed_flag = []
    not_processed_tree = []
    not_processed_p2s = []
    not_processed_ids = []
    print('Category:', target_category_agg)
    for sdf_filename in tqdm(os.listdir(SDF_FILES)[:]):

        try:
            scan_id = sdf_filename.split('-')[0]
            cat_id = sdf_filename.split('-')[1]
            instance_id = sdf_filename.split('-')[2].split('_')[0]

            s2c_scene_annotation = scan2cad_anno[scan_id]
            t = s2c_scene_annotation['trs']['translation']
            r = s2c_scene_annotation['trs']['rotation']
            s = s2c_scene_annotation['trs']['scale']
            scan_transform = from_tqs_to_matrix(t, r, s)

            scan_points_original = np.array(trimesh.load(os.path.join(SCANNET_DIR, scan_id, scan_id + '_vh_clean_2.ply')).vertices)
            scan_points_axis_aligned = apply_transform(scan_points_original, scan_transform)

            scene_correspondences = correspondences[scan_id]
            flag_found = False
            for corr in scene_correspondences:
                if corr[1] == int(instance_id):
                    s2c_instance_id = corr[0]
                    aligned_shape_data = s2c_scene_annotation['aligned_models'][corr[0]]
                    obj_id = aligned_shape_data['id_cad']
                    t = aligned_shape_data['trs']['translation']
                    r = aligned_shape_data['trs']['rotation']
                    s = aligned_shape_data['trs']['scale']
                    align_transform = from_tqs_to_matrix(t, r, s)

                    rotation_transform = from_tqs_to_matrix([0, 0, 0], r, [1, 1, 1])
                    flag_found = True
                    break
            #         print(flag_found)
            if not flag_found:
                not_processed_flag += [sdf_filename]
                continue
            if obj_id not in obj_to_partnetid:
                not_processed_partnet += [sdf_filename]
                continue
            else:
                partnet_id = obj_to_partnetid[obj_id]

            box_filename = '-'.join([scan_id, target_cat_id, instance_id]) + '.ply'
            box_mesh = trimesh.load(os.path.join(BOXES_DIR, box_filename), process=False)

            try:
                sdf_sample = load_sample(os.path.join(SDF_FILES, sdf_filename))
            except:
                not_processed_bytes += [sdf_filename]
                continue
            g2w = sdf_sample.grid2world.T
            w2g = np.linalg.inv(g2w)

            sdf_locations = sdf_sample.locations
            sdf_locations = apply_transform(sdf_locations, w2g)
            sdfs = sdf_sample.sdfs
            sdf_locations_axis_aligned = apply_transform(sdf_locations, scan_transform)
            sdf_locations_axis_aligned_unrot = apply_inverse_transform(sdf_locations_axis_aligned, align_transform)
            if scan_id in scannet_val:
                sdf_locations_axis_aligned_shapenet, shift_p, scale_p = bounding_cube_normalization \
                    (sdf_locations_axis_aligned)
                sdf_locations_axis_aligned_shapenet = np.hstack \
                    ([sdf_locations_axis_aligned_shapenet, sdf_sample.sdfs[:, None]])

                indices = np.where(sdf_locations_axis_aligned_shapenet[:, 3] >= 0)[0]
                sdf_locations_axis_aligned_shapenet_pos = sdf_locations_axis_aligned_shapenet[indices]
                indices = np.where(sdf_locations_axis_aligned_shapenet[:, 3] < 0)[0]
                sdf_locations_axis_aligned_shapenet_neg = sdf_locations_axis_aligned_shapenet[indices]

                indices = np.random.choice(len(sdf_locations_axis_aligned_shapenet_pos),
                                           min(800000, len(sdf_locations_axis_aligned_shapenet_pos)),
                                           replace=False)
                sdf_locations_axis_aligned_shapenet_pos = sdf_locations_axis_aligned_shapenet_pos[indices]
                sdf_locations_axis_aligned_shapenet_pos[:, 3] = sdf_locations_axis_aligned_shapenet_pos[:, 3] * 0.4
                indices = np.random.choice(len(sdf_locations_axis_aligned_shapenet_neg),
                                           min(800000, len(sdf_locations_axis_aligned_shapenet_neg)),
                                           replace=False)
                sdf_locations_axis_aligned_shapenet_neg = sdf_locations_axis_aligned_shapenet_neg[indices]
                sdf_locations_axis_aligned_shapenet_neg[:, 3] = sdf_locations_axis_aligned_shapenet_neg[:, 3] * 0.4

                sdf_locations_axis_aligned_shapenet_pos = sdf_locations_axis_aligned_shapenet_pos.astype('float32')
                sdf_locations_axis_aligned_shapenet_neg = sdf_locations_axis_aligned_shapenet_neg.astype('float32')

                indices = np.where(sdf_locations_axis_aligned_shapenet_pos[:, 3] <= 0.071)[0]
                sdf_locations_axis_aligned_shapenet_pos = sdf_locations_axis_aligned_shapenet_pos[indices]
                indices = np.where(sdf_locations_axis_aligned_shapenet_neg[:, 3] >= -0.071)[0]
                sdf_locations_axis_aligned_shapenet_neg = sdf_locations_axis_aligned_shapenet_neg[indices]

                meta_data = {}
                meta_data['t'] = t
                meta_data['r'] = r
                meta_data['s'] = s
                meta_data['shift'] = list(shift_p)
                meta_data['scale'] = scale_p

                SAVEDIR = os.path.join(SAVEDIR_PREFIX, 'test', f'SdfSamples/ShapeNetV2/{target_cat_id}')
                os.makedirs(SAVEDIR, exist_ok=True)

                outfile = os.path.join(SAVEDIR, f'{scan_id}_{obj_id}_{partnet_id}_{s2c_instance_id}' + '.npz')
                np.savez(outfile,
                         pos=sdf_locations_axis_aligned_shapenet_pos,
                         neg=sdf_locations_axis_aligned_shapenet_neg)
                with open(os.path.join(SAVEDIR, f'{scan_id}_{obj_id}_{partnet_id}_{s2c_instance_id}.json'), 'w') as fout:
                    json.dump(meta_data, fout)

            try:
                tree = VoxelisedScanNetAllShapesGNNDataset.load_object(os.path.join(VOXEL_TREES, f'{target_category_agg}_hier', f'{partnet_id}.json'),
                                                                       load_geo=True,
                                                                       geo_fn=os.path.join(VOXEL_TREES, f'{target_category_agg}_geo', f'{partnet_id}.npy'))
            except:
                not_processed_tree += [sdf_filename]
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

            try:
                p2s_transform = np.array(p2s_transforms[partnet_id])
            except:
                not_processed_p2s += [sdf_filename]
                continue

            root_geo = np.squeeze(tree.root.geo.numpy())
            geo_vertices = np.where(root_geo)
            geo_vertices_stacked = np.vstack([geo_vertices[2], geo_vertices[1], geo_vertices[0]]).T
            geo_vertices_stacked = apply_transform(geo_vertices_stacked, grid2world)
            geo_vertices_stacked += max_2
            geo_vertices_stacked *= max_1
            geo_vertices_stacked += min_1
            geo_vertices_stacked = apply_transform(geo_vertices_stacked, p2s_transform)
            geo_vertices_stacked, shift, scale = bounding_cube_normalization(geo_vertices_stacked)

            pos_entries = np.where(sdf_sample.sdfs >= 0)[0]
            pos_sdf = sdf_sample.sdfs[pos_entries] / scale
            pos_points = sdf_locations_axis_aligned_unrot[pos_entries]
            pos_points -= shift
            pos_points /= scale
            pos_samples = np.hstack([pos_points, pos_sdf[:, None]])
            neg_entries = np.where(sdf_sample.sdfs < 0)[0]
            neg_sdf = sdf_sample.sdfs[neg_entries] / scale
            neg_points = sdf_locations_axis_aligned_unrot[neg_entries]
            neg_points -= shift
            neg_points /= scale
            neg_samples = np.hstack([neg_points, neg_sdf[:, None]])

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

            all_child_geos = []
            all_grid_points_pos_sampled = []
            for child_name in part_to_geo:
                child_geo = part_to_geo[child_name]

                geo_vertices = np.where(child_geo)
                geo_vertices_stacked = np.vstack([geo_vertices[2], geo_vertices[1], geo_vertices[0]]).T
                geo_vertices_stacked = apply_transform(geo_vertices_stacked, grid2world)

                geo_vertices_stacked += max_2
                geo_vertices_stacked *= max_1
                geo_vertices_stacked += min_1

                geo_vertices_stacked = apply_transform(geo_vertices_stacked, p2s_transform)
                geo_vertices_stacked -= shift
                geo_vertices_stacked /= scale

                all_child_geos += [geo_vertices_stacked]

                kd_tree = cKDTree(geo_vertices_stacked)
                min_dist_pos, min_idx_pos = kd_tree.query(pos_points)
                min_dist_neg, min_idx_neg = kd_tree.query(neg_points)

                idx_pos = np.where(np.abs(min_dist_pos) < dist_thr)[0]
                grid_points_pos_sampled = pos_samples[idx_pos]
                grid_points_pos_indices_sampled = np.random.choice(len(grid_points_pos_sampled),
                                                                   min(300000, len(grid_points_pos_sampled)),
                                                                   replace=False)
                grid_points_pos_sampled = grid_points_pos_sampled[grid_points_pos_indices_sampled]
                grid_points_pos_sampled[:, 3] = grid_points_pos_sampled[:, 3] * 0.2

                idx_neg = np.where(np.abs(min_dist_neg) < dist_thr)[0]
                grid_points_neg_sampled = neg_samples[idx_neg]
                grid_points_neg_sampled[:, 3] = grid_points_neg_sampled[:, 3] * 0.2

                grid_points_pos_sampled = grid_points_pos_sampled.astype('float32')
                grid_points_neg_sampled = grid_points_neg_sampled.astype('float32')

                indices = np.where(grid_points_pos_sampled[:, 3] <= 0.071)[0]
                grid_points_pos_sampled = grid_points_pos_sampled[indices]
                indices = np.where(grid_points_neg_sampled[:, 3] >= -0.071)[0]
                grid_points_neg_sampled = grid_points_neg_sampled[indices]

                all_grid_points_pos_sampled += [grid_points_pos_sampled]

                if scan_id in scannet_val:
                    SAVEDIR = os.path.join(SAVEDIR_PREFIX, 'val', f'SdfSamples/ShapeNetV2/{target_cat_id}-' + child_name)
                else:
                    SAVEDIR = os.path.join(SAVEDIR_PREFIX, 'train', f'SdfSamples/ShapeNetV2/{target_cat_id}-' + child_name)
                os.makedirs(SAVEDIR, exist_ok=True)
                outfile = os.path.join(SAVEDIR, f'{scan_id}_{obj_id}_{partnet_id}_{s2c_instance_id}' + '.npz')
                np.savez(outfile, pos=grid_points_pos_sampled, neg=grid_points_neg_sampled)

            all_grid_points_pos_sampled = np.vstack(all_grid_points_pos_sampled)

            all_child_geos = np.vstack(all_child_geos)
            kd_tree = cKDTree(all_child_geos)
            min_dist_pos, min_idx_pos = kd_tree.query(pos_points)
            min_dist_neg, min_idx_neg = kd_tree.query(neg_points)

            idx_pos = np.where(np.abs(min_dist_pos) > 0.15)[0]
            grid_points_pos_back = pos_samples[idx_pos]
            grid_points_pos_back[:, 3] = grid_points_pos_back[:, 3] * 0.2

            idx_neg = np.where(np.abs(min_dist_neg) > 0.15)[0]
            grid_points_neg_back = neg_samples[idx_neg]
            grid_points_neg_back[:, 3] = grid_points_neg_back[:, 3] * 0.2

            indices = np.where(grid_points_pos_back[:, 3] <= 0.071)[0]
            grid_points_pos_back = grid_points_pos_back[indices]
            indices = np.where(grid_points_neg_back[:, 3] >= -0.071)[0]
            grid_points_neg_back = grid_points_neg_back[indices]

            num_samples_random = 30000
            random_samples = np.random.uniform(-1.0, 1.0, (num_samples_random, 3))
            min_dist_rand, min_idx_rand = kd_tree.query(random_samples)
            above_thr_indices = np.where(min_dist_rand > 0.18)[0]
            above_thr_points = random_samples[above_thr_indices]
            above_thr_points = np.hstack([above_thr_points, 0.07*np.ones((len(above_thr_points), 1))]).astype('float32')
            grid_points_pos_back = np.vstack([above_thr_points, grid_points_pos_back])

            meta_data = {}
            meta_data['t'] = t
            meta_data['r'] = r
            meta_data['s'] = s
            meta_data['shift'] = list(shift)
            meta_data['scale'] = scale

            if scan_id in scannet_val:
                SAVEDIR = os.path.join(SAVEDIR_PREFIX, 'val', f'SdfSamples/ShapeNetV2/{target_cat_id}-background')
            else:
                SAVEDIR = os.path.join(SAVEDIR_PREFIX, 'train', f'SdfSamples/ShapeNetV2/{target_cat_id}-background')
            os.makedirs(SAVEDIR, exist_ok=True)
            outfile = os.path.join(SAVEDIR, f'{scan_id}_{obj_id}_{partnet_id}_{s2c_instance_id}' + '.npz')
            np.savez(outfile, pos=grid_points_pos_back, neg=grid_points_neg_back)
            with open(os.path.join(SAVEDIR, f'{scan_id}_{obj_id}_{partnet_id}_{s2c_instance_id}.json'), 'w') as fout:
                json.dump(meta_data, fout)
        except:
            not_processed_ids += [sdf_filename]

    print('Not processed (not in obj_to_partnetid):', len(not_processed_partnet))
    print('Not processed (not in p2s_transforms):', len(not_processed_p2s))
    print('Not processed (non-existing trees):', len(not_processed_tree))
    print('Not processed (false flag):', len(not_processed_flag))
    print('Not processed (0 bytes):', len(not_processed_bytes))
    print('Not processed (another reason):', len(not_processed_ids))
    print()
    print('Not processed (another reason):', not_processed_ids)