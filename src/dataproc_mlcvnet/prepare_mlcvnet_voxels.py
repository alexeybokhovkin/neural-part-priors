import os, sys
from tqdm.autonotebook import tqdm
import json
import numpy as np
import pickle
from copy import deepcopy
import trimesh
import gc

sys.path.append('/rhome/abokhovkin/projects/scannet-relationships')
from src.data_utils.transformations import apply_transform, apply_inverse_transform, from_tqs_to_matrix
from src.data_utils.vox import load_sample

if __name__ == '__main__':

    target_category = 'chair'
    target_category_agg = 'chair'
    target_cat_id = '03001627'

    SAVE_VOXELS_DIR = f'/cluster/pegasus/abokhovkin/part-segmentation/hierarchies_32_1lvl_v3/{target_category_agg}_mlcvnet_canonical'
    os.makedirs(SAVE_VOXELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_VOXELS_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(SAVE_VOXELS_DIR, 'val'), exist_ok=True)

    SDF_FILES = f'/cluster_HDD/sorona/abokhovkin/ScanNetGrids/mlcvnet_sdf_256_{target_category_agg}s'
    BOXES_DIR = f'/cluster_HDD/sorona/abokhovkin/ScanNetGrids/MLCVNet_boxes'
    CORRESPONDENCES = f'/cluster_HDD/sorona/abokhovkin/ScanNetGrids/mlcvnet_corr_{target_category_agg}.pkl'
    SCAN2CAD_DIR = '/canis/Datasets/Scan2CAD/public'
    SCANNET_DIR = '/canis/Datasets/ScanNet/public/v2/scans'
    S2C_ANNOTATION = '/rhome/abokhovkin/projects/Scan2CAD/public/full_annotations2.json'
    SCANNET_BENCHMARK = '/cluster/pegasus/abokhovkin/ScanNet/Tasks/Benchmark'

    with open(os.path.join(SCANNET_BENCHMARK, 'scannetv2_train.txt'), 'r') as fin:
        scannet_train = [x[:-1] for x in fin.readlines()]
    with open(os.path.join(SCANNET_BENCHMARK, 'scannetv2_val.txt'), 'r') as fin:
        scannet_val = [x[:-1] for x in fin.readlines()]

    with open(S2C_ANNOTATION, 'r') as fin:
        s2c_annotation = json.load(fin)

    DICTIONARIES = '/rhome/abokhovkin/projects/scannet-relationships/dicts'
    with open(os.path.join(DICTIONARIES, 'obj_to_cat.json'), 'r') as fin:
        obj_to_cat = json.load(fin)
    with open(os.path.join(DICTIONARIES, 'parts_to_shapes.json'), 'r') as fin:
        partnetid_to_obj = json.load(fin)
    obj_to_partnetid = {u: v for v, u in partnetid_to_obj.items()}

    with open(os.path.join(SCAN2CAD_DIR, 'full_annotations.json'), 'rb') as fin:
        anno = json.load(fin)
    scan2cad_anno = {}
    for item in anno:
        scan2cad_anno[item['id_scan']] = item

    with open(CORRESPONDENCES, 'rb') as fin:
        correspondences = pickle.load(fin)

    size = 32

    voxel_transform = np.array([[0.03333334, 0., 0., -0.53125],
                                [0., 0.03333334, 0., -0.53125],
                                [0., 0., 0.03333334, -0.53125],
                                [0., 0., 0., 1.]])
    rotation_matrix = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])

    extra_samples = [('scene0653_01', '39589', '2'),
                     ('scene0653_01', '39589', '1'),
                     ('scene0663_01', '39589', '1'),
                     ('scene0222_01', '36827', '10'),
                     ('scene0403_01', '2644', '6'),
                     ('scene0653_01', '39589', '20'),
                     ('scene0474_03', '39589', '2'),
                     ('scene0653_01', '39589', '19'),
                     ('scene0653_01', '39589', '16'),
                     ('scene0025_01', '39589', '12'),
                     ('scene0653_01', '39589', '18'),
                     ('scene0496_00', '39589', '4'),
                     ('scene0496_00', '39589', '5'),
                     ('scene0257_00', '35737', '10'),
                     ('scene0627_01', '3377', '13'),
                     ('scene0603_00', '43615', '14'),
                     ('scene0569_01', '41055', '17'),
                     ('scene0426_01', '39589', '3'),
                     ('scene0222_01', '36827', '6'),
                     ('scene0653_01', '39589', '17'),
                     ('scene0653_01', '39589', '15')]

    extra_scan_ids = [x[0] for x in extra_samples]

    print('Category:', target_category_agg)
    for sdf_filename in tqdm(os.listdir(SDF_FILES)[:]):

        scan_id = sdf_filename.split('-')[0]
        cat_id = sdf_filename.split('-')[1]
        instance_id = sdf_filename.split('-')[2].split('_')[0]

        if scan_id not in extra_scan_ids:
            continue

        if scan_id in scannet_train:
            SAVE_DIR = os.path.join(SAVE_VOXELS_DIR, 'train')
        else:
            SAVE_DIR = os.path.join(SAVE_VOXELS_DIR, 'test')

        if scan_id not in scannet_val:
            continue

        s2c_scene_annotation = s2c_annotation[scan_id]
        t = s2c_scene_annotation['trs']['translation']
        r = s2c_scene_annotation['trs']['rotation']
        s = s2c_scene_annotation['trs']['scale']
        scan_transform = from_tqs_to_matrix(t, r, s)

        scan_points_original = np.array(
            trimesh.load(os.path.join(SCANNET_DIR, scan_id, scan_id + '_vh_clean_2.ply')).vertices)
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
                rotation_transform = from_tqs_to_matrix(t, r, s)
                flag_found = True
                break
        #     print(flag_found)
        if not flag_found:
            # print(1)
            continue
        if obj_id not in obj_to_partnetid:
            print('PartNet id not found')
            partnet_id = '0000'
        else:
            partnet_id = obj_to_partnetid[obj_id]

        sample_id = (scan_id, partnet_id, str(s2c_instance_id))
        if sample_id not in extra_samples:
            continue
        print(sample_id)

        box_filename = '-'.join([scan_id, target_cat_id, instance_id]) + '.ply'
        box_mesh = trimesh.load(os.path.join(BOXES_DIR, box_filename), process=False)

        try:
            sdf_sample = load_sample(os.path.join(SDF_FILES, sdf_filename))
        except:
            print(sdf_filename, '0 bytes')
            del sdf_sample
            continue
        g2w = sdf_sample.grid2world.T
        w2g = np.linalg.inv(g2w)

        sdf_locations = sdf_sample.locations
        sdf_locations = apply_transform(sdf_locations, w2g)
        sdfs = sdf_sample.sdfs
        indices = np.where(np.abs(sdfs) < 0.02)[0]
        sdf_locations_sampled = sdf_locations[indices]
        sdf_locations_sampled_axis_aligned = apply_transform(sdf_locations_sampled, scan_transform)
        if scan_id in scannet_val:
            sdf_locations_sampled_axis_aligned_unrot = sdf_locations_sampled_axis_aligned
            # sdf_locations_sampled_axis_aligned_unrot = apply_inverse_transform(sdf_locations_sampled_axis_aligned, rotation_transform)
        else:
            sdf_locations_sampled_axis_aligned_unrot = apply_inverse_transform(sdf_locations_sampled_axis_aligned, rotation_transform)

        try:
            sdf_locations_sampled_processed = deepcopy(sdf_locations_sampled_axis_aligned_unrot)
            if len(sdf_locations_sampled_axis_aligned_unrot) < 10:
                print('Few sdf locations')
                continue
            min_1 = deepcopy(sdf_locations_sampled_processed.min(0))
            sdf_locations_sampled_processed -= sdf_locations_sampled_processed.min(0)
            max_1 = deepcopy(sdf_locations_sampled_processed.max() / 0.95)
            sdf_locations_sampled_processed /= sdf_locations_sampled_processed.max() / 0.95
            max_2 = deepcopy(sdf_locations_sampled_processed.max(0) / 2)
            sdf_locations_sampled_processed -= sdf_locations_sampled_processed.max(0) / 2

            sdf_locations_sampled_processed = apply_inverse_transform(sdf_locations_sampled_processed, rotation_matrix)
            instance_grid = apply_inverse_transform(sdf_locations_sampled_processed, voxel_transform)

            if np.max(instance_grid) > 100 or np.min(instance_grid) < -100:
                print('Invalid limits')
                continue

            instance_grid_int = instance_grid.astype('int')
            instance_grid_int = np.maximum(0, np.minimum(size - 1, instance_grid_int))
            voxel_points = np.zeros((size, size, size)).astype('uint8')
            voxel_points[instance_grid_int[:, 2], instance_grid_int[:, 1], instance_grid_int[:, 0]] = 1
        except FileNotFoundError:
            print('Bad execution:', sdf_filename)
            continue

        shape_transform = {}
        shape_transform['t'] = t
        shape_transform['r'] = r
        shape_transform['s'] = s
        shape_transform['min_1'] = list(min_1)
        shape_transform['max_1'] = max_1
        shape_transform['max_2'] = list(max_2)

        print('Saved:', f'{partnet_id}_{scan_id}_{s2c_instance_id}.npy')
        np.save(os.path.join(SAVE_DIR, f'{partnet_id}_{scan_id}_{s2c_instance_id}.npy'), voxel_points)
        with open(os.path.join(SAVE_DIR, f'{partnet_id}_{scan_id}_{s2c_instance_id}.json'), 'w') as fout:
            json.dump(shape_transform, fout)

        gc.collect()

