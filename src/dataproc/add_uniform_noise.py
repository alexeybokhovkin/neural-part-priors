import os, sys
from tqdm.autonotebook import tqdm
import numpy as np
from scipy.spatial import cKDTree


if __name__ == '__main__':
    FULL_SHAPES = '/cluster/pegasus/abokhovkin/DeepSDF/ShapeNetV2_dim256_parts/SdfSamples/ShapeNetV2'
    SAVE_DIR = '/cluster/pegasus/abokhovkin/DeepSDF/ShapeNetV2_dim256_parts_uniform/SdfSamples/ShapeNetV2'
    os.makedirs(SAVE_DIR, exist_ok=True)

    dist_thr = 0.065

    not_processed_ids = []
    for part_dir in os.listdir(FULL_SHAPES):
        PART_DIR = os.path.join(FULL_SHAPES, part_dir)
        for filename in tqdm(os.listdir(PART_DIR)):
            try:
                shape_id = filename.split('.')[0]
                os.makedirs(os.path.join(SAVE_DIR, part_dir), exist_ok=True)
                outfile = os.path.join(SAVE_DIR, part_dir, shape_id + '.npz')
                # if os.path.exists(outfile):
                #     continue

                grid_points = np.load(os.path.join(PART_DIR, filename))

                # filter points
                idx_pos = np.where(np.abs(grid_points['pos'][:, 3]) < dist_thr)[0]
                grid_points_pos_sampled = grid_points['pos'][idx_pos]

                grid_points_pos_indices_sampled = np.random.choice(len(grid_points_pos_sampled),
                                                                   min(250000, len(grid_points_pos_sampled)),
                                                                   replace=False)
                grid_points_pos_sampled = grid_points_pos_sampled[grid_points_pos_indices_sampled]

                idx_neg = np.where(np.abs(grid_points['neg'][:, 3] < dist_thr))[0]
                grid_points_neg_sampled = grid_points['neg'][idx_neg]

                grid_points_neg_indices_sampled = np.random.choice(len(grid_points_neg_sampled),
                                                                   min(250000, len(grid_points_neg_sampled)),
                                                                   replace=False)
                grid_points_neg_sampled = grid_points_neg_sampled[grid_points_neg_indices_sampled]

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

                np.savez(outfile, pos=grid_points_pos_sampled, neg=grid_points_neg_sampled,
                         random_part=random_part)
            except:
                not_processed_ids += [filename]
                continue

    print('Not processed ids:', len(not_processed_ids))