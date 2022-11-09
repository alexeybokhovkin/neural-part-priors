import os, sys
import numpy as np
from tqdm import tqdm
from sample import load_sample

if __name__ == '__main__':
    DATADIR = '/cluster_HDD/sorona/adai/data/shapenet_core/shapenetv2_dim256_partial/03337140'
    SAVEDIR = '/cluster/daidalos/abokhovkin/DeepSDF/ShapeNetV2_dim256_partial/SdfSamples/ShapeNetV2/03337140'
    os.makedirs(SAVEDIR, exist_ok=True)
    sdf_filenames = sorted([x for x in os.listdir(DATADIR) if x.endswith('frame0.sdf')])

    for sdf_filename in tqdm(sdf_filenames):

        save_filename = sdf_filename.split('_')[0] + '_' + sdf_filename.split('_')[2] + '.npz'
        outfile = os.path.join(SAVEDIR, save_filename)
        if not os.path.exists(outfile):
            sample = load_sample(os.path.join(DATADIR, sdf_filename))

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

            points_pos = np.hstack([locations_ones_world_pos, sdfs_pos[:, None]]).astype('float32')
            points_neg = np.hstack([locations_ones_world_neg, sdfs_neg[:, None]]).astype('float32')

            np.savez(outfile, pos=points_pos, neg=points_neg)