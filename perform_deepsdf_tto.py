import os
import gc

import torch
import numpy as np
import argparse

from src.lightning_models.gnn_deepsdf import GNNPartnetLightning
from src.datasets.partnet import VoxelisedImplicitScanNetDataset
from src.utils.transformations import convert_sdf_samples_to_ply
from src.utils.config import load_config, _load_config_yaml


def main(args):

    config = _load_config_yaml(args.config)
    config = argparse.Namespace(**config)
    
    print('Cat name:', config.cat_name)

    train_dataset = VoxelisedImplicitScanNetDataset(datadir=config.datadir_trees,
                                                    dataset=config.dataset,
                                                    object_list='test.txt',
                                                    data_features=['object'],
                                                    load_geo=True,
                                                    num_subsample_points=config.num_subsample_points, 
                                                    sdf_data_source=config.sdf_data_source,
                                                    cat_name=config.cat_name,
                                                    eval_mode=False,
                                                    full_shape_list_path=config.full_shape_list_path, 
                                                    parts_list_path=config.parts_list_path,
                                                    data_mode=config.data_mode, 
                                                    partnet_noise_dir=config.partnet_noise_dir, 
                                                    mlcvnet_noise_dir=config.mlcvnet_noise_dir,
                                                    partnet_noise_dummy=config.partnet_noise_dummy)

    with open(os.path.join(config.datadir_trees, config.dataset, 'test.txt'), 'r') as fin:
        test_samples = fin.readlines()
        test_samples = [x[:-1] for x in test_samples]

    device = torch.device('cuda:0')

    level_inference = 0.01
    level_tto = 0.005

    for i in range(0, len(train_dataset)):
    # for i in range(0, 1):

        model = GNNPartnetLightning(config, cat_name=config.cat_name, mode='tto')

        model.to(device)
        model.eval()
        model.freeze()

        print('Object:', i)
        print('Only align:', args.only_align)

        batch = list(train_dataset[i])

        orig_scan_geo = batch[0].cpu().numpy().astype('uint8')
        rotated_scan_geo_torch = torch.FloatTensor(orig_scan_geo)[None, ...]

        scan_geo = rotated_scan_geo_torch.to(device)
        batch[0] = (scan_geo,)
        shape = batch[1][0].to(device)
        batch[1] = (shape,)
        batch[3] = (batch[3],)
        batch[5] = (batch[5],)
        parts_indices = batch[7]
        shape_idx = batch[8]

        SAVE_DIR = os.path.join(config.save_basedir, test_samples[i])
        os.makedirs(SAVE_DIR, exist_ok=True)

        output = model.tto_two_stage(batch, only_align=args.only_align,
                                     constr_mode=args.constr_mode, cat_name=config.cat_name,
                                     scale=args.scale, wconf=args.wconf,
                                     w_full_noise=args.w_full_noise, w_part_u_noise=args.w_part_u_noise,
                                     w_part_part_noise=args.w_part_part_noise, lr_dec_full=args.lr_dec_full,
                                     lr_dec_part=args.lr_dec_part, parts_indices=parts_indices, shape_idx=shape_idx,
                                     store_dir=os.path.join(config.save_basedir, test_samples[i]))


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
                        except:
                            print('Bad part after TTO:', test_samples[i])
                            continue

                    np.save(os.path.join(SAVE_DIR, part_name + '.pts.npy'), output[5][0][part_name][0])
                    np.save(os.path.join(SAVE_DIR, part_name + '.sdf.npy'), output[5][0][part_name][2])
                    np.save(os.path.join(SAVE_DIR, part_name + '.loss.' + str(0) + '.npy'), output[5][0][part_name][1])
                    np.save(os.path.join(SAVE_DIR, part_name + '.loss.' + str(1) + '.npy'), output[5][1][part_name][1])
                    if j == 0:
                        np.save(os.path.join(SAVE_DIR, 'full.pts.npy'), output[5][0][part_name][3])
                except:
                    print(part_name, k)

            SAVE_DIR = os.path.join(config.save_basedir, test_samples[i] + '_tto')
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
                    except:
                        print('Bad full shape after TTO:', test_samples[i])

        del output
        gc.collect()


if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()
    # data params
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

    parser.add_argument('--saveid', required=True, default=1, type=int)

    parser.add_argument('--config', required=True, type=str)

    args = parser.parse_args()

    main(args)
