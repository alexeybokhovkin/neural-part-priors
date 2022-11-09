import os
import pickle

import torch
import torch.nn as nn

from ..buildingblocks import FeatureVector, ConvEncoder, ConvDecoder
# from .gnn import RecursiveDecoder
from .gnn_contrast import RecursiveDecoder
from .hier_decoder_deepsdf import RecursiveDeepSDFDecoder


class GeoEncoderProj(nn.Module):
    def __init__(self, enc_in_f_maps=None, enc_out_f_maps=None,
                 layer_order='crg', num_groups=8, enc_strides=1, enc_paddings=1, enc_number_of_fmaps=5,
                 enc_conv_kernel_sizes=3, num_convs_per_block=1, last_pooling_size=1,
                 device='gpu', recursive_feat_size=256, recursive_hidden_size=256, variational=False,
                 last_pooling=True, layer_orders=None, **kwargs):

        super(GeoEncoderProj, self).__init__()

        self.device = device
        self.last_pooling = last_pooling

        if not isinstance(enc_conv_kernel_sizes, list):
            enc_conv_kernel_sizes = [enc_conv_kernel_sizes] * enc_number_of_fmaps
        if not isinstance(enc_strides, list):
            enc_strides = [enc_strides] * enc_number_of_fmaps
        if not isinstance(enc_paddings, list):
            enc_paddings = [enc_paddings] * enc_number_of_fmaps

        self.voxel_encoder = ConvEncoder(enc_in_f_maps, enc_out_f_maps, layer_order, num_groups,
                                         enc_strides, enc_paddings, enc_conv_kernel_sizes, num_convs_per_block,
                                         layer_orders)
        self.shape_feature_vector = FeatureVector(last_pooling_size, pooling=True)

        self.mlp = nn.ModuleList([nn.Linear(256, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 256)
                                  ])

        self.mseLoss = nn.MSELoss()

    def forward(self, x_foreground):
        x_root, features = self.voxel_encoder(x_foreground)
        if self.last_pooling:
            x_root = self.shape_feature_vector(x_root)
        for layer in self.mlp:
            x_root = layer(x_root)
        return x_root, features

    def encode_loss(self, x_root, gt_root):
        return self.mseLoss(x_root, gt_root)


class GeoEncoder(nn.Module):
    def __init__(self, enc_in_f_maps=None, enc_out_f_maps=None,
                 layer_order='crg', num_groups=8, enc_strides=1, enc_paddings=1, enc_number_of_fmaps=5,
                 enc_conv_kernel_sizes=3, num_convs_per_block=1, last_pooling_size=1,
                 device='gpu', recursive_feat_size=256, recursive_hidden_size=256, variational=False,
                 last_pooling=True, layer_orders=None, **kwargs):

        super(GeoEncoder, self).__init__()

        self.device = device
        self.last_pooling = last_pooling

        if not isinstance(enc_conv_kernel_sizes, list):
            enc_conv_kernel_sizes = [enc_conv_kernel_sizes] * enc_number_of_fmaps
        if not isinstance(enc_strides, list):
            enc_strides = [enc_strides] * enc_number_of_fmaps
        if not isinstance(enc_paddings, list):
            enc_paddings = [enc_paddings] * enc_number_of_fmaps

        self.voxel_encoder = ConvEncoder(enc_in_f_maps, enc_out_f_maps, layer_order, num_groups,
                                         enc_strides, enc_paddings, enc_conv_kernel_sizes, num_convs_per_block,
                                         layer_orders)
        self.shape_feature_vector = FeatureVector(last_pooling_size, pooling=True)

        self.mseLoss = nn.MSELoss()

    def root_latents(self, x_foreground):
        x_root, features = self.voxel_encoder(x_foreground)
        if self.last_pooling:
            x_root = self.shape_feature_vector(x_root)
        return x_root, features

    def encode_loss(self, x_root, gt_root):
        return self.mseLoss(x_root, gt_root)


class GeoDecoder(nn.Module):

    def __init__(self, dec_in_f_maps=None, dec_out_f_maps=None,
                 layer_order='crg', num_groups=8, dec_strides=1, dec_paddings=1,
                 dec_number_of_fmaps=6, dec_conv_kernel_sizes=3,
                 num_convs_per_block=1, scale_factors=None, output_paddings=None, joins=None,
                 device='gpu', **kwargs):

        super(GeoDecoder, self).__init__()

        self.device = device

        if not isinstance(dec_conv_kernel_sizes, list):
            dec_conv_kernel_sizes = [dec_conv_kernel_sizes] * dec_number_of_fmaps
        if not isinstance(dec_strides, list):
            dec_strides = [dec_strides] * dec_number_of_fmaps
        if not isinstance(dec_paddings, list):
            dec_paddings = [dec_paddings] * dec_number_of_fmaps

        self.node_decoder = ConvDecoder(dec_in_f_maps, dec_out_f_maps, num_convs_per_block, layer_order, num_groups,
                                        scale_factors, dec_conv_kernel_sizes, dec_strides, dec_paddings,
                                        output_paddings, joins)

        self.voxelLoss = nn.BCELoss()
        self.mseLoss = nn.MSELoss()

    def forward(self, x_root, encoder_feature=None):
        x_root = x_root[..., None, None, None]
        pred_masks = self.node_decoder(x_root, encoder_feature)

        return pred_masks

    def loss(self, pred, gt):
        pred = torch.sigmoid(pred)
        loss = self.voxelLoss(pred, gt)
        avg_loss = loss.mean()

        return avg_loss

    def mse_loss(self, pred, gt):
        loss = self.mseLoss(pred, gt)
        avg_loss = loss.mean()

        return avg_loss


class HierarchicalDecoder(nn.Module):
    def __init__(self, dec_in_f_maps=None, dec_out_f_maps=None, layer_order='crg', num_groups=8,
                 dec_strides=1, dec_paddings=1, dec_number_of_fmaps=6, dec_conv_kernel_sizes=3,
                 num_convs_per_block=1, scale_factors=None, enc_hier=False,
                 recursive_feat_size=256, geo_feature_size=256, recursive_hidden_size=256,
                 max_depth=10, max_child_num=10, device='gpu', edge_symmetric_type='avg', num_iterations=0,
                 edge_type_num=0, split_subnetworks=False, loss_children=False, split_enc_children=False,
                 encode_mask=False, shape_priors=False, priors_path=None, enc_in_f_maps=None, enc_out_f_maps=None,
                 enc_strides=1, enc_paddings=1, enc_conv_kernel_sizes=3, last_pooling_size=1, enc_number_of_fmaps=5,
                 parts_to_ids_path=None, base=None, output_paddings=None, joins=None,
                 **kwargs):

        super(HierarchicalDecoder, self).__init__()

        self.max_depth = max_depth

        if not isinstance(dec_conv_kernel_sizes, list):
            dec_conv_kernel_sizes = [dec_conv_kernel_sizes] * dec_number_of_fmaps
        if not isinstance(dec_strides, list):
            dec_strides = [dec_strides] * dec_number_of_fmaps
        if not isinstance(dec_paddings, list):
            dec_paddings = [dec_paddings] * dec_number_of_fmaps
        if not isinstance(enc_conv_kernel_sizes, list):
            enc_conv_kernel_sizes = [enc_conv_kernel_sizes] * enc_number_of_fmaps
        if not isinstance(enc_strides, list):
            enc_strides = [enc_strides] * enc_number_of_fmaps
        if not isinstance(enc_paddings, list):
            enc_paddings = [enc_paddings] * enc_number_of_fmaps

        with open(os.path.join(parts_to_ids_path), 'rb') as f:
            parts_dict = pickle.load(f)

        self.recursive_decoder = RecursiveDecoder(recursive_feat_size, geo_feature_size, recursive_hidden_size,
                                                  max_child_num, dec_in_f_maps, dec_out_f_maps, num_convs_per_block,
                                                  layer_order, num_groups, scale_factors, dec_conv_kernel_sizes,
                                                  dec_strides, dec_paddings, device, edge_symmetric_type,
                                                  num_iterations, edge_type_num, enc_hier, split_subnetworks,
                                                  loss_children, split_enc_children, encode_mask, shape_priors,
                                                  priors_path, parts_dict, enc_in_f_maps, enc_out_f_maps,
                                                  enc_strides, enc_paddings,
                                                  enc_conv_kernel_sizes, last_pooling_size,
                                                  base=base, output_paddings=output_paddings, joins=joins)

        self.mseLoss = nn.MSELoss()

    def forward(self, x_root, mask_code=None, mask_feature=None, full_label=None, encoder_features=None,
                rotation=None, gt_tree=None, scan_geo=None):
        output = self.recursive_decoder.decode_structure(x_root, self.max_depth, mask_code,
                                                         mask_feature,
                                                         full_label=full_label,
                                                         encoder_features=encoder_features,
                                                         rotation=rotation,
                                                         gt_tree=gt_tree,
                                                         scan_geo=scan_geo)
        return output

    def inference_from_latents(self, root_latent, leaf_latents, mask_code=None, mask_feature=None,
                               children_names=None):
        output = self.recursive_decoder.decode_from_latents(root_latent, leaf_latents, mask_code, mask_feature,
                                                            children_names)
        return output

    def structure_recon_loss(self, x_root, gt_tree, mask_code=None, mask_feature=None,
                             encoder_features=None, rotation=None, scan_geo=None):
        return self.recursive_decoder.structure_recon_loss(x_root, gt_tree, mask_code=mask_code,
                                                           mask_feature=mask_feature,
                                                           encoder_features=encoder_features,
                                                           rotation=rotation, scan_geo=scan_geo)

    def tto_loss(self, x_root, gt_tree, mask_code=None, mask_feature=None, scan_geo=None,
                       encoder_features=None, rotation=None, scan_sdf=None, predicted_tree=None):
        return self.recursive_decoder.tto_recon_loss(x_root, gt_tree, mask_code=mask_code,
                                                     mask_feature=mask_feature,
                                                     scan_geo=scan_geo, encoder_features=encoder_features,
                                                     rotation=rotation,
                                                     scan_sdf=scan_sdf,
                                                     predicted_tree=predicted_tree)

    def tto_leaves_loss(self, x_root_learnable, child_feats_learnable, mask_code=None,
                        mask_feature=None, scan_geo=None, scan_sdf=None,
                        predicted_tree=None):
        return self.recursive_decoder.tto_leaves_recon_loss(x_root_learnable, child_feats_learnable, mask_code,
                        mask_feature, scan_geo, scan_sdf, predicted_tree)

    def children_mse_loss(self, pred_children, gt_children):
        total_loss = 0
        for i in range(len(pred_children)):
            total_loss += self.mseLoss(pred_children[i], gt_children[i])
        avg_loss = total_loss.mean()

        return avg_loss

    def children_geo_sum_loss(self, pred_geos, gt_geos):
        total_loss = 0
        for i in range(len(pred_geos)):
            total_loss += self.mseLoss(torch.sum(pred_geos[i]), torch.sum(gt_geos[i]))
        avg_loss = total_loss.mean()

        return avg_loss


class HierarchicalDeepSDFDecoder(nn.Module):
    def __init__(self, recursive_feat_size=128, recursive_hidden_size=128,
                 max_child_num=10, device='gpu', edge_symmetric_type='avg', num_iterations=0,
                 edge_type_num=0, num_parts=0, num_shapes=0, deep_sdf_specs=None,
                 cat_name=None, class2id=None, **kwargs):

        super(HierarchicalDeepSDFDecoder, self).__init__()

        self.recursive_decoder = RecursiveDeepSDFDecoder(recursive_feat_size, recursive_hidden_size,
                                                         max_child_num, device, edge_symmetric_type,
                                                         num_iterations, edge_type_num,
                                                         num_parts, num_shapes, deep_sdf_specs, cat_name,
                                                         class2id)

        self.mseLoss = nn.MSELoss()

    def forward(self, x_root, sdf_data, full_label=None, encoder_features=None, rotation=None, gt_tree=None,
                index=0):
        output = self.recursive_decoder.decode_structure(x_root, sdf_data,
                                                         full_label=full_label,
                                                         encoder_features=encoder_features,
                                                         rotation=rotation,
                                                         gt_tree=gt_tree,
                                                         index=index)
        return output

    def forward_two_stage(self, x_root, sdf_data, full_label=None, encoder_features=None, rotation=None, gt_tree=None,
                          index=0, parts_indices=None, full_shape_idx=None, noise_full=None,
                          rot_aug=0, shift_x=0, shift_y=0, bck_thr=0.5, cat_name=None,
                          scale=1):
        output = self.recursive_decoder.decode_structure_two_stage(x_root, sdf_data,
                                                                   full_label=full_label,
                                                                   encoder_features=encoder_features,
                                                                   rotation=rotation,
                                                                   gt_tree=gt_tree,
                                                                   index=index,
                                                                   parts_indices=parts_indices,
                                                                   full_shape_idx=full_shape_idx,
                                                                   noise_full=noise_full,
                                                                   rot_aug=rot_aug, shift_x=shift_x, shift_y=shift_y,
                                                                   bck_thr=bck_thr, cat_name=cat_name,
                                                                   scale=scale)
        return output

    def structure_recon_loss(self, x_root, gt_tree, sdf_data,
                             encoder_features=None, rotation=None,
                             parts_indices=None, epoch=0,
                             full_shape_idx=None,
                             noise_full=None):
        return self.recursive_decoder.structure_recon_loss(x_root, gt_tree, sdf_data,
                                                           encoder_features=encoder_features, rotation=rotation,
                                                           parts_indices=parts_indices, epoch=epoch,
                                                           full_shape_idx=full_shape_idx,
                                                           noise_full=noise_full)

    def latent_recon_loss(self, x_root, gt_tree, sdf_data,
                             encoder_features=None, rotation=None,
                             parts_indices=None, epoch=0,
                             full_shape_idx=None,
                             noise_full=None, rotations=None, class2id=None):
        return self.recursive_decoder.latent_recon_loss(x_root, gt_tree, sdf_data,
                                                        encoder_features=encoder_features, rotation=rotation,
                                                        parts_indices=parts_indices, epoch=epoch,
                                                        full_shape_idx=full_shape_idx,
                                                        noise_full=noise_full,
                                                        rotations=rotations,
                                                        class2id=class2id)

    def deepsdf_recon_loss(self, sdf_data, parts_indices=None, epoch=0,
                           full_shape_idx=None, noise_full=None):
        return self.recursive_decoder.deepsdf_recon_loss(sdf_data, parts_indices=parts_indices, epoch=epoch,
                                                         full_shape_idx=full_shape_idx, noise_full=noise_full)

    def get_latent_vecs(self):
        return self.recursive_decoder.get_latent_vecs()

    def get_latent_shape_vecs(self):
        return self.recursive_decoder.get_latent_shape_vecs()

    def tto(self, children_initial_data, shape_initial_data, only_align=False, constr_mode=0, cat_name=None,
            num_shapes=0, k_near=0, scene_id='0', wconf=0, w_full_noise=1, w_part_u_noise=1,
            w_part_part_noise=1, lr_dec_full=0, lr_dec_part=0, target_sample_names=None,
            sa_mode=None, parts_indices=None, shape_idx=None, store_dir=None):
        return self.recursive_decoder.tto(children_initial_data, shape_initial_data,
                                          only_align=only_align,
                                          constr_mode=constr_mode,
                                          cat_name=cat_name,
                                          num_shapes=num_shapes,
                                          k_near=k_near,
                                          scene_id=scene_id,
                                          wconf=wconf,
                                          w_full_noise=w_full_noise, w_part_u_noise=w_part_u_noise,
                                          w_part_part_noise=w_part_part_noise, lr_dec_full=lr_dec_full,
                                          lr_dec_part=lr_dec_part,
                                          target_sample_names=target_sample_names, sa_mode=sa_mode,
                                          parts_indices=parts_indices, shape_idx=shape_idx,
                                          store_dir=store_dir)



