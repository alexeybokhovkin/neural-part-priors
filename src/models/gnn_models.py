import os
import importlib

import torch
import torch.nn as nn

# import models
from ..buildingblocks import FeatureVector, ConvEncoder, ConvDecoder
from .hier_decoder_deepsdf_chair import RecursiveDeepSDFDecoder as RecursiveDeepSDFDecoder_chair
from .hier_decoder_deepsdf_bed import RecursiveDeepSDFDecoder as RecursiveDeepSDFDecoder_bed
from .hier_decoder_deepsdf_storagefurniture import RecursiveDeepSDFDecoder as RecursiveDeepSDFDecoder_storagefurniture
from .hier_decoder_deepsdf_table import RecursiveDeepSDFDecoder as RecursiveDeepSDFDecoder_table
from .hier_decoder_deepsdf_trashcan import RecursiveDeepSDFDecoder as RecursiveDeepSDFDecoder_trashcan


class GeoEncoder(nn.Module):
    def __init__(self, enc_in_f_maps=None, enc_out_f_maps=None,
                 layer_order='crg', num_groups=8, enc_strides=1, enc_paddings=1, enc_number_of_fmaps=5,
                 enc_conv_kernel_sizes=3, num_convs_per_block=1, last_pooling_size=1,
                 device='gpu', last_pooling=True, layer_orders=None, **kwargs):

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


class HierarchicalDeepSDFDecoder(nn.Module):
    def __init__(self, recursive_feat_size=128, recursive_hidden_size=128,
                 max_child_num=10, device='gpu', edge_symmetric_type='avg', num_iterations=0,
                 edge_type_num=0, num_parts=0, num_shapes=0, deep_sdf_specs=None,
                 cat_name=None, class2id=None, scene_aware_points_path=None, **kwargs):

        super(HierarchicalDeepSDFDecoder, self).__init__()

        RecursiveDeepSDFDecoder = eval(f'RecursiveDeepSDFDecoder_{cat_name}')
        self.recursive_decoder = RecursiveDeepSDFDecoder(recursive_feat_size, recursive_hidden_size,
                                                         max_child_num, device, edge_symmetric_type,
                                                         num_iterations, edge_type_num,
                                                         num_parts, num_shapes, deep_sdf_specs, cat_name,
                                                         class2id, scene_aware_points_path)

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
                          cat_name=None, scale=1):
        output = self.recursive_decoder.decode_structure_two_stage(x_root, sdf_data,
                                                                   full_label=full_label,
                                                                   encoder_features=encoder_features,
                                                                   rotation=rotation,
                                                                   gt_tree=gt_tree,
                                                                   cat_name=cat_name,
                                                                   scale=scale)
        return output

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
    
    def tto(self, children_initial_data, shape_initial_data, only_align=False,
            num_shapes=0, k_near=0, wconf=0, w_full_noise=1, w_part_u_noise=1,
            w_part_part_noise=1, lr_dec_full=0, lr_dec_part=0, target_sample_names=None,
            sa_mode=None, parts_indices=None, shape_idx=None, store_dir=None, class2id=None):
        return self.recursive_decoder.tto(children_initial_data, 
                                          shape_initial_data,
                                          only_align=only_align,
                                          num_shapes=num_shapes,
                                          k_near=k_near,
                                          wconf=wconf,
                                          w_full_noise=w_full_noise, 
                                          w_part_u_noise=w_part_u_noise,
                                          w_part_part_noise=w_part_part_noise, 
                                          lr_dec_full=lr_dec_full,
                                          lr_dec_part=lr_dec_part,
                                          target_sample_names=target_sample_names, 
                                          sa_mode=sa_mode,
                                          parts_indices=parts_indices, 
                                          shape_idx=shape_idx,
                                          store_dir=store_dir,
                                          class2id=class2id)



