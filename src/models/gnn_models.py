import os
import pickle

import torch
import torch.nn as nn

from ..buildingblocks import FeatureVector, ConvEncoder, ConvDecoder
from .gnn import RecursiveDecoder


class GeoEncoder(nn.Module):
    def __init__(self, enc_in_f_maps=None, enc_out_f_maps=None,
                 layer_order='crg', num_groups=8, enc_strides=1, enc_paddings=1, enc_number_of_fmaps=5,
                 enc_conv_kernel_sizes=3, num_convs_per_block=1, last_pooling_size=1,
                 device='gpu', recursive_feat_size=256, recursive_hidden_size=256, variational=False,
                 last_pooling=True, **kwargs):

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
                                         enc_strides, enc_paddings, enc_conv_kernel_sizes, num_convs_per_block)
        self.shape_feature_vector = FeatureVector(last_pooling_size, pooling=True)

        self.mseLoss = nn.MSELoss()

    def root_latents(self, x_foreground):
        x_root, features = self.voxel_encoder(x_foreground)
        if self.last_pooling:
            x_root = self.shape_feature_vector(x_root)
        return x_root, features

    def encode_loss(self, x_root, gt_root):
        return self.mseLoss(x_root, gt_root)


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

    def forward(self, x_root, mask_code=None, mask_feature=None, scan_geo=None, full_label=None, encoder_features=None,
                rotation=None):
        tree, S_priors, pred_rotation = self.recursive_decoder.decode_structure(x_root, self.max_depth, mask_code,
                                                                                mask_feature, scan_geo=scan_geo,
                                                                                full_label=full_label,
                                                                                encoder_features=encoder_features,
                                                                                rotation=rotation)
        return tree, S_priors, pred_rotation

    def structure_recon_loss(self, x_root, gt_tree, mask_code=None, mask_feature=None, scan_geo=None,
                             encoder_features=None, rotation=None):
        return self.recursive_decoder.structure_recon_loss(x_root, gt_tree, mask_code=mask_code,
                                                           mask_feature=mask_feature,
                                                           scan_geo=scan_geo, encoder_features=encoder_features,
                                                           rotation=rotation)


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

    def forward(self, x_root, encoder_feature=None):
        x_root = x_root[..., None, None, None]
        pred_masks = self.node_decoder(x_root, encoder_feature)

        return pred_masks

    def loss(self, pred, gt):
        pred = torch.sigmoid(pred)
        loss = self.voxelLoss(pred, gt)
        avg_loss = loss.mean()

        return avg_loss