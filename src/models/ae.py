import torch
from torch import nn

from ..buildingblocks import ConvEncoder, ConvDecoder

class AE_Encoder(nn.Module):

    def __init__(self, enc_in_f_maps=None, enc_out_f_maps=None,
                 layer_order='crg', num_groups=8, enc_strides=1, enc_paddings=1,
                 enc_conv_kernel_sizes=3, num_convs_per_block=1, device='gpu', last_pooling=True,
                 **kwargs):
        super(AE_Encoder, self).__init__()

        self.device = device
        self.last_pooling = last_pooling

        enc_number_of_fmaps = len(enc_in_f_maps)
        if not isinstance(enc_conv_kernel_sizes, list):
            enc_conv_kernel_sizes = [enc_conv_kernel_sizes] * enc_number_of_fmaps
        if not isinstance(enc_strides, list):
            enc_strides = [enc_strides] * enc_number_of_fmaps
        if not isinstance(enc_paddings, list):
            enc_paddings = [enc_paddings] * enc_number_of_fmaps

        self.encoder = ConvEncoder(enc_in_f_maps, enc_out_f_maps, layer_order, num_groups,
                                   enc_strides, enc_paddings, enc_conv_kernel_sizes, num_convs_per_block)

    def forward(self, x):

        fmap, features = self.encoder(x)
        return fmap, features


class AE_Decoder(nn.Module):

    def __init__(self, dec_in_f_maps=None, dec_out_f_maps=None,
                 layer_order='crg', num_groups=8, dec_strides=1, dec_paddings=1,
                 dec_conv_kernel_sizes=3, num_convs_per_block=1, scale_factors=None, device='gpu',
                 output_paddings=1,
                 **kwargs):
        super(AE_Decoder, self).__init__()

        self.device = device

        self.node_decoder = ConvDecoder(dec_in_f_maps, dec_out_f_maps, num_convs_per_block, layer_order, num_groups,
                                        scale_factors, dec_conv_kernel_sizes, dec_strides, dec_paddings,
                                        output_paddings)

        self.voxelLoss = nn.BCELoss()

    def forward(self, fmap):

        pred_masks = self.node_decoder(fmap)
        return pred_masks

    def loss(self, x_pred, x_gt):

        x_pred = torch.sigmoid(x_pred)
        loss = self.voxelLoss(x_pred, x_gt)
        return loss