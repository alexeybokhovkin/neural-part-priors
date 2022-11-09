import torch
from torch import nn as nn
from torch.nn import functional as F


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=1, stride=1):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int): add zero-padding to the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order or 'i' in order)
            # modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
            modules.append(('conv', nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, stride=stride)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            assert not is_before_conv, 'GroupNorm MUST go after the Conv3d'
            # number of groups must be less or equal the number of channels
            if out_channels < num_groups:
                num_groups = out_channels
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        elif char == 'i':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('instancenorm', nn.InstanceNorm3d(in_channels)))
            else:
                modules.append(('instancenorm', nn.InstanceNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c', 'i']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8, padding=1, stride=1):
        super(SingleConv, self).__init__()
        
        self.weight_size = [in_channels, out_channels, kernel_size, kernel_size, kernel_size]

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding, stride=stride):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='crg', num_groups=8, padding=1, stride=1):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups,
                                   padding=padding, stride=stride))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups,
                                   padding=padding, stride=stride))


class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, padding=1, stride=1, **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order,
                                num_groups=num_groups, padding=padding, stride=stride)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order,
                                num_groups=num_groups, padding=padding)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups, padding=padding)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.non_linearity(out)

        return out


class ConvBlock(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (tuple): the size of the window to take a max over
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=(2, 2, 2), pool_type='max', basic_module=DoubleConv, conv_layer_order='crg',
                 num_groups=8, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        padding = tuple([padding]) * 3
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        if basic_module == SingleConv:
            self.basic_module = basic_module(in_channels, out_channels,
                                             kernel_size=conv_kernel_size,
                                             order=conv_layer_order,
                                             num_groups=num_groups,
                                             stride=stride,
                                             padding=padding)
        elif basic_module == DoubleConv:
            self.basic_module = basic_module(in_channels, out_channels,
                                             encoder=True,
                                             kernel_size=conv_kernel_size,
                                             order=conv_layer_order,
                                             num_groups=num_groups)
        elif basic_module == ExtResNetBlock:
            self.basic_module = basic_module(in_channels, out_channels,
                                             kernel_size=conv_kernel_size,
                                             order=conv_layer_order,
                                             num_groups=num_groups,
                                             stride=stride,
                                             padding=padding)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)

        return x


class ConvEncoder(nn.Module):

    def __init__(self, enc_in_f_maps=None, enc_out_f_maps=None,
                 layer_order=None, num_groups=8, enc_strides=1, enc_paddings=1, enc_conv_kernel_sizes=3,
                 num_convs_per_block=1, layer_orders=None, **kwargs):
        super(ConvEncoder, self).__init__()

        if num_convs_per_block == 1:
            basic_module = SingleConv
        elif num_convs_per_block == 2:
            basic_module = DoubleConv
        elif num_convs_per_block == 0:
            basic_module = ExtResNetBlock

        encoders = []
        for i, out_feature_num in enumerate(enc_out_f_maps):
            if i == 0:
                print(i)
                print(enc_in_f_maps[i], out_feature_num, enc_conv_kernel_sizes[i], enc_strides[i], enc_paddings[i])
                encoder = ConvBlock(enc_in_f_maps[i], out_feature_num, apply_pooling=False,
                                    basic_module=basic_module,
                                    conv_layer_order=layer_orders[i], num_groups=num_groups,
                                    conv_kernel_size=enc_conv_kernel_sizes[i], stride=enc_strides[i],
                                    padding=enc_paddings[i])
            else:
                print(i)
                print(enc_in_f_maps[i], out_feature_num, enc_conv_kernel_sizes[i], enc_strides[i], enc_paddings[i])
                encoder = ConvBlock(enc_in_f_maps[i], out_feature_num, basic_module=basic_module,
                                    conv_layer_order=layer_orders[i], num_groups=num_groups,
                                    conv_kernel_size=enc_conv_kernel_sizes[i], stride=enc_strides[i],
                                    padding=enc_paddings[i])
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x):
        encoder_features = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encoder_features.insert(0, x)

        return x, encoder_features


class DeconvBlock(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=2, basic_module=DoubleConv, conv_layer_order='crg', num_groups=8, stride=1, padding=1,
                 output_padding=1, join=None):
        super(DeconvBlock, self).__init__()

        scale_factor = tuple([scale_factor]) * 3
        self.scale_factor = scale_factor
        self.join = join
        if basic_module == DoubleConv:
            # if DoubleConv is the basic_module use nearest neighbor interpolation for upsampling
            self.upsample = None
        else:
            # otherwise use ConvTranspose3d (bear in mind your GPU memory)
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            # also scale the number of channels from in_channels to out_channels so that summation joining
            # works correctly
            self.upsample = nn.ConvTranspose3d(in_channels,
                                               out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1,
                                               output_padding=output_padding)
            # adapt the number of in_channels for the ExtResNetBlock
            in_channels = out_channels

        if basic_module == SingleConv:
            self.basic_module = basic_module(in_channels, out_channels,
                                             kernel_size=kernel_size,
                                             order=conv_layer_order,
                                             num_groups=num_groups,
                                             stride=stride,
                                             padding=padding)
        elif basic_module == DoubleConv:
            self.basic_module = basic_module(in_channels, out_channels,
                                             encoder=False,
                                             kernel_size=kernel_size,
                                             order=conv_layer_order,
                                             num_groups=num_groups)

    def forward(self, x, feature=None):

        if self.upsample is None:
            # print('Input dec:', x.shape)
            # print('Scale factor:', self.scale_factor)
            x_feature = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear')
            # print('After up:', x_feature.shape)
            x = self.basic_module(x_feature)
            # print('After conv:', x.shape)
            # print()
        else:

            # print('Input dec:', x.shape)
            x_feature = self.upsample(x)
            # print('After up:', x.shape)
            x = self.basic_module(x_feature)
            # print('After conv:', x.shape)
            # if self.join == 1:
            #     x = x + feature
            # elif self.join == 2:
                # print('Feat:', feature.shape)
                # print(x.shape)
                # x = torch.cat([x, feature], dim=1)
            # print('After join:', x.shape)
            # print()

        return x, x_feature


class ConvDecoder(nn.Module):

    def __init__(self, dec_in_f_maps, dec_out_f_maps, num_convs_per_block, conv_layer_order, num_groups,
                 scale_factors, kernel_sizes, strides, paddings, output_paddings, joins, last_conv=False):
        super(ConvDecoder, self).__init__()

        if num_convs_per_block == 1:
            basic_module = SingleConv
        elif num_convs_per_block == 2:
            basic_module = DoubleConv
        elif num_convs_per_block == 0:
            basic_module = ExtResNetBlock

        self.last_conv = last_conv

        decoders = []
        for i in range(len(dec_in_f_maps)):
            in_feature_num = dec_in_f_maps[i]
            out_feature_num = dec_out_f_maps[i]
            decoder = DeconvBlock(in_feature_num, out_feature_num, basic_module=basic_module,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups,
                                  scale_factor=scale_factors[i], kernel_size=kernel_sizes[i],
                                  stride=strides[i], padding=paddings[i], join=joins[i],
                                  output_padding=output_paddings[i])
            if i == 0:
                decoder = nn.Identity()

            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)
        self.decoder_last_conv = SingleConv(out_feature_num, 1,
                                            kernel_size=1,
                                            order='c',
                                            stride=1,
                                            padding=0)

    def forward(self, x, features=None):
        x_features = []

        for i, decoder in enumerate(self.decoders):
            if i >= 1:
                feature = None
                # if i >= len(features) or not features or features is None:
                #     feature = None
                # else:
                #     feature = features[i]
                x, x_feature = decoder(x, feature)
                x_features.append(x_feature)
        if self.last_conv:
            x = self.decoder_last_conv(x)

        return x, x_features


class FeatureVector(nn.Module):

    def __init__(self, pool_kernel_size, pooling=True):
        super(FeatureVector, self).__init__()

        self.pooling = pooling

        if self.pooling:
            self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            kernel_size = (pool_kernel_size, pool_kernel_size, pool_kernel_size)
            self.conv = nn.Conv3d(128, 128, kernel_size, padding=(0, 0, 0), bias=True, stride=(1, 1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        if self.pooling:
            x = self.pooling(x)
        else:
            x = self.conv(x)
        x = self.flatten(x)

        return x