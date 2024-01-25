#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F


cat_to_parts = {
    'chair': 7,
    'table': 7,
    'storagefurniture': 5,
    'bed': 4,
    'trashcan': 5
}


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        cat_name,
        learn_parts,
        pe,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        if pe:
            self.pos_dim = 63 # for positional encoding
        else:
            self.pos_dim = 3

        if learn_parts:
            dims_sdf = [latent_size + cat_to_parts[cat_name] + self.pos_dim] + dims + [1]
        else:
            dims_sdf = [latent_size + self.pos_dim] + dims + [1]

        self.num_layers = len(dims_sdf)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        # SDF branch
        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims_sdf[layer + 1] - dims_sdf[0]
            else:
                out_dim = dims_sdf[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= self.pos_dim

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims_sdf[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims_sdf[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L + pos_dim)
    def forward(self, input):
        xyz = input[:, -self.pos_dim:]

        if input.shape[1] > self.pos_dim and self.latent_dropout:
            latent_vecs = input[:, :-self.pos_dim]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        # SDF branch
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
                
        if hasattr(self, "th"):
            x = self.th(x)

        return x
