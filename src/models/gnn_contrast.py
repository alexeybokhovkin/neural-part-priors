import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import numpy as np
import os
import pickle

from ..data_utils.hierarchy import Tree
from ..utils.gnn import linear_assignment
from ..buildingblocks import ConvDecoder


class LeafClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(LeafClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.mlp1(x))
        x = self.mlp2(x)
        x = self.activation(x)

        return x


class NumChildrenClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size, max_child_num):
        super(NumChildrenClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, max_child_num)

    def forward(self, x):
        x = torch.relu(self.mlp1(x))
        x = self.mlp2(x)

        return x


class LatentDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(LatentDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)

    def forward(self, x):
        x = torch.relu(self.mlp1(x))
        x = torch.relu(self.mlp2(x))

        return x


class NodeDecoder(nn.Module):

    def __init__(self, geo_feat_len, node_feat_len, dec_in_f_maps, dec_out_f_maps, num_convs_per_block, layer_order,
                 num_groups, scale_factors, dec_conv_kernel_sizes, dec_strides, dec_paddings, encode_mask,
                 output_paddings=None, joins=None):
        super(NodeDecoder, self).__init__()

        self.part_decoder = PartDecoder(geo_feat_len, dec_in_f_maps, dec_out_f_maps, num_convs_per_block,
                                        layer_order, num_groups, scale_factors, dec_conv_kernel_sizes,
                                        dec_strides, dec_paddings,
                                        encode_mask=encode_mask, output_paddings=output_paddings, joins=joins)

        self.mlp = nn.Linear(node_feat_len, geo_feat_len)
        self.activation = nn.Sigmoid()

        if encode_mask:
            self.mask_mlp = nn.Linear(2*node_feat_len, geo_feat_len)

    def forward(self, x, mask_code=None, mask_feature=None):
        geo_feat = self.mlp(x)
        if mask_code is not None:
            geo_feat = torch.cat([geo_feat, mask_code.repeat(geo_feat.shape[0], 1)], dim=1)
            geo_feat = self.mask_mlp(geo_feat)
        geo_mask, decoder_features = self.part_decoder(geo_feat, mask_feature)
        geo_mask = self.activation(geo_mask)

        return geo_mask, geo_feat, decoder_features


class NodeDecoderPrior(nn.Module):

    def __init__(self, geo_feat_len, node_feat_len, dec_in_f_maps, dec_out_f_maps, num_convs_per_block, layer_order,
                 num_groups, scale_factors, dec_conv_kernel_sizes, dec_strides, dec_paddings, encode_mask, shape_priors,
                 output_paddings, joins,
                 **kwargs):
        super(NodeDecoderPrior, self).__init__()

        self.part_decoder = PartDecoder(geo_feat_len, dec_in_f_maps, dec_out_f_maps, num_convs_per_block,
                                        layer_order, num_groups, scale_factors, dec_conv_kernel_sizes,
                                        dec_strides, dec_paddings, encode_mask, output_paddings, joins)

        self.mlp = nn.Linear(node_feat_len, geo_feat_len)
        self.activation = nn.Sigmoid()

        if encode_mask:
            self.mask_mlp = nn.Linear(2 * node_feat_len, geo_feat_len)
        if shape_priors:
            self.prior_mlp = nn.Linear(geo_feat_len, 10)

        self.softmax_layer = nn.Softmax()

    def forward(self, x, mask_code=None, mask_feature=None, priors=None):
        geo_feat = self.mlp(x)
        if mask_code is not None:
            geo_feat = torch.cat([geo_feat, mask_code.repeat(geo_feat.shape[0], 1)], dim=1)
            geo_feat = self.mask_mlp(geo_feat)

        # compute prior
        geo_feat_prior = self.prior_mlp(geo_feat)
        geo_feat_prior = self.softmax_layer(geo_feat_prior)
        S_prior = geo_feat_prior.view((-1, 1, 1, 1)) * priors
        S_prior = torch.sum(S_prior, dim=0, keepdim=True)[None, ...]

        return geo_feat, S_prior


class PriorRefiner(nn.Module):

    def __init__(self):
        super(PriorRefiner, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(2, 8, (3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(8)
        self.conv2 = nn.Conv3d(8, 16, (3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 8, (3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(8)
        self.conv4 = nn.Conv3d(8, 1, (1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.activation = nn.Sigmoid()

    def forward(self, x, scan_geo=None, node_latent=None, encoder_features=None):
        y = torch.cat([x, scan_geo], dim=1)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu(y)

        y2 = self.conv4(y)

        y2 = y2 + x
        y2 = self.activation(y2)

        return y2, y


class MasksRefiner(nn.Module):

    def __init__(self):
        super(MasksRefiner, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(80, 16, (3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 10, (3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.activation = nn.Sigmoid()

    def forward(self, geo_features):
        y = torch.cat(geo_features, dim=1)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)

        y = self.activation(y)

        return y


class PartDecoder(nn.Module):

    def __init__(self, feat_len, dec_in_f_maps, dec_out_f_maps, num_convs_per_block, layer_order,
                 num_groups, scale_factors, dec_conv_kernel_sizes, dec_strides, dec_paddings,
                 encode_mask=False, output_paddings=None, joins=None):
        super(PartDecoder, self).__init__()

        self.encode_mask = encode_mask

        self.geo_decoder = ConvDecoder(dec_in_f_maps, dec_out_f_maps, num_convs_per_block, layer_order, num_groups,
                                       scale_factors, dec_conv_kernel_sizes, dec_strides, dec_paddings, output_paddings,
                                       joins)

    def forward(self, x, mask_feature=None):

        x = x[..., None, None, None]
        decoder_features = None
        if self.encode_mask:
            x, decoder_features = self.geo_decoder(x, mask_feature)
        else:
            x = self.geo_decoder(x)

        return x, decoder_features


class GNNChildDecoder(nn.Module):

    def __init__(self, node_feat_size, hidden_size, max_child_num,
                 edge_symmetric_type, num_iterations, edge_type_num):
        super(GNNChildDecoder, self).__init__()

        self.max_child_num = max_child_num
        self.hidden_size = hidden_size

        self.edge_symmetric_type = edge_symmetric_type
        self.num_iterations = num_iterations
        self.edge_type_num = edge_type_num

        # self.mlp_parent = nn.Linear(node_feat_size, hidden_size * max_child_num)
        # ==================================
        self.mlp_parent = nn.Linear(node_feat_size + 26, hidden_size * max_child_num)
        # ==================================

        self.mlp_exists = nn.Linear(hidden_size, 1)
        self.mlp_sem = nn.Linear(hidden_size, Tree.num_sem)
        self.mlp_edge_latent = nn.Linear(hidden_size * 2, hidden_size)

        self.mlp_edge_exists = nn.ModuleList()
        for i in range(self.edge_type_num):
            self.mlp_edge_exists.append(nn.Linear(hidden_size, 1))

        self.node_edge_op = torch.nn.ModuleList()
        for i in range(self.num_iterations):
            self.node_edge_op.append(nn.Linear(hidden_size * 3 + self.edge_type_num, hidden_size))

        self.mlp_child = nn.Linear(hidden_size * (self.num_iterations + 1), hidden_size)
        self.mlp_child2 = nn.Linear(hidden_size, node_feat_size)

    def forward(self, parent_feature, gt_children_code=None, gt_num_code=None):
        batch_size = parent_feature.shape[0]
        feat_size = parent_feature.shape[1]

        # ==================================
        parent_feature = torch.cat([parent_feature, gt_children_code, gt_num_code], dim=1)
        # ==================================

        parent_feature = torch.relu(self.mlp_parent(parent_feature))
        child_feats = parent_feature.view(batch_size, self.max_child_num, self.hidden_size)

        # node existence
        child_exists_logits = self.mlp_exists(child_feats.view(batch_size * self.max_child_num, self.hidden_size))
        child_exists_logits = child_exists_logits.view(batch_size, self.max_child_num, 1)

        # edge features
        edge_latents = torch.cat([
            child_feats.view(batch_size, self.max_child_num, 1, feat_size).expand(-1, -1, self.max_child_num, -1),
            child_feats.view(batch_size, 1, self.max_child_num, feat_size).expand(-1, self.max_child_num, -1, -1)
        ], dim=3)
        edge_latents = torch.relu(self.mlp_edge_latent(edge_latents))

        # edge existence prediction
        edge_exists_logits_per_type = []
        for i in range(self.edge_type_num):
            edge_exists_logits_cur_type = self.mlp_edge_exists[i](edge_latents).view(
                batch_size, self.max_child_num, self.max_child_num, 1)
            edge_exists_logits_per_type.append(edge_exists_logits_cur_type)
        edge_exists_logits = torch.cat(edge_exists_logits_per_type, dim=3)

        """
            decoding stage message passing
            there are several possible versions, this is a simple one:
            use a fixed set of edges, consisting of existing edges connecting existing nodes
            this set of edges does not change during iterations
            iteratively update the child latent features
            then use these child latent features to compute child features and semantics
        """
        # get edges that exist between nodes that exist
        edge_indices = torch.nonzero(edge_exists_logits > 0)
        edge_types = edge_indices[:, 3]
        edge_indices = edge_indices[:, 1:3]
        nodes_exist_mask = (child_exists_logits[0, edge_indices[:, 0], 0] > 0) \
                           & (child_exists_logits[0, edge_indices[:, 1], 0] > 0)
        edge_indices = edge_indices[nodes_exist_mask, :]
        edge_types = edge_types[nodes_exist_mask]

        # get latent features for the edges
        edge_feats_mp = edge_latents[0:1, edge_indices[:, 0], edge_indices[:, 1], :]

        # append edge type to edge features, so the network has information which
        # of the possibly multiple edges between two nodes it is working with
        edge_type_logit = edge_exists_logits[0:1, edge_indices[:, 0], edge_indices[:, 1], :]
        edge_type_logit = edge_feats_mp.new_zeros(edge_feats_mp.shape[:2] + (self.edge_type_num,))
        edge_type_logit[0:1, range(edge_type_logit.shape[1]), edge_types] = \
            edge_exists_logits[0:1, edge_indices[:, 0], edge_indices[:, 1], edge_types]
        edge_feats_mp = torch.cat([edge_feats_mp, edge_type_logit], dim=2)

        num_edges = edge_indices.shape[0]
        max_childs = child_feats.shape[1]

        iter_child_feats = [child_feats]  # zeroth iteration

        if self.num_iterations > 0 and num_edges > 0:
            edge_indices_from = edge_indices[:, 0].view(-1, 1).expand(-1, self.hidden_size)

        for i in range(self.num_iterations):
            if num_edges > 0:
                node_edge_feats = torch.cat([
                    child_feats[0:1, edge_indices[:, 0], :],  # start node features
                    child_feats[0:1, edge_indices[:, 1], :],  # end node features
                    edge_feats_mp], dim=2)  # edge features

                node_edge_feats = node_edge_feats.view(num_edges, -1)
                node_edge_feats = torch.relu(self.node_edge_op[i](node_edge_feats))

                # aggregate information from neighboring nodes
                new_child_feats = child_feats.new_zeros(max_childs, self.hidden_size)
                if self.edge_symmetric_type == 'max':
                    new_child_feats, _ = torch_scatter.scatter_max(node_edge_feats, edge_indices_from, dim=0,
                                                                   out=new_child_feats)
                elif self.edge_symmetric_type == 'sum':
                    new_child_feats = torch_scatter.scatter_add(node_edge_feats, edge_indices_from, dim=0,
                                                                out=new_child_feats)
                elif self.edge_symmetric_type == 'avg':
                    new_child_feats = torch_scatter.scatter_mean(node_edge_feats, edge_indices_from, dim=0,
                                                                 out=new_child_feats)
                else:
                    raise ValueError(f'Unknown edge symmetric type: {self.edge_symmetric_type}')

                child_feats = new_child_feats.view(1, max_childs, self.hidden_size)

            # save child features of this iteration
            iter_child_feats.append(child_feats)

        # concatenation of the child features from all iterations (as in GIN, like skip connections)
        child_feats = torch.cat(iter_child_feats, dim=2)

        # transform concatenation back to original feature space size
        child_feats = child_feats.view(-1, self.hidden_size * (self.num_iterations + 1))
        child_feats = torch.relu(self.mlp_child(child_feats))
        child_feats = child_feats.view(batch_size, self.max_child_num, self.hidden_size)

        # node semantics
        child_sem_logits = self.mlp_sem(child_feats.view(-1, self.hidden_size))
        child_sem_logits = child_sem_logits.view(batch_size, self.max_child_num, Tree.num_sem)

        # node features
        child_feats = self.mlp_child2(child_feats.view(-1, self.hidden_size))
        child_feats = child_feats.view(batch_size, self.max_child_num, feat_size)
        child_feats = torch.relu(child_feats)

        return child_feats, child_sem_logits, child_exists_logits, edge_exists_logits


class RecursiveDecoder(nn.Module):

    def __init__(self, feature_size, geo_feature_size, hidden_size, max_child_num,
                 dec_in_f_maps, dec_out_f_maps, num_convs_per_block, layer_order, num_groups, scale_factors,
                 dec_conv_kernel_sizes, dec_strides, dec_paddings, device,
                 edge_symmetric_type, num_iterations, edge_type_num, enc_hier, split_subnetworks, loss_children,
                 split_enc_children, encode_mask, shape_priors, priors_path=None, parts_dict=None,
                 enc_in_f_maps=None, enc_out_f_maps=None, enc_strides=None, enc_paddings=None,
                 enc_conv_kernel_sizes=None, last_pooling_size=None, output_paddings=None, joins=None,
                 base=None):
        super(RecursiveDecoder, self).__init__()

        self.label_to_id = {
            'chair': 0,
            'bed': 1,
            'storage_furniture': 2,
            'table': 3,
            'trash_can': 4
        }

        self.parts_dict = parts_dict

        self.priors = {}
        for path in os.listdir(priors_path):
            p = torch.FloatTensor(np.load(os.path.join(priors_path, path)))
            self.priors[int(path.split('.')[0])] = p

        # self.priors = {}
        # for rot in range(8):
        #     self.priors[rot] = {}
        # for path in os.listdir(priors_path):
        #     p = torch.FloatTensor(np.load(os.path.join(priors_path, path)))
        #     path = path.split('.')[0]
        #     rot = int(path.split('_')[1])
        #     prior_id = int(path.split('_')[0])
        #     self.priors[rot][prior_id] = p

        self.edge_types = ['ADJ', 'SYM']
        self.device = device
        self.split_subnetworks = split_subnetworks
        self.max_child_num = max_child_num
        self.split_enc_children = split_enc_children

        self.latent_decoder = LatentDecoder(feature_size, hidden_size)
        self.node_decoder_prior = NodeDecoderPrior(geo_feature_size, feature_size, dec_in_f_maps, dec_out_f_maps,
                                                   num_convs_per_block, layer_order, num_groups, scale_factors,
                                                   dec_conv_kernel_sizes, dec_strides, dec_paddings, encode_mask,
                                                   shape_priors, output_paddings, joins)
        self.prior_refiner = PriorRefiner()
        self.masks_refiner = MasksRefiner()

        self.root_classifier = NumChildrenClassifier(feature_size, hidden_size, 5)
        self.rotation_classifier = NumChildrenClassifier(feature_size, hidden_size, 8)


        self.leaf_classifiers = nn.ModuleList([LeafClassifier(feature_size, hidden_size)])
        self.child_decoders = nn.ModuleList([GNNChildDecoder(feature_size, hidden_size,
                                                             max_child_num, edge_symmetric_type,
                                                             num_iterations, edge_type_num)])
        self.node_decoders = nn.ModuleList(
            [NodeDecoder(geo_feature_size, feature_size, dec_in_f_maps, dec_out_f_maps, num_convs_per_block,
                         layer_order, num_groups, scale_factors,
                         dec_conv_kernel_sizes, dec_strides, dec_paddings, encode_mask, output_paddings, joins)])

        self.softmax_layer = nn.Softmax()

        self.mseLoss = nn.MSELoss(reduction='none')
        self.voxelLoss = nn.BCELoss(reduction='none')
        self.bceLoss = nn.BCELoss(reduction='none')
        self.semCELoss = nn.CrossEntropyLoss(reduction='none')
        self.childrenCELoss = nn.CrossEntropyLoss()

        self.enc_hier = enc_hier

        # ========================================================
        self.gt_label_encoder = nn.Linear(Tree.num_sem, 16)
        self.latent_sem_encoder = nn.Linear(feature_size + 16, feature_size)
        self.gt_label_full_encoder = nn.Linear(32, 16)

        self.root_geo = None
        self.children_geos = []
        # ========================================================

    def isLeafLossEstimator(self, is_leaf_logit, gt_is_leaf):
        return self.bceLoss(is_leaf_logit, gt_is_leaf).view(-1)

    # decode a root code into a tree structure
    def decode_structure(self, z, max_depth, mask_code=None, mask_feature=None, scan_geo=None, full_label=None,
                         encoder_features=None, rotation=None, gt_tree=None):
        root_latent = self.latent_decoder(z)  # torch.Tensor[bsize, 256], torch.Tensor[bsize, 256]
        if full_label is None:
            full_label = Tree.root_sem
        root, S_priors, rotation, child_nodes = self.decode_node(root_latent, max_depth, full_label=full_label, level=0,
                                                    mask_code=mask_code, mask_feature=mask_feature, scan_geo=scan_geo,
                                                    encoder_features=encoder_features, rotation=rotation,
                                                    gt_tree=gt_tree)

        obj = Tree(root=root)
        return obj, S_priors, rotation, child_nodes

    # decode a part node
    def decode_node(self, node_latent, max_depth, full_label, is_leaf=False, level=0, mask_code=None,
                    mask_feature=None, child_sem_logit=None, scan_geo=None, encoder_features=None, rotation=None,
                    pred_rotation=None, gt_sem_code=None, gt_tree=None):
        if self.split_subnetworks:
            if level < 1:
                clip_level = level
            else:
                clip_level = 0
        else:
            clip_level = -1

        size = 32

        # check node is leaf
        is_leaf_logit = self.leaf_classifiers[clip_level](node_latent)
        node_is_leaf = is_leaf_logit.item() > 0.5  # bool (bsize = 1!)

        # use maximum depth to avoid potential infinite recursion
        if max_depth < 1:
            is_leaf = True  # bool

        # ================================
        # if gt_sem_code is None:
        #     gt_sem_code = torch.zeros((1, 16)).cuda()
        # node_latent = self.latent_sem_encoder(torch.cat([node_latent, gt_sem_code], dim=1))
        # ================================

        geo_mask, geo_feature, _ = self.node_decoders[clip_level](node_latent, mask_code, mask_feature)

        cuda_device = is_leaf_logit.get_device()

        # if level == 0:
        #     rotation_cls_pred = self.rotation_classifier(node_latent)
        #     pred_rotation = self.softmax_layer(rotation_cls_pred)
        #     pred_rotation = int(torch.argmax(pred_rotation).cpu().detach().numpy())
        #
        # if level > 0:
        #     with torch.no_grad():
        #         child_sem_logit = self.softmax_layer(child_sem_logit)
        #         child_sem = int(torch.argmax(child_sem_logit).cpu().detach().numpy())
        #         if child_sem > 0:
        #             full_part_name = Tree.part_id2name[child_sem]
        #             prior = self.priors[self.parts_dict[full_part_name]].to(cuda_device)
        #
        #             # prior = self.priors[pred_rotation][self.parts_dict[full_part_name]].to(cuda_device)
        #     if child_sem > 0:
        #         geo_feat, S_prior = self.node_decoder_prior(node_latent, mask_code, mask_feature, prior)
        #         prior.to('cpu')
        #         geo_mask, geo_feature = self.prior_refiner(S_prior, scan_geo, node_latent, encoder_features)
        #     else:
        #         geo_mask = torch.zeros((1, 1, size, size, size))
        #         S_prior = torch.zeros((1, 1, size, size, size))
        # else:
        #     geo_mask = torch.zeros((1, 1, size, size, size))
        #     S_prior = torch.zeros((1, 1, size, size, size))

        # print(node_is_leaf, is_leaf)
        if node_is_leaf or is_leaf:
            return Tree.Node(is_leaf=True, full_label=full_label, label=full_label.split('/')[-1],
                             geo=geo_mask), geo_feature, 0, 0
        else:

            # ======================
            child_gt_sem_embeddings = []
            for j, child_node in enumerate(gt_tree.root.children):
                child_gt_sem_vector = torch.zeros((1, Tree.num_sem))
                child_gt_sem_vector[0, child_node.get_semantic_id()] = 1
                child_gt_sem_vector = child_gt_sem_vector.to(cuda_device)
                child_gt_sem_embedding = self.gt_label_encoder(child_gt_sem_vector)
                child_gt_sem_embeddings += [child_gt_sem_embedding]
            num_gt_children = len(child_gt_sem_embeddings)
            if num_gt_children >= 2:
                child_gt_sem_full_embedding = self.gt_label_full_encoder(
                    torch.cat([child_gt_sem_embeddings[0], child_gt_sem_embeddings[1]], dim=1))
            else:
                child_gt_sem_full_embedding = self.gt_label_full_encoder(
                    torch.cat([child_gt_sem_embeddings[0], child_gt_sem_embeddings[0]], dim=1))
            if num_gt_children > 2:
                for j in range(num_gt_children - 2):
                    child_gt_sem_full_embedding = self.gt_label_full_encoder(
                        torch.cat([child_gt_sem_full_embedding, child_gt_sem_embeddings[j + 2]], dim=1))
            child_gt_num_vector = torch.zeros((1, 10))
            child_gt_num_vector[0, num_gt_children] = 1
            child_gt_num_vector = child_gt_num_vector.to(cuda_device)
            # ======================

            child_feats, child_sem_logits, child_exists_logit, edge_exists_logits = \
                self.child_decoders[clip_level](node_latent, child_gt_sem_full_embedding, child_gt_num_vector)
            # child_feats, child_sem_logits, child_exists_logit, edge_exists_logits = \
            #     self.child_decoders[clip_level](node_latent)
            # torch.Tensor[bsize, max_child_num, 256]
            # torch.Tensor[bsize, max_child_num, tree.num_sem]
            # torch.Tensor[bsize, max_child_num, 1]
            # torch.Tensor[bsize, max_child_num, max_child_num, 4]

            child_sem_logits_numpy = child_sem_logits.cpu().detach().numpy().squeeze()
            # torch.Tensor[max_child_num, tree.num_sem]

            # children
            child_nodes = []  # list[exist_child_num]<Tree.Node>
            child_idx = {}
            all_geo_features = []
            for ci in range(child_feats.shape[1]):
                if torch.sigmoid(child_exists_logit[:, ci, :]).item() > 0.5:
                    idx = np.argmax(child_sem_logits_numpy[ci, Tree.part_name2cids[full_label]])  # int
                    idx = Tree.part_name2cids[full_label][idx]  # int (1 <= idx <= tree.num_sem)
                    child_full_label = Tree.part_id2name[idx]
                    # child_node, geo_feature = self.decode_node(
                    #     child_feats[:, ci, :], max_depth - 1, child_full_label,
                    #     is_leaf=(child_full_label not in Tree.part_non_leaf_sem_names), level=level + 1,
                    #     mask_code=mask_code, mask_feature=mask_feature, child_sem_logit=child_sem_logits[0, ci, :],
                    #     scan_geo=scan_geo, encoder_features=encoder_features, rotation=rotation,
                    #     pred_rotation=pred_rotation)
                    child_node, geo_feature, _, _ = self.decode_node(
                        child_feats[:, ci, :], max_depth - 1, child_full_label,
                        is_leaf=(child_full_label not in Tree.part_non_leaf_sem_names), level=level + 1,
                        mask_code=mask_code, mask_feature=mask_feature, child_sem_logit=child_sem_logits[0, ci, :],
                        scan_geo=scan_geo, encoder_features=encoder_features, rotation=rotation,
                        pred_rotation=pred_rotation, gt_tree=gt_tree, gt_sem_code=gt_sem_code)
                    child_nodes.append(child_node)
                    all_geo_features.append(geo_feature)
                    child_idx[ci] = len(child_nodes) - 1

            # edges
            child_edges = []  # list[<=nnz_num]<dict['part_a', 'part_b', 'type']>
            nz_inds = torch.nonzero(torch.sigmoid(edge_exists_logits) > 0.5)  # torch.Tensor[nnz_num, 4]
            edge_from = nz_inds[:, 1]  # torch.Tensor[nnz_num]
            edge_to = nz_inds[:, 2]  # torch.Tensor[nnz_num]
            edge_type = nz_inds[:, 3]  # torch.Tensor[nnz_num]

            for i in range(edge_from.numel()):
                cur_edge_from = edge_from[i].item()
                cur_edge_to = edge_to[i].item()
                cur_edge_type = edge_type[i].item()

                if cur_edge_from in child_idx and cur_edge_to in child_idx:
                    child_edges.append({
                        'part_a': child_idx[cur_edge_from],
                        'part_b': child_idx[cur_edge_to],
                        'type': self.edge_types[cur_edge_type]})

            return Tree.Node(is_leaf=False, children=child_nodes, edges=child_edges,
                             full_label=full_label, label=full_label.split('/')[-1], geo=geo_mask), [0], 0, child_feats

    def structure_recon_loss(self, z, gt_tree, children_roots=None, mask_code=None, mask_feature=None,
                             scan_geo=None, encoder_features=None, rotation=None):
        root_latent = self.latent_decoder(z)  # torch.Tensor[bsize, 256], torch.Tensor[bsize, 256]
        output = self.node_recon_loss_latentless(root_latent, gt_tree.root, 0, children_roots,
                                                 mask_code, mask_feature, scan_geo=scan_geo,
                                                 encoder_features=encoder_features, rotation=rotation)
        return output

    def tto_recon_loss(self, z, gt_tree, children_roots=None, mask_code=None, mask_feature=None,
                             scan_geo=None, encoder_features=None, rotation=None):
        root_latent = self.latent_decoder(z)  # torch.Tensor[bsize, 256], torch.Tensor[bsize, 256]
        output = self.tto_loss(root_latent, gt_tree.root, 0, children_roots,
                               mask_code, mask_feature, scan_geo=scan_geo,
                               encoder_features=encoder_features, rotation=rotation)
        return output

    def node_recon_loss_latentless(self, node_latent, gt_node, level=0, children_roots=None,
                                   mask_code=None, mask_feature=None, child_sem_logit=None,
                                   scan_geo=None, encoder_features=None, rotation=None, pred_rotation=None,
                                   gt_sem_code=None):
        if self.split_subnetworks:
            if level < 1:
                clip_level = level
            else:
                clip_level = 0
        else:
            clip_level = -1

        gt_geo = gt_node.geo  # torch.Tensor[1, 64, 64, 64]
        cuda_device = gt_geo.get_device()

        if level == 0:
            root_cls_pred = self.root_classifier(node_latent)
            root_cls_gt = torch.zeros(1, dtype=torch.long)
            root_cls_gt[0] = self.label_to_id[gt_node.label]
            root_cls_gt = root_cls_gt.to("cuda:{}".format(cuda_device))
            root_cls_loss = self.childrenCELoss(root_cls_pred, root_cls_gt)

            # rotation_cls_pred = self.rotation_classifier(node_latent)
            # rotation_cls_gt = torch.zeros(1, dtype=torch.long)
            # rotation_cls_gt[0] = rotation
            # rotation_cls_gt = rotation_cls_gt.to("cuda:{}".format(cuda_device))
            # rotation_cls_loss = self.childrenCELoss(rotation_cls_pred, rotation_cls_gt)
            # pred_rotation = self.softmax_layer(rotation_cls_pred)
            # pred_rotation = int(torch.argmax(pred_rotation).cpu().detach().numpy())
        else:
            root_cls_loss = 0

            rotation_cls_loss = 0

        # ================================
        # if gt_sem_code is None:
        #     gt_sem_code = torch.zeros((1, 16)).cuda()
        # node_latent = self.latent_sem_encoder(torch.cat([node_latent, gt_sem_code], dim=1))
        # ================================

        geo_mask, geo_feat, decoder_features = self.node_decoders[clip_level](node_latent, mask_code, mask_feature)
        S_prior = torch.zeros_like(gt_geo)[None, ...]
        geo_loss = self.voxelLoss(geo_mask[:, 0, ...], gt_geo).mean()

        if gt_node.is_leaf:
            loss_dict = {}

            is_leaf_logit = self.leaf_classifiers[clip_level](node_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit,
                                                    is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1))

            loss_dict['leaf'] = is_leaf_loss
            loss_dict['geo'] = geo_loss
            loss_dict['geo_prior'] = 0
            loss_dict['exists'] = torch.zeros_like(is_leaf_loss)
            loss_dict['semantic'] = torch.zeros_like(is_leaf_loss)
            loss_dict['edge_exists'] = torch.zeros_like(is_leaf_loss)
            loss_dict['root_cls'] = root_cls_loss

            return loss_dict, geo_mask, geo_mask, gt_geo, geo_feat, S_prior, decoder_features
        else:

            # ======================
            child_gt_sem_embeddings = []
            for j, child_node in enumerate(gt_node.children):
                child_gt_sem_vector = torch.zeros((1, Tree.num_sem))
                child_gt_sem_vector[0, child_node.get_semantic_id()] = 1
                child_gt_sem_vector = child_gt_sem_vector.to(cuda_device)
                child_gt_sem_embedding = self.gt_label_encoder(child_gt_sem_vector)
                child_gt_sem_embeddings += [child_gt_sem_embedding]
            num_gt_children = len(child_gt_sem_embeddings)
            if num_gt_children >= 2:
                child_gt_sem_full_embedding = self.gt_label_full_encoder(torch.cat([child_gt_sem_embeddings[0], child_gt_sem_embeddings[1]], dim=1))
            else:
                child_gt_sem_full_embedding = self.gt_label_full_encoder(torch.cat([child_gt_sem_embeddings[0], child_gt_sem_embeddings[0]], dim=1))
            if num_gt_children > 2:
                for j in range(num_gt_children - 2):
                    child_gt_sem_full_embedding = self.gt_label_full_encoder(torch.cat([child_gt_sem_full_embedding, child_gt_sem_embeddings[j + 2]], dim=1))
            child_gt_num_vector = torch.zeros((1, 10))
            child_gt_num_vector[0, num_gt_children] = 1
            child_gt_num_vector = child_gt_num_vector.to(cuda_device)
            # ======================


            loss_dict = {}
            # child_feats, child_sem_logits, child_exists_logits, edge_exists_logits = \
            #     self.child_decoders[clip_level](node_latent)
            child_feats, child_sem_logits, child_exists_logits, edge_exists_logits = \
                self.child_decoders[clip_level](node_latent, child_gt_sem_full_embedding, child_gt_num_vector)

            all_geo = []
            all_gt_geo = []
            all_leaf_geo = []
            all_geo_features = []
            all_leaf_S_priors = []
            all_child_decoder_features = []

            num_pred = child_feats.size(1)

            with torch.no_grad():
                child_gt_geo = torch.cat(
                    [child_node.geo for child_node in gt_node.children], dim=0)
                child_gt_geo = child_gt_geo.unsqueeze(dim=1)
                num_gt = child_gt_geo.size(0)

                child_gt_sem_vectors = []
                child_gt_sem_classes = torch.zeros(num_gt)
                for j, child_node in enumerate(gt_node.children):
                    child_gt_sem_vector = torch.zeros((1, Tree.num_sem))
                    child_gt_sem_vector[0, child_node.get_semantic_id()] = 1
                    child_gt_sem_vectors += [child_gt_sem_vector]
                    child_gt_sem_classes[j] = child_node.get_semantic_id()
                child_gt_sem_classes = child_gt_sem_classes.long()

                child_sem_logits_tiled = child_sem_logits[0].unsqueeze(dim=0).repeat(num_gt, 1, 1)
                child_gt_sem_classes_tiled = child_gt_sem_classes.unsqueeze(dim=1).repeat(1, num_pred).to(cuda_device)

                # get edge ground truth
                edge_type_list_gt, edge_indices_gt = gt_node.edge_tensors(
                    edge_types=self.edge_types, device=child_feats.device, type_onehot=False)

                dist_mat = self.semCELoss(child_sem_logits_tiled.view(-1, Tree.num_sem),
                                          child_gt_sem_classes_tiled.view(-1)).view(1, num_gt, num_pred)

                _, matched_gt_idx, matched_pred_idx = linear_assignment(dist_mat)

                gt2pred = {gt_idx: pred_idx for gt_idx, pred_idx in zip(matched_gt_idx, matched_pred_idx)}
                edge_exists_gt = torch.zeros_like(edge_exists_logits)

                adj_from = []
                adj_to = []
                sym_from = []
                sym_to = []
                for i in range(edge_indices_gt.shape[1] // 2):
                    if edge_indices_gt[0, i, 0].item() not in gt2pred or edge_indices_gt[0, i, 1].item() not in gt2pred:
                        """
                            one of the adjacent nodes of the current gt edge was not matched
                            to any node in the prediction, ignore this edge
                        """
                        continue

                    # correlate gt edges to pred edges
                    edge_from_idx = gt2pred[edge_indices_gt[0, i, 0].item()]
                    edge_to_idx = gt2pred[edge_indices_gt[0, i, 1].item()]
                    edge_exists_gt[:, edge_from_idx, edge_to_idx, edge_type_list_gt[0:1, i]] = 1
                    edge_exists_gt[:, edge_to_idx, edge_from_idx, edge_type_list_gt[0:1, i]] = 1

                    # compute binary edge parameters for each matched pred edge
                    if edge_type_list_gt[0, i].item() == 0:  # ADJ
                        adj_from.append(edge_from_idx)
                        adj_to.append(edge_to_idx)
                    else:  # SYM
                        sym_from.append(edge_from_idx)
                        sym_to.append(edge_to_idx)

            # train the current node to be non-leaf
            is_leaf_logit = self.leaf_classifiers[clip_level](node_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit,
                                                    is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1))

            # gather information
            child_sem_gt_labels = []
            child_sem_pred_logits = []
            child_exists_gt = torch.zeros_like(child_exists_logits)
            for i in range(len(matched_gt_idx)):
                child_sem_gt_labels.append(gt_node.children[matched_gt_idx[i]].get_semantic_id())
                child_sem_pred_logits.append(child_sem_logits[0, matched_pred_idx[i], :].view(1, -1))
                child_exists_gt[:, matched_pred_idx[i], :] = 1

            # train semantic labels
            child_sem_pred_logits = torch.cat(child_sem_pred_logits, dim=0)
            child_sem_gt_labels = torch.tensor(child_sem_gt_labels, dtype=torch.int64,
                                               device=child_sem_pred_logits.device)
            semantic_loss = self.semCELoss(child_sem_pred_logits, child_sem_gt_labels)
            semantic_loss = semantic_loss.sum()

            # train exist scores
            child_exists_loss = F.binary_cross_entropy_with_logits(
                input=child_exists_logits, target=child_exists_gt, reduction='none')
            child_exists_loss = child_exists_loss.sum()

            # train edge exists scores
            edge_exists_loss = F.binary_cross_entropy_with_logits(
                input=edge_exists_logits, target=edge_exists_gt, reduction='none')
            edge_exists_loss = edge_exists_loss.sum()
            # rescale to make it comparable to other losses,
            # which are in the order of the number of child nodes
            edge_exists_loss = edge_exists_loss / (edge_exists_gt.shape[2] * edge_exists_gt.shape[3])

            # call children + aggregate losses
            pred2allgeo = dict()
            pred2allleafgeo = dict()
            for i in range(len(matched_gt_idx)):
                # child_losses, child_all_geo, child_all_leaf_geo, child_all_gt_geo, child_all_geo_features, child_all_leaf_S_priors, child_decoder_features = self.node_recon_loss_latentless(
                #     child_feats[:, matched_pred_idx[i], :], gt_node.children[matched_gt_idx[i]], level + 1,
                #     children_roots, mask_code, mask_feature, child_sem_pred_logits[i], scan_geo=scan_geo,
                #     encoder_features=encoder_features, rotation=rotation, pred_rotation=pred_rotation)
                child_losses, child_all_geo, child_all_leaf_geo, child_all_gt_geo, child_all_geo_features, child_all_leaf_S_priors, child_decoder_features = self.node_recon_loss_latentless(
                    child_feats[:, matched_pred_idx[i], :], gt_node.children[matched_gt_idx[i]], level + 1,
                    children_roots, mask_code, mask_feature, child_sem_pred_logits[i], scan_geo=scan_geo,
                    encoder_features=encoder_features, rotation=rotation, pred_rotation=pred_rotation,
                    gt_sem_code=child_gt_sem_embeddings[matched_gt_idx[i]])

                pred2allgeo[matched_pred_idx[i]] = child_all_geo
                pred2allleafgeo[matched_pred_idx[i]] = child_all_leaf_geo

                all_geo.append(child_all_geo)
                all_gt_geo.append(child_all_gt_geo)
                all_leaf_geo.append(child_all_leaf_geo)
                all_geo_features.append(child_all_geo_features)
                all_leaf_S_priors.append(child_all_leaf_S_priors)
                all_child_decoder_features.append(child_decoder_features)

                geo_loss = geo_loss + child_losses['geo']
                root_cls_loss = root_cls_loss + child_losses['root_cls']
                is_leaf_loss = is_leaf_loss + child_losses['leaf']
                child_exists_loss = child_exists_loss + child_losses['exists']
                semantic_loss = semantic_loss + child_losses['semantic']
                edge_exists_loss = edge_exists_loss + child_losses['edge_exists']

            loss_dict['leaf'] = is_leaf_loss.view((1))
            loss_dict['geo'] = geo_loss.view((1))
            loss_dict['geo_prior'] = torch.zeros_like(loss_dict['leaf']).view((1))
            loss_dict['exists'] = child_exists_loss.view((1))
            loss_dict['semantic'] = semantic_loss.view((1))
            loss_dict['edge_exists'] = edge_exists_loss.view((1))
            loss_dict['root_cls'] = root_cls_loss.view((1))

            child_feats_output = child_feats[:, matched_pred_idx, :]

            return loss_dict, torch.cat(all_geo, dim=0), torch.cat(all_leaf_geo, dim=0), torch.cat(all_leaf_S_priors, dim=0), \
                   child_feats_output, child_sem_pred_logits, all_child_decoder_features, child_sem_gt_labels

    def tto_loss(self, node_latent, gt_node, level=0, children_roots=None,
                                   mask_code=None, mask_feature=None, child_sem_logit=None,
                                   scan_geo=None, encoder_features=None, rotation=None, pred_rotation=None,
                                   gt_sem_code=None):
        if self.split_subnetworks:
            if level < 1:
                clip_level = level
            else:
                clip_level = 0
        else:
            clip_level = -1

        gt_geo = gt_node.geo  # torch.Tensor[1, 64, 64, 64]
        cuda_device = gt_geo.get_device()

        if level == 0:
            root_cls_pred = self.root_classifier(node_latent)
            root_cls_gt = torch.zeros(1, dtype=torch.long)
            root_cls_gt[0] = self.label_to_id[gt_node.label]
            root_cls_gt = root_cls_gt.to("cuda:{}".format(cuda_device))
            root_cls_loss = self.childrenCELoss(root_cls_pred, root_cls_gt)
        else:
            root_cls_loss = 0

        # ================================
        # if gt_sem_code is None:
        #     gt_sem_code = torch.zeros((1, 16)).cuda()
        # node_latent = self.latent_sem_encoder(torch.cat([node_latent, gt_sem_code], dim=1))
        # ================================

        geo_mask, geo_feat, decoder_features = self.node_decoders[clip_level](node_latent, mask_code, mask_feature)
        S_prior = torch.zeros_like(gt_geo)[None, ...]
        if level == 0:
            geo_loss = self.voxelLoss(geo_mask[:, 0, ...], scan_geo[0]).mean()
        else:
            # self.children_geos += [geo_mask]
            geo_loss = 0

        if gt_node.is_leaf:
            loss_dict = {}

            is_leaf_logit = self.leaf_classifiers[clip_level](node_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit,
                                                    is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1))

            loss_dict['leaf'] = is_leaf_loss
            loss_dict['geo'] = geo_loss
            loss_dict['geo_prior'] = 0
            loss_dict['exists'] = torch.zeros_like(is_leaf_loss)
            loss_dict['semantic'] = torch.zeros_like(is_leaf_loss)
            loss_dict['edge_exists'] = torch.zeros_like(is_leaf_loss)
            loss_dict['root_cls'] = root_cls_loss

            return loss_dict, geo_mask, geo_mask, gt_geo, geo_feat, S_prior, decoder_features
        else:

            # ======================
            child_gt_sem_embeddings = []
            for j, child_node in enumerate(gt_node.children):
                child_gt_sem_vector = torch.zeros((1, Tree.num_sem))
                child_gt_sem_vector[0, child_node.get_semantic_id()] = 1
                child_gt_sem_vector = child_gt_sem_vector.to(cuda_device)
                child_gt_sem_embedding = self.gt_label_encoder(child_gt_sem_vector)
                child_gt_sem_embeddings += [child_gt_sem_embedding]
            num_gt_children = len(child_gt_sem_embeddings)
            if num_gt_children >= 2:
                child_gt_sem_full_embedding = self.gt_label_full_encoder(torch.cat([child_gt_sem_embeddings[0], child_gt_sem_embeddings[1]], dim=1))
            else:
                child_gt_sem_full_embedding = self.gt_label_full_encoder(torch.cat([child_gt_sem_embeddings[0], child_gt_sem_embeddings[0]], dim=1))
            if num_gt_children > 2:
                for j in range(num_gt_children - 2):
                    child_gt_sem_full_embedding = self.gt_label_full_encoder(torch.cat([child_gt_sem_full_embedding, child_gt_sem_embeddings[j + 2]], dim=1))
            child_gt_num_vector = torch.zeros((1, 10))
            child_gt_num_vector[0, num_gt_children] = 1
            child_gt_num_vector = child_gt_num_vector.to(cuda_device)
            # ======================


            loss_dict = {}
            # child_feats, child_sem_logits, child_exists_logits, edge_exists_logits = \
            #     self.child_decoders[clip_level](node_latent)
            child_feats, child_sem_logits, child_exists_logits, edge_exists_logits = \
                self.child_decoders[clip_level](node_latent, child_gt_sem_full_embedding, child_gt_num_vector)

            all_geo = []
            all_gt_geo = []
            all_leaf_geo = []
            all_geo_features = []
            all_leaf_S_priors = []
            all_child_decoder_features = []

            num_pred = child_feats.size(1)

            with torch.no_grad():
                child_gt_geo = torch.cat(
                    [child_node.geo for child_node in gt_node.children], dim=0)
                child_gt_geo = child_gt_geo.unsqueeze(dim=1)
                num_gt = child_gt_geo.size(0)

                child_gt_sem_vectors = []
                child_gt_sem_classes = torch.zeros(num_gt)
                for j, child_node in enumerate(gt_node.children):
                    child_gt_sem_vector = torch.zeros((1, Tree.num_sem))
                    child_gt_sem_vector[0, child_node.get_semantic_id()] = 1
                    child_gt_sem_vectors += [child_gt_sem_vector]
                    child_gt_sem_classes[j] = child_node.get_semantic_id()
                child_gt_sem_classes = child_gt_sem_classes.long()

                child_sem_logits_tiled = child_sem_logits[0].unsqueeze(dim=0).repeat(num_gt, 1, 1)
                child_gt_sem_classes_tiled = child_gt_sem_classes.unsqueeze(dim=1).repeat(1, num_pred).to(cuda_device)

                # get edge ground truth
                edge_type_list_gt, edge_indices_gt = gt_node.edge_tensors(
                    edge_types=self.edge_types, device=child_feats.device, type_onehot=False)

                dist_mat = self.semCELoss(child_sem_logits_tiled.view(-1, Tree.num_sem),
                                          child_gt_sem_classes_tiled.view(-1)).view(1, num_gt, num_pred)

                _, matched_gt_idx, matched_pred_idx = linear_assignment(dist_mat)

                gt2pred = {gt_idx: pred_idx for gt_idx, pred_idx in zip(matched_gt_idx, matched_pred_idx)}
                edge_exists_gt = torch.zeros_like(edge_exists_logits)

                adj_from = []
                adj_to = []
                sym_from = []
                sym_to = []
                for i in range(edge_indices_gt.shape[1] // 2):
                    if edge_indices_gt[0, i, 0].item() not in gt2pred or edge_indices_gt[0, i, 1].item() not in gt2pred:
                        """
                            one of the adjacent nodes of the current gt edge was not matched
                            to any node in the prediction, ignore this edge
                        """
                        continue

                    # correlate gt edges to pred edges
                    edge_from_idx = gt2pred[edge_indices_gt[0, i, 0].item()]
                    edge_to_idx = gt2pred[edge_indices_gt[0, i, 1].item()]
                    edge_exists_gt[:, edge_from_idx, edge_to_idx, edge_type_list_gt[0:1, i]] = 1
                    edge_exists_gt[:, edge_to_idx, edge_from_idx, edge_type_list_gt[0:1, i]] = 1

                    # compute binary edge parameters for each matched pred edge
                    if edge_type_list_gt[0, i].item() == 0:  # ADJ
                        adj_from.append(edge_from_idx)
                        adj_to.append(edge_to_idx)
                    else:  # SYM
                        sym_from.append(edge_from_idx)
                        sym_to.append(edge_to_idx)

            # train the current node to be non-leaf
            is_leaf_logit = self.leaf_classifiers[clip_level](node_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit,
                                                    is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1))

            # gather information
            child_sem_gt_labels = []
            child_sem_pred_logits = []
            child_exists_gt = torch.zeros_like(child_exists_logits)
            for i in range(len(matched_gt_idx)):
                child_sem_gt_labels.append(gt_node.children[matched_gt_idx[i]].get_semantic_id())
                child_sem_pred_logits.append(child_sem_logits[0, matched_pred_idx[i], :].view(1, -1))
                child_exists_gt[:, matched_pred_idx[i], :] = 1

            # train semantic labels
            child_sem_pred_logits = torch.cat(child_sem_pred_logits, dim=0)
            child_sem_gt_labels = torch.tensor(child_sem_gt_labels, dtype=torch.int64,
                                               device=child_sem_pred_logits.device)
            semantic_loss = self.semCELoss(child_sem_pred_logits, child_sem_gt_labels)
            semantic_loss = semantic_loss.sum()

            # train exist scores
            child_exists_loss = F.binary_cross_entropy_with_logits(
                input=child_exists_logits, target=child_exists_gt, reduction='none')
            child_exists_loss = child_exists_loss.sum()

            # train edge exists scores
            edge_exists_loss = F.binary_cross_entropy_with_logits(
                input=edge_exists_logits, target=edge_exists_gt, reduction='none')
            edge_exists_loss = edge_exists_loss.sum()
            # rescale to make it comparable to other losses,
            # which are in the order of the number of child nodes
            edge_exists_loss = edge_exists_loss / (edge_exists_gt.shape[2] * edge_exists_gt.shape[3])

            # call children + aggregate losses
            pred2allgeo = dict()
            pred2allleafgeo = dict()
            for i in range(len(matched_gt_idx)):
                # child_losses, child_all_geo, child_all_leaf_geo, child_all_gt_geo, child_all_geo_features, child_all_leaf_S_priors, child_decoder_features = self.tto_loss(
                #     child_feats[:, matched_pred_idx[i], :], gt_node.children[matched_gt_idx[i]], level + 1,
                #     children_roots, mask_code, mask_feature, child_sem_pred_logits[i], scan_geo=scan_geo,
                #     encoder_features=encoder_features, rotation=rotation, pred_rotation=pred_rotation)
                child_losses, child_all_geo, child_all_leaf_geo, child_all_gt_geo, child_all_geo_features, child_all_leaf_S_priors, child_decoder_features = self.tto_loss(
                    child_feats[:, matched_pred_idx[i], :], gt_node.children[matched_gt_idx[i]], level + 1,
                    children_roots, mask_code, mask_feature, child_sem_pred_logits[i], scan_geo=scan_geo,
                    encoder_features=encoder_features, rotation=rotation, pred_rotation=pred_rotation,
                    gt_sem_code=child_gt_sem_embeddings[matched_gt_idx[i]])

                pred2allgeo[matched_pred_idx[i]] = child_all_geo
                pred2allleafgeo[matched_pred_idx[i]] = child_all_leaf_geo

                all_geo.append(child_all_geo)
                all_gt_geo.append(child_all_gt_geo)
                all_leaf_geo.append(child_all_leaf_geo)
                all_geo_features.append(child_all_geo_features)
                all_leaf_S_priors.append(child_all_leaf_S_priors)
                all_child_decoder_features.append(child_decoder_features)

                geo_loss = geo_loss + child_losses['geo']
                root_cls_loss = root_cls_loss + child_losses['root_cls']
                is_leaf_loss = is_leaf_loss + child_losses['leaf']
                child_exists_loss = child_exists_loss + child_losses['exists']
                semantic_loss = semantic_loss + child_losses['semantic']
                edge_exists_loss = edge_exists_loss + child_losses['edge_exists']

            loss_dict['leaf'] = is_leaf_loss.view((1))
            loss_dict['geo'] = geo_loss.view((1))
            loss_dict['geo_prior'] = torch.zeros_like(loss_dict['leaf']).view((1))
            loss_dict['exists'] = child_exists_loss.view((1))
            loss_dict['semantic'] = semantic_loss.view((1))
            loss_dict['edge_exists'] = edge_exists_loss.view((1))
            loss_dict['root_cls'] = root_cls_loss.view((1))

            geo_children_loss = self.voxelLoss(torch.clamp(torch.sum(torch.stack(all_geo), dim=0), 0, 1), scan_geo).mean()
            loss_dict['geo_children'] = geo_children_loss

            child_feats_output = child_feats[:, matched_pred_idx, :]

            return loss_dict, torch.cat(all_geo, dim=0), torch.cat(all_leaf_geo, dim=0), torch.cat(all_leaf_S_priors, dim=0), \
                   child_feats_output, child_sem_pred_logits, all_child_decoder_features, child_sem_gt_labels