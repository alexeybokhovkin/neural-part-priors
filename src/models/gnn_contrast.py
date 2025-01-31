import torch
import torch.nn as nn
import torch_scatter

from ..utils.hierarchy import Tree


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

    def __init__(self, feature_size, hidden_size, feature_out_size):
        super(LatentDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_out_size)

    def forward(self, x):
        x = torch.relu(self.mlp1(x))
        x = torch.relu(self.mlp2(x))

        return x


class LatentProjector(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(LatentProjector, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, hidden_size)
        self.mlp4 = nn.Linear(hidden_size, hidden_size)
        self.mlp5 = nn.Linear(hidden_size, feature_size)

    def forward(self, x):
        x = torch.relu(self.mlp1(x))
        x = torch.relu(self.mlp2(x))
        x = torch.relu(self.mlp3(x))
        x = torch.relu(self.mlp4(x))
        x = self.mlp5(x)

        return x


class GNNChildDecoder(nn.Module):

    def __init__(self, node_feat_size, hidden_size, max_child_num,
                 edge_symmetric_type, num_iterations, edge_type_num):
        super(GNNChildDecoder, self).__init__()

        self.max_child_num = max_child_num
        self.hidden_size = hidden_size

        self.edge_symmetric_type = edge_symmetric_type
        self.num_iterations = num_iterations
        self.edge_type_num = edge_type_num

        self.mlp_parent = nn.Linear(node_feat_size, hidden_size * max_child_num)
        # ==================================
        # self.mlp_parent = nn.Linear(node_feat_size + 26, hidden_size * max_child_num)
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
        # parent_feature = torch.cat([parent_feature, gt_children_code, gt_num_code], dim=1)
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

        # print('Edge latents', edge_latents.shape)
        # print(edge_latents)

        # edge existence prediction
        edge_exists_logits_per_type = []
        for i in range(self.edge_type_num):
            edge_exists_logits_cur_type = self.mlp_edge_exists[i](edge_latents).view(
                batch_size, self.max_child_num, self.max_child_num, 1)
            edge_exists_logits_per_type.append(edge_exists_logits_cur_type)
        edge_exists_logits = torch.cat(edge_exists_logits_per_type, dim=3)
        # print('Edge exists', edge_exists_logits.shape)
        # print(edge_exists_logits)

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
        if edge_indices.shape[0] == 0:
            edge_indices = torch.LongTensor([[0, 0]]).to(edge_exists_logits.device)
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
