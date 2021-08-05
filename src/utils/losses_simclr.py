import torch
from torch import nn
import numpy as np
import gc


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, partnet_ids):
        representations = torch.cat([zjs, zis], dim=0)
        # print('Rep shape', representations.shape)

        similarity_matrix = self.similarity_function(representations, representations)
        # print('Matrix shape', similarity_matrix.shape)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        # print('L pos shape', l_pos.shape)
        # print('R pos shape', r_pos.shape)
        # print('Pos shape', positives.shape)
        # print('Neg shape', negatives.shape)
        # print(similarity_matrix[self.mask_samples_from_same_repr].shape)
        # print()

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        # print('Root logits:', logits.shape)
        # print('Root labels:', labels.shape)
        # print()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class NTXentLossClean(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLossClean, self).__init__()

        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device

        self.similarity = torch.nn.CosineSimilarity()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, zis, zjs, partnet_ids):

        final_loss = 0

        for i, zi in enumerate(zis):
            negative_similarities = []
            with torch.no_grad():
                zi_ = zi[None, ...]
                for j, zj in enumerate(zjs):
                    zj_ = zj[None, ...]
                    if i != j and partnet_ids[i] != partnet_ids[j]:
                        negative_similarities += [self.similarity(zi_, zj_) / self.temperature]
                    else:
                        positive_similarity = self.similarity(zi_, zj_) / self.temperature
                for k, zk in enumerate(zis):
                    zk_ = zk[None, ...]
                    if k != i and partnet_ids[i] != partnet_ids[k]:
                        negative_similarities += [self.similarity(zi_, zk_) / self.temperature]

            negative_similarities = [positive_similarity] + negative_similarities

            labels = torch.zeros(1).to(self.device).long()
            loss = self.criterion(torch.cat(negative_similarities)[None, ...], labels)

            final_loss += loss

            labels = labels.to('cpu')
            del labels

        del negative_similarities, positive_similarity
        gc.collect()

        return final_loss / (2 * self.batch_size)


class NTXentLossCleanProj(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLossCleanProj, self).__init__()

        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device

        self.similarity = torch.nn.CosineSimilarity()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        # size = 128
        # self.projection_head = nn.Sequential(
        #     nn.Linear(size, size),
        #     nn.ReLU(),
        #     nn.Linear(size, size),
        #     nn.ReLU(),
        #     nn.Linear(size, 128)
        # )

        size = 256
        self.projection_head = nn.Sequential(
            nn.Linear(size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, size)
        )

    def forward(self, zis, zjs, partnet_ids):

        final_loss = 0

        for i, zi in enumerate(zis):
            negative_similarities = []
            with torch.no_grad():
                zi_ = zi[None, ...]
                zi_ = self.projection_head(zi_)
                for j, zj in enumerate(zjs):
                    zj_ = zj[None, ...]
                    zj_ = self.projection_head(zj_)
                    if i != j and partnet_ids[i] != partnet_ids[j]:
                        negative_similarities += [self.similarity(zi_, zj_) / self.temperature]
                    else:
                        positive_similarity = self.similarity(zi_, zj_) / self.temperature
                for k, zk in enumerate(zis):
                    zk_ = zk[None, ...]
                    zk_ = self.projection_head(zk_)
                    if k != i and partnet_ids[i] != partnet_ids[k]:
                        negative_similarities += [self.similarity(zi_, zk_) / self.temperature]

            negative_similarities = [positive_similarity] + negative_similarities

            labels = torch.zeros(1).to(self.device).long()
            loss = self.criterion(torch.cat(negative_similarities)[None, ...], labels)

            final_loss += loss

            labels = labels.to('cpu')
            del labels

        del negative_similarities, positive_similarity
        gc.collect()

        return final_loss / (2 * self.batch_size)


class NTXentLoss2(torch.nn.Module):

    def __init__(self, device, temperature):
        super(NTXentLoss2, self).__init__()

        self.temperature = temperature
        self.device = device

        self.similarity = torch.nn.CosineSimilarity()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, x_roots, x_roots_pos, x_roots_negs):

        final_loss = 0

        for i, x_root in enumerate(x_roots):
            negative_similarities = []
            x_root_ = x_root[None, ...]
            for j, x_root_neg in enumerate(x_roots_negs[i]):
                x_root_neg_ = x_root_neg
                negative_similarities += [self.similarity(x_root_, x_root_neg_) / self.temperature]
            x_root_pos_ = x_roots_pos
            positive_similarity = self.similarity(x_root_, x_root_pos_)

            negative_similarities = [positive_similarity] + negative_similarities

            labels = torch.zeros(1).to(self.device).long()
            loss = self.criterion(torch.cat(negative_similarities)[None, ...], labels)

            final_loss += loss

        return final_loss / (2 * x_roots.shape[0])


class NTXentLossChildren(torch.nn.Module):

    def __init__(self, device, batch_size, temperature):
        super(NTXentLossChildren, self).__init__()

        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device

        self.similarity = torch.nn.CosineSimilarity()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, children_is, children_js, partnet_ids):

        final_loss = 0

        for i, children_i in enumerate(children_is):
            children_i_concat = torch.flatten(children_i)[None, ...]
            num_children_i = len(children_i)
            negative_similarities = []
            # with torch.no_grad():
            for j, children_j in enumerate(children_js):
                num_children_j = len(children_j)
                # negatives
                if i != j:
                    negative_indices = np.random.choice(num_children_j, num_children_i)
                    children_j_concat = torch.flatten(children_j[negative_indices])[None, ...]
                    negative_similarities += [self.similarity(children_i_concat, children_j_concat) / self.temperature]
                    # print('i shape', children_i_concat.shape)
                    # print('j shape', children_j_concat.shape)
                    # print('negative indices', negative_indices)
                    # print('sim shape', self.similarity(children_i_concat, children_j_concat).shape)
                    # print()
                # positive
                else:
                    children_j_concat_pos = torch.flatten(children_j)[None, ...]
                    positive_similarity = self.similarity(children_i_concat, children_j_concat_pos) / self.temperature
            # negatives from another samples in batch
            for k, children_k in enumerate(children_is):
                num_children_k = len(children_k)
                if k != i:
                    negative_indices = np.random.choice(num_children_k, num_children_i)
                    children_k_concat = torch.flatten(children_k[negative_indices])[None, ...]
                    negative_similarities += [self.similarity(children_i_concat, children_k_concat) / self.temperature]
            # end of torch.no_grad()

            negative_similarities = [positive_similarity] + negative_similarities

            labels = torch.zeros(1).to(self.device).long()
            loss = self.criterion(torch.cat(negative_similarities)[None, ...], labels)

            final_loss += loss

        return final_loss / (2 * self.batch_size)


class NTXentLossChildrenClean(torch.nn.Module):

    def __init__(self, device, batch_size, temperature):
        super(NTXentLossChildrenClean, self).__init__()

        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device

        self.similarity = torch.nn.CosineSimilarity()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, children_is, children_js, partnet_ids):

        final_loss = 0

        for i, children_i in enumerate(children_is):
            if len(children_i.shape) == 1:
                children_i = children_i[None, ...]
            children_i_concat = torch.flatten(children_i)[None, ...]
            num_children_i = len(children_i)
            negative_similarities = []
            # with torch.no_grad():
            for j, children_j in enumerate(children_js):
                if len(children_j.shape) == 1:
                    children_j = children_j[None, ...]
                num_children_j = len(children_j)
                # negatives
                if i != j and partnet_ids[i] != partnet_ids[j]:
                    negative_indices = np.random.choice(num_children_j, num_children_i)
                    children_j_concat = torch.flatten(children_j[negative_indices])[None, ...]
                    negative_similarities += [self.similarity(children_i_concat, children_j_concat) / self.temperature]
                # positive
                else:
                    children_j_concat_pos = torch.flatten(children_j)[None, ...]
                    positive_similarity = self.similarity(children_i_concat, children_j_concat_pos) / self.temperature
            # negatives from another samples in batch
            for k, children_k in enumerate(children_is):
                if len(children_k.shape) == 1:
                    children_k = children_k[None, ...]
                num_children_k = len(children_k)
                if k != i and partnet_ids[i] != partnet_ids[k]:
                    negative_indices = np.random.choice(num_children_k, num_children_i)
                    children_k_concat = torch.flatten(children_k[negative_indices])[None, ...]
                    negative_similarities += [self.similarity(children_i_concat, children_k_concat) / self.temperature]
            # end of torch.no_grad()

            negative_similarities = [positive_similarity] + negative_similarities

            labels = torch.zeros(1).to(self.device).long()
            loss = self.criterion(torch.cat(negative_similarities)[None, ...], labels)

            final_loss += loss

        return final_loss / (2 * self.batch_size)


class NTXentLossChildrenCleanProj(torch.nn.Module):

    def __init__(self, device, batch_size, temperature):
        super(NTXentLossChildrenCleanProj, self).__init__()

        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device

        self.similarity = torch.nn.CosineSimilarity()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        # size = 128
        # self.projection_head = nn.Sequential(
        #     nn.Linear(size, size),
        #     nn.ReLU(),
        #     nn.Linear(size, size),
        #     nn.ReLU(),
        #     nn.Linear(size, 128)
        # )

        size = 256
        self.projection_head = nn.Sequential(
            nn.Linear(size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, size)
        )

    def forward(self, children_is, children_js, partnet_ids):

        final_loss = 0

        for i, children_i in enumerate(children_is):
            if len(children_i.shape) == 1:
                children_i = children_i[None, ...]
            children_i = self.projection_head(children_i)
            children_i_concat = torch.flatten(children_i)[None, ...]
            num_children_i = len(children_i)
            negative_similarities = []
            for j, children_j in enumerate(children_js):
                if len(children_j.shape) == 1:
                    children_j = children_j[None, ...]
                children_j = self.projection_head(children_j)
                num_children_j = len(children_j)
                # negatives
                if i != j and partnet_ids[i] != partnet_ids[j]:
                    negative_indices = np.random.choice(num_children_j, num_children_i)
                    children_j_concat = torch.flatten(children_j[negative_indices])[None, ...]
                    negative_similarities += [self.similarity(children_i_concat, children_j_concat) / self.temperature]
                # positive
                else:
                    children_j_concat_pos = torch.flatten(children_j)[None, ...]
                    positive_similarity = self.similarity(children_i_concat, children_j_concat_pos) / self.temperature
            # negatives from another samples in batch
            for k, children_k in enumerate(children_is):
                if len(children_k.shape) == 1:
                    children_k = children_k[None, ...]
                children_k = self.projection_head(children_k)
                num_children_k = len(children_k)
                if k != i and partnet_ids[i] != partnet_ids[k]:
                    negative_indices = np.random.choice(num_children_k, num_children_i)
                    children_k_concat = torch.flatten(children_k[negative_indices])[None, ...]
                    negative_similarities += [self.similarity(children_i_concat, children_k_concat) / self.temperature]

            negative_similarities = [positive_similarity] + negative_similarities

            labels = torch.zeros(1).to(self.device).long()
            loss = self.criterion(torch.cat(negative_similarities)[None, ...], labels)

            final_loss += loss

        return final_loss / (2 * self.batch_size)


class NTXentLossChildren2(torch.nn.Module):

    def __init__(self, device, temperature):
        super(NTXentLossChildren2, self).__init__()

        self.temperature = temperature
        self.device = device

        self.similarity = torch.nn.CosineSimilarity()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, childrens, childrens_pos, childrens_negs, childrens_gt_labels, childrens_gt_label_negs):

        final_loss = 0

        for i, children in enumerate(childrens):
            for j, child in enumerate(children):
                negative_similarities = []
                child_gt_label = childrens_gt_labels[i][j]
                child_pos_ = childrens_pos[i][j][None, ...]
                child_ = child[None, ...]
                positive_similarity = self.similarity(child_, child_pos_) / self.temperature

                for k, children_gt_label_neg in enumerate(childrens_gt_label_negs[i]):
                    for m in range(len(children_gt_label_neg)):
                        child_gt_label_neg = children_gt_label_neg[m]
                        if child_gt_label_neg == child_gt_label:
                            child_neg_ = childrens_negs[i][k][m][None, ...]
                            negative_similarities += [self.similarity(child_, child_neg_) / self.temperature]
                negative_similarities = [positive_similarity] + negative_similarities

            labels = torch.zeros(1).to(self.device).long()
            loss = self.criterion(torch.cat(negative_similarities)[None, ...], labels)

            final_loss += loss

        return final_loss / (2 * len(childrens))


class NTXentLossChildrenPartwise(torch.nn.Module):

    def __init__(self, device, batch_size, temperature):
        super(NTXentLossChildrenPartwise, self).__init__()

        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device

        self.similarity = torch.nn.CosineSimilarity()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, children_ts, children_rs, partnet_ids):

        final_loss = 0

        for t, children_t in enumerate(children_ts):
            children_r = children_rs[t]
            if len(children_t.shape) == 1 or len(children_r.shape) == 1:
                continue
            for i, zi in enumerate(children_t):
                negative_similarities = []
                zi_ = zi[None, ...]
                for j, zj in enumerate(children_r):
                    zj_ = zj[None, ...]
                    if i != j:
                        negative_similarities += [self.similarity(zi_, zj_) / self.temperature]
                    else:
                        positive_similarity = self.similarity(zi_, zj_) / self.temperature
                for k, zk in enumerate(children_t):
                    zk_ = zk[None, ...]
                    if k != i:
                        negative_similarities += [self.similarity(zi_, zk_) / self.temperature]

                negative_similarities = [positive_similarity] + negative_similarities

                labels = torch.zeros(1).to(self.device).long()
                loss = self.criterion(torch.cat(negative_similarities)[None, ...], labels)

                final_loss += loss

        return final_loss / (2 * self.batch_size)


class NTXentLossChildrenFmaps(torch.nn.Module):

    def __init__(self, device, batch_size, temperature):
        super(NTXentLossChildrenFmaps, self).__init__()

        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device

        self.similarity = torch.nn.CosineSimilarity()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, children_is_fmaps, children_js_fmaps, partnet_ids):

        final_loss = 0

        for fmap_num in range(3):
            for i, children_i_fmap in enumerate(children_is_fmaps[fmap_num]):
                children_i_concat = torch.flatten(torch.cat(children_i_fmap))[None, ...]
                num_children_i = len(children_i_fmap)
                negative_similarities = []
                # with torch.no_grad():
                for j, children_j_fmap in enumerate(children_js_fmaps[fmap_num]):
                    num_children_j = len(children_j_fmap)
                    # negatives
                    if i != j:
                        negative_indices = np.random.choice(num_children_j, num_children_i)
                        children_j_concat = torch.flatten(torch.cat(children_j_fmap)[negative_indices])[None, ...]
                        negative_similarities += [self.similarity(children_i_concat, children_j_concat) / self.temperature]
                    # positive
                    else:
                        children_j_concat_pos = torch.flatten(torch.cat(children_j_fmap))[None, ...]
                        positive_similarity = self.similarity(children_i_concat, children_j_concat_pos) / self.temperature
                # negatives from another samples in batch
                for k, children_k_fmap in enumerate(children_is_fmaps[fmap_num]):
                    num_children_k = len(children_k_fmap)
                    if k != i:
                        negative_indices = np.random.choice(num_children_k, num_children_i)
                        children_k_concat = torch.flatten(torch.cat(children_k_fmap)[negative_indices])[None, ...]
                        negative_similarities += [self.similarity(children_i_concat, children_k_concat) / self.temperature]
                # end of torch.no_grad()

                negative_similarities = [positive_similarity] + negative_similarities

                labels = torch.zeros(1).to(self.device).long()
                loss = self.criterion(torch.cat(negative_similarities)[None, ...], labels)

                final_loss += loss

        return final_loss / (2 * self.batch_size)


class NTXentLossChildrenFmaps2(torch.nn.Module):

    def __init__(self, device, temperature):
        super(NTXentLossChildrenFmaps2, self).__init__()

        self.temperature = temperature
        self.device = device

        self.similarity = torch.nn.CosineSimilarity()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, childrens, childrens_pos, childrens_negs, childrens_gt_labels, childrens_gt_label_negs):

        final_loss = 0

        for i, children in enumerate(childrens):
            for j, child_fmaps in enumerate(children):
                child_gt_label = childrens_gt_labels[i][j]
                negative_similarities_0 = []
                negative_similarities_1 = []
                negative_similarities_2 = []
                for k in range(len(child_fmaps) - 1):
                    child_fmap_ = torch.flatten(child_fmaps[k])[None, ...]
                    child_fmap_pos_ = torch.flatten(childrens_pos[i][j][k])[None, ...]
                    if k == 0:
                        positive_similarity_0 = self.similarity(child_fmap_, child_fmap_pos_) / self.temperature
                    elif k == 1:
                        positive_similarity_1 = self.similarity(child_fmap_, child_fmap_pos_) / self.temperature
                    elif k == 2:
                        positive_similarity_2 = self.similarity(child_fmap_, child_fmap_pos_) / self.temperature

                for n, children_gt_label_neg in enumerate(childrens_gt_label_negs[i]):
                    for m in range(len(children_gt_label_neg)):
                        child_gt_label_neg = children_gt_label_neg[m]
                        if child_gt_label_neg == child_gt_label:
                            for k in range(len(child_fmaps) - 1):
                                child_fmap_ = torch.flatten(child_fmaps[k])[None, ...]
                                child_fmap_neg_ = torch.flatten(childrens_negs[i][n][m][k])[None, ...]
                                if k == 0:
                                    negative_similarities_0 += [self.similarity(child_fmap_, child_fmap_neg_) / self.temperature]
                                elif k == 1:
                                    negative_similarities_1 += [self.similarity(child_fmap_, child_fmap_neg_) / self.temperature]
                                elif k == 2:
                                    negative_similarities_2 += [self.similarity(child_fmap_, child_fmap_neg_) / self.temperature]
                negative_similarities_0 = [positive_similarity_0] + negative_similarities_0
                negative_similarities_1 = [positive_similarity_1] + negative_similarities_1
                negative_similarities_2 = [positive_similarity_2] + negative_similarities_2

            labels_0 = torch.zeros(1).to(self.device).long()
            labels_1 = torch.zeros(1).to(self.device).long()
            labels_2 = torch.zeros(1).to(self.device).long()
            loss_0 = self.criterion(torch.cat(negative_similarities_0)[None, ...], labels_0)
            loss_1 = self.criterion(torch.cat(negative_similarities_1)[None, ...], labels_1)
            loss_2 = self.criterion(torch.cat(negative_similarities_2)[None, ...], labels_2)

            final_loss += (loss_0 + loss_1 + loss_2)

        return final_loss / (2 * len(childrens))


class NTXentLossChildrenFmapsClean(torch.nn.Module):

    def __init__(self, device, batch_size, temperature):
        super(NTXentLossChildrenFmapsClean, self).__init__()

        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device

        self.similarity = torch.nn.CosineSimilarity()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, children_is_fmaps, children_js_fmaps, partnet_ids):

        final_loss = 0

        for fmap_num in range(3):
            for i, children_i_fmap in enumerate(children_is_fmaps[fmap_num]):
                num_children_i = len(children_i_fmap)
                if num_children_i == 0:
                    continue
                children_i_concat = torch.flatten(torch.cat(children_i_fmap))[None, ...]
                negative_similarities = []
                # with torch.no_grad():
                for j, children_j_fmap in enumerate(children_js_fmaps[fmap_num]):
                    num_children_j = len(children_j_fmap)
                    # negatives
                    if i != j and partnet_ids[i] != partnet_ids[j] and num_children_j != 0:
                        negative_indices = np.random.choice(num_children_j, num_children_i)
                        children_j_concat = torch.flatten(torch.cat(children_j_fmap)[negative_indices])[None, ...]
                        negative_similarities += [
                            self.similarity(children_i_concat, children_j_concat) / self.temperature]
                    # positive
                    elif num_children_j != 0:
                        children_j_concat_pos = torch.flatten(torch.cat(children_j_fmap))[None, ...]
                        positive_similarity = self.similarity(children_i_concat,
                                                              children_j_concat_pos) / self.temperature
                # negatives from another samples in batch
                for k, children_k_fmap in enumerate(children_is_fmaps[fmap_num]):
                    num_children_k = len(children_k_fmap)
                    if k != i and partnet_ids[i] != partnet_ids[k] and num_children_k != 0:
                        negative_indices = np.random.choice(num_children_k, num_children_i)
                        children_k_concat = torch.flatten(torch.cat(children_k_fmap)[negative_indices])[None, ...]
                        negative_similarities += [
                            self.similarity(children_i_concat, children_k_concat) / self.temperature]
                # end of torch.no_grad()

                negative_similarities = [positive_similarity] + negative_similarities

                labels = torch.zeros(1).to(self.device).long()
                loss = self.criterion(torch.cat(negative_similarities)[None, ...], labels)

                final_loss += loss

        return final_loss / (2 * self.batch_size)


class NTXentLossChildrenFmapsPartwise(torch.nn.Module):

    def __init__(self, device, batch_size, temperature):
        super(NTXentLossChildrenFmapsPartwise, self).__init__()

        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device

        self.similarity = torch.nn.CosineSimilarity()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, children_ts_fmaps, children_rs_fmaps, partnet_ids):

        final_loss = 0

        for fmap_num in range(3):
            for t, children_t_fmap in enumerate(children_ts_fmaps[fmap_num]):
                children_r_fmap = children_rs_fmaps[fmap_num][t]
                num_children_r = len(children_r_fmap)
                num_children_t = len(children_t_fmap)
                if num_children_r == 0 or num_children_t == 0:
                    continue
                for i, zi in enumerate(children_t_fmap):
                    negative_similarities = []
                    zi_ = torch.flatten(zi)[None, ...]
                    for j, zj in enumerate(children_r_fmap):
                        zj_ = torch.flatten(zj)[None, ...]
                        if i != j:
                            negative_similarities += [self.similarity(zi_, zj_) / self.temperature]
                        else:
                            positive_similarity = self.similarity(zi_, zj_) / self.temperature
                    for k, zk in enumerate(children_t_fmap):
                        zk_ = torch.flatten(zk)[None, ...]
                        if k != i:
                            negative_similarities += [self.similarity(zi_, zk_) / self.temperature]

                    negative_similarities = [positive_similarity] + negative_similarities

                    labels = torch.zeros(1).to(self.device).long()
                    loss = self.criterion(torch.cat(negative_similarities)[None, ...], labels)

                    final_loss += loss

        return final_loss / (2 * self.batch_size)