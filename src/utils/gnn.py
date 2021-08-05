from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
from scipy.ndimage import rotate


def linear_assignment(distance_mat, row_counts=None, col_counts=None):
    batch_ind = []
    row_ind = []
    col_ind = []
    for i in range(distance_mat.shape[0]):
        dmat = distance_mat[i, :, :]
        if row_counts is not None:
            dmat = dmat[:row_counts[i], :]
        if col_counts is not None:
            dmat = dmat[:, :col_counts[i]]

        rind, cind = linear_sum_assignment(dmat.to('cpu').numpy())
        rind = list(rind)
        cind = list(cind)

        if len(rind) > 0:
            rind, cind = zip(*sorted(zip(rind, cind)))
            rind = list(rind)
            cind = list(cind)

        batch_ind += [i]*len(rind)
        row_ind += rind
        col_ind += cind

    return batch_ind, row_ind, col_ind


def collate_feats(b):
    return list(zip(*b))


def one_hot(inp, label_count):
    out = torch.zeros(label_count, inp.numel(), dtype=torch.uint8, device=inp.device)
    out[inp.view(-1), torch.arange(out.shape[1])] = 1
    out = out.view((label_count,) + inp.shape)
    return out


def sym_reflect_tree(tree):
    root_geo = tree.root.geo
    root_geo = torch.flip(root_geo, dims=(-1,))
    tree.root.geo = root_geo

    for i in range(len(tree.root.children)):
        child_geo = tree.root.children[i].geo
        child_geo = torch.flip(child_geo, dims=(3,))
        tree.root.children[i].geo = child_geo
    return tree


def rotate_tree_geos(tree, angle):
    root_geo = tree.root.geo[0].cpu().numpy().astype('uint8')
    rotated_root_geo = rotate(root_geo, angle, axes=[0, 2], reshape=False).astype('float32')
    tree.root.geo = torch.FloatTensor(rotated_root_geo)[None, ...]

    for i, child in enumerate(tree.root.children):
        child_geo = child.geo[0].cpu().numpy().astype('uint8')
        rotated_child_geo = rotate(child_geo, angle, axes=[0, 2], reshape=False).astype('float32')
        tree.root.children[i].geo = torch.FloatTensor(rotated_child_geo)[None, ...]

    return tree


def rotate_tensor_by_unique_values(tensor, angle):
    rotated_tensor = np.zeros_like(tensor)
    unique_values = np.unique(tensor)
    for unique_value in unique_values:
        if unique_value != 0:
            coords = np.where(tensor == unique_value, 1, 0).astype('uint8')
            rotated_coords = rotate(coords, angle, axes=[0, 2], reshape=False)
            rotated_tensor[np.where(rotated_coords)] = unique_value

    return rotated_tensor