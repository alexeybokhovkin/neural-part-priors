import sys
import numpy as np
import torch
from scipy.spatial import cKDTree as KDTree

sys.path.append('/home/bohovkin/cluster/abokhovkin_home/external_packages/chamferdist') # path to chamfer distance
from chamferdist import ChamferDistance
chamfer_module = ChamferDistance()


def compute_classification_metrics(pred_points, gt_points):
    cd_all_parts = {}
    cd_exist_parts = {}
    iou_all_parts = {}
    iou_exist_parts = {}

    all_parts = list(set(list(pred_points.keys())).union(set(list(gt_points.keys()))))
    all_parts = list(set(all_parts) - {'full'})

    part_to_id = {part: i for i, part in enumerate(all_parts)}
    id_to_part = {u: v for v, u in part_to_id.items()}

    all_pred_points, all_pred_labels = [], []
    for part in pred_points:
        if part != 'full':
            all_pred_points += [pred_points[part]]
            all_pred_labels += [part_to_id[part] for _ in range(len(pred_points[part]))]
    all_pred_points = np.vstack(all_pred_points)
    all_pred_labels = np.array(all_pred_labels)
    all_gt_points, all_gt_labels = [], []
    for part in gt_points:
        if part != 'full':
            all_gt_points += [gt_points[part]]
            all_gt_labels += [part_to_id[part] for _ in range(len(gt_points[part]))]
    all_gt_points = np.vstack(all_gt_points)
    all_gt_labels = np.array(all_gt_labels)

    kd_tree = KDTree(all_pred_points)
    min_dist, min_idx = kd_tree.query(all_gt_points)
    projected_pred_labels = all_pred_labels[min_idx]
    all_unique_labels = np.unique(np.hstack([all_gt_labels, projected_pred_labels]))

    for part in all_parts:
        if part not in pred_points:
            pc_2 = [[0, 0, 0]]
        else:
            pc_2 = pred_points[part]
        if part not in gt_points:
            pc_1 = [[0, 0, 0]]
        else:
            pc_1 = gt_points[part]
        pc_1 = torch.FloatTensor(pc_1)[None, ...].cuda()
        pc_2 = torch.FloatTensor(pc_2)[None, ...].cuda()
        dist1, dist2, idx1, idx2 = chamfer_module(pc_1, pc_2)
        distance = float(0.5 * (dist1.mean() + dist2.mean()).cpu().numpy())

        cd_all_parts[part] = distance
        if part in pred_points and part in gt_points:
            cd_exist_parts[part] = distance

    for label in all_unique_labels:
        mask_gt = np.where(all_gt_labels == label, 1, 0)
        mask_pred = np.where(projected_pred_labels == label, 1, 0)
        iou_score = float(np.sum(mask_gt & mask_pred) / (1. + np.sum(mask_gt | mask_pred)))

        part_name = id_to_part[label]
        iou_all_parts[part_name] = iou_score
        if part_name in pred_points and part_name in gt_points:
            iou_exist_parts[part_name] = iou_score

    return cd_all_parts, cd_exist_parts, iou_all_parts, iou_exist_parts