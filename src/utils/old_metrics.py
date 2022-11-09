import numpy as np
import torch
import math
from scipy.optimize import linear_sum_assignment
from chamferdist import ChamferDistance
from copy import deepcopy
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree as KDTree

from .transformations import apply_transform, apply_inverse_transform

chamfer_module = ChamferDistance()


def chamfer_distance(cloud_1, cloud_2):
    if len(cloud_1) == 0 or len(cloud_2) == 0:
        distance = 0
    else:
        dist_matrix = cdist(cloud_1, cloud_2)
        distance = dist_matrix.min(axis=0).mean() + dist_matrix.min(axis=1).mean()
    return distance

def make_assignment(gt_nodes, pred_nodes, thr):
    assignment = {}
    gt_nodes_with_labels = {}
    for i, node in enumerate(gt_nodes):
        if node.get_semantic_id() not in gt_nodes_with_labels:
            gt_nodes_with_labels[node.get_semantic_id()] = [i]
        else:
            gt_nodes_with_labels[node.get_semantic_id()] += [i]

    pred_nodes_with_labels = {}
    for i, node in enumerate(pred_nodes):
        if node.get_semantic_id() not in pred_nodes_with_labels:
            pred_nodes_with_labels[node.get_semantic_id()] = [i]
        else:
            pred_nodes_with_labels[node.get_semantic_id()] += [i]

    for gt_label in gt_nodes_with_labels:
        if gt_label not in pred_nodes_with_labels:
            for node in gt_nodes_with_labels[gt_label]:
                assignment[node] = -1
        else:
            if len(gt_nodes_with_labels[gt_label]) == 1 and len(pred_nodes_with_labels[gt_label]) == 1:
                assignment[gt_nodes_with_labels[gt_label][0]] = pred_nodes_with_labels[gt_label][0]
            else:
                dist_matrix = np.zeros((len(gt_nodes_with_labels[gt_label]), len(pred_nodes_with_labels[gt_label])))
                for i, gt_node in enumerate(gt_nodes_with_labels[gt_label]):
                    for j, pred_node in enumerate(pred_nodes_with_labels[gt_label]):
                        pc_1 = np.where(pred_nodes[pred_node].geo.cpu().numpy() > thr)
                        pc_1 = np.vstack([pc_1[2], pc_1[1], pc_1[0]]).T
                        pc_2 = np.where(gt_nodes[gt_node].geo.cpu().numpy())
                        pc_2 = np.vstack([pc_2[2], pc_2[1], pc_2[0]]).T
                        distance = chamfer_distance(pc_1, pc_2)
                        dist_matrix[i, j] = distance
                row_ind, col_ind = linear_sum_assignment(dist_matrix)
                for i in range(len(row_ind)):
                    assignment[gt_nodes_with_labels[gt_label][row_ind[i]]] = pred_nodes_with_labels[gt_label][
                        col_ind[i]]
    return assignment

def grid_chamfer_distance_by_part(gt_tree, pred_tree, voxel_transform):
    grid = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)
    output_dict = {}
    gt_leaves = [x for x in gt_tree.depth_first_traversal() if x.is_leaf]
    pred_leaves = [x for x in pred_tree.depth_first_traversal() if x.is_leaf]
    for thr in grid:
        children_metric = chamfer_distance_by_part_metric(gt_tree.root.children, pred_tree.root.children,
                                                          thr=thr, voxel_transform=voxel_transform)
        output_dict[thr] = [children_metric]
    return output_dict

def grid_chamfer_distance_object(gt_tree, pred_tree, voxel_transform):
    grid = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)
    output_dict = {}
    gt_leaves = [gt_tree.root]
    pred_leaves = [pred_tree.root]
    for thr in grid:
        metric = chamfer_distance_by_part_metric(gt_leaves, pred_leaves,
                                                 thr=thr, voxel_transform=voxel_transform)
        output_dict[thr] = [metric]
    return output_dict

def chamfer_distance_by_part_metric(gt_nodes, pred_nodes, thr, voxel_transform):

    assignment = make_assignment(gt_nodes, pred_nodes, 0.5)
    pred_nodes_in_assignment = [v for u, v in assignment.items()]
    extra_pred_nodes = list(set(range(len(pred_nodes))) - set(pred_nodes_in_assignment))
    part_metrics = {}
    for gt_node_id in assignment:
        if assignment[gt_node_id] != -1:
            pred_geo = np.squeeze(pred_nodes[assignment[gt_node_id]].geo.cpu().numpy())
            pc_1 = np.where(pred_geo > thr)
            pc_1 = np.vstack([pc_1[2], pc_1[1], pc_1[0]]).T

            pc_1 = apply_transform(pc_1, voxel_transform)

            if pc_1.shape[0] == 0:
                pc_1 = np.vstack([[0, 0, 0]])

            gt_geo = np.squeeze(gt_nodes[gt_node_id].geo.cpu().numpy())
            pc_2 = np.where(gt_geo)
            pc_2 = np.vstack([pc_2[2], pc_2[1], pc_2[0]]).T

            pc_2 = apply_transform(pc_2, voxel_transform)

            if pc_2.shape[0] == 0:
                pc_2 = np.vstack([[0, 0, 0]])

            pc_1 = torch.FloatTensor(pc_1)[None, ...].cuda()
            pc_2 = torch.FloatTensor(pc_2)[None, ...].cuda()
            dist1, dist2, idx1, idx2 = chamfer_module(pc_1, pc_2)
            distance = 0.5 * (dist1.mean() + dist2.mean()).cpu().numpy()

            if math.isnan(distance):
                continue
            else:
                if gt_nodes[gt_node_id].get_semantic_id() in part_metrics:
                    part_metrics[gt_nodes[gt_node_id].get_semantic_id()] += [distance]
                else:
                    part_metrics[gt_nodes[gt_node_id].get_semantic_id()] = [distance]
        else:
            gt_geo = np.squeeze(gt_nodes[gt_node_id].geo.cpu().numpy())
            pc_2 = np.where(gt_geo)
            pc_2 = np.vstack([pc_2[2], pc_2[1], pc_2[0]]).T

            pc_2 = apply_transform(pc_2, voxel_transform)

            pc_1 = np.vstack([[0, 0, 0]])

            if pc_2.shape[0] == 0:
                pc_2 = np.vstack([[0, 0, 0]])

            pc_1 = torch.FloatTensor(pc_1)[None, ...].cuda()
            pc_2 = torch.FloatTensor(pc_2)[None, ...].cuda()

            dist1, dist2, idx1, idx2 = chamfer_module(pc_1, pc_2)
            distance = 0.5 * (dist1.mean() + dist2.mean()).cpu().numpy()

            if pc_1.shape[0] == 0 or pc_2.shape[0] == 0 or math.isnan(distance):
                continue
            else:
                if gt_nodes[gt_node_id].get_semantic_id() in part_metrics:
                    part_metrics[gt_nodes[gt_node_id].get_semantic_id()] += [distance]
                else:
                    part_metrics[gt_nodes[gt_node_id].get_semantic_id()] = [distance]
    for extra_pred_node in extra_pred_nodes:
        pred_geo = np.squeeze(pred_nodes[extra_pred_node].geo.cpu().numpy())
        pc_1 = np.where(pred_geo > thr)
        pc_1 = np.vstack([pc_1[2], pc_1[1], pc_1[0]]).T

        pc_1 = apply_transform(pc_1, voxel_transform)

        if pc_1.shape[0] == 0:
            pc_1 = np.vstack([[0, 0, 0]])

        pc_2 = np.vstack([[0, 0, 0]])

        pc_1 = torch.FloatTensor(pc_1)[None, ...].cuda()
        pc_2 = torch.FloatTensor(pc_2)[None, ...].cuda()
        dist1, dist2, idx1, idx2 = chamfer_module(pc_1, pc_2)
        distance = 0.5 * (dist1.mean() + dist2.mean()).cpu().numpy()

        if math.isnan(distance):
            continue
        else:
            if pred_nodes[extra_pred_node].get_semantic_id() in part_metrics:
                part_metrics[pred_nodes[extra_pred_node].get_semantic_id()] += [distance]
            else:
                part_metrics[pred_nodes[extra_pred_node].get_semantic_id()] = [distance]
    return part_metrics


def iou(gt_nodes, pred_nodes, thr=0.5, size=32):
    smooth = 1e-6
    gt_mask, pred_mask = np.zeros((size, size, size)), np.zeros((size, size, size))
    assignment = make_assignment(gt_nodes, pred_nodes, 0.5)
    part_metrics = []
    for gt_node_id in assignment:
        if assignment[gt_node_id] != -1:
            gt_node = gt_nodes[gt_node_id]
            pd_node = pred_nodes[assignment[gt_node_id]]

            gt_mask, pred_mask = np.zeros((size, size, size)), np.zeros((size, size, size))
            geometry = np.where(gt_node.geo.cpu().numpy()[0] > thr)
            gt_mask[geometry] = 1
            geometry = np.where(pd_node.geo.cpu().numpy()[0][0] > thr)
            pred_mask[geometry] = 1

            intersection = gt_mask * pred_mask
            union = gt_mask + pred_mask - intersection
            metric = np.sum(intersection) / (np.sum(union) + smooth)

            part_metrics += [metric]
        else:
            part_metrics += [0]
    return np.mean(part_metrics)


def grid_iou(gt_tree, pred_tree):
    grid = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    output_dict = {}
    output_dict['names'] = ['object', 'children', 'leaves']
    gt_leaves = [x for x in gt_tree.depth_first_traversal() if x.is_leaf]
    pred_leaves = [x for x in pred_tree.depth_first_traversal() if x.is_leaf]
    for thr in grid:
        children_iou = iou(gt_tree.root.children, pred_tree.root.children, thr=thr)
        output_dict[thr] = [children_iou]
    return output_dict


def grid_iou_by_part(gt_tree, pred_tree):
    grid = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)
    output_dict = {}
    gt_leaves = [x for x in gt_tree.depth_first_traversal() if x.is_leaf]
    pred_leaves = [x for x in pred_tree.depth_first_traversal() if x.is_leaf]
    for thr in grid:
        children_metric = iou_by_part_metric(gt_tree.root.children, pred_tree.root.children, thr=thr)
        output_dict[thr] = [children_metric]
    return output_dict


def grid_iou_object(gt_tree, pred_tree):
    grid = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)
    output_dict = {}
    gt_leaves = [gt_tree.root]
    pred_leaves = [pred_tree.root]
    for thr in grid:
        metric = iou_by_part_metric(gt_leaves, pred_leaves, thr=thr)
        output_dict[thr] = [metric]
    return output_dict


def iou_by_part_metric(gt_nodes, pred_nodes, thr=0.5, size=32):
    smooth = 1e-6

    assignment = make_assignment(gt_nodes, pred_nodes, 0.5)
    pred_nodes_in_assignment = [v for u, v in assignment.items()]
    extra_pred_nodes = list(set(range(len(pred_nodes))) - set(pred_nodes_in_assignment))
    part_metrics = {}
    for gt_node_id in assignment:
        if assignment[gt_node_id] != -1:
            gt_mask, pred_mask = np.zeros((size, size, size)), np.zeros((size, size, size))
            geometry = np.where(np.squeeze(gt_nodes[gt_node_id].geo.cpu().numpy()) > thr)
            gt_mask[geometry] = 1
            geometry = np.where(np.squeeze(pred_nodes[assignment[gt_node_id]].geo.cpu().numpy()) > thr)
            pred_mask[geometry] = 1

            intersection = gt_mask * pred_mask
            union = gt_mask + pred_mask - intersection
            distance = np.sum(intersection) / (np.sum(union) + smooth)

            if math.isnan(distance):
                continue
            else:
                if gt_nodes[gt_node_id].get_semantic_id() in part_metrics:
                    part_metrics[gt_nodes[gt_node_id].get_semantic_id()] += [distance]
                else:
                    part_metrics[gt_nodes[gt_node_id].get_semantic_id()] = [distance]
        else:
            gt_mask, pred_mask = np.zeros((size, size, size)), np.zeros((size, size, size))
            geometry = np.where(np.squeeze(gt_nodes[gt_node_id].geo.cpu().numpy()) > thr)
            gt_mask[geometry] = 1

            intersection = gt_mask * pred_mask
            union = gt_mask + pred_mask - intersection
            distance = np.sum(intersection) / (np.sum(union) + smooth)

            if math.isnan(distance):
                continue
            else:
                if gt_nodes[gt_node_id].get_semantic_id() in part_metrics:
                    part_metrics[gt_nodes[gt_node_id].get_semantic_id()] += [distance]
                else:
                    part_metrics[gt_nodes[gt_node_id].get_semantic_id()] = [distance]
    for extra_pred_node in extra_pred_nodes:
        gt_mask, pred_mask = np.zeros((size, size, size)), np.zeros((size, size, size))
        geometry = np.where(np.squeeze(pred_nodes[assignment[gt_node_id]].geo.cpu().numpy()) > thr)
        pred_mask[geometry] = 1

        intersection = gt_mask * pred_mask
        union = gt_mask + pred_mask - intersection
        distance = np.sum(intersection) / (np.sum(union) + smooth)

        if math.isnan(distance):
            continue
        else:
            if pred_nodes[extra_pred_node].get_semantic_id() in part_metrics:
                part_metrics[pred_nodes[extra_pred_node].get_semantic_id()] += [distance]
            else:
                part_metrics[pred_nodes[extra_pred_node].get_semantic_id()] = [distance]
    return part_metrics


def mean_chamfer(chamfer_object, thr=0.2):
    all_scores = []
    for item in chamfer_object[thr]:
        for key in item:
            for x in item[key]:
                all_scores += [x]
    return np.mean(all_scores)


def mean_iou(iou_object, thr=0.2):
    all_scores = []
    for item in iou_object[thr]:
        for key in item:
            for x in item[key]:
                all_scores += [x]
    return np.mean(all_scores)


def chamfer_distance_by_parts(chamfer_stats, grid, part_id2name):
    all_parts_chamfer = {}
    for thr in grid:
        chamfer_by_parts = {}
        for obj_metric_tuple in chamfer_stats[thr]:
            obj_metric = obj_metric_tuple[0][0]
            obj_name = obj_metric_tuple[1]
            for sem_id in obj_metric:
                sem_name = part_id2name[sem_id]
                if obj_name != 'other':
                    sem_name = obj_name + '/' + sem_name.split('/')[1]
                if sem_name not in chamfer_by_parts:
                    chamfer_by_parts[sem_name] = deepcopy(obj_metric[sem_id])
                else:
                    chamfer_by_parts[sem_name] += deepcopy(obj_metric[sem_id])
        for sem_name in chamfer_by_parts:
            chamfer_by_parts[sem_name] = np.mean(chamfer_by_parts[sem_name])
        all_parts_chamfer[thr] = deepcopy(chamfer_by_parts)
    return all_parts_chamfer


def chamfer_distance_by_objects(chamfer_stats, grid):
    all_objects_chamfer = {}
    for thr in grid:
        chamfer_by_objects = []
        for obj_metric in chamfer_stats[thr]:
            total_metric = []
            for sem_id in obj_metric:
                total_metric += deepcopy(obj_metric[sem_id])
            total_metric = np.mean(total_metric)
            if not math.isnan(total_metric):
                chamfer_by_objects += deepcopy([total_metric])
        all_objects_chamfer[thr] = np.mean(chamfer_by_objects)
    return all_objects_chamfer


def chamfer_distance_by_categories(chamfer_stats, grid, part_id2name):
    all_categories_chamfer = {}
    for thr in grid:
        chamfer_by_categories = {}
        for obj_metric_tuple in chamfer_stats[thr]:
            obj_metric = obj_metric_tuple[0][0]
            obj_name = obj_metric_tuple[1]
            for sem_id in obj_metric:
                sem_name = part_id2name[sem_id]
                if obj_name != 'other':
                    sem_name = obj_name + '/' + sem_name.split('/')[1]
                sem_category = sem_name.split('/')[0]
                if sem_category not in chamfer_by_categories:
                    chamfer_by_categories[sem_category] = deepcopy(obj_metric[sem_id])
                else:
                    chamfer_by_categories[sem_category] += deepcopy(obj_metric[sem_id])
        for sem_category in chamfer_by_categories:
            chamfer_by_categories[sem_category] = np.mean(chamfer_by_categories[sem_category])
        all_categories_chamfer[thr] = deepcopy(chamfer_by_categories)
    return all_categories_chamfer


def instance_average(chamfer_stats, grid):
    all_numbers_by_thr = {}
    for thr in grid:
        all_numbers = []
        for obj_metric_tuple in chamfer_stats[thr]:
            obj_metric = obj_metric_tuple[0][0]
            for sem_id in obj_metric:
                all_numbers += deepcopy(obj_metric[sem_id])
        all_numbers_by_thr[thr] = np.mean(all_numbers)
    return all_numbers_by_thr
