import numpy as np

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