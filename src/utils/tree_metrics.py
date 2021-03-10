import numpy as np
from scipy.optimize import linear_sum_assignment

def compare_two_nodes(gt_node, pd_node):
    compare_log_local = {}
    compare_log_local['n_children_diff'] = abs(len(gt_node.children) - len(pd_node.children))
    gt_labels = []
    for child in gt_node.children:
        gt_labels += [child.get_semantic_id()]
    pd_labels = []
    for child in pd_node.children:
        pd_labels += [child.get_semantic_id()]
    compare_log_local['gt_semantic_sup'] = len(set(gt_labels) - set(pd_labels))
    compare_log_local['pd_semantic_sup'] = len(set(pd_labels) - set(gt_labels))
    return compare_log_local


def compare_by_dfs(gt_root, pd_root, cur_level=0, compare_log={}):
    if 'pd_not_leaves_trav_len' not in compare_log:
        compare_log['pd_not_leaves_trav_len'] = 0
    if 'pd_spare_leaves_trav_len' not in compare_log:
        compare_log['pd_spare_leaves_trav_len'] = 0
    if cur_level not in compare_log:
        compare_log[cur_level] = []
    log = compare_two_nodes(gt_root, pd_root)
    compare_log[cur_level] += [log]
    gt_labels = []
    if len(gt_root.children) > 0:
        for child in gt_root.children:
            gt_labels += [child.get_semantic_id()]
    pd_labels = []
    if len(pd_root.children) > 0:
        for child in pd_root.children:
            pd_labels += [child.get_semantic_id()]

    assignment = {}
    gt_nodes = gt_root.children
    gt_nodes_with_labels = {}
    for i, node in enumerate(gt_nodes):
        if node.get_semantic_id() not in gt_nodes_with_labels:
            gt_nodes_with_labels[node.get_semantic_id()] = [i]
        else:
            gt_nodes_with_labels[node.get_semantic_id()] += [i]
    pred_nodes = pd_root.children
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
                        gt_traversal = gt_nodes[gt_node].depth_first_traversal()
                        pred_traversal = pred_nodes[pred_node].depth_first_traversal()
                        distance = abs(len(gt_traversal) - len(pred_traversal))
                        dist_matrix[i, j] = distance
                row_ind, col_ind = linear_sum_assignment(dist_matrix)
                for i in range(len(row_ind)):
                    assignment[gt_nodes_with_labels[gt_label][row_ind[i]]] = pred_nodes_with_labels[gt_label][
                        col_ind[i]]

    for gt_node in assignment:
        if assignment[gt_node] != -1:
            compare_by_dfs(gt_root.children[gt_node], pd_root.children[assignment[gt_node]], cur_level + 1, compare_log)
    if len(gt_labels) == 0 and len(pd_labels) > 0:
        compare_log['pd_not_leaves_trav_len'] += len(pd_root.depth_first_traversal()) - 1
    for pd_sem_id in pred_nodes_with_labels:
        if pd_sem_id not in gt_nodes_with_labels:
            for pd_node in pred_nodes_with_labels[pd_sem_id]:
                compare_log['pd_spare_leaves_trav_len'] += len(pred_nodes[pd_node].depth_first_traversal())


def aggregate_dfs_log(log):
    global_log = {}
    for level in log:
        if isinstance(level, int):
            level_log = {}
            level_log['n_children_diff'] = 0
            level_log['gt_semantic_sup'] = 0
            level_log['pd_semantic_sup'] = 0
            for local_log in log[level]:
                level_log['n_children_diff'] += local_log['n_children_diff']
                level_log['gt_semantic_sup'] += local_log['gt_semantic_sup']
                level_log['pd_semantic_sup'] += local_log['pd_semantic_sup']
            global_log[level] = level_log
    global_log['pd_not_leaves_trav_len'] = log['pd_not_leaves_trav_len']
    global_log['pd_spare_leaves_trav_len'] = log['pd_spare_leaves_trav_len']
    return global_log


def compare_trees(gt_tree, pd_tree):
    gt_depth = gt_tree.depth()
    pd_depth = pd_tree.depth()

    compare_log = {}
    compare_by_dfs(gt_tree.root, pd_tree.root, 0, compare_log)
    compare_log = aggregate_dfs_log(compare_log)
    compare_log['gt_depth'] = gt_depth
    compare_log['pd_depth'] = pd_depth
    compare_log['depth_diff'] = gt_depth - pd_depth

    return compare_log


def aggregate_statistics(path):
    total_log = {}
    total_log_general = {}

    total_log_general['total'] = {}
    total_log_general['total']['depth_diff_num'] = 0
    total_log_general['total']['depth_diff_total'] = 0
    total_log_general['total']['depth_diff_num_ratio'] = 0
    total_log_general['total']['pd_not_leaves_num'] = 0
    total_log_general['total']['pd_not_leaves_total'] = 0
    total_log_general['total']['pd_not_leaves_num_ratio'] = 0
    total_log_general['total']['pd_spare_leaves_num'] = 0
    total_log_general['total']['pd_spare_leaves_total'] = 0
    total_log_general['total']['pd_spare_leaves_num_ratio'] = 0
    total_log_general['total']['n_objects'] = 0

    for i in range(5):
        total_log[str(i)] = {}
        total_log[str(i)]['n_children_diff_num'] = 0
        total_log[str(i)]['n_children_diff_total'] = 0
        total_log[str(i)]['n_children_diff_num_ratio'] = 0
        total_log[str(i)]['gt_semantic_sup_num'] = 0
        total_log[str(i)]['gt_semantic_sup_total'] = 0
        total_log[str(i)]['gt_semantic_sup_num_ratio'] = 0
        total_log[str(i)]['pd_semantic_sup_num'] = 0
        total_log[str(i)]['pd_semantic_sup_total'] = 0
        total_log[str(i)]['pd_semantic_sup_num_ratio'] = 0

    n_logs = 0
    for file in os.listdir(path):
        if file.endswith('.json'):
            n_logs += 1
            with open(os.path.join(path, file), 'r') as file:
                log = json.load(file)

                if log['depth_diff'] > 0:
                    total_log_general['total']['depth_diff_num'] += 1
                    total_log_general['total']['depth_diff_total'] += log['depth_diff']

                if log['pd_not_leaves_trav_len'] > 0:
                    total_log_general['total']['pd_not_leaves_num'] += 1
                    total_log_general['total']['pd_not_leaves_total'] += log['pd_not_leaves_trav_len']

                if log['pd_spare_leaves_trav_len'] > 0:
                    total_log_general['total']['pd_spare_leaves_num'] += 1
                    total_log_general['total']['pd_spare_leaves_total'] += log['pd_spare_leaves_trav_len']

                for i in range(5):
                    if str(i) in log:
                        if log[str(i)]['n_children_diff'] > 0:
                            total_log[str(i)]['n_children_diff_num'] += 1
                            total_log[str(i)]['n_children_diff_total'] += log[str(i)]['n_children_diff']

                        if log[str(i)]['gt_semantic_sup'] > 0:
                            total_log[str(i)]['gt_semantic_sup_num'] += 1
                            total_log[str(i)]['gt_semantic_sup_total'] += log[str(i)]['gt_semantic_sup']

                        if log[str(i)]['pd_semantic_sup'] > 0:
                            total_log[str(i)]['pd_semantic_sup_num'] += 1
                            total_log[str(i)]['pd_semantic_sup_total'] += log[str(i)]['pd_semantic_sup']
    total_log_general['total']['n_objects'] = n_logs

    total_log_general['total']['depth_diff_num_ratio'] = total_log_general['total']['depth_diff_num'] / n_logs
    total_log_general['total']['pd_not_leaves_num_ratio'] = total_log_general['total']['pd_not_leaves_num'] / n_logs
    total_log_general['total']['pd_spare_leaves_num_ratio'] = total_log_general['total']['pd_spare_leaves_num'] / n_logs

    for i in range(5):
        total_log[str(i)]['n_children_diff_num_ratio'] = total_log[str(i)]['n_children_diff_num'] / n_logs
        total_log[str(i)]['gt_semantic_sup_num_ratio'] = total_log[str(i)]['gt_semantic_sup_num'] / n_logs
        total_log[str(i)]['pd_semantic_sup_num_ratio'] = total_log[str(i)]['pd_semantic_sup_num'] / n_logs

    return total_log, total_log_general