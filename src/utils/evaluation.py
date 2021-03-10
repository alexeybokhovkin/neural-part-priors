import os
import time

import torch
from tqdm import tqdm
from scipy.ndimage import rotate
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns
import umap

from ..lightning_models.gnn_scannet_contrastive import GNNPartnetLightning
from .gnn import rotate_tree_geos
from .vox import load_sample
from .metrics import grid_iou_by_part, grid_chamfer_distance_by_part, mean_chamfer, mean_iou
from .metrics import chamfer_distance_by_parts, chamfer_distance_by_categories, instance_average
from ..data_utils.hierarchy import Tree


def get_semantic_ids(tree):
    sem_ids = []
    for child in tree.root.children:
        sem_ids += [child.get_semantic_id()]
    return sem_ids


def get_pred_gt_combinations(gt_ids, pred_ids):
    y_gt = []
    y_pred = []

    tp_set = set(gt_ids).intersection(set(pred_ids))
    for x in tp_set:
        y_gt += [x]
        y_pred += [x]

    gt_mispredicted_ids = list(set(gt_ids) - tp_set)
    pred_extra_ids = list(set(pred_ids) - tp_set)
    for x in pred_extra_ids:
        for y in gt_ids:
            y_gt += [y]
            y_pred += [x]
    return y_gt, y_pred


class Evaluator:

    def __init__(self,
                 shapes_to_parts=None,
                 obj_to_cat=None,
                 cat_ids_to_name=None,
                 partnet_voxelized=None,
                 device=None,
                 checkpoint=None,
                 config=None):

        # meta
        self.shapes_to_parts = shapes_to_parts
        self.obj_to_cat = obj_to_cat
        self.cat_ids_to_name = cat_ids_to_name
        self.partnet_voxelized = partnet_voxelized

        self.device = device

        model = GNNPartnetLightning(config)
        pretrained_model = model.load_from_checkpoint(checkpoint_path=checkpoint)
        pretrained_model.to(device)
        pretrained_model.eval()
        pretrained_model.freeze()
        self.pretrained_model = pretrained_model

        self.grid = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)
        self.part_name2id = Tree.part_name2id
        self.part_id2name = Tree.part_id2name

    def evaluate_experiment(self, dataloader):

        scans_to_shapes = {}
        self.part_latents = {}
        self.root_latents = {}

        iou_stats = {}
        chamfer_stats = {}
        chamfer_stats_full_shape = {}
        chamfer_priors_stats = {}
        for thr in self.grid:
            iou_stats[thr] = []
            chamfer_stats[thr] = []
            chamfer_stats_full_shape[thr] = []
            chamfer_priors_stats[thr] = []
        partnet_ids = []

        all_chamfer_singles = []
        all_iou_singles = []
        self.y_gts, self.y_preds = [], []

        n_objects = 0
        with torch.no_grad():
            for batch_ind, batch in enumerate(tqdm(dataloader)):
                try:
                    tokens = batch[6][0]
                    partnet_id = tokens[0]
                    scan_id = '_'.join([tokens[1], tokens[2]])
                    instance_id = int(tokens[3])
                    obj_id = self.shapes_to_parts[partnet_id]
                    cat_id = self.obj_to_cat[obj_id]

                    if cat_id in self.cat_ids_to_name:
                        cat_name = self.cat_ids_to_name[cat_id]
                    else:
                        cat_name = 'other'

                    # predicted_root = np.load(os.path.join(PREDICTED_ROOTS, f'{"_".join(tokens)}.npy'))

                    # matched_items, matched_ids, angle, _, _, _ = get_angle(scan_id, partnet_id, instance_id)
                    angle = 0

                    orig_scan_geo = batch[0][0].cpu().numpy().astype('uint8')
                    rotated_scan_geo = rotate(orig_scan_geo, -angle, axes=[0, 2], reshape=False).astype('float32')
                    rotated_scan_geo_torch = torch.FloatTensor(rotated_scan_geo)[None, ...]

                    scan_geo = rotated_scan_geo_torch.to(self.device)
                    batch[0] = (scan_geo,)
                    shape = batch[2][0].to(self.device)
                    batch[2] = (shape,)
                    output = self.pretrained_model.inference(batch)
                    predicted_tree = output[0][0]
                    gt_tree = batch[3][0][0]
                    rotated_gt_tree = rotate_tree_geos(gt_tree, angle)

                    x_root = np.squeeze(output[1][0].cpu().detach().numpy())
                    x_children = output[8][0].cpu().detach().numpy()

                    for i, child in enumerate(predicted_tree.root.children):
                        if i < len(x_children):
                            if child.label not in self.part_latents:
                                self.part_latents[child.label] = [x_children[i]]
                            else:
                                self.part_latents[child.label] += [x_children[i]]

                    if partnet_id not in self.root_latents:
                        self.root_latents[partnet_id] = [x_root]
                    else:
                        self.root_latents[partnet_id] += [x_root]

                    shape_vox = load_sample(os.path.join(self.partnet_voxelized, partnet_id, 'full_vox.df'))
                    voxel_transform = shape_vox.grid2world

                    iou_object = grid_iou_by_part(rotated_gt_tree, predicted_tree)
                    chamfer_object = grid_chamfer_distance_by_part(rotated_gt_tree, predicted_tree, voxel_transform)
                    for thr in self.grid:
                        iou_stats[thr] += [(iou_object[thr], cat_name)]
                        chamfer_stats[thr] += [(chamfer_object[thr], cat_name)]

                    cd_single = mean_chamfer(chamfer_object)
                    all_chamfer_singles += [cd_single]

                    iou_single = mean_iou(iou_object)
                    all_iou_singles += [iou_single]

                    partnet_ids += [partnet_id]

                    if scan_id not in scans_to_shapes:
                        scans_to_shapes[scan_id] = ["_".join(tokens)]
                    else:
                        scans_to_shapes[scan_id] += ["_".join(tokens)]

                    gt_sem_ids = get_semantic_ids(gt_tree)
                    pred_sem_ids = get_semantic_ids(predicted_tree)
                    y_gt, y_pred = get_pred_gt_combinations(pred_sem_ids, gt_sem_ids)
                    self.y_gts += y_gt
                    self.y_preds += y_pred

                    n_objects += 1
                except:
                    continue

        return chamfer_stats, iou_stats

    def calculate_metrics(self, chamfer_stats, iou_stats, SAVE_PATH, thr=0.2):

        all_parts_chamfer = chamfer_distance_by_parts(chamfer_stats, self.grid, self.part_id2name)
        all_categories_chamfer = chamfer_distance_by_categories(chamfer_stats, self.grid, self.part_id2name)
        instance_average_chamfer = instance_average(chamfer_stats, self.grid)

        all_parts_iou = chamfer_distance_by_parts(iou_stats, self.grid, self.part_id2name)
        all_categories_iou = chamfer_distance_by_categories(iou_stats, self.grid, self.part_id2name)
        instance_average_iou = instance_average(iou_stats, self.grid)

        with open(os.path.join(SAVE_PATH, 'all_parts_chamfer.txt'), 'w') as fout:
            for item in all_parts_chamfer[thr]:
                fout.write("%s: %s\n" % (item, all_parts_chamfer[thr][item]))
        with open(os.path.join(SAVE_PATH, 'all_categories_chamfer.txt'), 'w') as fout:
            for item in all_categories_chamfer[thr]:
                fout.write("%s: %s\n" % (item, all_categories_chamfer[thr][item]))
        with open(os.path.join(SAVE_PATH, 'instance_average_chamfer.txt'), 'w') as fout:
            for item in instance_average_chamfer:
                fout.write("%s: %s\n" % (item, instance_average_chamfer[item]))

        with open(os.path.join(SAVE_PATH, 'all_parts_iou.txt'), 'w') as fout:
            for item in all_parts_iou[thr]:
                fout.write("%s: %s\n" % (item, all_parts_iou[thr][item]))
        with open(os.path.join(SAVE_PATH, 'all_categories_iou.txt'), 'w') as fout:
            for item in all_categories_iou[thr]:
                fout.write("%s: %s\n" % (item, all_categories_iou[thr][item]))
        with open(os.path.join(SAVE_PATH, 'instance_average_iou.txt'), 'w') as fout:
            for item in instance_average_iou:
                fout.write("%s: %s\n" % (item, instance_average_iou[item]))

        return all_parts_chamfer, all_categories_chamfer, instance_average_chamfer, \
               all_parts_iou, all_categories_iou, instance_average_iou

    def save_visuals(self, SAVE_PATH):

        all_latents = []
        for partnet_id in self.root_latents:
            all_latents += self.root_latents[partnet_id]

        all_norms = []
        for latent in all_latents:
            all_norms += [np.sqrt(np.sum(latent ** 2))]

        all_nnz = []
        for latent in all_latents:
            all_nnz += [np.count_nonzero(latent)]

        # HISTS
        plt.figure()
        plt.hist(all_norms, bins=100)
        plt.title('Part latents norms')
        plt.savefig(os.path.join(SAVE_PATH, 'part_latents_norms.png'), dpi=120)

        plt.figure()
        plt.hist(all_nnz, bins=100)
        plt.title('Part latents nnz')
        plt.savefig(os.path.join(SAVE_PATH, 'part_latents_nnz.png'), dpi=120)

        # T-SNE
        print('Start part t-SNE')
        latents = []
        latent_labels = []
        for i, key in enumerate(self.part_latents):
            for item in self.part_latents[key]:
                latents += [item]
                latent_labels += [i]

        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=0, perplexity=60, n_iter=300)
        tsne_results = tsne.fit_transform(latents)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

        data_tsne = {}
        data_tsne['tsne-2d-one'] = tsne_results[:, 0]
        data_tsne['tsne-2d-two'] = tsne_results[:, 1]
        data_tsne['color'] = latent_labels

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            palette=sns.color_palette("hls", 8),
            hue="color",
            data=data_tsne,
            legend="full",
            alpha=0.5
        )
        plt.title('Part latents, t-SNE')
        plt.savefig(os.path.join(SAVE_PATH, 'part_latents_tsne.png'), dpi=120)

        # UMAP
        reducer = umap.UMAP(random_state=42, angular_rp_forest=True, local_connectivity=50, n_neighbors=30)
        reducer.fit(latents)
        embedding = reducer.transform(latents)
        data_umap = {}
        data_umap['umap-2d-one'] = embedding[:, 0]
        data_umap['umap-2d-two'] = embedding[:, 1]
        data_umap['color'] = latent_labels

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="umap-2d-one", y="umap-2d-two",
            palette=sns.color_palette("hls", 8),
            hue="color",
            data=data_umap,
            legend="full",
            alpha=0.5
        )
        plt.title('Part latents, UMAP')
        plt.savefig(os.path.join(SAVE_PATH, 'part_latents_umap.png'), dpi=120)

        # T-SNE
        print('Start root t-SNE')
        latents = []
        latent_labels = []
        j = 0
        for i, partnet_id in enumerate(self.root_latents):
            if len(self.root_latents[partnet_id]) < 5:
                continue
            if j >= 30:
                break
            j += 1
            latents += self.root_latents[partnet_id]
            latent_labels += [j for k in range(len(self.root_latents[partnet_id]))]

        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=0, perplexity=60, n_iter=300)
        tsne_results = tsne.fit_transform(latents)

        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

        data_tsne = {}
        data_tsne['tsne-2d-one'] = tsne_results[:, 0]
        data_tsne['tsne-2d-two'] = tsne_results[:, 1]
        data_tsne['color'] = latent_labels

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            palette=sns.color_palette("Spectral", np.max(latent_labels)),
            hue="color",
            legend=None,
            data=data_tsne,
            alpha=1
        )
        plt.title('Root latents, t-SNE')
        plt.savefig(os.path.join(SAVE_PATH, 'root_latents_tsne.png'), dpi=120)

        # UMAP
        latents = []
        latent_labels = []
        j = 0
        for i, partnet_id in enumerate(self.root_latents):
            if len(self.root_latents[partnet_id]) < 5:
                continue
            if j >= 30:
                break
            j += 1
            latents += self.root_latents[partnet_id]
            latent_labels += [j for k in range(len(self.root_latents[partnet_id]))]

        reducer = umap.UMAP(random_state=42, angular_rp_forest=True, local_connectivity=50, n_neighbors=30)
        reducer.fit(latents)
        embedding = reducer.transform(latents)

        data_umap = {}
        data_umap['umap-2d-one'] = embedding[:, 0]
        data_umap['umap-2d-two'] = embedding[:, 1]
        data_umap['color'] = latent_labels

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="umap-2d-one", y="umap-2d-two",
            palette=sns.color_palette("Spectral", np.max(latent_labels)),
            hue="color",
            data=data_umap,
            legend=None,
            alpha=1
        )
        plt.title('Root latents, UMAP')
        plt.savefig(os.path.join(SAVE_PATH, 'root_latents_umap.png'), dpi=120)

        # Confusion matrices
        ids_to_names = {x: self.part_id2name[x].split('/')[1] for x in sorted(self.y_gts)}
        label_ids_ordered = list(ids_to_names.keys())
        label_names_ordered = list(ids_to_names.values())
        sns.set(font_scale=1.4)
        conf_matrix = confusion_matrix(self.y_gts, self.y_preds, labels=label_ids_ordered)
        plt.figure(figsize=(20, 16))
        sns.heatmap(conf_matrix / np.sum(conf_matrix), annot=True, fmt='.1%', cmap='rocket_r',
                    xticklabels=label_names_ordered, yticklabels=label_names_ordered)
        plt.savefig(os.path.join(SAVE_PATH, 'parts_confusion_matrix.png'), dpi=60)
