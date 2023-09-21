"""
Goal: Align images of several spots and cell detections to common coordinates.

Steps:
1) Identify spots that are near each other. We want to find a group of 7 spots in a hexagonal shape (with
one spot in the center) that we can stitch together. For most cases, this just means finding one spot and
taking the 6 nearest neighbors.

2) Given the spots, crop the WSI such that the spots are in the center of the image. We can then just draw
our cell detections on this image.

3) Align the coordinate systems of each of the spot detections.

Each spot is identified by its index within the dataset.
"""

import io
import json
import os
import pickle
import sys

# to draw lines
import cv2
import matplotlib.pyplot as plt
import numpy as np
import omicsio.datasets
import PIL.Image
import torch
import torch.utils.data
from omicsio.datasets import Slide

from adapt_for_cell_graph_models import get_valid_indexes
from cytassist_segmentation import (
    get_cytassist_subsets_dict_and_cut_cell_detections,
    reconstruct_validation_data)
from load_datasets import ALL_PATIENTS

def crop_image_for_spots(slide: Slide, spot_indexes: list, downsample: int):
    """
    Returns a cropped image and the position of image(0, 0) on the original image.
    Format is numpy array.
    """

    buffer = 256
    min_x = slide.spot_locations.image_x[spot_indexes].min() - buffer
    min_y = slide.spot_locations.image_y[spot_indexes].min() - buffer
    max_x = slide.spot_locations.image_x[spot_indexes].max() + buffer
    max_y = slide.spot_locations.image_y[spot_indexes].max() + buffer

    image = slide.image_region(min_x, min_y, max_x - min_x, max_y - min_y, downsample)
    # Clip the alpha channel if it existss
    image = np.ascontiguousarray(image.numpy().transpose((1, 2, 0))[:, :, :3])

    return (image, min_x, min_y)

def get_nearest_neighbors(slide: Slide, spot_index: int, num_neighbors: int):
    spot_x = slide.spot_locations.image_x[spot_index]
    spot_y = slide.spot_locations.image_y[spot_index]

    neighbors_by_distance = []

    for neighbor_index in range(len(slide.spot_locations.image_x)):
        if neighbor_index == spot_index:
            continue

        neighbor_x = slide.spot_locations.image_x[neighbor_index]
        neighbor_y = slide.spot_locations.image_y[neighbor_index]

        distance = ((spot_x - neighbor_x) ** 2 + (spot_y - neighbor_y) ** 2) ** 0.5

        neighbors_by_distance.append((distance, neighbor_index))
    
    neighbors_by_distance.sort()

    return [neighbor_index for (distance, neighbor_index) in neighbors_by_distance[:num_neighbors]]

def scale_to_minmax(value, min, max):
    return (value - min) / (max - min)

# dataset is wrapped using `adapt_for_cell_graph_models.py`
def predict_and_explain_with_nodes(model, dataset, spot_index, gene_index):
    (cell_images, cell_locations, cell_detection_indexes), label = dataset[spot_index]

    return predict_and_explain_with_nodes_helper(model, cell_images, cell_locations, cell_detection_indexes, gene_index)

def predict_and_explain_with_nodes_helper(model, cell_images, cell_locations, cell_detection_indexes, gene_index):
    from torch_geometric.explain import Explainer, GNNExplainer

    from create_graph import create_graph

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
        threshold_config=dict(threshold_type='topk', value=10)
    )

    with torch.no_grad():
        cell_embeddings = model.image_embedder(cell_images)
    
    data = create_graph(cell_embeddings, cell_locations)

    explanation = explainer(
        data.x,
        data.pos,
        # This is a keyword argument passed to the model.
        input_type='nodes_and_edges',
    )

    print(explanation.edge_mask)
    print(explanation.node_mask)

def transform_cell_boxes(boxes, origin_x, origin_y, spot_x, spot_y, downsample=1):
    # Adjust for spot position
    offset_x = spot_x - 256 - origin_x
    offset_y = spot_y - 256 - origin_y

    return torch.stack([
        boxes[:, 0] + offset_x,
        boxes[:, 1] + offset_y,
        boxes[:, 2] + offset_x,
        boxes[:, 3] + offset_y
    ], dim=-1) / downsample

def render_edges(image, boxes, edge_index, edge_color):
    # edge_index is a tensor of shape [2, num_edges]
    # therefore, we transpose it to be able to iterate
    # over index tuples.
    # This is based on COO format. See:
    # https://pytorch.org/docs/stable/sparse.html#sparse-coo-docs
    for (a, b) in edge_index.T:
        # Find position of cell
        box_a = boxes[a]
        box_b = boxes[b]
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ax = (ax1 + ax2) / (2)
        ay = (ay1 + ay2) / (2)
        bx = (bx1 + bx2) / (2)
        by = (by1 + by2) / (2)
        
        a_img = np.copy(image[int(ay1):int(ay2), int(ax1):int(ax2), :])
        b_img = np.copy(image[int(by1):int(by2), int(bx1):int(bx2), :])
        
        # draw line
        # make color palette consistent
        r,g,b,a = edge_color
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        cv2.line(image, (int(ax), int(ay)), (int(bx), int(by)), (r, g, b), 3)

        # recover boxes to prevent clutter
        image[int(ay1):int(ay2), int(ax1):int(ax2), :] = a_img
        image[int(by1):int(by2), int(bx1):int(bx2), :] = b_img

def render_boxes(image, boxes: torch.Tensor, box_colors: np.ndarray, edge_index=None, edge_color=None):
    # if edge_index is not None, then we draw lines between cell positions. to make it look nice,
    # we draw the line and then paint over it where the boxes are. this makes it look like the boxes
    # are only connected at the boundaries.
    if edge_index is not None:
        assert edge_color is not None, "Must provide edge_color if edge_index is not None"

        render_edges(image, boxes, edge_index, edge_color)

    for box, box_color in zip(boxes, box_colors):
        x1, y1, x2, y2 = box.int()

        # Draw cell to image
        # It would have been so nice to have masks here.
        translucency = box_color[3]
        rgb = box_color[:3]
        image[y1:y2, x1:x2, :] = image[y1:y2, x1:x2, :] * (1 - translucency) + rgb * translucency
    return image

def render_visium(image, image_x, image_y, colors, square_size):
    for (x, y, color) in zip(image_x, image_y, colors):
        rgb = color[:3]
        if len(color) == 4:
            a = color[3]
        else:
            a = 1
        image[y - square_size:y + square_size, x - square_size:x + square_size, :] = \
            a * rgb + (1 - a) * image[y - square_size:y + square_size, x - square_size:x + square_size, :]
    return image

def np_to_pil(image_np: np.ndarray):
    # format is currently [3, height, width], elements in range [0, 1)
    # output format is [height, width, 3], elements in range [0, 256)
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).astype(np.uint8)
    return PIL.Image.fromarray(image_np)

def plt_savefig_to_np():
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    return np.array(PIL.Image.open(buf)) / 255.0


VALIDATION_PATH_FORMATS = {
    'inception_models': "training_results_v2/v6_cnn_models_augmented_2_epochs/inception-holding-out-{patient}_validation_results.pt",
    'cells_models':     "training_results_v2/v7_cell_models_augmented_2_epochs/cellmodel_prior-none_cellreg-enabled_heldout-{patient}_validation_results.pt",
}

class ValidationRenderer:
    def __init__(self, slide_id: str, patient_id_in_slide: str, export_dir: str, validation_path_format=VALIDATION_PATH_FORMATS['cells_models']):
        """
        - slide_id: The whole slide image to use
        - patient_id_in_slide: Selects a subset of the image to render. If this is none, then collects validation data for all patients and renders them on the same slide.
        - export_dir: Where to output image files. If `None`, images are returned, but not saved to disk.
        """

        print("Creating ValidationRenderer:", slide_id, patient_id_in_slide, validation_path_format)

        if export_dir is not None and not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        with open("./training_results_v2/v2_repr_contrastive/genes.txt") as f:
            self.genes: list = f.read().split("\n")

        # Load slide and cell locations
        if slide_id == 'autostainer':
            with open(f"./input_data/all_visium_data/preprocessed/autostainer_40x.pkl", "rb") as f:
                self.slide: omicsio.datasets.Slide = pickle.load(f).select_genes(self.genes).log1p()

            self.validation_data = torch.load(
                validation_path_format.format(patient='autostainer'),
                map_location='cpu'
            )
        else:
            with open(f"./input_data/all_visium_data/preprocessed/cytassist_{slide_id}_40x.pkl", "rb") as f:
                self.slide: omicsio.datasets.Slide = pickle.load(f).select_genes(self.genes).log1p()

            # get predicted data
            self.validation_data = reconstruct_validation_data(self.slide, slide_id, {
                patient.split("_")[-1]: torch.load(
                    validation_path_format.format(patient=patient),
                    map_location='cpu'
                ) for patient in ALL_PATIENTS if slide_id in patient
            }, is_tensor_only='cnn' in validation_path_format)

        self.downsample = 32
        self.blank_image, self.origin_x, self.origin_y = get_or_save_thumbnail(slide_id, self.downsample, self.slide)

        # see if this validation data came with cell predictions
        if type(self.validation_data['predictions']) == dict and ('cell_vectors_batch' in self.validation_data['predictions']):
            if slide_id != 'autostainer':
                with open(f"./input_data/all_visium_data/annotations/json_files/_SS12251_{slide_id}.json", "rb") as f:
                    annotations = json.load(f)

                _patient_subsets, self.cell_detections = get_cytassist_subsets_dict_and_cut_cell_detections(
                    self.slide,
                    annotations,
                    # Cell detections
                    torch.load(f'./cell-detections/cytassist_{slide_id}/inferences.pt'),
                    512
                )

                # filter validation_data for patient subset
                if patient_id_in_slide is not None:
                    subset = set(_patient_subsets[patient_id_in_slide])

                    new_gvb = []
                    new_cvb = []
                    new_pi = []

                    for gv, cv, index in zip(
                        self.validation_data['predictions']['graph_vector_batch'],
                        self.validation_data['predictions']['cell_vectors_batch'],
                        self.validation_data['prediction_indices']
                    ):
                        if index in subset:
                            new_pi.append(index)
                            new_gvb.append(gv)
                            new_cvb.append(cv)

                    self.validation_data['predictions'] = {'graph_vector_batch': new_gvb, 'cell_vectors_batch': new_cvb}
                    self.validation_data['prediction_indices'] = new_pi

                boxes = [torch.tensor(self.cell_detections[i]['boxes']) * 2 for i in self.validation_data['prediction_indices']]
            else:
                self.cell_detections = torch.load("cell-detections/autostainer_orig/combined_nms.pt")
                boxes = [torch.stack(self.cell_detections[i]['boxes']) * 2 for i in self.validation_data['prediction_indices']]

            # load boxes
            # 20x --> 40x
            # boxes = [torch.tensor(detection['boxes']) * 2 for detection in self.cell_detections]
            # boxes = [boxes[i] for i in self.validation_data['prediction_indices']]
            boxes = [b[get_valid_indexes(b, 512, 64, magnify=1)] for b in boxes]
            # transform to absolute position
            self.boxes = [transform_cell_boxes(b, self.origin_x, self.origin_y, x, y, self.downsample) for b, x, y in zip(boxes, self.slide.spot_locations.image_x[self.validation_data['prediction_indices']], self.slide.spot_locations.image_y[self.validation_data['prediction_indices']])]
            # get normalized predicted expression
            self.pred_expr_per_cell = [torch.tensor(e) for e in self.validation_data['predictions']['cell_vectors_batch']]
        else:
            self.boxes = None
            self.pred_expr_per_cell = None

        if type(self.validation_data['predictions']) == torch.Tensor:
            self.pred_visium = self.validation_data['predictions'].cpu()
            if 'prediction_indices' not in self.validation_data:
                self.validation_data['prediction_indices'] = torch.arange(0, self.validation_data['predictions'].shape[0])
        else:
            # These correspond via the `prediction_indices`
            self.pred_visium = torch.stack(self.validation_data['predictions']['graph_vector_batch']).cpu()

        self.true_visium = self.slide.spot_counts[self.validation_data['prediction_indices']].cpu()

        self.visium_x = ((self.slide.spot_locations.image_x[self.validation_data['prediction_indices']] - self.origin_x) / self.downsample).int()
        self.visium_y = ((self.slide.spot_locations.image_y[self.validation_data['prediction_indices']] - self.origin_y) / self.downsample).int()

        self.cmap = plt.get_cmap('viridis')

        self.export_dir = export_dir

    def render_cell_prediction(self, gene):
        pred_exprs = torch.cat(self.pred_expr_per_cell)
        pred_exprs = pred_exprs[:, self.genes.index(gene)].clone()
        pred_exprs -= pred_exprs.min()
        pred_exprs /= pred_exprs.max()

        colors = self.cmap(pred_exprs.cpu().numpy())
        colors[3] = 0.5 # semi-transparent
        # filter out artifacts
        boxes = torch.cat(self.boxes)
        valid = ((boxes[:, 0] - boxes[:, 2]) * (boxes[:, 1] - boxes[:, 3])).abs() <= (64)
        boxes, colors = boxes[valid], colors[valid]

        return self.export(
            render_boxes(self.blank_image.copy(), boxes, colors),
            gene + '/cell_pred'
        )

    def render_predicted_visium(self, gene: str):
        pred_exprs = self.pred_visium[:, self.genes.index(gene)].clone()
        pred_exprs -= pred_exprs.min()
        pred_exprs /= pred_exprs.max()
        
        return self.export(
            render_visium(
                self.blank_image.copy(),
                self.visium_x,
                self.visium_y,
                self.cmap(pred_exprs.numpy()),
                128 // self.downsample
            ),
            gene + '/visium_pred'
        )
        
    def render_true_visium(self, gene: str):
        true_exprs = self.true_visium[:, self.genes.index(gene)].clone()
        true_exprs -= true_exprs.min()
        true_exprs /= true_exprs.max()

        return self.export(
            render_visium(
                self.blank_image.copy(),
                self.visium_x,
                self.visium_y,
                self.cmap(true_exprs.numpy()),
                128 // self.downsample
            ),
            gene + '/visium_true'
        )

    def render_hdbscan_clusters(self, **hdbscan_args):
        import hdbscan
        import hdbscan.flat
        import seaborn as sns

        from create_aligned_umap import plot_aligned_umap, create_umap_embeddings

        true_emb, pred_emb = create_umap_embeddings(self.true_visium, self.pred_visium)

        clusterer = hdbscan.flat.HDBSCAN_flat(true_emb, **hdbscan_args)
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=45, **hdbscan_args)
        # clusterer.fit(true_emb)
        
        color_palette = sns.color_palette()
        cluster_colors = np.array([
            color_palette[col % len(color_palette)] for col, sat in zip(clusterer.labels_, clusterer.probabilities_)
        ])
        
        plot_aligned_umap(true_emb, pred_emb, cluster_colors=cluster_colors)
        aligned_umap_clustered = plt_savefig_to_np()
        plt.clf()

        plot_aligned_umap(true_emb, pred_emb)
        aligned_umap_spectrum = plt_savefig_to_np()
        plt.clf()

        return {
            "cluster_overlay": self.export(
                render_visium(
                    self.blank_image.copy(),
                    self.visium_x,
                    self.visium_y,
                    cluster_colors,
                    128 // self.downsample
                ),
                'cluster_overlay',
            ),
            "umap_clustered": self.export(
                aligned_umap_clustered,
                'umap_clustered'
            ),
            "umap_spectrum": self.export(
                aligned_umap_spectrum,
                'umap_spectrum'
            )
        }

    def export(self, image: np.ndarray, name: str):
        """
        If `export_dir` is set to `None`, then this is a no-op.
        In any case, it returns the image, so I can use this function
        in a return statement.
        """

        if self.export_dir is not None:
            path = os.path.join(self.export_dir, name + ".png")
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            np_to_pil(image).save(os.path.join(self.export_dir, name + ".png"))

        return image

def get_or_save_thumbnail(slide_id, downsample, slide): # , spots):
    # import hashlib

    # spot_hash = hashlib.sha256(str(sorted(spots)).encode("utf-8")).hexdigest()
    image_path = f"figures/thumbnail_cache/{slide_id}/thumbnail_{downsample}.png"
    image_info_path = f"figures/thumbnail_cache/{slide_id}/thumbnail_{downsample}_info.json"
    if os.path.exists(image_path):
        with open(image_info_path) as f:
            image_info = json.load(f)

        image = np.array(PIL.Image.open(image_path)) / 255.0

        return (image, image_info['origin_x'], image_info['origin_y'])
    else:
        image, origin_x, origin_y = crop_image_for_spots(slide, [*range(len(slide.spot_locations.image_x))], downsample)

        if not os.path.exists(f"figures/{slide_id}_combined"):
            os.makedirs(f"figures/{slide_id}_combined")

        PIL.Image.fromarray((image * 255).astype(np.uint8)).save(image_path)

        with open(image_info_path, "w") as f:
            json.dump({'origin_x': origin_x.item(), 'origin_y': origin_y.item()}, f)

        return (image, origin_x, origin_y)

if __name__ == '__main__':
    pid = sys.argv[1]

    GENES_TO_USE = [
        'CDX2',
        # 'CDX1', # 'EPCAM', 'RAB25', 'KRT8',
        # # DNA Repair
        # 'CANT1',
        # # Angiogenesis
        # 'COL3A1', # 'COL5A2', 'FGFR1', 'S100A4',
        # # E2F Targets
        # 'CDK1', # 'CDKN3', 'TOP2A',
        # # TGF-beta Signaling
        # 'CDH1', # 'SERPINE1',
        # # Inflammatory Response
        # 'CXCL8', # 'CXCL9', 'CXCL10',
        # # WNT Signaling
        # 'CCND2',
        # # Epithelial Mesenchymal Transition
        # 'ABI3BP', # 'ADAM12', 'AREG', 'BGN', 'COL1A1', 'COL1A2', 'IGFBP2', 'IGFBP3'
    ]

    # # Fix progression images
    # for gene in GENES_TO_USE:
    #     root = 'figures/' + slide_id + '_combined'
    #     blank_image = PIL.Image.open(root + '/blank_image.png')
    #     cell_predictions = PIL.Image.open(root + '/cell_predictions.png')
    #     pred_visium = PIL.Image.open(root + '/pred_visium.png')
    #     true_visium = PIL.Image.open(root + '/true_visium.png')

    if pid == 'autostainer':
        sid = 'autostainer'
        pid_in_slide = None
    else:
        sid, pid_in_slide = pid.split("_")

    renderer = ValidationRenderer(sid, pid_in_slide, "figures/v7/" + pid)

    # These have already been generated for most.
    # print("Rendering HDBScan.")
    # hdbscan_results = renderer.render_hdbscan_clusters()
    # # renderer.export(hdbscan_results['cluster_overlay'], 'hdbscan_overlay')
    # # renderer.export(hdbscan_results['umap_comparison'], 'umap_comparison')
    # print("Done.")
    renderer.export(renderer.blank_image, 'blank_image')

    for gene_name in GENES_TO_USE:
        print("Rendering", gene_name)
        progression = np.concatenate([
            renderer.blank_image,
            renderer.render_cell_prediction(gene_name),
            renderer.render_predicted_visium(gene_name),
            renderer.render_true_visium(gene_name),
        ], axis=0)
        renderer.export(progression, 'progression')
