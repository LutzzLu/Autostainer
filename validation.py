from dataclasses import dataclass
import json
import pprint
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import scipy.stats
import sklearn.metrics
import torch
import torch.utils.data

import patch_model
from adapt_for_cell_graph_models import adapt_for_cell_graph_models
from cell_model import load_model_from_path, GraphSageModel
from infer import infer, infer_inception
from load_datasets import ALL_PATIENTS, load_patient
from dataset_wrapper import DatasetWrapper

PIL.Image.MAX_IMAGE_PIXELS = 1e10


@dataclass
class CellLevelValidation:
    spot_predictions: torch.Tensor
    cell_predictions: List[torch.Tensor]
    cell_boxes: List[torch.Tensor]
    indexes_of_spots_in_slide: torch.Tensor

def convert_to_cell_level_validation(raw_validation: dict, cell_detections_for_slide: list) -> CellLevelValidation:
    from adapt_for_cell_graph_models import get_valid_indexes

    ensure_tensor = lambda maybe_tensor: (torch.stack(maybe_tensor) if isinstance(maybe_tensor, (tuple, list)) else maybe_tensor)
    
    all_boxes = [ensure_tensor(cell_detections_for_slide[i]['boxes']) * 2 for i in raw_validation['prediction_indices']]
    all_boxes = [boxes_for_spot[get_valid_indexes(boxes_for_spot, 512, 64, magnify=1)] for boxes_for_spot in all_boxes]

    return CellLevelValidation(
        spot_predictions=raw_validation['predictions']['graph_vector_batch'],
        cell_predictions=raw_validation['predictions']['cell_vectors_batch'],
        cell_boxes=all_boxes,
        indexes_of_spots_in_slide=raw_validation['prediction_indices'],
    )

def add_histogram(title, values):
    ax = plt.gca()
    plt.title(title)
    plt.hist(values)
    plt.text(0.03, 0.90, f"Median: {np.nanmedian(values):.3f}", transform=ax.transAxes, fontsize=8)
    plt.xlabel(title)
    plt.ylabel("Count")

def add_histogram_four_panel(values_list):
    for i, (title, values) in enumerate(values_list):
        plt.subplot(2, 2, i + 1)
        add_histogram(title, values)

def analyze_continuous_predictions(log_counts: torch.Tensor, log_counts_pred):
    # valid_dataset[i] = (input, label)
    gene_count = log_counts_pred.shape[1]
    
    # Spearman correlation
    # spearmanr() -> (statistic, p-value)
    spearmen = np.array([
        scipy.stats.spearmanr(log_counts_pred[:, i], log_counts[:, i])[0] for i in range(gene_count)
    ])

    # Mean Squared Error
    mse = ((log_counts - log_counts_pred) ** 2).mean(axis=0)
    
    # Pearson r
    # pearsonr() -> (statistic, p-value)
    pearsonr = np.array([
        scipy.stats.pearsonr(log_counts_pred[:, i], log_counts[:, i])[0] for i in range(gene_count)
    ])

    log_counts = torch.tensor(log_counts)
    log_counts_pred = torch.tensor(log_counts_pred)

    # AUROC
    # Calculated from dichotomized expression
    # "scores" are sigmoid(amount above or below median expression)
    log_counts_binary = (log_counts > torch.quantile(log_counts, 0.5, dim=0)).detach().cpu().numpy()
    log_counts_median_pred = torch.quantile(log_counts_pred, 0.5, dim=0)
    scores = torch.sigmoid(log_counts_pred - log_counts_median_pred).detach().cpu().numpy()
    roc = np.array([
        sklearn.metrics.roc_auc_score(log_counts_binary[:, i], scores[:, i])
        if log_counts_binary[:, i].sum() else np.nan
        for i in range(gene_count)
    ])
    
    return {
        "mse_values": mse,
        "spearman_values": spearmen,
        "pearsonr_values": pearsonr,
        "auroc_values": roc,
    }

def bootstrap(values):
    values = values[~np.isnan(values)]
    np.random.seed(42)
    performances_bootstrapped=[np.quantile(np.random.choice(values,len(values),replace=True),0.5) for i in range(1000)]
    med = np.quantile(values,0.5)
    p975 = np.quantile(performances_bootstrapped,[0.025,0.975])
    return {"median": med, "width": (p975[1]-p975[0])/2}

def bootstrap_stats_dict(stats_dict: dict):
    return {key: bootstrap(stats_dict[key]) for key in stats_dict}

def convert_cell_graph_outputs_to_python_object(results):
    graph_vector_batch = []
    cell_vectors_batch = []

    for result in results:
        graph_vector_batch.append(result.graph_vector)
        cell_vectors_batch.append(result.cell_vectors)
    
    return {"graph_vector_batch": graph_vector_batch, "cell_vectors_batch": cell_vectors_batch}

def validate_cnn_model(model, valid_dataset):
    predicted_labels, true_labels = infer_inception(model, valid_dataset)
    log_counts = true_labels.cpu().numpy()
    log_counts_pred = predicted_labels.cpu().numpy()
    stats = analyze_continuous_predictions(log_counts, log_counts_pred)

    return {
        "predictions": predicted_labels,
        "true_log_counts": log_counts,
        "stats": stats,
        "stats_bootstrapped": bootstrap_stats_dict(stats),
    }

def validate_continuous_model(model, valid_dataset):
    # Cast model outputs to CPU, because otherwise we eat up too much of CUDA's memory.
    predicted_labels, prediction_indices = infer(model, valid_dataset)
    # Here, we make a prediction on the held out slide.
    # When serializing, we use raw data structures to prevent
    # problems where we can't unpickle something.
    predicted_labels_python = convert_cell_graph_outputs_to_python_object(predicted_labels)

    log_counts = torch.stack([valid_dataset[i][1] for i in prediction_indices]).cpu().numpy()
    log_counts_pred = torch.stack(predicted_labels_python['graph_vector_batch']).cpu().numpy()
    stats = analyze_continuous_predictions(log_counts, log_counts_pred)

    return {
        "predictions": predicted_labels_python,
        "prediction_indices": prediction_indices,
        "true_log_counts": log_counts,
        "stats": stats,
        "stats_bootstrapped": bootstrap_stats_dict(stats),
    }

def plot_continuous_model_stats(stats, root, name):
    plt.clf()
    plt.rcParams['figure.figsize'] = (12, 12)
    plt.rcParams['figure.dpi'] = 100
    add_histogram_four_panel([
        ("Spearman", stats['spearman_values']),
        ("MSE", stats['mse_values']),
        ("Pearson R", stats['pearsonr_values']),
        ("AUROC", stats['auroc_values'])
    ])
    plt.tight_layout()
    plt.savefig(root + "/" + name + "_validation_figure.png")

def patch_model_valid_transforms(ds, idx):
    image, label = ds[idx]
    return patch_model.validation_transforms(image), label

def cnn_model(root, name, heldout_patient):
    with open(root + "/genes.txt") as f:
        genes = f.read().split("\n")

    model = patch_model.get_model(len(genes))
    model.load_state_dict(
        torch.load(root + "/" + name + "_model.pt")
    )
    model = model.to('cuda')
    model = model.eval()

    # Get the valid dataset
    valid_dataset = load_patient(heldout_patient, genes, include_cells=False)
    valid_dataset = DatasetWrapper(valid_dataset, patch_model_valid_transforms)

    results = validate_cnn_model(model, valid_dataset)

    torch.save(results, root + "/" + name + "_validation_results.pt")
    plot_continuous_model_stats(results['stats'], root, name)

    print(f"Stats for {name}:")
    pprint.pprint(results['stats_bootstrapped'])

    with open(root + "/" + name + "_stats.json", "w") as f:
        json.dump({
            "genes": genes,
            "stats": {k: [float(x) for x in v] for k, v in results['stats'].items()},
            "stats_bootstrapped": {k: cast_dictkeys_to_float(v) for k, v in results['stats_bootstrapped'].items()}
        }, f)

def continuous_model(root, name, heldout_patient):
    with open(root + "/genes.txt") as f:
        genes = f.read().split("\n")
    
    model = load_model_from_path(root + "/" + name + "_model.pt", output_size=len(genes), use_pos_enc=False)
    model = model.to('cuda')
    # model = model.eval()

    # Get the valid dataset
    valid_dataset = load_patient(heldout_patient, genes)
    valid_dataset = adapt_for_cell_graph_models(valid_dataset, mode='validation', min_cells=10)

    results = validate_continuous_model(model, valid_dataset)

    torch.save(results, root + "/" + name + "_validation_results.pt")
    plot_continuous_model_stats(results['stats'], root, name)

    print(f"Stats for {name}:")
    pprint.pprint(results['stats_bootstrapped'])

    with open(root + "/" + name + "_stats.json", "w") as f:
        json.dump({
            "genes": genes,
            "stats": {k: [float(x) for x in v] for k, v in results['stats'].items()},
            "stats_bootstrapped": {k: cast_dictkeys_to_float(v) for k, v in results['stats_bootstrapped'].items()}
        }, f)

def sage_model(root, name, heldout_patient):
    with open(root + "/genes.txt") as f:
        genes = f.read().split("\n")

    model = GraphSageModel(output_size=len(genes), hidden_size=512, n_layers=4).to('cuda')
    model.load_state_dict(torch.load(root + "/" + name + "_model.pt"))

        # Get the valid dataset
    valid_dataset = load_patient(heldout_patient, genes)
    valid_dataset = adapt_for_cell_graph_models(valid_dataset, mode='validation', min_cells=10)

    results = validate_continuous_model(model, valid_dataset)

    torch.save(results, root + "/" + name + "_validation_results.pt")
    plot_continuous_model_stats(results['stats'], root, name)

    print(f"Stats for {name}:")
    pprint.pprint(results['stats_bootstrapped'])

    with open(root + "/" + name + "_stats.json", "w") as f:
        json.dump({
            "genes": genes,
            "stats": {k: [float(x) for x in v] for k, v in results['stats'].items()},
            "stats_bootstrapped": {k: cast_dictkeys_to_float(v) for k, v in results['stats_bootstrapped'].items()}
        }, f)

def cast_dictkeys_to_float(d):
    return {k1: float(v1) for k1, v1 in d.items()}

def load_gene_list(gene_set):
    with open(f'gene-sets/{gene_set}.json', 'r') as f:
        content = json.load(f)
    return content['geneSymbols']

def validate_with_gene_sets():
    gene_sets = [
        "HALLMARK_ANGIOGENESIS",
        "HALLMARK_DNA_REPAIR",
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_TGF_BETA_SIGNALING",
        "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
    ]

    names = [
        'ang',
        'dnarep',
        'e2f',
        'emt',
        'inf',
        'tgfb',
        'wntb',
    ]

    # Use sets here for faster lookups
    gene_sets_map = {s: set(load_gene_list(s)) for s in gene_sets}
    genes = torch.load("cell_model_with_pathways_genes_to_use.ptsave")

    load_cached = True
    
    # CELL_MODEL_TO_USE = "models/cell_model_g4_regressor_cytassist_e1_unifiedSingleCell_pathways"
    CELL_MODEL_TO_USE = "cell_model_from_repr_g4_regressor_cytassist_e1_unifiedSingleCell_pathways"

    autostainer_dataset = load_patient("autostainer", genes)
    autostainer_dataset = adapt_for_cell_graph_models(autostainer_dataset)
    model = load_model_from_path("models/" + CELL_MODEL_TO_USE + ".pt", len(genes))

    results = validate_continuous_model(model, autostainer_dataset)
    stats = results['stats']

    # Create violin plot from gene set
    spearman_by_gene_set = {k: [] for k in gene_sets}
    mse_by_gene_set = {k: [] for k in gene_sets}
    pearsonr_by_gene_set = {k: [] for k in gene_sets}

    for i, gene in enumerate(genes):
        for gene_set in gene_sets:
            if gene in gene_sets_map[gene_set]:
                mse_by_gene_set[gene_set].append(stats['mse_values'][i])
                spearman_by_gene_set[gene_set].append(stats['spearman_values'][i])
                pearsonr_by_gene_set[gene_set].append(stats['pearsonr_values'][i])

    def set_axis_style(ax, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Gene set name')

    # Create violin plots
    plt.clf()
    fig, (ax_mse, ax_spearman, ax_pearsonr) = plt.subplots(nrows=1, ncols=3, figsize=(24, 5))
    plt.subplots_adjust(wspace=0.4)

    ax_mse.set_title("MSE")
    ax_spearman.set_title("Spearman")
    ax_pearsonr.set_title("Pearsonr")

    gene_set_order = gene_sets

    for gene_set in gene_set_order:
        mse = np.median(mse_by_gene_set[gene_set])
        spearman = np.median(spearman_by_gene_set[gene_set])
        pearsonr = np.median(pearsonr_by_gene_set[gene_set])
        print(f"set {gene_set[9:]}: {mse=:.3f} {spearman=:.3f} {pearsonr=:.3f}")

    ax_mse.violinplot([mse_by_gene_set[gene_set] for gene_set in gene_set_order], showmedians=True)
    ax_spearman.violinplot([spearman_by_gene_set[gene_set] for gene_set in gene_set_order], showmedians=True)
    ax_pearsonr.violinplot([pearsonr_by_gene_set[gene_set] for gene_set in gene_set_order], showmedians=True)

    set_axis_style(ax_mse, names)
    set_axis_style(ax_spearman, names)
    set_axis_style(ax_pearsonr, names)

    plt.savefig(CELL_MODEL_TO_USE + "_violin.png")

def render_attributions(image, inference, cell_predictions, cell_indexes, gene_id, filename=None):
    import matplotlib
    import matplotlib.patches as patches
    
    cm = matplotlib.colormaps['viridis']
    
    # Create figure and axes
    fig, ax = plt.subplots()
    
    ax.set_title("Attribution map")
    
    # Class names (from `maskrcnn/visualize.py`)
    CLASSES = ["Neutrophil", "Epithelial", "Lymphocyte", "Plasma", "Eosinohil", "Connective"]

    # Display the image
    ax.imshow(image)
    
    # Choose the cell-level predictions for this specific gene
    cell_predictions = cell_predictions[:, gene_id]
    cell_predictions_for_colormap = (cell_predictions - cell_predictions.mean()) / (cell_predictions.std())
    
    attrib_by_cell_type = [0] * 6
    count_by_cell_type = [0] * 6
    
    max_prediction = abs(cell_predictions).max()
    total_prediction = abs(cell_predictions).sum()
    for prediction, idx in zip(cell_predictions_for_colormap, cell_indexes):
        # https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
        # Create a Rectangle patch
        x1, y1, x2, y2 = inference['boxes'][idx]
        rect = patches.Rectangle((int(x1)*2, int(y1)*2), int(x2-x1)*2, int(y2-y1)*2, linewidth=1, edgecolor=cm(prediction.item()), facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        attrib_by_cell_type[inference['labels'][idx]] += (prediction/total_prediction).item()
        count_by_cell_type[inference['labels'][idx]] += 1

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    
    plt.clf()

def tensor_image_to_numpy(image):
    # Converts (C, H, W) f32 to (H, W, C) u8
    return np.array(image.cpu().permute(1, 2, 0) * 255, dtype=np.uint8)

# def validate_and_render_attributions(CELL_MODEL_TO_USE):
#     genes = list(np.load("filtered_set_accessible_by_all_datasets.npy", allow_pickle=True))
#     autostainer_dataset = load_patient('autostainer', genes)
#     autostainer_dataset.patch_transform = None
    
#     # Now, show attribution results.
#     for gene in ['EPCAM', 'AXIN2', 'RAB25', 'COL6A1']:
#         for i in range(10):
#             image, _, detections = autostainer_dataset[i]
#             # render_attributions(image, inference, cell_predictions, cell_indexes, gene_id, filename=None)
#             render_attributions(
#                 tensor_image_to_numpy(image),
#                 detections,
#                 cell_pred['predictions']['cell'][i],
#                 cell_pred['metadata']['cell-indexes'][i],
#                 genes_to_use.index(gene),
#                 filename=f"models/cell_model_g4_regressor_trained_on_all_cytassist_epoch1_with_scRNA_regularization/attrib_{gene}_{i}.png"
#             )

if __name__ == '__main__':
    pass
    # sage_model('training_results_v2/v4_optimal_transport/v1_graphsage', 'scRNAmodel_patient3_prior-none-apply_reg-False', '091759_7')
    # sage_model('training_results_v2/v4_optimal_transport/v1_graphsage', 'scRNAmodel_patient3_prior-none-apply_reg-True', '091759_7')
    # sage_model('training_results_v2/v4_optimal_transport/v1_graphsage', 'scRNAmodel_patient7_prior-none-apply_reg-False', '092146_3')
    # sage_model('training_results_v2/v4_optimal_transport/v1_graphsage', 'scRNAmodel_patient7_prior-none-apply_reg-True', '092146_3')
    # continuous_model('training_results_v2/v4_optimal_transport', 'scRNAmodel_patient3_prior-repr-apply_reg-False', '091759_7')
    # continuous_model('training_results_v2/v4_optimal_transport', 'scRNAmodel_patient3_prior-repr-apply_reg-True', '091759_7')
    continuous_model('training_results_v2/v4_optimal_transport', 'scRNAmodel_patient7_prior-repr-apply_reg-False', '092146_3')
    continuous_model('training_results_v2/v4_optimal_transport', 'scRNAmodel_patient7_prior-repr-apply_reg-True', '092146_3')
    # sage_model('training_results_v2/v4_optimal_transport', 'scRNAmodel_patient7_prior-repr-apply_reg-False', '092146_3')
    # sage_model('training_results_v2/v4_optimal_transport', 'scRNAmodel_patient7_prior-repr-apply_reg-True', '092146_3')

    # src_folder = sys.argv[1]
    # model_name = sys.argv[2]
    # model_type = sys.argv[3]
    # patient = model_name.split("-")[-1]
    # if model_type == 'cnn':
    #     cnn_model(src_folder, model_name, patient)
    # elif model_type == 'cgnn':
    #     continuous_model(src_folder, model_name, patient)
    # elif model_type == 'cgnn:screg':
    #     heldout = sys.argv[4]
    #     continuous_model(src_folder, model_name, heldout)
    # # continuous_model(src_folder, model_name, patient)
    # # continuous_model('training_results_v2/v0', 'cellmodel_prior-none_cellreg-enabled_heldout-autostainer', 'autostainer')
    # # continuous_model('training_results_v2/v0', 'cellmodel_prior-none_heldout-autostainer', 'autostainer')
    # # for patient in ALL_PATIENTS:
    # #     continuous_model('training_results_v2/v2_repr_contrastive', f'cellmodel_prior-repr_cellreg-enabled_heldout-{patient}', patient)
    # # continuous_model('training_results_v2/v0_repr', 'cellmodel_prior-repr_cellreg-enabled_heldout-autostainer', 'autostainer')
