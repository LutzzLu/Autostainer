import traceback
from functools import cached_property

import numpy as np
import scipy
import scipy.stats
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
import loss


def spearman(pred: np.ndarray, true: np.ndarray):
    return np.array([
        scipy.stats.spearmanr(pred[:, i], true[:, i])
        for i in range(pred.shape[1])
    ])

def rmse(pred: np.ndarray, true: np.ndarray):
    return np.sqrt(((pred - true) ** 2).mean(axis=0))


class Validation:
    def __init__(self, true: torch.Tensor, pred: torch.Tensor):
        self.true = true.cpu()
        self.pred = pred.cpu()

    def __getstate__(self):
        return {
            'true': self.true.cpu(),
            'pred': self.pred.cpu()
        }

    def __setstate__(self, state):
        self.true = state['true'].cpu()
        self.pred = state['pred'].cpu()

    @cached_property
    def spearman(self):
        pred_np = self.pred.cpu().numpy()
        true_np = self.true.cpu().numpy()
        return torch.tensor(spearman(pred_np, true_np))

    @cached_property
    def rmse(self):
        pred_np = self.pred.cpu().numpy()
        true_np = self.true.cpu().numpy()
        return torch.tensor(rmse(pred_np, true_np))
    
    @cached_property
    def auroc(self):
        import sklearn.metrics

        true_np = self.true.cpu().numpy()
        pred_np = self.pred.cpu().numpy()

        return np.array([
            sklearn.metrics.roc_auc_score(true_np[:, i], pred_np[:, i]) if len(np.unique(true_np[:, i])) == 2 else np.nan
            for i in range(pred_np.shape[1])
        ])
    
    @cached_property
    def cross_entropy(self):
        return torch.tensor([
            F.binary_cross_entropy_with_logits(self.pred[:, i], self.true[:, i])
            for i in range(self.pred.shape[1])
        ])

    @cached_property
    def accuracy(self):
        return ((torch.sigmoid(self.pred) > 0.5) == self.true).float().mean(dim=0)

    @cached_property
    def tp_tn_fp_fn(self):
        return loss.tp_tn_fp_fn(self.pred, self.true)

    @cached_property
    def precision(self):
        tp, tn, fp, fn = self.tp_tn_fp_fn
        return tp / (tp + fp)

    @cached_property
    def recall(self):
        tp, tn, fp, fn = self.tp_tn_fp_fn
        return tp / (tp + fn)

def _show_text(text: str, x: float, y: float):
    plt.text(
        x,
        y,
        text,
        horizontalalignment='left',
        verticalalignment='center',
        transform=plt.gca().transAxes
    )

def _show_median(statname: str, values: np.ndarray):
    _show_text(f'Median {statname}: {np.nanmedian(values):.4f}', 0.1, 0.9)

def render_regression_analysis(validation: Validation, out_file):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.title('Spearman correlation')
    plt.xlabel('Spearman correlation')
    plt.ylabel('Frequency')
    plt.xlim(0, 1)
    plt.hist(validation.spearman[:, 0], label='Spearman', bins=int(np.sqrt(validation.pred.shape[1])))
    _show_median('Spearman', validation.spearman[:, 0])
    
    plt.subplot(1, 2, 2)
    plt.title('RMSE')
    plt.xlabel('RMSE')
    plt.ylabel('Frequency')
    # plt.xlim(0, 1)
    plt.hist(validation.rmse, label='RMSE', bins=int(np.sqrt(validation.pred.shape[1])))
    _show_median('RMSE', validation.rmse)

    plt.savefig(out_file)

def render_binary_analysis(validation: Validation, out_file):
    ce = validation.cross_entropy.cpu().numpy()
    # accuracy = self.accuracy.cpu().numpy()
    auroc = validation.auroc

    N = len(auroc)

    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1)
    plt.title('Crossentropy')
    plt.xlabel('Crossentropy')
    plt.ylabel('Frequency')
    plt.xlim(0, 1)
    plt.hist(ce, label='Crossentropy', bins=int(np.sqrt(N)))

    plt.subplot(2, 2, 2)
    plt.title('AUROC')
    plt.xlabel('AUROC')
    plt.ylabel('Frequency')
    plt.xlim(0, 1)
    plt.hist(auroc, label='AUROC', bins=int(np.sqrt(N)))
    _show_median('AUROC', auroc)
    
    # plt.subplot(2, 2, 2)
    # plt.title('Accuracy')
    # plt.xlabel('Accuracy')
    # plt.ylabel('Frequency')
    # plt.xlim(0, 1)
    # plt.hist(accuracy, label='Accuracy', bins=int(np.sqrt(len(accuracy))))
    # # Write median accuracy
    # plt.text(0.1, 0.9, f'Median accuracy: {np.median(accuracy):.4f}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    tp, tn, fp, fn = loss.tp_tn_fp_fn(validation.pred, validation.true)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    overall_precision = tp.sum() / (tp.sum() + fp.sum())
    overall_recall = tp.sum() / (tp.sum() + fn.sum())
    print("Overall precision:", overall_precision)
    print("Overall recall:", overall_recall)
    print("Overall tp:", tp.sum())
    print("Overall tn:", tn.sum())
    print("Overall fp:", fp.sum())
    print("Overall fn:", fn.sum())
    try:
        precision = precision.cpu().numpy()
        recall = recall.cpu().numpy()
        plt.subplot(2, 2, 3)
        plt.title('Precision')
        plt.xlabel('Precision')
        plt.ylabel('Frequency')
        plt.xlim(0, 1)
        plt.hist(precision, label='Precision', bins=int(np.sqrt(N)))
        _show_median('Precision', precision)
        
        plt.subplot(2, 2, 4)
        plt.title('Recall')
        plt.xlabel('Recall')
        plt.ylabel('Frequency')
        plt.xlim(0, 1)
        plt.hist(recall, label='Recall', bins=int(np.sqrt(N)))
        _show_median('Recall', recall)

        plt.savefig(out_file)
    except Exception as e:
        traceback.print_exc()
        print("No precision and recall")

def render_fsv_comparison(true: np.ndarray, pred: np.ndarray, out_file):
    """
    `true` and `pred` are 1D arrays of shape (n_genes,). They contain
    the FSV values for the true and predicted datasets.

    This function renders:
     * Scatterplot of FSV
     * Spearman correlation
    """

    plt.figure(figsize=(10, 10))
    plt.title('FSV')
    plt.xlabel('True FSV')
    plt.ylabel('Predicted FSV')
    plt.scatter(true, pred, s=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    _show_text(f'Spearman correlation: {scipy.stats.spearmanr(true, pred)[0]:.4f}', 0.1, 0.9)
    plt.savefig(out_file)

# From https://umap-learn.readthedocs.io/en/latest/aligned_umap_basic_usage.html
def _axis_bounds(embedding):
    left, right = embedding.T[0].min(), embedding.T[0].max()
    bottom, top = embedding.T[1].min(), embedding.T[1].max()
    adj_h, adj_v = (right - left) * 0.1, (top - bottom) * 0.1
    return [left - adj_h, right + adj_h, bottom - adj_v, top + adj_v]

def render_aligned_umap(validation: Validation, out_file: str, colors=None):
    """
    Render UMAP embedding of the true and predicted datasets.
    I use [AlignedUMAP](https://umap-learn.readthedocs.io/en/latest/aligned_umap_basic_usage.html)
    to align the embeddings.

    Accepts custom colors to be able to compare subsets of points.
    """

    import umap
    import colorsys

    pred = validation.pred.cpu().numpy()
    true = validation.true.cpu().numpy()

    # Create dictionary to relate first "dataset" to second "dataset" (predictions to ground truth).
    # This is so AlignedUMAP knows which points to align.
    mapping = {i: i for i in range(len(pred))}

    aligned_mapper = umap.AlignedUMAP().fit([pred, true], relations=[mapping])

    # Each of these has shape (num_spots, 2)
    pred_embeddings, true_embeddings = aligned_mapper.embeddings_

    # Generate colors
    if colors is None:
        colors = [colorsys.hsv_to_rgb((np.arctan2(*true_embeddings[i]) + np.pi) / (2 * np.pi), 1.0, 1.0) for i in range(len(true_embeddings))]

    # Convert this into scatterplots.
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.title('UMAP Embeddings of Ground Truth Expression')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.scatter(true_embeddings.T[0], true_embeddings.T[1], s=4, c=colors)
    plt.axis(_axis_bounds(true_embeddings))

    plt.subplot(1, 2, 2)
    plt.title('UMAP Embedding of Predicted Expression')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.scatter(pred_embeddings.T[0], pred_embeddings.T[1], s=4, c=colors)
    plt.axis(_axis_bounds(pred_embeddings))

    plt.savefig(out_file)
