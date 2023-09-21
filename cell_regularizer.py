import pickle
from typing import TYPE_CHECKING

import faiss
import numpy as np
import torch
import scipy.optimize
import scipy.spatial.distance
from omicsio.datasets import Slide

if TYPE_CHECKING:
    from cell_model import CellGraphModelOutputs

class CellRegularizer:
    def __init__(self, counts):
        self.counts = counts
        self.index = faiss.IndexFlatL2(counts.shape[1])
        self.index.add(counts.cpu().numpy())

    def single(self, cell_predictions):
        np_vectors = cell_predictions.cpu().detach().numpy()
        L2_distances, nearest_neighbor_indexes = self.index.search(np_vectors, k=1)
        # select the 0th nearest neighbor. the shape is (n, k)
        nearest_neighbor_indexes = nearest_neighbor_indexes[:, 0]
        nearest_neighbor_counts = self.counts[nearest_neighbor_indexes].to(cell_predictions.device)
        return torch.nn.functional.mse_loss(nearest_neighbor_counts, cell_predictions)
    
    def batch(self, cell_predictions_batch: 'CellGraphModelOutputs', *_):
        if len(cell_predictions_batch.cell_vectors) == 0:
            return torch.tensor(0).requires_grad_()
        
        return torch.stack([self.single(p) for p in cell_predictions_batch.cell_vectors]).mean()

class CellRegularizerWithOptimalMatching:
    def __init__(self, matrix: np.ndarray, scData: np.ndarray):
        # matrixx[i, j] = P(spot j | cell i)
        self.matrix = matrix
        self.scData = scData

    def single(self, predicted_cells_expression, spot_id):
        num_cells = predicted_cells_expression.shape[0]
        sc_indexes = np.argpartition(self.matrix[:, spot_id], num_cells)[:num_cells]
        sc_data = self.scData[sc_indexes]
        cost_matrix = scipy.spatial.distance.cdist(sc_data, predicted_cells_expression.detach().cpu().numpy(), metric='euclidean')
        # optimal matching
        row_indexes, col_indexes = scipy.optimize.linear_sum_assignment(cost_matrix)
        sc_tensor = torch.tensor(sc_data[row_indexes], device=predicted_cells_expression.device).log1p()
        return torch.nn.functional.mse_loss(predicted_cells_expression[col_indexes], sc_tensor)
    
    def batch(self, cell_predictions_batch: 'CellGraphModelOutputs', _labels, spot_ids):
        if len(cell_predictions_batch.cell_vectors) == 0:
            return torch.tensor(0).requires_grad_()
        
        return torch.stack([self.single(p, spot_id) for p, spot_id in zip(cell_predictions_batch.cell_vectors, spot_ids)]).mean()

# New loss function idea: Training based on similarity to scRNA-seq data
cell_data_root = "./input_data/all_single_cell_data/colon"

def load_scRNA_datasets():
    return load_A19_patient_3(), load_A3_patient_7()

def load_A19_patient_3() -> 'Slide':
    with open(cell_data_root + "/DH_feb_23/SP18_56_A19_preprocessed_cached.pkl", "rb") as f:
        sc1 = pickle.load(f)
    return sc1

def load_A3_patient_7() -> 'Slide':
    with open(cell_data_root + "/DH_feb_23/SP18_16438_A3_preprocessed_cached.pkl", "rb") as f:
        sc2 = pickle.load(f)
    return sc2

def create_cell_reg(datasets, sample_count=512):
    counts_unified = torch.cat(tuple(ds.spot_counts for ds in datasets), dim=0)
    counts_sample = counts_unified[torch.randperm(len(counts_unified))[:sample_count]]

    return CellRegularizer(counts_sample.log1p())

def get_default_cell_reg(genes):
    # scRNA_dataset_1: 56_A19
    # scRNA_dataset_2: 16438_A3
    scRNA_dataset_1, scRNA_dataset_2 = load_scRNA_datasets()

    return create_cell_reg([scRNA_dataset_1.select_genes(genes), scRNA_dataset_2.select_genes(genes)])
