import datetime
from collections import namedtuple
from typing import List, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
import torch_geometric.nn as gnn
import torchvision

from cell_regularizer import CellRegularizer
from create_graph import create_graph
from loss_composition import compose_losses
from optimization import pass_once

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_image_embedder(hidden_size):
    model = torchvision.models.resnet50('IMAGENET1K_V1')
    model.fc = torch.nn.Linear(2048, hidden_size)
    return model

CellGraphModelOutputs = namedtuple('CellGraphModelOutputs', ['cell_vectors', 'graph_vector'])

LAPLACIAN_K = 6

class GraphSageModel(torch.nn.Module):
    def __init__(self, output_size, hidden_size, n_layers):
        super().__init__()
        
        self.image_embedder = create_image_embedder(hidden_size)
        self.gnn = gnn.GraphSAGE(
            in_channels=hidden_size,
            hidden_channels=hidden_size,
            num_layers=n_layers,
            out_channels=output_size,
        )

    def forward(self, cell_images, cell_locations, *_, augment=False):
        cell_embeddings = self.image_embedder(cell_images)

        graph = create_graph(cell_embeddings, cell_locations, augment=augment)
        out = self.gnn.forward(graph.x, graph.edge_index)

        return CellGraphModelOutputs(
            cell_vectors=out,
            graph_vector=self.aggregate(out),
        )
    
    def aggregate(self, cell_predictions: torch.Tensor):
        rectified = torch.nn.functional.relu(cell_predictions)
        exponentiated = torch.exp(rectified) - 1
        summed = torch.sum(exponentiated, dim=0)
        logged_sum = torch.log(1 + summed)
        return logged_sum

class CellGraphModel(torch.nn.Module):
    def __init__(self, output_size, hidden_size, n_layers, aggregate_mode='log_exponent_sum', use_pos_enc=False):
        super().__init__()
        
        self.image_embedder = create_image_embedder(hidden_size)
        self.layers = torch.nn.ModuleList([
            gnn.GATConv(hidden_size, hidden_size)
            for idx in range(n_layers)
        ])
        self.projection = torch.nn.Linear(hidden_size, output_size)
        # Unused now
        self.graph_embedder = torch.nn.Linear(hidden_size, hidden_size)
        self.aggregate_mode = aggregate_mode
        self.use_pos_enc = use_pos_enc
        if self.use_pos_enc:
            # projects from k = 8 Laplacian eigenvector PE to hidden_size
            self.pos_enc = torch.nn.Linear(LAPLACIAN_K, hidden_size)

    def forward(self, *args, augment=False, input_type='images_and_locations'):
        """
        input_type is a workaround for pytorch-geometric.
        `images_and_locations`:
            cell_images, cell_locations: torch.Tensor
            Returns a CellGraphModelOutputs object.
        `nodes_and_edges`:
            x, edge_index: torch.Tensor
            Returns the prediction as a tensor.
        """

        if input_type == 'images_and_locations':
            cell_images, cell_locations, *_ = args
            cell_embeddings = self.image_embedder(cell_images)

            graph_data = create_graph(cell_embeddings, cell_locations, augment=augment, use_pos_enc=self.use_pos_enc)
            x = graph_data.x
            edge_index = graph_data.edge_index
            if self.use_pos_enc:
                x = x + self.pos_enc(graph_data.laplacian_eigenvector_pe.to(x.device))
        elif input_type == 'nodes_and_edges':
            x, edge_index = args
        else:
            raise ValueError("Invalid input_type: Must be one of `images_and_locations` or `nodes_and_edges`.")
        
        for layer in self.layers:
            x = x + layer(x, edge_index)
        
        x = self.projection(x)

        if input_type == 'nodes_and_edges':
            return self.aggregate(x)
        else:
            return CellGraphModelOutputs(
                cell_vectors=x,
                graph_vector=self.aggregate(x),
            )
    
    def aggregate(self, cell_predictions: torch.Tensor):
        if self.aggregate_mode == 'log_exponent_sum':
            # Returns estimated log prediction for the spot (log of sum of exponentiated cell log predictions)
            # uses gelu so it's still differentiable but doesn't dip below 0
            G = torch.nn.functional.relu(cell_predictions)
            E = torch.exp(G) - 1
            S = torch.sum(E, dim=0)
            # print("min(G), min(E), min(S))")
            # print(torch.min(G), torch.min(E), torch.min(S))
            return torch.log(1 + S)
        else:
            return torch.mean(cell_predictions, dim=0)

def _transpose_outputs(outputs: List[CellGraphModelOutputs]):
    return CellGraphModelOutputs(
        cell_vectors=[output.cell_vectors for output in outputs],
        graph_vector=[output.graph_vector for output in outputs],
    )

def _vectorize(model: CellGraphModel):
    # Input: list[tuple], where each tuple is an instance of the model's input
    def wrapper(args_batch, **kwargs):
        return _transpose_outputs([model(*args, **kwargs) for args in args_batch])
    return wrapper

"""
Cell models.

Type I: Representation-Learned Models

These models are trained with graph contrastive learning.

Training input:
 - A Torch dataset where each entry has (cell image, x position, y position) * N
"""

def contrastive_loss(embeddings_1_and_2, _label_batch):
    embeddings1, embeddings2 = embeddings_1_and_2
    # add loss term to make embeddings near unit length
    # target_length = torch.sqrt(torch.tensor(embeddings1.shape[-1], dtype=torch.float32, device=embeddings1.device))
    target_length = 1
    norm_loss = torch.mean((target_length - torch.norm(embeddings1, dim=-1)) ** 2 + (target_length - torch.norm(embeddings2, dim=-1)) ** 2)
    # embeddings1 = F.normalize(embeddings1, p=2, dim=-1)
    # embeddings2 = F.normalize(embeddings2, p=2, dim=-1)

    scores = (embeddings1 @ embeddings2.T)
    contr_loss = F.cross_entropy(scores, torch.arange(len(embeddings1), device=embeddings1.device))
    return contr_loss + norm_loss

def pass_once_with_representation_learning(model, optimizer, dataset):
    # Outputs original embeddings + positive pairs for a set of inputs
    def wrapped_model_contrastive(args_batch):
        embeddings1 = torch.stack(model(args_batch, augment=True).graph_vector)
        embeddings2 = torch.stack(model(args_batch, augment=False).graph_vector)
        return (embeddings1, embeddings2)

    return pass_once(wrapped_model_contrastive, optimizer, dataset, compose_losses({
        "contrastive": (contrastive_loss, 1.0),
    }))

def train_representation_model_from_scratch(dataset):
    model = CellGraphModel(output_size=512, hidden_size=512, n_layers=4, aggregate_mode='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    history = pass_once_with_representation_learning(_vectorize(model), optimizer, dataset)

    return model, history

"""
Type II: Cell Graph Models

These models can be based off of the pretrained ones, but this is not necessary.

Training input:
 - A Torch dataset where each entry has (cell image, x position, y position) * N and a target
 - Optionally, a CellRegularizer
"""

def mse_loss(pred_batch: CellGraphModelOutputs, label_batch, *_):
    if len(label_batch) == 0:
        return torch.tensor(0).requires_grad_()
    
    return torch.nn.functional.mse_loss(torch.stack(pred_batch.graph_vector), torch.stack(label_batch))

def pass_once_with_cell_model(model, optimizer, dataset, cell_regularizer: Optional[CellRegularizer]):
    return pass_once(model, optimizer, dataset, compose_losses({
        "mse": (mse_loss, 1.0),
        "cell": (cell_regularizer.batch if cell_regularizer else None, 1.0)
    }))

def create_cell_model(output_size, starter=None, use_pos_enc=False):
    if starter:
        model = CellGraphModel(output_size=starter.projection.out_features, hidden_size=512, n_layers=4, aggregate_mode='log_exponent_sum', use_pos_enc=use_pos_enc)
        model.load_state_dict(starter.state_dict())
        model.projection = torch.nn.Linear(512, output_size)
    else:
        model = CellGraphModel(output_size=output_size, hidden_size=512, n_layers=4, aggregate_mode='log_exponent_sum', use_pos_enc=use_pos_enc)

    return model

def load_representation_model_from_path(path):
    return load_model_from_path(path, output_size=512)

def load_model_from_path(path, output_size, use_pos_enc=False):
    model = CellGraphModel(output_size=output_size, hidden_size=512, n_layers=4, use_pos_enc=use_pos_enc)
    model.load_state_dict(torch.load(path))
    return model

def train_cell_model(dataset, output_size, cell_regularizer: Optional[CellRegularizer] = None, starter: Optional['CellGraphModel'] = None, use_pos_enc=False, epochs=1):
    model = create_cell_model(output_size, starter, use_pos_enc).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = []

    for epoch in range(epochs):
        # print timestamped message
        print(f"Epoch {epoch} started: {datetime.datetime.now().isoformat()}")
        history.extend(
            pass_once_with_cell_model(_vectorize(model), optimizer, dataset, cell_regularizer)
        )

    return model, history

def train_sage_model(dataset, output_size, cr: CellRegularizer, epochs=1):
    model = GraphSageModel(output_size=output_size, hidden_size=512, n_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = []

    for epoch in range(epochs):
        # print timestamped message
        print(f"Epoch {epoch} started: {datetime.datetime.now().isoformat()}")
        history.extend(
            pass_once_with_cell_model(_vectorize(model), optimizer, dataset, cr)
        )

    return model, history
