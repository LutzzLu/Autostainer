import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as T

import loss_composition
from optimization import pass_once


def standard_scaler_per_channel(patch: torch.Tensor):
    # patch.shape: (3, H, W)
    # (3, 1, 1)
    mean = patch.mean(dim=(1, 2), keepdim=True)
    std = patch.std(dim=(1, 2), keepdim=True)
    return (patch - mean) / std


training_transforms = T.Compose([
    standard_scaler_per_channel,
    T.ColorJitter(),
    T.RandomRotation(90),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
])
validation_transforms = standard_scaler_per_channel

def get_model(n_genes):
    model = torchvision.models.inception_v3(
        weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
    model.requires_grad_(True)

    # Remove the last layer and initialize with the same method
    model.fc = nn.Linear(model.fc.in_features, n_genes)
    torch.nn.init.trunc_normal_(model.fc.weight, mean=0.0, std=0.1, a=-2, b=2)

    # Just don't worry about AuxLogits for now
    model.aux_logits = False
    model.AuxLogits = None

    return model

def mse_loss(pred_batch, label_batch, *_):
    return torch.nn.functional.mse_loss(pred_batch, torch.stack(label_batch))

def stack_inputs(model):
    return lambda inputs: model(torch.stack(inputs))

def train_patch_model(dataset, output_size):
    model = get_model(output_size).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = []

    EPOCHS = 2

    for epoch in range(EPOCHS):
        # print timestamped message
        print(f"Epoch {epoch} started: {datetime.datetime.now().isoformat()}")

        history.extend(
            pass_once(stack_inputs(model), optimizer, dataset, loss_composition.compose_losses({"mse": (mse_loss, 1.0)}), batch_size=32)
        )

    return model, history
