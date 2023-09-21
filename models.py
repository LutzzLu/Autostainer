import os
from typing import Dict, Optional, Union, overload, Tuple

import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as T
import tqdm

import checkpointing
import datasets
import loss
import validation

PIL.Image.MAX_IMAGE_PIXELS = 800000000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class FeatureExtractor(nn.Module):
    def __init__(self, image_size: int, d_model: int, model: Optional[torchvision.models.Inception3] = None):
        super().__init__()

        if model is None:
            model = torchvision.models.inception_v3(
                weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
            model.requires_grad_(True)

        self.hp = (image_size, d_model)

        self.image_size = image_size
        self.d_model = d_model

        assert d_model == 2048, "InceptionV3 uses a d_model of 2048"

        model.fc = nn.Identity()
        # # Remove the last layer and initialize with the same method
        # model.fc = nn.Linear(model.fc.in_features, d_model)
        # torch.nn.init.trunc_normal_(model.fc.weight, mean=0.0, std=0.1, a=-2, b=2)

        # Just don't worry about AuxLogits for now
        model.aux_logits = False
        model.AuxLogits = None

        self.model = model

    @staticmethod
    def from_hp(hp):
        model = torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1).requires_grad_()
        return FeatureExtractor(*hp, model)

    def forward(self, image: torch.Tensor):
        return self.model(image)


class FeaturesToGenes(nn.Module):
    def __init__(self, d_model: int, n_genes: int, mode='regression'):
        super().__init__()

        assert mode in ['regression',
                        'classification'], "Invalid mode: " + str(mode)

        self.hp = (d_model, n_genes, mode)

        self.d_model = d_model
        self.n_genes = n_genes
        self.mode = mode
        self.fc = nn.Linear(d_model, n_genes if mode !=
                            'regression' else n_genes * 3)

    @staticmethod
    def from_hp(hp):
        return FeaturesToGenes(*hp)

    def forward(self, features: torch.Tensor, labels=None):
        if self.mode == 'regression':
            h = self.fc(features).view(-1, 3, self.n_genes)

            # mean = torch.clamp(torch.exp(h[:, 0]), min=1e-5, max=1e6)
            log_mean = h[:, 0]
            disp = torch.clamp(F.softplus(h[:, 1]), min=1e-4, max=1e4)
            pi = torch.sigmoid(h[:, 2])

            if labels is None:
                return torch.clamp(torch.exp(log_mean), min=1e-5, max=1e6) * (1 - pi)
            else:
                # yes it is supposed to accept log_mean and un-logged logits
                return loss.ZINB_loss(labels, log_mean, disp, pi)

        elif self.mode == 'classification':
            logits = self.fc(features)

            if labels is not None:
                pos = labels == 1
                neg = labels == 0
                pos_logits = logits[pos]
                neg_logits = logits[neg]
                pos_labels = labels[pos]
                neg_labels = labels[neg]

                return 0.5 * (F.binary_cross_entropy_with_logits(pos_logits, pos_labels) + F.binary_cross_entropy_with_logits(neg_logits, neg_labels))
            else:
                return logits
        else:
            raise ValueError(f'Unknown mode: {self.mode}')


class PatchModel(nn.Module):
    def __init__(self, image_to_features: FeatureExtractor, features_to_genes: FeaturesToGenes, uid: str):
        super().__init__()

        self.image_to_features = image_to_features
        self.features_to_genes = features_to_genes

        self.hp = (image_to_features.hp, features_to_genes.hp)
        self.uid = uid
        self.parent = None
        self.epoch = -1
        self.step = 0

    @staticmethod
    def from_hp(hp):
        return PatchModel(FeatureExtractor.from_hp(hp[0]), FeaturesToGenes.from_hp(hp[1]))

    def forward(self, image: torch.Tensor, labels=None):
        return self.features_to_genes(self.image_to_features(image), labels)


class PatchModel2Level(nn.Module):
    """ Modified version of PatchModel2Level. Features are combined with a linear layer + activation function *before* being sent to FeaturesToGenes. """
    def __init__(self, image_to_features1: FeatureExtractor, image_to_features2: FeatureExtractor, features_to_genes: FeaturesToGenes, uid: str):
        super().__init__()

        self.image_to_features1 = image_to_features1
        self.image_to_features2 = image_to_features2
        self.features_to_genes = features_to_genes
        self.proj = nn.Linear(
            image_to_features1.d_model + image_to_features2.d_model,
            features_to_genes.d_model
        )

        self.uid = uid
        self.parent = None
        self.epoch = -1
        self.step = 0

    def forward(self, images: Tuple[torch.Tensor], labels=None):
        image1, image2 = images
        features1 = self.image_to_features1(image1)
        features2 = self.image_to_features2(image2)
        features = torch.cat((features1, features2), dim=-1)
        # combine features with MLP
        features = self.proj(features)
        # apply activation
        features = F.tanh(features)
        return self.features_to_genes(features, labels)


def train(model: Union[str, PatchModel], train_dataset: torch.utils.data.ConcatDataset, valid_dataset: datasets.PatchDataset):
    if type(model) == str:
        checkpoint = checkpointing.load(model)
        assert checkpoint, "Checkpoint not found: " + model
        model = checkpoint.model
        optimizer = checkpoint.optimizer
    else:
        assert isinstance(
            model, torch.nn.Module), "Model must be a torch.nn.Module or a path to a checkpoint."
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=16, shuffle=False)

    for epoch in range(model.epoch + 1, 2):
        model.train()
        model.epoch = epoch
        epoch_loss = 0
        start_step = model.step
        with tqdm.tqdm(train_loader, total=len(train_dataset)) as pbar:
            for patches, labels in train_loader:
                loss = model(patches, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(len(patches))
                model.step += len(patches)

                epoch_loss += loss.item() * len(patches)
                pbar.set_description(
                    f'epoch: {epoch} loss: {loss.item():.4f} epoch_loss: {epoch_loss / (model.step - start_step):.4f}')

        model.eval()
        with torch.no_grad():
            validation_loss = (torch.tensor([
                model(patches, labels) * len(patches) for patches, labels in tqdm.tqdm(valid_loader, desc='Calculating validation loss')
            ]).sum() / len(valid_dataset)).item()
            print("Validation loss: " + str(validation_loss))

        if os.path.exists('checkpoints/' + model.uid + '/analysis.pt'):
            os.remove('checkpoints/' + model.uid + '/analysis.pt')

        checkpointing.save(model, optimizer)

    return checkpointing.Checkpoint(model.uid)


def train_2level(model: Union[str, PatchModel2Level], train_dataset: datasets.PatchDataset2Level, valid_dataset: datasets.PatchDataset2Level, max_epoch: int):
    if type(model) == str:
        checkpoint = checkpointing.load(model)
        assert checkpoint, "Checkpoint not found: " + model
        model = checkpoint.model
        optimizer = checkpoint.optimizer
    else:
        assert isinstance(
            model, torch.nn.Module), "Model must be a torch.nn.Module or a path to a checkpoint."
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model: PatchModel2Level

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=datasets.PatchDataset2Level.collate_fn)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=16, shuffle=False, collate_fn=datasets.PatchDataset2Level.collate_fn)

    for epoch in range(model.epoch + 1, max_epoch):
        model.train()
        model.epoch = epoch
        epoch_loss = 0
        start_step = model.step
        with tqdm.tqdm(train_loader, total=len(train_dataset)) as pbar:
            for patches, labels in train_loader:
                loss = model(patches, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                count = len(labels)

                pbar.update(count)
                model.step += count

                epoch_loss += loss.item() * count
                pbar.set_description(
                    f'epoch: {epoch} loss: {loss.item():.4f} epoch_loss: {epoch_loss / (model.step - start_step):.4f}')

        model.eval()
        with torch.no_grad():
            validation_loss = (torch.tensor([
                model(patches, labels) * count for patches, labels in tqdm.tqdm(valid_loader, desc='Calculating validation loss')
            ]).sum() / len(valid_dataset)).item()
            print("Validation loss: " + str(validation_loss))

        if os.path.exists('checkpoints/' + model.uid + '/analysis.pt'):
            os.remove('checkpoints/' + model.uid + '/analysis.pt')

        checkpointing.save(model, optimizer)

    return checkpointing.Checkpoint(model.uid)


def validate(
    val: validation.Validation,
    slide: datasets.Slide,
    label: str,
    mode: str,
    render_gene: int,
    spot_size: int,
    skip_spatialde=False,
):
    if mode == 'classification':
        validation.render_binary_analysis(
            val, f'checkpoints/{label}/aggregate_binary_analysis.png')
    elif mode == 'regression':
        validation.render_regression_analysis(
            val, f'checkpoints/{label}/aggregate_regression_analysis.png')
    else:
        raise ValueError(f'Unknown mode: {mode}')

    if not skip_spatialde:
        print(" * Running SpatialDE")
        slide.spatialde(val.pred).to_csv(
            f'checkpoints/{label}/SpatialDEResults.csv')
    else:
        print(f" * Skipping SpatialDE [{skip_spatialde=}]")

    slide \
        .render(downsample=16, spot_counts=val.pred[:, render_gene], spot_size=spot_size) \
        .save(f'checkpoints/{label}/heatmap_pred_{render_gene}.png')
    slide \
        .render(downsample=16, spot_counts=slide.spot_counts[:, render_gene], spot_size=spot_size) \
        .save(f'checkpoints/{label}/heatmap_true_{render_gene}.png')

    print(" * Done")


def train_slide_with_quadrant_method(
    slide: datasets.Slide,
    label: str,
    patch_size: int,
    mode='regression'
):
    predictions_by_location = {}

    # Contains a list of 4 tuples of (held_out_quadrant, remaining_quadrants)
    quadrants_and_complements = slide.create_quadrants()
    for i, (valid_quadrant, train_quadrants) in enumerate(quadrants_and_complements):
        print("Training on quadrant", i, "for run", label)
        train_dataset = datasets.PatchDataset(
            train_quadrants, patch_size, magnify=1, patch_transform=training_transforms, device=device)
        valid_dataset = datasets.PatchDataset(
            valid_quadrant, patch_size, magnify=1, patch_transform=validation_transforms, device=device)

        # Create the feature extractors
        feature_extractor = FeatureExtractor(image_size=512, d_model=2048)
        features_to_genes = FeaturesToGenes(
            d_model=2048, n_genes=len(slide.genes), mode=mode)

        # Create the model
        uid = f"{label}/quadrant_{i}"
        if os.path.exists('checkpoints/' + uid):
            model = uid
        else:
            model = PatchModel(feature_extractor,
                               features_to_genes, uid).to(device)

        # Train the model
        checkpoint = train(model, train_dataset, valid_dataset)
        model = checkpoint.model

        with torch.no_grad():
            predictions = torch.cat([
                model(patches) for patches, labels in tqdm.tqdm(torch.utils.data.DataLoader(valid_dataset, batch_size=32), desc='Predicting on validation set')
            ])

        # Reassemble the predictions into the original slide
        for col, row, pred in zip(valid_quadrant.spot_locations.col, valid_quadrant.spot_locations.row, predictions):
            predictions_by_location[col.item(), row.item()] = pred

        if mode == 'classification':
            # Run validation for this quadrant
            validation.render_binary_analysis(
                validation.Validation(
                    valid_dataset.slide.spot_counts, predictions),
                f'checkpoints/{model.uid}/analyses/binary_analysis.png',
            )
        elif mode == 'regression':
            # Run validation for this quadrant
            validation.render_regression_analysis(
                validation.Validation(
                    valid_dataset.slide.spot_counts, predictions),
                f'checkpoints/{model.uid}/analyses/regression_analysis.png',
            )

    reordered_predictions = []
    for col, row in zip(slide.spot_locations.col, slide.spot_locations.row):
        reordered_predictions.append(
            predictions_by_location[col.item(), row.item()])

    aggregate_validation = validation.Validation(
        slide.spot_counts, torch.stack(reordered_predictions))

    # Run validation for all quadrants
    torch.save(aggregate_validation,
               f'checkpoints/{label}/aggregate_validation.pt')

    return aggregate_validation


def train_slide_with_quadrant_method_2level(
    slide: datasets.Slide,
    label: str,
    patch_size: int,
    mode='regression'
):
    predictions_by_location = {}

    # Contains a list of 4 tuples of (held_out_quadrant, remaining_quadrants)
    quadrants_and_complements = slide.create_quadrants()
    for i, (valid_quadrant, train_quadrants) in enumerate(quadrants_and_complements):
        print("Training on quadrant", i, "for run", label)
        train_dataset = datasets.PatchDataset2Level(
            datasets.PatchDataset(
                train_quadrants, patch_size, magnify=1, patch_transform=training_transforms, device=device),
            datasets.PatchDataset(
                train_quadrants, patch_size, magnify=0.5, patch_transform=training_transforms, device=device)
        )
        valid_dataset = datasets.PatchDataset2Level(
            datasets.PatchDataset(
                valid_quadrant, patch_size, magnify=1, patch_transform=validation_transforms, device=device),
            datasets.PatchDataset(
                valid_quadrant, patch_size, magnify=0.5, patch_transform=validation_transforms, device=device)
        )

        # Create the feature extractors
        feature_extractor1 = FeatureExtractor(
            image_size=patch_size, d_model=2048)
        feature_extractor2 = FeatureExtractor(
            image_size=patch_size, d_model=2048)
        features_to_genes = FeaturesToGenes(
            d_model=2048, n_genes=len(slide.genes), mode=mode)

        # Create the model
        uid = f"{label}/quadrant_{i}"
        if os.path.exists('checkpoints/' + uid):
            model = uid
        else:
            model = PatchModel2Level(feature_extractor1, feature_extractor2, features_to_genes, uid).to(device)

        # Train the model
        print("NOTE: max_epoch=4")
        checkpoint = train_2level(model, train_dataset, valid_dataset, max_epoch=4)
        model = checkpoint.model

        with torch.no_grad():
            predictions = torch.cat([
                model(patches) for patches, labels in tqdm.tqdm(torch.utils.data.DataLoader(valid_dataset, batch_size=32), desc='Predicting on validation set')
            ])

        # Reassemble the predictions into the original slide
        for col, row, pred in zip(valid_quadrant.spot_locations.col, valid_quadrant.spot_locations.row, predictions):
            predictions_by_location[col.item(), row.item()] = pred

        if mode == 'classification':
            # Run validation for this quadrant
            validation.render_binary_analysis(
                validation.Validation(
                    valid_quadrant.spot_counts, predictions),
                f'checkpoints/{model.uid}/analyses/binary_analysis.png',
            )
        elif mode == 'regression':
            # Run validation for this quadrant
            validation.render_regression_analysis(
                validation.Validation(
                    valid_quadrant.spot_counts, predictions),
                f'checkpoints/{model.uid}/analyses/regression_analysis.png',
            )

    reordered_predictions = []
    for col, row in zip(slide.spot_locations.col, slide.spot_locations.row):
        reordered_predictions.append(
            predictions_by_location[col.item(), row.item()])

    aggregate_validation = validation.Validation(
        slide.spot_counts, torch.stack(reordered_predictions))

    # Run validation for all quadrants
    torch.save(aggregate_validation,
               f'checkpoints/{label}/aggregate_validation.pt')

    return aggregate_validation


def train_all():
    # datasets.cache_slides()

    # region Load slides
    autostained_slide_20x = datasets.Slide.load(
        'input_data/preprocessed/autostainer_40x_cropped_downsampled_by_2.pkl')
    autostained_slide_40x = datasets.Slide.load(
        'input_data/preprocessed/autostainer_40x_cropped.pkl')
    manual_slide_20x = datasets.Slide.load(
        'input_data/preprocessed/manual_20x.pkl')
    # These are 20x slides
    visium_slides = [datasets.Slide.load(
        f'input_data/preprocessed/visium_{s}.pkl') for s in ['A1', 'B1', 'C1', 'D1']]

    filtered_set = [g for g in np.load(
        '1k_spatially_variable.npy', allow_pickle=True) if g in autostained_slide_20x.genes]
    whole_set = [
        g for g in autostained_slide_20x.genes if g in visium_slides[0].genes]
    # endregion

    def fsv(path): return pd.read_csv(path, index_col=0)['FSV'].to_numpy()

    # region Train autostainer slides
    for mode in ['classification', 'regression']:
        for subset_name in ['filtered_set']:
            print(f"Training group {{{mode=}, {subset_name=}}}")

            subset = {'filtered_set': filtered_set,
                      'whole_set': whole_set}[subset_name]

            if mode == 'classification':
                def prepare_slide(slide): return slide.binary(
                ).select_genes(subset, suppress_errors=True)
            else:
                def prepare_slide(slide): return slide.select_genes(
                    subset, suppress_errors=True)

            prefix = '2level/' + mode + '/' + subset_name

            val = train_slide_with_quadrant_method_2level(
                prepare_slide(autostained_slide_40x),
                prefix + '/autostained_40x_512',
                patch_size=512,
                mode=mode,
            )
            validate(
                val,
                prepare_slide(autostained_slide_40x),
                prefix + '/autostained_40x_512',
                mode,
                render_gene=16,
                spot_size=128,
                skip_spatialde=True,
            )

    # endregion

    # region Train Visium slides
    # for i, name in enumerate(['A1', 'B1', 'C1', 'D1']):
    #     train_slide_with_quadrant_method(visium_slides[i], f'whole_set/visium_{name}_512', 512)
    #     train_slide_with_quadrant_method(visium_slides[i], f'whole_set/visium_{name}_256', 256)
    # endregion


def evaluate(subfolder: str):
    import pandas as pd

    results: Dict[str, validation.Validation] = {}
    folders = sorted(os.listdir(f'checkpoints/{subfolder}'))
    for label in folders:
        results[label] = torch.load(
            f'checkpoints/{subfolder}/{label}/aggregate_validation.pt')

    # region Create table of results
    results_table = {"label": [], "auroc": [], "precision": [],
                     "recall": [], "accuracy": [], "cross_entropy": []}
    for label in folders:
        val = results[label]
        auroc = np.nanmedian(val.auroc)
        if np.isnan(auroc):
            print("NaN AUROC. val.auroc =", val.auroc)
        precision = np.nanmedian(val.precision)
        recall = np.nanmedian(val.recall)
        accuracy = np.nanmedian(val.accuracy)
        cross_entropy = np.nanmedian(val.cross_entropy)
        results_table["label"].append(label)
        results_table["auroc"].append(auroc)
        results_table["precision"].append(precision)
        results_table["recall"].append(recall)
        results_table["accuracy"].append(accuracy)
        results_table["cross_entropy"].append(cross_entropy)

    df = pd.DataFrame(results_table)
    df.to_csv(f'checkpoints/{subfolder}/results_table.csv', index=False)

    print(df)
    # endregion

    # region Create correlation matrices

    # Compare autostainer 20x slide with 20x manual slide
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')

    autostained_20x = results['autostained_20x']
    manual_20x = results['visium_D1_256']

    plt.title("Autostainer 20x vs Manual 20x")
    plt.xlabel("Manual 20x")
    plt.ylabel("Autostainer 20x")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.scatter(manual_20x.auroc, autostained_20x.auroc, s=3)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.savefig(
        f'checkpoints/{subfolder}/autostained_20x_vs_manual_20x_auroc.png')

    # endregion


if __name__ == '__main__':
    train_all()
    # evaluate('whole_set')
