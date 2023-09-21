"""
### TRAINING SCHEME ###

1) Train model on *all data* using representation learning (without single cell penalization)
2) Train cells-based model using representation learning as pretraining
   - k-fold cross validation
   - discrete and continuous prediction

In order to streamline this as much as possible, we can define each training job with (k, 'discrete' | 'continuous')
Then, we can create a different Slurm job for each of these.

In order to ensure that we do not duplicate efforts across gene sets, we must make gene sets immutable. If we want to
change a gene set, we must clone it and create one with a new name. This is to prevent confusion on what the gene sets
are.

Usage:
- python trainer.py representation_learning out:<path>
- python trainer.py continuous_prediction gene_set_id:<id> heldout_patient:<id> prior:(none | repr) enable_cell_reg:(true | false) out:<path>
"""

__usage__ = __doc__[__doc__.index("Usage:"):]

import gc
import os
import sys
import traceback
from dataset_wrapper import DatasetWrapper

import anndata
import torch.utils.data
import tqdm

from adapt_for_cell_graph_models import adapt_for_cell_graph_models
from cell_model import (load_representation_model_from_path, train_cell_model, train_sage_model,
                        train_representation_model_from_scratch)
from cell_regularizer import (create_cell_reg, get_default_cell_reg,
                              load_A3_patient_7, load_A19_patient_3, CellRegularizerWithOptimalMatching)
import patch_model
from gene_sets import GENE_SETS
from load_datasets import ALL_PATIENTS, load_torch_datasets, load_patient


def validate_gene_set(gene_set_id):
    assert gene_set_id in GENE_SETS, f"Gene set {gene_set_id} is not a valid gene set. Valid ones are {list(GENE_SETS.keys())}"

    return GENE_SETS[gene_set_id]

def validate_path(path):
    if os.path.isfile(path):
        raise ValueError(f"Path {path} is a file. It must be a directory.")
    
    os.makedirs(path, exist_ok=True)

    return path.rstrip('/') # Remove trailing slash, if it's there.

def representation_learning(out):
    validate_gene_set("DUMMY")

    datasets = load_torch_datasets("DUMMY")
    combined_dataset = torch.utils.data.ConcatDataset(list(datasets.values()))
    combined_dataset = adapt_for_cell_graph_models(combined_dataset)
    model, history = train_representation_model_from_scratch(combined_dataset)

    out = validate_path(out)
    torch.save(model.state_dict(), out + "/representation_model.pt")
    torch.save(history, out + "/representation_history.pt")

def sage_trainer(gene_set_id, heldout_patient, enable_cell_reg, out, custom_train_list='', epochs=2):
    genes = validate_gene_set(gene_set_id)

    assert heldout_patient in ALL_PATIENTS, f"Held out patient {heldout_patient} is not in ALL_PATIENTS ({ALL_PATIENTS})"
    assert enable_cell_reg in [True, False], f"enable_cell_reg must be a Python boolean. Received: {repr(enable_cell_reg)}"
    # assert use_pos_enc in [True, False], f"use_pos_enc must be a Python boolean. Received: {repr(use_pos_enc)}"
    if type(epochs) != int:
        epochs = int(epochs)

    datasets = load_torch_datasets(gene_set_id)
    valid_dataset = datasets.pop(heldout_patient)
    if custom_train_list:
        keys = custom_train_list.split(",")
        train_dataset = torch.utils.data.ConcatDataset([datasets[key] for key in keys])
    else:
        # we have already popped the held out patient
        train_dataset = torch.utils.data.ConcatDataset(list(datasets.values()))
    train_dataset = adapt_for_cell_graph_models(train_dataset, mode='train', min_cells=11)

    if enable_cell_reg:
        cell_reg = get_default_cell_reg(genes)
    else:
        cell_reg = None
    
    # if prior == 'repr':
    #     starter = load_representation_model_from_path(out + "/representation_model.pt")
    # else:
    #     starter = None

    model, history = train_sage_model(train_dataset, len(genes), cell_reg, epochs)

    name = "cellmodel"
    # if prior:
    #     name += f"_prior-{prior}"
    if enable_cell_reg:
        name += "_cellreg-enabled"
    # if use_pos_enc:
    #     name += "_posenc"
    name += f"_heldout-{heldout_patient}"

    out = validate_path(out)
    torch.save(model.state_dict(), f"{out}/{name}_model.pt")
    torch.save(history, f"{out}/{name}_history.pt")

    with open(out + "/genes.txt", "w") as f:
        f.write('\n'.join(genes))
    
    from validation import sage_model as sage_model_validation
    sage_model_validation(out, name, heldout_patient)

def continuous_prediction(gene_set_id, heldout_patient, prior, enable_cell_reg, out, use_pos_enc, custom_train_list=''):
    genes = validate_gene_set(gene_set_id)

    assert heldout_patient in ALL_PATIENTS, f"Held out patient {heldout_patient} is not in ALL_PATIENTS ({ALL_PATIENTS})"
    assert enable_cell_reg in [True, False], f"enable_cell_reg must be a Python boolean. Received: {repr(enable_cell_reg)}"
    assert use_pos_enc in [True, False], f"use_pos_enc must be a Python boolean. Received: {repr(use_pos_enc)}"

    datasets = load_torch_datasets(gene_set_id)
    valid_dataset = datasets.pop(heldout_patient)
    if custom_train_list:
        keys = custom_train_list.split(",")
        train_dataset = torch.utils.data.ConcatDataset([datasets[key] for key in keys])
    else:
        # we have already popped the held out patient
        train_dataset = torch.utils.data.ConcatDataset(list(datasets.values()))
    train_dataset = adapt_for_cell_graph_models(train_dataset, mode='train', min_cells=11)

    if enable_cell_reg:
        cell_reg = get_default_cell_reg(genes)
    else:
        cell_reg = None
    
    if prior == 'repr':
        starter = load_representation_model_from_path(out + "/representation_model.pt")
    else:
        starter = None

    model, history = train_cell_model(train_dataset, len(genes), cell_reg, starter, use_pos_enc)

    name = "cellmodel"
    if prior:
        name += f"_prior-{prior}"
    if enable_cell_reg:
        name += "_cellreg-enabled"
    if use_pos_enc:
        name += "_posenc"
    name += f"_heldout-{heldout_patient}"

    out = validate_path(out)
    torch.save(model.state_dict(), f"{out}/{name}_model.pt")
    torch.save(history, f"{out}/{name}_history.pt")
    
    with open(out + "/genes.txt", "w") as f:
        f.write('\n'.join(genes))

def continuous_prediction_cellregeval(gene_set_id, prior, out, apply_reg_during_training, method='gat', validate_only=False):
    genes = validate_gene_set(gene_set_id)

    # datasets = load_torch_datasets(gene_set_id)
    datasets = {
        '092146_3': load_patient('092146_3', genes),
        '091759_7': load_patient('091759_7', genes),
    }
    patient_3_dataset = adapt_for_cell_graph_models(datasets['092146_3'])
    patient_3_scdata = load_A19_patient_3().select_genes(genes)
    patient_3_reg = create_cell_reg([patient_3_scdata], sample_count=512)
    patient_7_dataset = adapt_for_cell_graph_models(datasets['091759_7'])
    patient_7_scdata = load_A3_patient_7().select_genes(genes)
    patient_7_reg = create_cell_reg([patient_7_scdata], sample_count=512)
    
    if prior == 'repr':
        starter = load_representation_model_from_path(out + "/representation_model.pt")
    else:
        starter = None

    out = validate_path(out)

    mat3 = anndata.read("scRNA_tangram_matrix_patient3.h5")
    mat7 = anndata.read("scRNA_tangram_matrix_patient7.h5")

    patient_3_reg2 = CellRegularizerWithOptimalMatching(mat3.X, patient_3_scdata.spot_counts.numpy())
    patient_7_reg2 = CellRegularizerWithOptimalMatching(mat7.X, patient_7_scdata.spot_counts.numpy())

    if not validate_only:
        # ### TRAINING ON PATIENT 3 ###
        # if method == 'gat':
        #     model_3, history_3 = train_cell_model(patient_3_dataset, len(genes), patient_3_reg2 if apply_reg_during_training else None, starter)
        # elif method == 'sage':
        #     model_3, history_3 = train_sage_model(patient_3_dataset, len(genes), patient_3_reg2 if apply_reg_during_training else None)
        # else:
        #     raise ValueError("Invalid method specified: must be gat or sage, got " + str(method))
        # name_3 = f"scRNAmodel_patient3_prior-{prior}-apply_reg-{apply_reg_during_training}"
        # torch.save(model_3.state_dict(), f"{out}/{name_3}_model.pt")
        # torch.save(history_3, f"{out}/{name_3}_history.pt")

        # Evaluation
        # MSE_total, N_cells = evaluate_single_cell_match(model_3, patient_3_reg, patient_7_dataset)
        # print(f"[train on patient 3, hold out patient 7] {MSE_total=} {N_cells=} MSE_avg={MSE_total/N_cells}")

        torch.cuda.empty_cache()
        gc.collect()

        ### TRAINING ON PATIENT 7 ###

        model_7, history_7 = train_cell_model(patient_7_dataset, len(genes), patient_7_reg2 if apply_reg_during_training else None, starter)
        # model_7, history_7 = train_sage_model(patient_7_dataset, len(genes), patient_7_reg2 if apply_reg_during_training else None)
        name_7 = f"scRNAmodel_patient7_prior-{prior}-apply_reg-{apply_reg_during_training}"
        torch.save(model_7.state_dict(), f"{out}/{name_7}_model.pt")
        torch.save(history_7, f"{out}/{name_7}_history.pt")

        # Evaluation
        # MSE_total, N_cells = evaluate_single_cell_match(model_7, patient_7_reg, patient_3_dataset)
        # print(f"[train on patient 7, hold out patient 3] {MSE_total=} {N_cells=} MSE_avg={MSE_total/N_cells}")

    from validation import continuous_model

    continuous_model(out, f'scRNAmodel_patient7_prior-{prior}-apply_reg-{apply_reg_during_training}', '092146_3')
    # continuous_model(out, 'scRNAmodel_patient3_prior-{prior}-apply_reg-{apply_reg_during_training}', '091759_7')

    with open(out + "/genes.txt", "w") as f:
        f.write('\n'.join(genes))

def evaluate_single_cell_match(model, regularizer, heldout_dataset):
    MSE_total = 0
    cells_seen = 0

    with torch.no_grad():
        for entry in tqdm.tqdm(heldout_dataset, desc='Evaluating single cell match'):
            if entry is None:
                continue
            
            ((cell_images, cell_locations, valid_indexes), label) = entry
            pred = model(cell_images, cell_locations, valid_indexes)
            N_cells = len(pred.cell_vectors)
            MSE_total += regularizer.single(pred.cell_vectors) * N_cells
            cells_seen += N_cells
            
    return (MSE_total, cells_seen)

def patch_model_train_transforms(ds, idx):
    image, label = ds[idx]
    return patch_model.training_transforms(image), label

def cnn_trainer(gene_set_id, heldout_patient, out, custom_train_list=''):
    genes = validate_gene_set(gene_set_id)

    assert heldout_patient in ALL_PATIENTS, f"Held out patient {heldout_patient} is not in ALL_PATIENTS ({ALL_PATIENTS})"

    datasets = load_torch_datasets(gene_set_id, include_cells=False)

    # These are all PatchDatasetWithCellAnnotations
    # Each item of this gives you image, label, detections
    valid_dataset = datasets.pop(heldout_patient)
    if custom_train_list:
        keys = custom_train_list.split(",")
        train_dataset = torch.utils.data.ConcatDataset([datasets[key] for key in keys])
    else:
        # we have already popped the held out patient
        train_dataset = torch.utils.data.ConcatDataset(list(datasets.values()))
    train_dataset = DatasetWrapper(train_dataset, patch_model_train_transforms)

    model, history = patch_model.train_patch_model(train_dataset, len(genes))

    name = f"inception-holding-out-{heldout_patient}"

    torch.save(model.state_dict(), f"{out}/{name}_model.pt")
    torch.save(history, f"{out}/{name}_history.pt")

def validate_args(args):
    assert len(args) > 0, f"Invalid args format. Received: {args}"

    command = args[0]
    kwargs = {}

    for arg in args[1:]:
        assert ':' in arg, "Parameter is in invalid format. Must be in the form <key>:<value>."

        key = arg[:arg.index(':')]
        value = arg[arg.index(':')+1:]

        if value.lower() in ['true', 'false']:
            kwargs[key] = value.lower() == 'true'
            continue

        # try:
        #     kwargs[key] = int(value)
        #     continue
        # except ValueError:
        #     pass

        # try:
        #     kwargs[key] = float(value)
        #     continue
        # except ValueError:
        #     pass

        kwargs[key] = value

    if command == 'representation_learning':
        command = representation_learning
    elif command == 'continuous_prediction':
        command = continuous_prediction
    elif command == 'cnn_trainer':
        command = cnn_trainer
    elif command == 'sage_trainer':
        command = sage_trainer
    else:
        raise ValueError(f"Invalid command {command}. Valid commands are 'representation_learning' and 'continuous_prediction'.")

    return (command, kwargs)

def run(args):
    command, kwargs = validate_args(args)
    command(**kwargs)

# if __name__ == '__main__':
#     try:
#         run(sys.argv[1:])
#     except Exception as e:
#         print("Encountered an error.")
#         print(e)
#         traceback.print_exc()
#         print(__usage__)

if __name__ == '__main__':
    gene_set_id = 'cytassist-autostainer-top1k-by-rank-mean-thresholded'
    # continuous_prediction_cellregeval(gene_set_id, prior='repr', out='training_results_v2/v4_optimal_transport/v1_graphsage', apply_reg_during_training=True)
    # continuous_prediction_cellregeval(gene_set_id, prior='repr', out='training_results_v2/v4_optimal_transport/v1_graphsage', apply_reg_during_training=False)
    continuous_prediction_cellregeval(
        gene_set_id,
        prior='repr',
        out='training_results_v2/v4_optimal_transport/fr',
        apply_reg_during_training=True,
        validate_only=True,
        method='gat',
    )
    continuous_prediction_cellregeval(
        gene_set_id,
        prior='repr',
        out='training_results_v2/v4_optimal_transport/fr',
        apply_reg_during_training=False,
        method='gat',
    )
    # cnn_trainer(gene_set_id, 'autostainer', out='training_results_v2/v5_cnn_models')
