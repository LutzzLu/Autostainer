import os

# Proxy methods
def representation_learning(out):
    os.system(f"sbatch run_pyg.slurm python -u trainer.py representation_learning out:{out}")

def continuous_prediction(gene_set_id, heldout_patient, prior, enable_cell_reg, out, use_pos_enc='false', custom_train_list=''):
    # print(f"sh interactive_pyg.sh python -u trainer.py continuous_prediction gene_set_id:{gene_set_id} heldout_patient:{heldout_patient} prior:{prior} enable_cell_reg:{enable_cell_reg} out:{out}")
    os.system(f"sbatch run_pyg.slurm python -u trainer.py continuous_prediction gene_set_id:{gene_set_id} heldout_patient:{heldout_patient} prior:{prior} enable_cell_reg:{enable_cell_reg} out:{out} custom_train_list:{custom_train_list} use_pos_enc:{use_pos_enc}")

def sage_trainer(gene_set_id, heldout_patient, enable_cell_reg, out, custom_train_list='', epochs=2):
    # print(f"sh interactive_pyg.sh python -u trainer.py continuous_prediction gene_set_id:{gene_set_id} heldout_patient:{heldout_patient} prior:{prior} enable_cell_reg:{enable_cell_reg} out:{out}")
    os.system(f"sbatch run_pyg.slurm python -u trainer.py sage_trainer gene_set_id:{gene_set_id} heldout_patient:{heldout_patient} enable_cell_reg:{enable_cell_reg} out:{out} custom_train_list:{custom_train_list} epochs:{epochs}")

def cnn_trainer(gene_set_id, heldout_patient, out, custom_train_list=''):
    os.system(f"sbatch run_pyg.slurm python -u trainer.py cnn_trainer gene_set_id:{gene_set_id} heldout_patient:{heldout_patient} out:{out} custom_train_list:{custom_train_list}")

def validate_cnn_model(src_folder, model_name):
    os.system(f"sbatch run_pyg.slurm python -u validation.py \"{src_folder}\" \"{model_name}\" cnn")

def validate_continuous_prediction(src_folder, model_name):
    os.system(f"sbatch run_pyg.slurm python -u validation.py \"{src_folder}\" \"{model_name}\" cgnn")

def validate_cellreg_test(src_folder, model_name, held_out_patient):
    os.system(f"sbatch run_pyg.slurm python -u validation.py \"{src_folder}\" \"{model_name}\" cgnn:screg {held_out_patient}")

# validate_continuous_prediction("training_results_v2/v2_repr_contrastive", "cellmodel_prior-repr_heldout-092842_17")

gene_set_id = 'cytassist-autostainer-top1k-by-rank-mean-thresholded'

patients_to_hold_out = [
    'autostainer',
    '092534_24',
    '092534_35',
    '091759_4',
    '091759_7',
    '092146_33',
    '092146_3',
    '092842_17',
    '092842_16'
]

# continuous_prediction(gene_set_id, '092842_17', 'repr', False, 'training_results_v2/v2_repr_contrastive')
# continuous_prediction(gene_set_id, '092842_16', 'none', False, 'training_results_v2/v2_repr_contrastive')

# validate_continuous_prediction('training_results_v2/v2_repr_contrastive', 'cellmodel_prior-repr_heldout-092842_17')
# exit()

for PID in patients_to_hold_out:
    # sage_trainer(gene_set_id, PID, True, 'training_results_v2/v10_graphsage_cellreg')
    sage_trainer(gene_set_id, PID, False, 'training_results_v2/v10_graphsage')

# for PID in patients_to_hold_out:
#     os.system(f"sbatch run_pyg_cpu.slurm python -u render_3d_deconv.py {PID}")
    
# for SLIDE_ID iSn ['autostainer', '092534', '091759', '092146', '092842']:
# for SLIDE_ID in ['autostainer', '092534', '092146', '092842']:
    # if len(os.listdir("figures/" + SLIDE_ID)) == 0:
    # os.system(f"sbatch run_pyg_cpu.slurm python -u render_3d_deconv.py {SLIDE_ID}")

exit()

# representation_learning('./training_results_v2/v2_repr_contrastive')
# exit()

# validating optimal transport models

# # models are named after what they were TRAINED on, in this case.
# validate_cellreg_test("training_results_v2/v4_optimal_transport", "scRNAmodel_patient3_prior-none-apply_reg-False", "091759_7")
# validate_cellreg_test("training_results_v2/v4_optimal_transport", "scRNAmodel_patient3_prior-none-apply_reg-True", "091759_7")
# validate_cellreg_test("training_results_v2/v4_optimal_transport", "scRNAmodel_patient7_prior-none-apply_reg-False", "092146_3")
# validate_cellreg_test("training_results_v2/v4_optimal_transport", "scRNAmodel_patient7_prior-none-apply_reg-True", "092146_3")

# training mini graph laplacian PE test

def train_mini_graph_laplacian_test():
    # continuous_prediction(
    #     gene_set_id=gene_set_id,
    #     heldout_patient='092146_3',
    #     prior='none',
    #     enable_cell_reg='false',
    #     out="training_results_v2/v8_mini_graph_laplacian",
    #     use_pos_enc='true',
    #     custom_train_list='091759_7'
    # )

    validate_continuous_prediction(
        src_folder="training_results_v2/v8_mini_graph_laplacian",
        model_name="cellmodel_prior-none_posenc_heldout-091759_7",
    )

    # continuous_prediction(
    #     gene_set_id=gene_set_id,
    #     heldout_patient='091759_7',
    #     prior='none',
    #     enable_cell_reg='false',
    #     out="training_results_v2/v8_mini_graph_laplacian",
    #     use_pos_enc='true',
    #     custom_train_list='092146_3'
    # )

    validate_continuous_prediction(
        src_folder="training_results_v2/v8_mini_graph_laplacian",
        model_name="cellmodel_prior-none_posenc_heldout-092146_3",
    )

    return

    # cnn_trainer(
    #     gene_set_id=gene_set_id,
    #     heldout_patient='092146_3',
    #     out="training_results_v2/v9_mini_graph_laplacian_cnn_comparison",
    #     custom_train_list='091759_7'
    # )

    validate_cnn_model(
        src_folder="training_results_v2/v9_mini_graph_laplacian_cnn_comparison",
        model_name="inception-holding-out-091759_7",
    )

    # cnn_trainer(
    #     gene_set_id=gene_set_id,
    #     heldout_patient='091759_7',
    #     out="training_results_v2/v9_mini_graph_laplacian_cnn_comparison",
    #     custom_train_list='092146_3'
    # )

    validate_cnn_model(
        src_folder="training_results_v2/v9_mini_graph_laplacian_cnn_comparison",
        model_name="inception-holding-out-092146_3",
    )

# train_mini_graph_laplacian_test()

# exit()

for patient in patients_to_hold_out:
    out = "training_results_v2/v6_cnn_models_augmented_2_epochs"
    name = f"inception-holding-out-{patient}"
    out_model_path = f"{out}/{name}_model.pt"
    if os.path.exists(out_model_path):
        print("skip training", out_model_path)
        out_validation_path = f"{out}/{name}_validation_results.pt"
        if os.path.exists(out_validation_path):
            print("skip validation", out_validation_path)
        else:
            print("validate", out_validation_path)
            validate_cnn_model(out, name)
    else:
        print("train", out_model_path)
        cnn_trainer(gene_set_id, patient, out)

    for enable_cell_reg in ['true']: # ['true', 'false']:
        for prior in ['none']: # ['repr', 'none']:
            # out = f"./training_results_v2/v2_repr_contrastive"
            out = "training_results_v2/v7_cell_models_augmented_2_epochs"
            cellregenabled_string = "cellreg-enabled_" if enable_cell_reg == 'true' else ""
            model_identifier = f"cellmodel_prior-{prior}_{cellregenabled_string}heldout-{patient}"

            # For training
            out_model_path = out + f"/{model_identifier}_model.pt"
            if os.path.exists(out_model_path):
                print("skip training", out_model_path)

                out_validation_path = out + f"/{model_identifier}_validation_results.pt"
                if os.path.exists(out_validation_path):
                    print("skip validation", out_validation_path)
                else:
                    print("validate", out_validation_path)
                    validate_continuous_prediction(out, model_identifier)
            else:
                print("train", out_model_path)
                continuous_prediction(gene_set_id=gene_set_id, heldout_patient=patient, prior=prior, enable_cell_reg=enable_cell_reg, out=out)

