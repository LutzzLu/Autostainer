import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

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

def bootstrap(values):
    values = values[~np.isnan(values)]
    np.random.seed(42)
    performances_bootstrapped=[np.quantile(np.random.choice(values,len(values),replace=True),0.5) for i in range(1000)]
    med = np.quantile(values,0.5)
    p975 = np.quantile(performances_bootstrapped,[0.025,0.975])
    width = (p975[1]-p975[0])/2
    return {"median": med, "width": width, "formatted": f"{med:.4f} ± {width:.4f}"}

def aggregate_cross_val_data(model_identifier_format: str):
    variant_aurocs = []
    variant_spearmans = []
    variant_mses = []
    variant_maes = []

    # take mean of each stat for each gene
    for patient in patients_to_hold_out:
        model_identifier = model_identifier_format.format(patient=patient)

        try:
            # load stats
            with open(model_identifier + "_stats.json") as f:
                stats = json.load(f)['stats']

            variant_aurocs.append(stats['auroc_values'])
            variant_spearmans.append(stats['spearman_values'])
            variant_mses.append(stats['mse_values'])

            results = torch.load(model_identifier + "_validation_results.pt", map_location='cpu')
            if type(results['predictions']) == torch.Tensor:
                preds = results['predictions']
            else:
                assert type(results['predictions']) == dict, "Got invalid type; type was " + str(type(results['predictions']))
                preds = torch.stack(results['predictions']['graph_vector_batch'])
            trues = results['true_log_counts']
            variant_maes.append(torch.abs(preds - trues).mean(dim=0).cpu().numpy())

        except Exception as e:
            print("Skipping " + model_identifier + " due to " + str(e))
    
    return (
        np.array(variant_aurocs).mean(axis=0),
        np.array(variant_spearmans).mean(axis=0),
        np.array(variant_mses).mean(axis=0),
        np.array(variant_maes).mean(axis=0),
    )

def create_violin_plots(aurocs_dict, spearmans_dict):
    # Create violin plots
    sns.set_theme(style="whitegrid")

    aurocs_df = pd.DataFrame.from_dict(aurocs_dict)
    aurocs_df = aurocs_df.melt(var_name='variant', value_name='AUROC')

    spearmans_df = pd.DataFrame.from_dict(spearmans_dict)
    spearmans_df = spearmans_df.melt(var_name='variant', value_name='Spearman')

    sns.violinplot(x="variant", y="AUROC", data=aurocs_df)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/all_variants_auroc_violin_plot.png")

    plt.clf()

    sns.violinplot(x="variant", y="Spearman", data=spearmans_df)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/all_variants_spearman_violin_plot.png")

def create_violin_plot_comparing_methods():
    aurocs_dict = {}
    spearmans_dict = {}

    for enable_cell_reg in ['true', 'false']:
        for prior in ['repr', 'none']:
            out = f"./training_results_v2/v2_repr_contrastive"
            cellregenabled_string = "cellreg-enabled_" if enable_cell_reg == 'true' else ""
            model_identifier_format = f"cellmodel_prior-{prior}_{cellregenabled_string}heldout-{{patient}}"

            aurocs, spearmans = aggregate_cross_val_data(out + "/" + model_identifier_format)

            key = 'cell_reg:' + enable_cell_reg + ",prior:" + prior
            aurocs_dict[key] = aurocs
            spearmans_dict[key] = spearmans

            print(key)
            print("spearman:", bootstrap(spearmans)['formatted'])
            print("auroc:", bootstrap(aurocs)['formatted'])

    create_violin_plots(aurocs_dict, spearmans_dict)


if __name__ == '__main__':
    genes = open("training_results_v2/v2_repr_contrastive/genes.txt").read().split("\n")

    # aurocs, spearmans, mses, maes = aggregate_cross_val_data(
    #     "./training_results_v2/v7_cell_models_augmented_2_epochs/cellmodel_prior-none_cellreg-enabled_heldout-{patient}"
    # )

    # print("*** CELL MODELS ***")
    # print("spearman:", bootstrap(spearmans)['formatted'])
    # print("auroc:", bootstrap(aurocs)['formatted'])
    # print("mse:", bootstrap(mses)['formatted'])
    # print("mae:", bootstrap(maes)['formatted'])

    # aurocs, spearmans, mses, maes = aggregate_cross_val_data(
    #     "./training_results_v2/v6_cnn_models_augmented_2_epochs/inception-holding-out-{patient}"
    # )

    # print("*** INCEPTION MODELS ***")
    # print("spearman:", bootstrap(spearmans)['formatted'])
    # print("auroc:", bootstrap(aurocs)['formatted'])
    # print("mse:", bootstrap(mses)['formatted'])
    # print("mae:", bootstrap(maes)['formatted'])

    genes = open("training_results_v2/v2_repr_contrastive/genes.txt").read().split("\n")
    cv_results = {}
    root = 'training_results_v2/v10_graphsage'
    # for prior in ['none', 'repr']:
    #     for cre in [False, True]:
    for prior in ['none']: # , 'repr']:
        for cre in [False]: # , True]:
            cre_str = 'cellreg-enabled_' if cre else ''
            aurocs, spearmans, mses, maes = aggregate_cross_val_data(
                # f"./{root}/cellmodel_prior-{prior}_{cre_str}heldout-{{patient}}"
                f"./{root}/cellmodel_{cre_str}heldout-{{patient}}"
            )
            cre_str2 = 'CellRegularized' if cre else 'NonCellRegularized'
            prior_str = 'Unpretrained' if prior == 'none' else 'ContrastivePretrained'
            out_file = f'./{root}/crossval_{cre_str2}_{prior}.csv'
            df = pd.DataFrame({
                "AUROC": list(aurocs),
                "Spearman": list(spearmans),
                "MSE": list(mses),
                "MAE": list(maes),
            }, index=genes)
            df.index.name = 'Gene'
            df.to_csv(out_file)
            print(out_file)
            print("spearman:", bootstrap(spearmans)['formatted'])
            print("auroc:", bootstrap(aurocs)['formatted'])
            print("mse:", bootstrap(mses)['formatted'])
            print("rmse:", bootstrap(mses ** 0.5)['formatted'])
            print("mae:", bootstrap(maes)['formatted'])

    with open(f"./{root}/cv_results.json", "w") as f:
        json.dump({"results": cv_results, "genes": genes}, f)
    # aurocs, spearmans, mses = aggregate_cross_val_data(
    #     "./training_results_v2/v2_repr_contrastive/cellmodel_prior-none_cellreg-enabled_heldout-{patient}"
    # )
