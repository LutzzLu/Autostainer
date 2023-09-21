import matplotlib.pyplot as plt
import numpy as np
from umap import AlignedUMAP
import seaborn as sns


def compute_colors(vectors):
    # Compute the angles relative to the mean vector
    mean_vector = np.mean(vectors, axis=0)
    xoff, yoff = (vectors - mean_vector).T
    angles = np.arctan2(yoff, xoff) + np.pi

    # Map the angles to hue values between 0 and 1
    hues = angles / (2 * np.pi)

    return hues

def create_umap_embeddings(ground_truth, predicted):
    # Compute UMAP embeddings using AlignedUMAP
    relation = {i: i for i in range(len(ground_truth))}
    slices = [ground_truth, predicted]
    aligned_umap = AlignedUMAP()
    aligned_umap.fit(slices, relations=[relation])

    ground_truth_embeddings, predicted_embeddings = aligned_umap.embeddings_

    return ground_truth_embeddings, predicted_embeddings

def plot_aligned_umap(ground_truth_embeddings, predicted_embeddings, cluster_colors=None):
    if cluster_colors is None:
        # Compute colors based on the angles relative to the mean vector
        color_dict = {'c': compute_colors(ground_truth_embeddings), 'cmap': 'hsv'}
    else:
        color_dict = {'c': cluster_colors}

    sns.set_style('whitegrid')

    minx = min(ground_truth_embeddings[:, 0].min(), predicted_embeddings[:, 0].min())
    maxx = max(ground_truth_embeddings[:, 0].max(), predicted_embeddings[:, 0].max())
    miny = min(ground_truth_embeddings[:, 1].min(), predicted_embeddings[:, 1].min())
    maxy = max(ground_truth_embeddings[:, 1].max(), predicted_embeddings[:, 1].max())

    # Plot the ground truth vectors
    plt.subplot(1, 2, 1)
    plt.scatter(ground_truth_embeddings[:, 0], ground_truth_embeddings[:, 1], **color_dict, s=2)
    plt.title('Ground Truth')
    # plt.xlim(minx - 0.1, maxx + 0.1)
    # plt.ylim(miny - 0.1, maxy + 0.1)
    ax = plt.gca()
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    # ax.set_xticks([])
    # for minor ticks
    # ax.set_xticks([], minor=True)
    plt.axis('off')
    # sns.despine()

    # Plot the predicted vectors
    plt.subplot(1, 2, 2)
    plt.scatter(predicted_embeddings[:, 0], predicted_embeddings[:, 1], **color_dict, s=2)
    plt.title('Predicted')
    # plt.xlim(minx - 0.1, maxx + 0.1)
    # plt.ylim(miny - 0.1, maxy + 0.1)
    ax = plt.gca()
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    # ax.set_xticks([])
    # for minor ticks
    # ax.set_xticks([], minor=True)
    plt.axis('off')
    # sns.despine()

    # Adjust the layout and show the plot
    plt.tight_layout()

def main():
    import hdbscan
    import torch

    heldout_patient = 'autostainer'
    with open('training_results_v2/v0_repr/genes.txt') as f:
        genes = f.read().split("\n")
    vname = 'training_results_v2/v0_repr/cellmodel_prior-repr_heldout-autostainer_validation_results.pt'
    validation = torch.load(vname, map_location='cpu')

    if 'ground_truth_log_counts_all' not in validation:
        from load_datasets import load_patient

        valid_dataset = load_patient(heldout_patient, genes)
        log_counts = torch.stack([valid_dataset[i][1] for i in range(len(valid_dataset))]).cpu().numpy()
        validation['ground_truth_log_counts'] = log_counts
        # torch.save(validation, vname)

    ground_truth_vectors = log_counts[validation['prediction_indices']]
    predicted_vectors = torch.stack(validation['predictions']['graph_vector_batch'])

    # Create HDBScan clusters
    # See: https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    clusterer.fit(ground_truth_vectors)
    color_palette = sns.color_palette()
    # first color is ugly :(
    cluster_colors = [sns.desaturate(color_palette[col % len(color_palette)], sat)
                    if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                    zip(clusterer.labels_, clusterer.probabilities_)]
    # plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)

    try:
        print("number of clusters:", np.max(clusterer.labels_))
        print("number of colors:", len(color_palette))
    except Exception as e:
        print(e)

    # ground_truth_vectors = np.random.rand(50, 2)  # Replace with your ground truth vectors
    # predicted_vectors = np.random.rand(50, 2)  # Replace with your predicted vectors

    plot_aligned_umap(ground_truth_vectors, predicted_vectors, cluster_colors=cluster_colors)
    plt.savefig("figures/aligned-umap_clustered.png")

def test_plot_aligned_umap():
    items1 = np.random.randn(80, 2)
    items2 = np.random.randn(80, 2)

    plot_aligned_umap(items1, items2)
    plt.savefig("test_plot.png")

# test_plot_aligned_umap()
