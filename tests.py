import adapt_for_cell_graph_models
import gene_sets
import load_datasets

dataset = load_datasets.get_autostainer_dataset(gene_sets.GENE_SETS['cytassist-autostainer-top1k-by-rank-mean'])

print(dataset[0])

dataset_adapted = adapt_for_cell_graph_models.adapt_for_cell_graph_models(dataset)

print(dataset_adapted[0][0])

print(len(dataset_adapted[0][1]))
