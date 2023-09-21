import torch_geometric
import torch_geometric.data
import torch_geometric.nn as gnn
import torch_geometric.nn.aggr as gpool
import torch_geometric.transforms as GT  # GT = Graph Transforms
import GCL.augmentors as A

# Changing positions of nodes before graph construction (uses pytorch-geometric)
pre_construction_augmentations = GT.Compose([
    GT.RandomJitter(5),
])

# Ground-truth connections (Uses pytorch-geometric)
graph_construction = GT.Compose([
    # force_undirected is used to make eigenvector/eigenvalue calculation
    # better, as part of laplacian positional encoding.
    GT.KNNGraph(k=6, force_undirected=True),
])

# Fuzzing up the constructed graph (Uses PyGCL)
post_construction_augmentations = A.Compose([
    A.RandomChoice([
        # A.NodeDropping(pn=0.1),
        A.EdgeRemoving(pe=0.1),
    ], num_choices=1),
    A.FeatureMasking(pf=0.3),
])

add_posenc = GT.Compose([
    GT.AddLaplacianEigenvectorPE(k=6, attr_name='laplacian_eigenvector_pe'),
])

# Creates a graph in the Pytorch-Geometric format.
# https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
def create_graph(cell_embeddings, cell_locations, augment=False, use_pos_enc=False):
    """
    data.x: Node feature matrix with shape [num_nodes, num_node_features]
    data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]
    data.pos: Node position matrix with shape [num_nodes, num_dimensions]
    """
    
    graph = torch_geometric.data.Data(
        x=cell_embeddings,
        pos=cell_locations,
    )
    
    if not augment:
        graph = graph_construction(graph)
    else:
        graph = pre_construction_augmentations(graph)
        graph = graph_construction(graph)
        x, ei, _ = post_construction_augmentations(graph.x, graph.edge_index)
        graph = torch_geometric.data.Data(x=x, edge_index=ei)
    if use_pos_enc:
        graph = add_posenc(graph)
    return graph
