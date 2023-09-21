import json

def load_gene_list_from_pathway(pathway):
    with open(f'pathways/{pathway}.json', 'r') as f:
        content = json.load(f)
    return content['geneSymbols']

def create_gene_list_from_pathways():
    pathways = [
        "HALLMARK_ANGIOGENESIS",
        "HALLMARK_DNA_REPAIR",
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_TGF_BETA_SIGNALING",
        "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
    ]

    results = set()

    for pathway in pathways:
        results.update(load_gene_list_from_pathway(pathway))
        
    return list(sorted(results))

GENE_SETS = {
    'msigdb-pathways7': create_gene_list_from_pathways(),
    # Can be any gene. This is the "gene set" used for the representation model, which ignores the spot annotations.
    'DUMMY': ['EPCAM']
}

with open("./spatialde-cache/cytassist_autostainer_top1k_genes_set.txt", "r") as f:
    GENE_SETS['cytassist-autostainer-top1k-by-rank-mean'] = sorted(f.read().split("\n"))

with open("./spatialde-cache/cytassist_autostainer_top1k_genes_set_thresholded.txt", "r") as f:
    GENE_SETS['cytassist-autostainer-top1k-by-rank-mean-thresholded'] = sorted(f.read().split("\n"))
