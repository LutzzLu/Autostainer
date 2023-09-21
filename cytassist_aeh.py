import json
import pickle
import sys
import pandas as pd

# For SpatialDE, make sure to run:
# pip install spatialde
# pip install patsy
# On the Conda environment that will be running your programs.

from cytassist_segmentation import create_cytassist_patient_segmentation
from omicsio.spatial_analysis import run_aeh
from omicsio.datasets import Slide

# cytassist_slide_ids = ['092534', '091759', '092146', '092842']

# Done like this so we can run batches with Slurm
slide_id = sys.argv[1]

with open(f"./input_data/all_visium_data/preprocessed/cytassist_{slide_id}_40x.pkl", "rb") as f:
    slide: Slide = pickle.load(f)

with open(f"./input_data/all_visium_data/annotations/json_files/_SS12251_{slide_id}.json", "rb") as f:
    annotation = json.load(f)

patient_torchdatasets = create_cytassist_patient_segmentation(
    slide,
    annotation,
    magnify=1,
    patch_transform=None,
    device='cpu',
)

for patient_key in patient_torchdatasets.keys():
    subset = patient_torchdatasets[patient_key].indices

    spatialde_result = pd.read_csv(f'./spatialde-cache/spatialde_cytassist_{slide_id}_{patient_key}.csv')
    num_groups = 5
    
    aeh_histology_results, aeh_patterns = run_aeh(
        slide.spot_counts.cpu().numpy(),
        slide.genes,
        slide.spot_locations.image_x.numpy(),
        slide.spot_locations.image_y.numpy(),
        num_groups=num_groups,
        spatialde_results=spatialde_result,
        length=0.1,
        significance_filter='qval < 0.05'
    )

    print(type(aeh_histology_results), type(aeh_patterns))

    with open(f'./spatialde-cache/aeh_cytassist_{slide_id}_{patient_key}.pkl', 'wb') as f:
        pickle.dump((aeh_histology_results, aeh_patterns), f)
