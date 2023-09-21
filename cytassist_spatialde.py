import json
import pickle
import sys

# For SpatialDE, make sure to run:
# pip install spatialde
# pip install patsy
# On the Conda environment that will be running your programs.

from cytassist_segmentation import create_cytassist_patient_segmentation
from omicsio.spatial_analysis import run_spatialde_and_aeh_on_slide, run_spatialde
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

    spatialde_result = run_spatialde(
        slide.spot_counts.cpu().numpy()[subset],
        slide.genes,
        slide.spot_locations.image_x.numpy()[subset],
        slide.spot_locations.image_y.numpy()[subset],
    )
    spatialde_result.to_csv(f"spatialde-cache/spatialde_cytassist_{slide_id}_{patient_key}.csv")
