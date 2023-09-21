
import json
import pickle

import PIL.Image
import torch
from omicsio.datasets import Slide

from cytassist_segmentation import create_cytassist_patient_segmentation
from gene_sets import GENE_SETS
from patch_dataset_factory import create_patch_dataset

PIL.Image.MAX_IMAGE_PIXELS = 1e10

ALL_PATIENTS = [
    'autostainer',
    '092534_24',
    '092534_35',
    '091759_4',
    '091759_7', # A3
    '092146_33',
    '092146_3', # A19
    '092842_17',
    '092842_16',
]

def load_torch_datasets(gene_set_id: str, include_cells=True):
    if gene_set_id != '__load_as_slide_objects__':
        genes = GENE_SETS[gene_set_id]
    else:
        genes = '__load_as_slide_objects__'

    slides = {
        'autostainer': get_autostainer_dataset(genes, include_cells),
        **get_cytassist_slides_as_torch_datasets(genes, include_cells),
    }

    assert sorted(slides.keys()) == sorted(ALL_PATIENTS), "Mismatch between slides and patients. Slides: {}, Patients: {}".format(sorted(slides.keys()), sorted(ALL_PATIENTS))

    return slides


"""
This corresponds to the single autostainer slide.
"""
def get_autostainer_dataset(genes, include_cells=True):
    slide = Slide.load("input_data/preprocessed/autostainer_40x_cropped.pkl")
    if genes == '__load_as_slide_objects__':
        return slide.log1p()
    
    slide = slide.select_genes(genes).log1p() # .binary()

    if include_cells:
        cells = torch.load("cell-detections/autostainer_orig/combined_nms.pt")
    else:
        cells = None

    return create_patch_dataset(slide, magnification='40x', image_size_in_40x=512, cell_detections=cells)

"""
These are the IDs of the CytAssist slides we have access to:
cytassist_slide_ids = ['092534', '091759', '092146', '092842']

These are the IDs of the patients that correspond to each slide:
cytassist_patient_keys = [[24, 35], [4, 7], [33, 3], [17, 16]]

There are multiple patients per slide, because we packed together
patients in such a way that we could get samples from different
tissue for the price of one slide.
"""
def get_cytassist_slides_as_torch_datasets(genes, include_cells=True):
    cytassist_patients = _get_cytassist_patients_as_torch_datasets(genes, include_cells)
    
    return {
        '092534_24': cytassist_patients['24'],
        '092534_35': cytassist_patients['35'],
        '091759_4': cytassist_patients['4'],
        '091759_7': cytassist_patients['7'],
        '092146_33': cytassist_patients['33'],
        '092146_3': cytassist_patients['3'],
        '092842_17': cytassist_patients['17'],
        '092842_16': cytassist_patients['16'],
    }

"""
Helper method for loading a specific patient.
"""
def load_patient(patient, genes, include_cells=True):
    assert patient in ALL_PATIENTS, f"Patient {patient} is not found in ALL_PATIENTS ({ALL_PATIENTS})"

    if patient == 'autostainer':
        return get_autostainer_dataset(genes, include_cells)
    else:
        slide_id, patient_id = patient.split("_")
        patient_dict = _load_cytassist_patient_dict(slide_id, genes, include_cells)
        return patient_dict[patient_id]

def load_whole_cytassist(slide_id, genes, include_cells=True):
    with open(f"./input_data/all_visium_data/preprocessed/cytassist_{slide_id}_40x.pkl", "rb") as f:
        slide = pickle.load(f).select_genes(genes).log1p()

    if include_cells:
        cell_detections = torch.load(f'./cell-detections/cytassist_{slide_id}/inferences.pt')
    else:
        cell_detections = None

    return create_patch_dataset(slide, magnification='40x', image_size_in_40x=512, cell_detections=cell_detections)

def _load_cytassist_patient_dict(slide_id, genes, include_cells=True):
    with open(f"./input_data/all_visium_data/preprocessed/cytassist_{slide_id}_40x.pkl", "rb") as f:
        slide = pickle.load(f)

    with open(f"./input_data/all_visium_data/annotations/json_files/_SS12251_{slide_id}.json", "rb") as f:
        annotation = json.load(f)
    
    if include_cells:
        cell_detections = torch.load(f'./cell-detections/cytassist_{slide_id}/inferences.pt')
    else:
        cell_detections = None

    if genes != '__load_as_slide_objects__':
        slide = slide.select_genes(genes).log1p()
    else:
        slide = slide.log1p()

    patient_dict = create_cytassist_patient_segmentation(
        slide,
        annotation,
        cell_detections,
        # yea this is probably not a good practice but whatever
        return_type='slides' if genes == '__load_as_slide_objects__' else 'torch-subsets'
    )

    return patient_dict

def _get_cytassist_patients_as_torch_datasets(genes, include_cells):
    cytassist_patients = {}

    cytassist_slide_ids = ['092534', '091759', '092146', '092842']

    for slide_id in cytassist_slide_ids:
        cytassist_patients.update(_load_cytassist_patient_dict(slide_id, genes, include_cells))

    return cytassist_patients

