import json
import pickle
from typing import Dict, List, Union

import numpy as np
import omicsio.datasets
import shapely
import torch.utils.data
import tqdm

from patch_dataset_factory import create_patch_dataset


def simple_svg_to_shapely_polygon(svg: str):
    """ Only works if there is only one <polygon/> in the svg"""
    quote_start = svg.find("\"") + 1
    quote_end = svg.rfind("\"")
    points_string = svg[quote_start:quote_end]
    points = []
    for xy_string in points_string.split(" "):
        x_string, y_string = xy_string.split(",")
        x = float(x_string)
        y = float(y_string)
        points.append((x, y))
    return shapely.Polygon(points)

def get_cytassist_subsets_dict_and_cut_cell_detections(
    slide: omicsio.datasets.Slide,
    annotations,
    cell_detections: Union[List, None] = None,
    image_size: int = 512,
):
    if cell_detections is None:
        cell_detections = [None for _ in range(len(slide.spot_locations.image_x))]

    # Each spot only belongs to one patient. Therefore, we can save some time.
    polygons: Dict[str, shapely.Polygon] = {}

    for annotation in annotations:
        if not annotation['body'][0]['value'].isdigit():
            continue

        patient_id = annotation['body'][0]['value']

        # This annotation corresponds to a general patient
        svg = annotation['target']['selector']['value']
        poly = simple_svg_to_shapely_polygon(svg)
        polygons[patient_id] = poly

    cell_detections_cut = []

    patient_subsets: Dict[str, List[int]] = {
        patient_id: []
        for patient_id in polygons.keys()
    }

#     print(cell_detections, slide.spot_locations.image_x, slide.spot_locations.image_y)

    # Determine whether cells should be filtered out of some samples.
    # If the centroid is at least image_size * sqrt 2 away from the exterior of the polygon,
    # all of the cells will be preserved.

    CUT_DISTANCE = np.sqrt(2) * image_size

    for spot_index, (cell_detection, spot_x, spot_y) in enumerate(zip(
        cell_detections,
        slide.spot_locations.image_x,
        slide.spot_locations.image_y,
    )):
        origin = shapely.Point(spot_x, spot_y)

        found_polygon = None

        # Find the polygon this spot is located in
        for patient_id, polygon in polygons.items():
            if polygon.contains(origin):
                found_polygon = polygon
                patient_subsets[patient_id].append(spot_index)
                break

        # If spot is not inside any polygon, find the closest one
        if not found_polygon:
            #             print("WARN: Spot was not in any polygon")

            closest_polygon = None
            closest_polygon_distance = None
            for patient_id, polygon in polygons.items():
                if closest_polygon_distance is None or polygon.exterior.distance(origin) < closest_polygon_distance:
                    closest_polygon = polygon

            found_polygon = closest_polygon
            patient_subsets[patient_id].append(spot_index)

        # Everything below is for ensuring cell graphs don't get connected across the cut.
        if cell_detection is None:
            continue

        # All of the cells will be preserved.
        if found_polygon.exterior.distance(origin) > CUT_DISTANCE:
            cell_detections_cut.append(cell_detection)
            continue

        # Cut out boxes which are outside of the polygon
        valid_cell_indexes = []
        for cell_index, (x1, y1, x2, y2) in enumerate(cell_detection['boxes']):
            # A box at (image_size/2, image_size/2) is equivalent to (spot_x, spot_y)
            cell_centroid = shapely.Point(
                (x1 + x2) / 2 + spot_x - image_size // 2,
                (y1 + y2) / 2 + spot_y - image_size // 2,
            )
            if found_polygon.contains(cell_centroid):
                valid_cell_indexes.append(cell_index)

        # Filter detections based on valid index
        cell_detections_cut.append({
            k: v[valid_cell_indexes]
            for k, v in cell_detection.items()
        })

    cell_detections = cell_detections_cut if len(cell_detections_cut) > 0 else None

    return (patient_subsets, cell_detections)

def slide_subset(slide, subset):
    return omicsio.datasets.Slide(
        image_path=slide.image_path,
        spot_locations=omicsio.datasets.SpotLocations(
            image_x=slide.spot_locations.image_x[subset],
            image_y=slide.spot_locations.image_y[subset],
            row=slide.spot_locations.row[subset],
            col=slide.spot_locations.col[subset],
            dia=None,
        ),
        spot_counts=slide.spot_counts[subset, :],
        genes=slide.genes,
    )

def create_cytassist_patient_segmentation(
    slide,
    annotations,
    cell_detections=None,
    image_size=512,
    return_type='torch-subsets',
    **dataset_kwargs,
):
    patient_subsets, cell_detections = get_cytassist_subsets_dict_and_cut_cell_detections(slide, annotations, cell_detections, image_size)

    if return_type == 'torch-subsets':
        return {
            patient_id: create_patch_dataset(slide_subset(slide, subset), magnification='40x', image_size_in_40x=image_size, cell_detections=[cell_detections[i] for i in subset])
            for patient_id, subset in patient_subsets.items()
        }
    elif return_type == 'slides':
        return {
            patient_id: slide_subset(slide, subset)
            for patient_id, subset in patient_subsets.items()
        }

def reconstruct_validation_data(slide, slide_id, validation_dict, is_tensor_only=False):
    with open(f"./input_data/all_visium_data/annotations/json_files/_SS12251_{slide_id}.json", "rb") as f:
        annotation = json.load(f)

    patient_subsets, cell_detections = get_cytassist_subsets_dict_and_cut_cell_detections(slide, annotation, None, 512)

    if is_tensor_only:
        predictions = []
        prediction_indices = []

        for patient_id, validation_results in validation_dict.items():
            for pred_i in range(validation_results['predictions'].shape[0]):
                predictions.append(validation_results['predictions'][pred_i])
                index_in_subset = pred_i
                index_in_true = patient_subsets[patient_id][index_in_subset]
                prediction_indices.append(index_in_true)
        predictions = torch.stack(predictions)
        return {"predictions": predictions, "prediction_indices": prediction_indices}
    else:
        predictions = {
            'graph_vector_batch': [],
            'cell_vectors_batch': [],
        }
        prediction_indices = []

        for patient_id, validation_results in validation_dict.items():
            for pred_i in range(len(validation_results['predictions']['graph_vector_batch'])):
                graph_vector = validation_results['predictions']['graph_vector_batch'][pred_i]
                cell_vectors = validation_results['predictions']['cell_vectors_batch'][pred_i]
                predictions['graph_vector_batch'].append(graph_vector)
                predictions['cell_vectors_batch'].append(cell_vectors)
                index_in_subset = validation_results['prediction_indices'][pred_i]
                index_in_true = patient_subsets[patient_id][index_in_subset]
                prediction_indices.append(index_in_true)

        return {"predictions": predictions, "prediction_indices": prediction_indices}
    
