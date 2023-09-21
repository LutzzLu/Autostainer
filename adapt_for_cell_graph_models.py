"""
Graph datasets are unique because we need at least two cells within the frame to be able
to create a prediction. We aren't always sure which spots have enough cells. We could cache
them I guess but this would be more trouble than it's worth. Instead, I'm going to write a custom
DataLoader that populated batches while ignoring entries in the Dataset that are `None`.
"""
import functools

import numpy as np
import torch

import patch_model
from dataset_wrapper import DatasetWrapper


def _cell_boxes_to_cell_locations(boxes):
    # boxes: (x1, y1, x2, y2)[]
    return (torch.stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))).T

def _crop_box(image, box_size, cx, cy):
    half_box_size = box_size // 2
    return image[:, int(cy) - half_box_size:int(cy) + half_box_size, int(cx) - half_box_size:int(cx) + half_box_size]

def get_valid_indexes(boxes, image_size, cell_image_crop_size, magnify=2):
    cell_locations = _cell_boxes_to_cell_locations(boxes) * magnify
    half_box_size = cell_image_crop_size // 2
    cx = cell_locations[:, 0]
    cy = cell_locations[:, 1]

    # torch.where returns a tuple, with one element for each dimension.
    # Here, there is only one dimension
    return torch.where(
        (half_box_size <= cx) & (cx <= image_size - half_box_size)
      & (half_box_size <= cy) & (cy <= image_size - half_box_size)
    )[0]

def _image_and_boxes_to_cropped_images_locations_and_valid_indexes(
    image,
    boxes,
    cell_image_crop_size=64,
    image_size=512,
):
    valid_indexes = get_valid_indexes(boxes, image_size, cell_image_crop_size, magnify=2)

    cell_locations = _cell_boxes_to_cell_locations(boxes) * 2
    cell_locations = cell_locations[valid_indexes]

    if len(valid_indexes) == 0:
        return ([], [], [])
    
    cell_images = torch.stack([
        _crop_box(image, cell_image_crop_size, cx, cy)
        for cx, cy in cell_locations
    ])
    
    return cell_images, cell_locations, valid_indexes

def _transform_regular_input_to_cell_graph_input(ds, idx, mode, min_cells):
    """
    Input format:
    (image, label, cell detections)

    Output format (that can be directly passed to a CellGraphModel):
    (cell images [N x 64 x 64], cell_locations [N x 2], valid_indexes [N]), spot label [G]
    """
    instance = ds[idx]

    image, label, cell_detections = instance

    return _transform_image_and_detections_to_cell_crops(image, label, cell_detections, mode=mode, min_cells=min_cells)

def _transform_image_and_detections_to_cell_crops(image, label, cell_detections, mode, min_cells):
    # `cell_detections['boxes']` is initially `list[Tensor]`.
    boxes = cell_detections['boxes']

    if len(boxes) < min_cells:
        return None

    if isinstance(boxes, list):
        boxes = torch.stack(cell_detections['boxes'])

    # Ensure that everything is on the same device.
    # Source of truth for the device is `image`.
    boxes = boxes.to(image.device)
    label = label.to(image.device)
    
    cell_data = _image_and_boxes_to_cropped_images_locations_and_valid_indexes(image, boxes)

    # SafeDataLoader in optimization.py automatically ignores None's.
    if len(cell_data[0]) < min_cells:
        return None
    
    if mode == 'train':
        cell_data = (
            patch_model.training_transforms(cell_data[0]),
            *cell_data[1:]
        )
    elif mode == 'validation':
        cell_data = (
            patch_model.validation_transforms(cell_data[0]),
            *cell_data[1:]
        )

    return cell_data, label

def adapt_for_cell_graph_models(dataset, mode='train', min_cells=2):
    return DatasetWrapper(dataset, functools.partial(_transform_regular_input_to_cell_graph_input, mode=mode, min_cells=min_cells))

def extend_with_single_cell_data(dataset, scData, scMatrix):
    def _add_cell_data(ds, spot_id):
        # Get the data for this spot
        cell_data, label = ds[spot_id]
        # Get the most likely cells for this spot
        num_cells = len(cell_data[0])
        sc_indexes = np.argpartition(scMatrix[:, spot_id], num_cells)[:num_cells]
        # Get the data for these cells
        sc_data = scData[sc_indexes]
        return cell_data, label, sc_data

    return DatasetWrapper(dataset, _add_cell_data)
