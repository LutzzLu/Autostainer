### unify cells in the slide through non maximum suppression ###

import time

import torch
from omicsio.datasets import Slide

SLIDES = [
    '091759',
    '092146',
    '092534',
    '092842',
    'autostainer',
]

for slide_id in SLIDES:
    print("Running for slide", slide_id)

    start_time = time.time()

    if slide_id == 'autostainer':
        slide = Slide.load("input_data/all_visium_data/preprocessed/autostainer_40x.pkl")
        cell_detections_per_spot = torch.load(f'./cell-detections/autostainer_orig/combined_nms.pt')
    else:
        slide = Slide.load(f"input_data/all_visium_data/preprocessed/cytassist_{slide_id}_40x.pkl")
        cell_detections_per_spot = torch.load(f'./cell-detections/cytassist_{slide_id}/inferences.pt')

    """
    An array with each index containing:
    A dictionary with key "boxes"
    Of format (x1, y1, x2, y2)[]
    """

    all_boxes = []
    all_scores = []

    for image_x, image_y, detection in zip(slide.spot_locations.image_x, slide.spot_locations.image_y, cell_detections_per_spot):
        if len(detection['boxes']) == 0:
            continue # No cells were detected. Skip this spot.
        if slide_id == 'autostainer':
            boxes = torch.stack(detection['boxes'])
            scores = torch.tensor(detection['scores'])
        else:
            boxes = detection['boxes']
            scoers = detection['scores']
        boxes[:, [0, 2]] += image_x
        boxes[:, [1, 3]] += image_y
        all_boxes.append(boxes)
        all_scores.append(scores)

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)

    import celldetection_nms

    remaining = celldetection_nms.remaining_indexes_after_non_max_suppression(
        {"boxes": boxes, "scores": scores}, overlap_thres=0.3333, custom_square_size=64
    )
    remaining = sorted(remaining)

    resulting_cells = {
        "boxes": boxes[remaining],
        "scores": scores[remaining],
    }

    print("Cells have been filtered. Elapsed time:", time.time() - start_time)

    print("Proportion of cells that remain:", len(remaining) / len(boxes))

    torch.save(resulting_cells, f"cell-detections/slide_{slide_id}_stitched.pt")
