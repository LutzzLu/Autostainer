import sys
import time
from collections import defaultdict

import torch
import tqdm


def box_area(box):
    x1, y1, x2, y2 = box
    return abs((x2 - x1) * (y2 - y1))

def box_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    w = xB - xA
    h = yB - yA

    if w < 0 or h < 0:
        return 0

    # compute the area of intersection rectangle
    interArea = w * h
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(box_area(boxA) + box_area(boxB) - interArea)

    # return the intersection over union value
    return iou

# non maximum suppression - look for boxes with an IOU > 0.5, and keep the box with the higher score
def remaining_indexes_after_non_max_suppression(inferences, overlap_thres, custom_square_size=None):
    boxes = inferences['boxes']
    scores = inferences['scores']
    if len(boxes) == 0:
        return set()
    min_x = min([box[0] for box in boxes])
    max_x = max([box[2] for box in boxes])
    min_y = min([box[1] for box in boxes])
    max_y = max([box[3] for box in boxes])
    if custom_square_size is None:
        # Create a 16x16 grid of cells
        # Automatically detects the size of the image based on the range of X values in inferences
        # Add 1 to the width and height to ensure that the last cell is included
        grid_square_width = (max_x - min_x + 1) / 16
        grid_square_height = (max_y - min_y + 1) / 16
    else:
        grid_square_width = custom_square_size
        grid_square_height = custom_square_size
    # Use a defaultdict to save memory
    grid = defaultdict(set)
    create_grid = time.time()
    # For each box, we add the cell to any grid squares it overlaps with.
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        min_grid_x = int((min(x1, x2) - min_x) / grid_square_width)
        max_grid_x = int((max(x1, x2) - min_x) / grid_square_width)
        min_grid_y = int((min(y1, y2) - min_y) / grid_square_height)
        max_grid_y = int((max(y1, y2) - min_y) / grid_square_height)
        for grid_x in range(min_grid_x, max_grid_x + 1):
            for grid_y in range(min_grid_y, max_grid_y + 1):
                grid[grid_x, grid_y].add(i)
    removed_indexes = set()
    # For each grid square, see if there are any boxes that overlap with each other significantly
    for (grid_x, grid_y) in tqdm.tqdm(grid.keys(), 'iterating over grid squares'):
        for i in grid[grid_x, grid_y]:
            if i in removed_indexes:
                continue
            for j in grid[grid_x, grid_y]:
                if i == j or j in removed_indexes:
                    continue
                iou = box_iou(boxes[i], boxes[j])
                if iou >= overlap_thres:
                    if scores[j] > scores[i]:
                        removed_indexes.add(i)
                        break
                    else:
                        removed_indexes.add(j)
    return set(range(len(boxes))) - removed_indexes

def apply_nms_to_inferences(inferences):
    result = []
    for inference in tqdm.tqdm(inferences, total=len(inferences), desc="running non-maximum suppression"):
        idxs = remaining_indexes_after_non_max_suppression(inference, 0.5)
        idxs = sorted(list(idxs))
        result.append({
            key: inference[key][idxs]
            for key in inference.keys()
        })
    return result

def main():
    for slide_id in ['092534', '091759', '092146', '092842']:
    # for slide_id in ['091759', '092146', '092534', '092842']:
        dir = "cell-detections/cytassist_" + slide_id

        raw_inferences = torch.load(dir + "/inferences_without_nms.pt")

        try:
            inferences = apply_nms_to_inferences(raw_inferences)
            torch.save(inferences, dir + "/inferences.pt")
        except Exception as e:
            print("Error applying NMS to inferences:", e, file=sys.stderr)

if __name__ == '__main__':
    main()
