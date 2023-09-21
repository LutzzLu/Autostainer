import gc
import os
import sys
from collections import defaultdict

import numpy as np
import PIL.Image
import torch
import tqdm
from omicsio.datasets import PatchDataset, Slide

import maskrcnn.model

# Ensure that PIL doesn't think some malicious entity is trying to hijack our machine learning project
PIL.Image.MAX_IMAGE_PIXELS = 1e10

def tensor_image_to_numpy(image):
    # Converts (C, H, W) f32 to (H, W, C) u8
    return np.array(image.cpu().permute(1, 2, 0) * 255, dtype=np.uint8)

def standard_scaler(patch_or_patches: torch.Tensor):
    # patch.shape: ([B,] 3, H, W)
    # ([B,] 3, 1, 1)
    mean = patch_or_patches.mean(dim=(-2, -1), keepdim=True)
    std = patch_or_patches.std(dim=(-2, -1), keepdim=True)
    # Ensure that we only have three channels
    return ((patch_or_patches - mean) / std)[:3, :, :]

# delete masks, too much memory
def inference_to_cpu(inference):
    cpu = {}
    for key in inference.keys():
        if type(inference[key]) == torch.Tensor:
            cpu[key] = inference[key].cpu()
        else:
            cpu[key] = inference[key]
    if 'masks' in cpu:
        del cpu['masks']
    return cpu

# This is a generator that returns the results in chunks. This is so we don't lose all progress if something goes wrong.
def infer_for_dataset(model, dataset):
    inferences = []
    
    model.eval()
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        for batch in tqdm.tqdm(dataloader, desc="Inferring for slide"):
            images, labels = batch

            # Convert inference to CPU to save CUDA memory
            inferences.extend([inference_to_cpu(inference) for inference in model(list(images))])

            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()

    return inferences

def box_area(box):
    x1, y1, x2, y2 = box
    return abs((x2 - x1) * (y2 - y1))

def box_intersection(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(box_area(boxA) + box_area(boxB) - interArea)

    # return the intersection over union value
    return iou

def box_iou(box1, box2):
    intersection = box_intersection(box1, box2)
    union = box_area(box1) + box_area(box2) - intersection
    return intersection / union

# non maximum suppression - look for boxes with an IOU > 0.5, and keep the box with the higher score
def remaining_indexes_after_non_max_suppression(inferences, overlap_thres):
    boxes = inferences['boxes']
    scores = inferences['scores']
    # Create a 16x16 grid of cells
    # Automatically detects the size of the image based on the range of X values in inferences
    min_x = min([box[0] for box in boxes])
    max_x = max([box[2] for box in boxes])
    min_y = min([box[1] for box in boxes])
    max_y = max([box[3] for box in boxes])
    # Add 1 to the width and height to ensure that the last cell is included
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    # Use a defaultdict to save memory
    grid = defaultdict(set)
    # For each box, we add the cell to any grid squares it overlaps with.
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        min_grid_x = int((x1 - min_x) / width * 16)
        max_grid_x = int((x2 - min_x) / width * 16)
        min_grid_y = int((y1 - min_y) / height * 16)
        max_grid_y = int((y2 - min_y) / height * 16)
        for grid_x in range(min_grid_x, max_grid_x + 1):
            for grid_y in range(min_grid_y, max_grid_y + 1):
                grid[grid_x, grid_y].add(i)
    removed_indexes = set()
    # For each grid square, see if there are any boxes that overlap with each other significantly
    for (grid_x, grid_y) in grid.keys():
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
    return [{
        key: [v for i, v in enumerate(inference[key]) if i in remaining_indexes_after_non_max_suppression(inference, 0.0005)]
        for key in inference.keys()
    } for inference in tqdm.tqdm(inferences, total=len(inferences), desc="running non-maximum suppression")]

def display_cell_count_information(inferences):
    cell_counts = [len(inference['boxes']) for inference in inferences]
    print("Total cell detections:", sum(cell_counts))
    print("Mean cells per spot:", sum(cell_counts) / len(inferences))

def main():
    model = maskrcnn.model.get_model(
        num_classes=6,
        path="/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/spatial_omics/code/MaskRCNN/models/maskrcnn_resnet50_1_10.pt"
    )

    for slide_id in ['091759', '092146', '092534', '092842']:
        slide = Slide.load("input_data/all_visium_data/preprocessed/cytassist_" + slide_id + "_40x.pkl")
        dataset = PatchDataset(slide, 512, magnify=0.5, patch_transform=standard_scaler, device='cuda')

        dir = "cell-detections/cytassist_" + slide_id
        os.makedirs(dir, exist_ok=True)

        raw_inferences = infer_for_dataset(model, dataset)
        torch.save(raw_inferences, dir + "/inferences_without_nms.pt")

        try:
            inferences = apply_nms_to_inferences(raw_inferences)
            torch.save(inferences, dir + "/inferences.pt")
        except Exception as e:
            print("Error applying NMS to inferences:", e, file=sys.stderr)

if __name__ == '__main__':
    main()
