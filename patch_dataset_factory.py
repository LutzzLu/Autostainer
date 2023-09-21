from omicsio.datasets import PatchDataset, PatchDatasetWithCellAnnotations, Slide
import re
import torch

def standard_scaler(patch_or_patches: torch.Tensor):
    # patch.shape: ([B,] 3, H, W)
    # ([B,] 3, 1, 1)
    mean = patch_or_patches.mean(dim=(-2, -1), keepdim=True)
    std = patch_or_patches.std(dim=(-2, -1), keepdim=True)
    result = (patch_or_patches - mean) / std
    
    if len(result.shape) == 4:
        result = result[:, :3, :, :]
    else:
        result = result[:3, :, :]
    
    return result

def create_patch_dataset(slide: Slide, magnification: str, image_size_in_40x: int, cell_detections=None):
    assert re.match(r"\d+[xX]", magnification), "Magnification must be of the format <number>x, such as 40x or 20x."

    zoom_level = int(magnification[:-1])
    # 20x slides need to be magnified 2x
    magnification_for_this = 40 / zoom_level
    # Patch is selected, and *then* magnified.
    image_size_for_this = int(image_size_in_40x / magnification_for_this)

    if cell_detections is None:
        return PatchDataset(
            slide,
            patch_size=image_size_for_this,
            magnify=magnification_for_this,
            patch_transform=standard_scaler,
            device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return PatchDatasetWithCellAnnotations(
            slide,
            patch_size=image_size_for_this,
            magnify=magnification_for_this,
            patch_transform=standard_scaler,
            detections=cell_detections,
            device='cuda' if torch.cuda.is_available() else 'cpu')
