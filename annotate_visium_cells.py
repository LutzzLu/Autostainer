print("Started")

import datasets
import torch
import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = 1e10

slide = datasets.Slide.load("input_data/preprocessed/autostainer_20x_cropped.pkl")

print("Loaded slide")

# training_transforms = T.Compose([
#     standard_scaler_per_channel,
#     T.ColorJitter(),
#     T.RandomRotation(90),
#     T.RandomHorizontalFlip(),
#     T.RandomVerticalFlip(),
# ])

def standard_scaler_per_channel(patch: torch.Tensor):
    # patch.shape: (3, H, W)
    # (3, 1, 1)
    mean = patch.mean(dim=(1, 2), keepdim=True)
    std = patch.std(dim=(1, 2), keepdim=True)
    return (patch - mean) / std

dataset = datasets.PatchDataset(slide, 256, 1, patch_transform=None, device=torch.device('cuda'))

import matplotlib.pyplot as plt

plt.imsave("image_0.png", dataset[0][0].cpu().permute(1, 2, 0).numpy())
# dataset[0]

# img = slide.image

# class PatchDataset(torch.utils.data.Dataset):
#     def __init__(self, slide: Slide, patch_size: int, magnify: int, patch_transform, device):

# slide.

# print(img.shape)
