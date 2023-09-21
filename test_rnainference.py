import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = 1e12

from datasets import Slide, PatchDataset

def rotate90(x, y, k):
    if k == 0:
        return x, y
    return rotate90(-y, x, k - 1)

def rotate90_around(x, y, cx, cy, k):
    x2, y2 = rotate90(x - cx, y - cy, k)
    return x2 + cx, y2 + cy

def forward(model, fc, image, boxes, apply_rotations=False):
    # get cell locations
    cell_locations = [
        (((box[0] + box[2]) / 2).item() * 2, ((box[1] + box[3]) / 2).item() * 2)
        for box in boxes
    ]
    box_size = 96
    rad = box_size // 2
    if apply_rotations:
        k = torch.randint(low=0, high=3, size=()).item()
        image = torch.rot90(image, k, dims=[-2, -1])
        cell_locations = [
            rotate90_around(cx, cy, 256, 256, k)
            for cx, cy in cell_locations
        ]
    # filter cell locations
    cell_locations = [
        (cx, cy)
        for cx, cy in cell_locations
        if 512 - rad > cx > rad and 512 - rad > cy > rad
    ]
    cell_images = [
        image[:, int(cy) - rad:int(cy) + rad, int(cx) - rad:int(cx) + rad]
        for cx, cy in cell_locations
    ]
    if len(cell_images) == 0:
        return None
    cell_image_embeddings = model(torch.stack(cell_images))
    # distances: softmax(-r^2)
    distances = torch.tensor([((2-cx/64)**2 + (2-cy/64)**2) for cx, cy in cell_locations], device='cuda')
    distances = distances - distances.min()
    distances = torch.softmax(distances, dim=0)
    distances[distances < 1e-8] = 0
    # Instead of using distances, just use mean
#     cell_image_embeddings = cell_image_embeddings.T @ distances
    results = fc(cell_image_embeddings).T @ distances # .mean(dim=0)
    return results

# Try focal loss and margin loss
def balanced_bce(logits, labels):
    pos = labels == 1
    neg = ~pos
    pos_logits = logits[pos]
    neg_logits = logits[neg]
    pos_labels = labels[pos]
    neg_labels = labels[neg]

    return 0.5 * (F.binary_cross_entropy_with_logits(pos_logits, pos_labels) + F.binary_cross_entropy_with_logits(neg_logits, neg_labels))

def standard_scaler(patch_or_patches: torch.Tensor):
    # patch.shape: ([B,] 3, H, W)
    # ([B,] 3, 1, 1)
    mean = patch_or_patches.mean(dim=(-2, -1), keepdim=True)
    std = patch_or_patches.std(dim=(-2, -1), keepdim=True)
    return (patch_or_patches - mean) / std

genes_to_use = list(np.load("filtered_set.npy"))
slide = Slide.load("input_data/preprocessed/autostainer_40x_cropped.pkl")
slide = slide.select_genes(genes_to_use).binary()

features = len(slide.genes)

import tqdm

torch.manual_seed(0)

quadrants = slide.create_quadrants()

valid_quadrant, train_quadrant = quadrants[0]
whole_ds = PatchDataset(slide, 512, 1, device=torch.device('cuda'), patch_transform=standard_scaler)
valid_ds = PatchDataset(valid_quadrant, 512, 1, device=torch.device('cuda'), patch_transform=standard_scaler)
train_ds = PatchDataset(train_quadrant, 512, 1, device=torch.device('cuda'), patch_transform=standard_scaler)
inferences = torch.load('cell-detections/autostainer_orig/1675456635/combined_nms.pt')

# New model
in_quadrant = []
out_quadrant = []

mid_x = slide.spot_locations.image_x.median()
mid_y = slide.spot_locations.image_y.median()

batch_size = 32

for i in range(len(slide.spot_locations)):
    if slide.spot_locations.image_x[i] <= mid_x and slide.spot_locations.image_y[i] <= mid_y:
        in_quadrant.append(i)
    else:
        out_quadrant.append(i)

train_perm = torch.randperm(len(out_quadrant))

def test_cells_model():
    # can use strings for this
    # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights
    model = torchvision.models.resnet50('IMAGENET1K_V1').cuda()
    model = model.train()
    # replace FC layer
    model.fc = torch.nn.Identity()
    fc = torch.nn.Linear(in_features=2048, out_features=features, bias=True).cuda()
    optim = torch.optim.Adam([*model.parameters(), *fc.parameters()])

    batch_size = 32

    for epoch in range(5):
        with tqdm.tqdm(total=len(train_perm), desc='Training epoch ' + str(epoch)) as pbar:
            i = 0
            running_loss = 0
            while i < len(train_perm):
                idxs = [out_quadrant[train_perm[j]] for j in range(i, min(i + batch_size, len(train_perm)))]
                optim.zero_grad()
                preds = []
                labels = []
                for idx in idxs:
                    image, label = whole_ds[idx]
                    # cell based model
                    result = forward(model, fc, image, inferences[idx]['boxes'], apply_rotations=True)
                    if result is not None:
                        preds.append(result)
                        labels.append(label.float())

                N = len(preds)
                    
                loss = balanced_bce(torch.stack(preds), torch.stack(labels))
                loss.backward()
                optim.step() 
                running_loss += loss.item() * N

                i += N
                pbar.update(N)
                pbar.set_postfix(loss=loss.item(), running_loss=running_loss/i)

    torch.save(model.state_dict(), "cell_model_resnet50_0.pt")

def test_baseline_model():
    print("Training the baseling model")
    # Baseline model
    baseline_model = torchvision.models.inception_v3('IMAGENET1K_V1').cuda()
    baseline_model.aux_logits = False
    baseline_model.fc = torch.nn.Linear(in_features=2048, out_features=features, bias=True).cuda()
    baseline_model_optim = torch.optim.Adam(baseline_model.parameters())
    baseline_model.train()

    for epoch in range(2):
        with tqdm.tqdm(total=len(train_ds), desc='Training baseline model...') as pbar:
            i = 0
            running_loss = 0
            running_loss_baseline = 0
            dataloader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
            transform = T.Compose([
                T.ColorJitter(),
                T.RandomRotation(90),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip()
            ])
            for images, labels in dataloader:
                images = transform(images)
                preds = baseline_model(images)
                baseline_model_optim.zero_grad()
                loss = balanced_bce(preds, labels.float())
                loss.backward()
                baseline_model_optim.step()
                
                running_loss += loss.item() * len(images)
                i += len(images)
                
                pbar.update(len(images))
                pbar.set_postfix(loss=loss.item(), running_loss=running_loss/i)

    torch.save(baseline_model.state_dict(), "baseline_model_inceptionv3_1.pt")

def validate(pred_logits, spot_counts):
    import matplotlib.pyplot as plt
    
    pred_logits = pred_logits.cpu()

    # tp = pred_tensor_bool == dataset.labels
    true_bool = spot_counts.bool()
    pred_scores = torch.sigmoid(pred_logits)
    pred_bool = pred_logits > 0

    tp = true_bool & pred_bool
    tn = ~(true_bool | pred_bool)
    fp = ~true_bool & pred_bool
    fn = true_bool & ~pred_bool

    acc = true_bool == pred_bool

    accuracy = acc.sum(axis=0) / true_bool.shape[0]
    plt.title("Histogram of accuracy")
    plt.hist(accuracy)
    plt.savefig("hists/accuracy.png")
    # plt.show()

    accuracy = tp.sum(axis=0) / true_bool.shape[0]
    plt.title("Histogram of TP")
    plt.hist(accuracy)
    plt.xlim(0, 1)
    plt.savefig("hists/tp.png")
    # plt.show()

    accuracy = tn.sum(axis=0) / true_bool.shape[0]
    plt.title("Histogram of TN")
    plt.hist(accuracy)
    plt.xlim(0, 1)
    plt.savefig("hists/tn.png")
    # plt.show()

    accuracy = fp.sum(axis=0) / true_bool.shape[0]
    plt.title("Histogram of FP")
    plt.hist(accuracy)
    plt.xlim(0, 1)
    plt.savefig("hists/fp.png")
    # plt.show()

    accuracy = fn.sum(axis=0) / true_bool.shape[0]
    plt.title("Histogram of FN")
    plt.xlim(0, 1)
    plt.hist(accuracy)
    plt.savefig("hists/fn.png")
    # plt.show()

    precision = tp.sum(axis=0) / (tp.sum(axis=0) + fp.sum(axis=0))
    plt.title("Histogram of Precision")
    plt.xlim(0, 1)
    plt.hist(precision)
    plt.savefig("hists/precision.png")
    # plt.show()

    recall = tp.sum(axis=0) / (tp.sum(axis=0) + fn.sum(axis=0))
    plt.title("Histogram of Recall")
    plt.xlim(0, 1)
    plt.hist(recall)
    plt.savefig("hists/recall.png")
    # plt.show()
    
    import sklearn
    
    auroc = sklearn.metrics.roc_auc_score(true_bool.numpy(), pred_scores.numpy(), average=None)
    plt.title("Histogram of AUROC")
    plt.xlim(0, 1)
    plt.hist(auroc)
    plt.savefig("hists/auroc.png")
    # plt.show()

    print("Median precision:", precision.nanmedian())
    print("Median recall:", recall.median())
    print("Median AUROC:", np.nanmedian(auroc))

def validate_cells_model(model, fc):
    predictions = []

    model.train()

    with tqdm.tqdm(total=len(valid_ds), desc='Validating...') as pbar:
        i = 0
        running_loss = 0
        while i < len(in_quadrant):
            idxs = [in_quadrant[j] for j in range(i, min(i + batch_size, len(in_quadrant)))]
            with torch.no_grad():
                preds = []
                labels = []
                for idx in idxs:
                    image, label = whole_ds[idx]
                    # cell based model
                    preds.append(forward(model, fc, image, inferences[idx]['boxes']))
                    labels.append(label.float())

                predictions.extend(preds)
                loss = balanced_bce(torch.stack(preds), torch.stack(labels))
            running_loss += loss.item() * len(idxs)

            i += len(idxs)
            pbar.update(len(idxs))
            pbar.set_postfix(loss=loss.item(), running_loss=running_loss/i)
    validate(torch.stack(predictions), slide.spot_counts[in_quadrant])

validate_cells_model()
