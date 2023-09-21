import torch
import torch.utils.data
import tqdm

def infer(model, dataset):
    """
    Create predictions from unlabeled slides.

    Returns a list of CellGraphModelOutputs aligned to the dataset.
    Batch size is 1, because the model can only handle one instance
    at a time for cell graphs. The reason we have a different batch
    size in the optimization step is that we accumulate gradients
    across the batch, but compute each instance separately.

    If an entry in the dataset is invalid (i.e., `None`), then we skip
    the result. The corresponding indexes of `results` are stored in a
    parallel array, `valid_indexes`.
    """

    with torch.no_grad():
        valid_indexes = []
        results = []
        for i, entry in tqdm.tqdm(enumerate(dataset), desc='Inferring for dataset...', total=len(dataset)):
            # Clear cache every so often for CUDA purposes.
            if len(results) % 1000 == 0:
                torch.cuda.empty_cache()

            if entry is not None:
                input, label = entry
                results.append(model(*input))
                valid_indexes.append(i)
    
    return results, valid_indexes

def infer_inception(model, dataset):
    """
    Create predictions from unlabeled slides.

    Returns a list of CellGraphModelOutputs aligned to the dataset.
    Batch size is 1, because the model can only handle one instance
    at a time for cell graphs. The reason we have a different batch
    size in the optimization step is that we accumulate gradients
    across the batch, but compute each instance separately.

    If an entry in the dataset is invalid (i.e., `None`), then we skip
    the result. The corresponding indexes of `results` are stored in a
    parallel array, `valid_indexes`.
    """

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        results = []
        labels = []
        for batch in tqdm.tqdm(dataloader, desc='Inferring for dataset...', total=len(dataset) // 32):
            inputs, labels_ = batch
            results.append(model(inputs))
            labels.append(labels_)
    
    return torch.cat(results), torch.cat(labels)

