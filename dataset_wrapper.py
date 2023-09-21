import torch.utils.data

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, inner: torch.utils.data.Dataset, transform):
        self.inner = inner
        self.transform = transform

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx: int):
        return self.transform(self.inner, idx)
