import os
from functools import cached_property
from typing import Union


class Checkpoint:
    def __init__(self, uid):
        self.uid = uid

    @cached_property
    def model(self):
        import torch
        model = torch.load(f'checkpoints/{self.uid}/model.pt')
        model.uid = self.uid
        return model
    
    @cached_property
    def optimizer(self):
        import torch
        return torch.load(f'checkpoints/{self.uid}/optim.pt')

def save(model, optim):
    import torch

    uid = model.uid

    if not os.path.exists(f'checkpoints/{uid}/analyses'):
        os.makedirs(f'checkpoints/{uid}/analyses', exist_ok=True)

    torch.save(model, f'checkpoints/{uid}/model.pt')
    torch.save(optim, f'checkpoints/{uid}/optim.pt')

    return Checkpoint(uid)

def load(uid) -> Union[Checkpoint, None]:
    if not os.path.exists(f'checkpoints/{uid}'):
        return None
    
    return Checkpoint(uid)
