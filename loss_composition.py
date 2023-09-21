import torch

def compose_losses(loss_map):
    def wrapped(*args):
        loss_results = {
            loss_name + "_loss": loss_function(*args) * loss_proportion
            for loss_name, (loss_function, loss_proportion) in loss_map.items()
            if loss_function is not None
        }

        return torch.stack([*loss_results.values()]).sum(), loss_results

    return wrapped
