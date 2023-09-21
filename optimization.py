import gc

import torch
import torch.utils.data
import tqdm


def collate(batch):
    inputs = [b[0] for b in batch]
    outputs = [b[1] for b in batch]
    # single_cell_data = [b[2] for b in batch]
    return inputs, outputs# , single_cell_data

class SafeDataLoader():
    def __init__(self, dataset, batch_size, shuffle, return_index=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.order = torch.randperm(len(dataset)) if shuffle else torch.arange(len(dataset))
        self.return_index = return_index
    
    def generate_batches(self):
        batch = []
        batch_indexes = []
        for idx in self.order:
            result = self.dataset[idx]
            if result is not None:
                batch.append(result)
                batch_indexes.append(idx)
            if len(batch) == self.batch_size:
                yield batch_indexes, collate(batch)
                batch = []
        if len(batch) > 0:
            yield batch_indexes, collate(batch)

def pass_once(model, optimizer, dataset, loss_fn, batch_size=16):
    dataloader = SafeDataLoader(dataset, batch_size=batch_size, shuffle=True, return_index=True)
    history = []

    with tqdm.tqdm(dataloader.generate_batches(), desc=f'Passing over dataset...', total=len(dataset) // batch_size) as pbar:
        for index_batch, (input_batch, label_batch) in pbar:
            torch.cuda.empty_cache()
            gc.collect()

            pred_batch = model(input_batch)
            
            # print("VERY FIRST PREDICTION:", pred_batch.graph_vector[0], pred_batch.cell_vectors[0])
            # print(label_batch[0])

            loss, info = loss_fn(pred_batch, label_batch, index_batch) # , single_cell_data_batch)
            # print(loss)
            # exit()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history.append(info)
            pbar.set_postfix({**{k: v.item() for k, v in info.items()}, "loss": loss.item()})
    
    return history
