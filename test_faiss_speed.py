import faiss
import torch
import time
import numpy as np

vectors = torch.randn(512, 512)
vectors_cuda = vectors.to('cuda')
test_vectors = torch.randn(512, 32) # a reasonable number of cells
test_vectors_numpy = np.ascontiguousarray(test_vectors.numpy().T)
test_vectors_cuda = test_vectors.to('cuda')

# naïve approach
start_time = time.time()
distances = vectors_cuda @ test_vectors_cuda
closest = distances.argmin(dim=0)
print(closest)
print(f"Time elapsed for Torch [gpu]: {time.time() - start_time:.3f}")

start_time = time.time()

# Create a FAISS index
index = faiss.IndexFlatL2(512)
index.add(vectors.numpy())

start_time = time.time()
_, nearest = index.search(test_vectors_numpy, k=1)
print(nearest)
print(f"Time elapsed for FAISS [cpu]: {time.time() - start_time:.3f}")

# start_time = time.time()
# index_gpu = faiss.index_cpu_to_gpu(index)
# nearest = index_gpu.search(test_vectors_numpy, k=1)
# print(nearest)
# print(f"Time elapsed for FAISS [gpu]: {time.time() - start_time:.3f}")
