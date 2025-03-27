import torch
import h5py
from pathlib import Path
import time

from lmi import LMI

print(torch.__version__)

dataset_path = Path("../data_toy/laion2B-en-clip768v2-n=100K.h5")
X = torch.from_numpy(h5py.File(dataset_path, "r")["emb"][:]).to(torch.float32)  # type: ignore
n, d = X.shape

# Create an instance of the LMI
lmi = LMI(n_buckets=320, data_dimensionality=d, epochs=1)
lmi.train(X)

n = 200
print(f"Evaluating on {n} queries...")
now = time.time()

queries = X[range(n)]

k = 10

recall_sum = 0

for query in queries:
    nearest_neighbors = lmi.search(query.unsqueeze(0), k)
    ground_truth = torch.argsort(torch.cdist(query.unsqueeze(0), X)).reshape(-1)[:k]
    recall = len(set(nearest_neighbors.tolist()).intersection(set(ground_truth.tolist())))/k
    recall_sum += recall

print(f"Ground truth: {ground_truth}")
print(f"Predicted: {nearest_neighbors}")

print(f"Avg. Recall: {recall_sum/n}")

print("Recall evaluated in", time.time() - now, "seconds.")