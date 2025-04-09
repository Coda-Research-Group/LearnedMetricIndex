import torch
import numpy as np
import h5py
from pathlib import Path
import time
from tqdm import tqdm
from loguru import logger

from lmi import LMI

logger.debug(f"Torch version: {torch.__version__}")

import utils

LMI.run_kmeans = utils.measure_runtime(LMI.run_kmeans)
LMI.train_model = utils.measure_runtime(LMI.train_model)
LMI.create_buckets = utils.measure_runtime(LMI.create_buckets)

torch.manual_seed(42)

logger.debug("Loading dataset...")
dataset_size = "300K"
dataset = Path(f"../data2024/laion2B-en-clip768v2-n={dataset_size}.h5")
sample_size = 100000
chunk_size = 100000
d = 768

X = torch.from_numpy(h5py.File(dataset, "r")["emb"][:]).to(torch.float32)  # type: ignore

n_data, data_dim = utils.get_dataset_shape(dataset)
X_train = utils.sample_train_subset(
    dataset, n_data, data_dim, sample_size, chunk_size
).to(torch.float32)

# Create an instance of the LMI
logger.debug("Creating LMI instance...")
lmi = LMI(n_buckets=320, data_dimensionality=d)

logger.debug("Running tests...")
lmi.tests()

logger.debug("Running kmeans...")
y = lmi.run_kmeans(X_train)

logger.debug("Training model...")
lmi.train_model(X_train, y, 15, 0.001)

logger.debug("Creating buckets...")
lmi.create_buckets(X)

now = time.time()

logger.debug("Loading queries...")
queries = utils.load_queries()

@utils.measure_runtime
def search(queries, k):
    nearest_neighbors = np.zeros((len(queries), k))
    # nearest_neighbors = lmi.search_multiple(queries, k)
    for i, query in enumerate(tqdm(queries)):
        result = (
                lmi.search_raw_parallel(query.unsqueeze(0), k).detach().cpu().numpy()
            )
        nearest_neighbors[i][:len(result)] = result # If lmi returns less than k results, the rest is left as 0
    return nearest_neighbors

logger.debug("Searching...")
k = 30
nearest_neighbors = search(queries, k)

identifier = f"lmi"
utils.store_results(
    Path("result/") / "task1" / "300K" / f"{identifier}.h5",
    "lmi",
    np.zeros((len(queries), k)),
    nearest_neighbors + 1,
    0,
    0,
    0,
    0,
    0,
    identifier,
    "300K",
)