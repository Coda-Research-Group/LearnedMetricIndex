import torch
import numpy as np
import h5py
from pathlib import Path
import time
from tqdm import tqdm
from loguru import logger
from torch.nn import Sequential, Linear, ReLU
from lmi import LMI

from math import sqrt

logger.debug(f"Torch version: {torch.__version__}")

import utils

torch.manual_seed(42)

logger.debug("Loading dataset...")
dataset_size = "300K"
dataset = Path(f"../../data2024/laion2B-en-clip768v2-n={dataset_size}.h5")
sample_size = 100000
chunk_size = 100000
d = 768

X = torch.from_numpy(h5py.File(dataset, "r")["emb"][:]).to(torch.float32)  # type: ignore

n_data, data_dim = utils.get_dataset_shape(dataset)
X_train = utils.sample_train_subset(
    dataset, n_data, data_dim, sample_size, chunk_size
).to(torch.float32)

n_buckets = int(sqrt(n_data))

# model = torch.nn.Sequential(
#     torch.nn.Linear(d, 512),
#     torch.nn.ReLU(),
#     torch.nn.Linear(512, 384),
#     torch.nn.ReLU(),
#     torch.nn.Linear(384, n_buckets),
# )

model = Sequential(
    Linear(data_dim, 512),
    ReLU(),
    Linear(512, n_buckets),
)

# Create an instance of the LMI
logger.debug("Creating LMI instance...")
lmi = LMI(model=model, n_buckets=n_buckets, data_dimensionality=d)

lmi._run_kmeans = utils.measure_runtime(lmi._run_kmeans)
lmi._train_model = utils.measure_runtime(lmi._train_model)
lmi._create_buckets = utils.measure_runtime(lmi._create_buckets)
lmi.search_raw_multiple = utils.measure_runtime(lmi.search_raw_multiple)
lmi.search_raw_multiple_nprobe = utils.measure_runtime(lmi.search_raw_multiple_nprobe)

logger.debug("Running tests...")
lmi.run_tests()

logger.debug("Running kmeans...")
y = lmi._run_kmeans(X_train)

logger.debug("Training model...")
lmi._train_model(X_train, y, epochs=15, lr=0.001)

logger.debug("Creating buckets...")
lmi._create_buckets(X)

# logger.debug("Building model...")
# lmi.build(X, 15, 0.001)


@utils.measure_runtime
def search(queries, k):
    nearest_neighbors = np.zeros((len(queries), k))
    # nearest_neighbors = lmi.search_multiple(queries, k)
    for i, query in enumerate(tqdm(queries)):
        result = lmi.search_raw(query.unsqueeze(0), k).detach().cpu().numpy()
        nearest_neighbors[i][
            : len(result)
        ] = result  # If lmi returns less than k results, the rest is left as 0
    return nearest_neighbors

logger.debug("Loading queries...")
queries = utils.load_queries()

# nprobes = [1, 2, 5, 10, 20]
nprobes = [5]

for nprobe in nprobes:
    now = time.time()

    logger.debug("Searching...")
    k = 30
    # nearest_neighbors = search(queries, k)
    # nearest_neighbors = lmi.search_raw_multiple(queries, k).detach().cpu().numpy()
    nearest_neighbors = (
        lmi.search_raw_multiple_nprobe(queries, k, nprobe).detach().cpu().numpy()
    )

    querytime = time.time() - now

    identifier = f"lmi-nprobe={nprobe}"
    utils.store_results(
        Path("result/") / "task1" / "300K" / f"{identifier}.h5",
        "lmi",
        np.zeros((len(queries), k)),
        nearest_neighbors + 1,
        0,
        0,
        0,
        0,
        querytime,
        identifier,
        "300K",
    )
