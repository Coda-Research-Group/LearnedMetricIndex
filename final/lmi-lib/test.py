import torch
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
from loguru import logger
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
n_buckets = int(sqrt(300000))

queries = None
k = 30
nprobe = 1

LMI.init_logging()
lmi = LMI.create(dataset, 1, 0.001, sample_size, n_buckets, chunk_size,
                       reduced_dim=128
                    )

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

nprobes = [1]
# nprobes = [1, 2, 5, 10, 20]
# nprobes = [5]

for nprobe in nprobes:
    now = time.time()

    logger.debug("Searching...")
    k = 30
    # nearest_neighbors = search(queries, k)
    # nearest_neighbors = lmi.search_raw_multiple(queries, k).detach().cpu().numpy()
    # nearest_neighbors, dists = (
    #     lmi.search_raw_multiple_nprobe(queries, k, nprobe)
    # )
    # nearest_neighbors = nearest_neighbors.detach().cpu().numpy()
    nearest_neighbors, dists = (
        lmi.search_with_reranking(queries, str(dataset), k, nprobe, 100)
    )
    nearest_neighbors = nearest_neighbors.detach().cpu().numpy()
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
