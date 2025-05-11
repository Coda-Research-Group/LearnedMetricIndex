import torch
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
from loguru import logger
from lmi import LMI

from math import sqrt
import argparse

import utils

SEED = 42
torch.manual_seed(SEED)

DEFAULT_DATASET_SIZE = "300K"
DEFAULT_EPOCHS = 15
DEFAULT_LR = 0.00098
DEFAULT_SAMPLE_SIZE = 1_000_000
DEFAULT_CHUNK_SIZE = 1_000_000
DEFAULT_ALPHA = 1.0
DEFAULT_K = 30
DEFAULT_NPROBES = range(1, 6)

parser = argparse.ArgumentParser(description="Run Rust LMI Task 1 Test")
parser.add_argument(
    "--dataset-size",
    type=str,
    default=DEFAULT_DATASET_SIZE,
    help="Dataset size identifier (e.g., 300K, 10M)",
)
parser.add_argument(
    "--dataset-base-path",
    type=str,
    default="/storage/brno12-cerit/home/prochazka/datasets/sisap24",
    help="Base path for datasets",
)
parser.add_argument(
    "--reduced-dim",
    type=int,
    default=None,
    help="Reduced dimensionality for training",
)
parser.add_argument(
    "--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs"
)
parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
parser.add_argument(
    "--sample-size",
    type=int,
    default=DEFAULT_SAMPLE_SIZE,
    help="Number of samples for training",
)
parser.add_argument(
    "--chunk-size",
    type=int,
    default=DEFAULT_CHUNK_SIZE,
    help="Chunk size for processing",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=DEFAULT_ALPHA,
    help="Factor for calculating n_buckets (alpha * sqrt(N))",
)
parser.add_argument(
    "--k", type=int, default=DEFAULT_K, help="Number of nearest neighbors to find"
)
parser.add_argument(
    "--nprobes",
    nargs="+",
    type=int,
    default=DEFAULT_NPROBES,
    help="List of nprobe values to test",
)
parser.add_argument(
    "--output-dir", type=str, default="result", help="Directory to store results"
)

args = parser.parse_args()

logger.info(f"Torch version: {torch.__version__}")
logger.info(f"Using arguments: {args}")

dataset_file = f"laion2B-en-clip768v2-n={args.dataset_size}.h5"
dataset_path = Path(args.dataset_base_path) / dataset_file

if not dataset_path.exists():
    logger.error(f"Dataset not found at: {dataset_path}")
    exit(1)

logger.info(f"Loading dataset info from: {dataset_path}")
try:
    n_data, data_dim = utils.get_dataset_shape(dataset_path)
    logger.info(f"Dataset shape: N={n_data}, D={data_dim}")
except Exception as e:
    logger.error(f"Could not read dataset shape from {dataset_path}: {e}")
    data_dim = 768
    logger.warning(
        f"Could not read N from dataset, estimating based on dataset_size name '{args.dataset_size}' for n_buckets calculation."
    )
    size_suffix = args.dataset_size[-1].upper()
    num_part = args.dataset_size[:-1]
    if size_suffix == "K":
        n_data = int(float(num_part) * 1000)
    elif size_suffix == "M":
        n_data = int(float(num_part) * 1000000)
    else:
        n_data = int(args.dataset_size)  # Assume it's just a number
    logger.info(f"Estimated N={n_data}, Assumed D={data_dim}")


n_buckets = int(args.alpha * sqrt(n_data))
logger.info(f"Calculated n_buckets: {n_buckets} (alpha={args.alpha}, N={n_data})")

LMI.init_logging()

logger.info("Creating LMI index (Rust backend)...")
build_start_time = time.time()
try:
    lmi, tsvd = LMI.create(
        dataset=dataset_path,
        epochs=args.epochs,
        lr=args.lr,
        sample_size=min(args.sample_size, n_data),
        n_buckets=n_buckets,
        chunk_size=args.chunk_size,
        SEED=SEED,
        reduced_dim=args.reduced_dim,
    )
except Exception as e:
    logger.error(f"Error during LMI creation: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

buildtime = time.time() - build_start_time
logger.success(f"LMI index created in {buildtime:.2f} seconds.")

modelingtime = 0.0
encdatabasetime = 0.0
encqueriestime = 0.0

logger.info("Loading queries...")
queries = utils.load_queries()
logger.info(f"Loaded {queries.shape[0]} queries with dimension {queries.shape[1]}")

queries = queries.to(torch.float32)

if args.reduced_dim is not None:
    queries = torch.tensor(tsvd.transform(queries))

output_base_path = Path(args.output_dir) / "task3" / args.dataset_size
output_base_path.mkdir(parents=True, exist_ok=True)
logger.info(f"Will store results in: {output_base_path}")

for nprobe in args.nprobes:
    logger.info(f"Starting search with nprobe={nprobe}, k={args.k}...")
    search_start_time = time.time()

    try:
        actual_k = min(args.k, n_data)
        nearest_neighbors = (
            lmi.search_raw_multiple_nprobe(queries, actual_k, nprobe)
            .detach()
            .cpu()
            .numpy()
        )
    except Exception as e:
        logger.error(f"Error during search (nprobe={nprobe}): {e}")
        import traceback

        traceback.print_exc()
        continue

    querytime = time.time() - search_start_time
    logger.success(f"Search completed for nprobe={nprobe} in {querytime:.2f} seconds.")

    if nearest_neighbors.shape != (queries.shape[0], actual_k):
        logger.warning(
            f"Unexpected shape for nearest_neighbors: {nearest_neighbors.shape}. Expected: {(queries.shape[0], actual_k)}"
        )

    identifier = f"rust-lmi-task3-ds={args.dataset_size}-reduced-dim={args.reduced_dim}-ep={args.epochs}-lr={args.lr}-sample={args.sample_size}-alpha={args.alpha}-chunk={args.chunk_size}-nprobe={nprobe}"
    output_file = output_base_path / f"{identifier}.h5"

    logger.info(f"Storing results to: {output_file}")
    utils.store_results(
        dst=output_file,
        algo="rust-lmi-task3",
        D=np.zeros((len(queries), actual_k), dtype=np.float32),
        I=nearest_neighbors + 1,
        modelingtime=modelingtime,
        encdatabasetime=encdatabasetime,
        encqueriestime=encqueriestime,
        buildtime=buildtime,
        querytime=querytime,
        params=identifier,
        size=args.dataset_size,
    )

logger.info("Testing finished.")
