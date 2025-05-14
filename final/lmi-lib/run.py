import torch
import numpy as np
from pathlib import Path
import time
from loguru import logger
from lmi import LMI
from math import sqrt
import argparse
import utils

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEFAULT_DATASET_SIZE = "300K"
DEFAULT_EPOCHS = 15
DEFAULT_LR = 0.00098
DEFAULT_SAMPLE_SIZE = 1_000_000
DEFAULT_CHUNK_SIZE_BUILD = 1_000_000
DEFAULT_ALPHA = 1.0
DEFAULT_K = 30
DEFAULT_NPROBES = list(range(1, 5))

parser = argparse.ArgumentParser(description="Run Rust LMI SISAP'24 Tests")

parser.add_argument(
    "--task",
    type=int,
    choices=[1, 2, 3],
    required=True,
    help="Task number to evaluate (1, 2, or 3).",
)

parser.add_argument(
    "--dataset-size",
    type=str,
    default=DEFAULT_DATASET_SIZE,
    help="Dataset size identifier (e.g., '300K', '10M').",
)
parser.add_argument(
    "--dataset-base-path",
    type=str,
    default="/storage/brno12-cerit/home/prochazka/datasets/sisap24",
    help="Base path for datasets.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="results_rust_lmi",
    help="Directory to store HDF5 results.",
)

parser.add_argument(
    "--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs."
)
parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
parser.add_argument(
    "--sample-size",
    type=int,
    default=DEFAULT_SAMPLE_SIZE,
    help="Number of samples for training K-Means and the LMI model.",
)
parser.add_argument(
    "--chunk-size-build",
    type=int,
    default=DEFAULT_CHUNK_SIZE_BUILD,
    help="Chunk size for processing during index construction.",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=DEFAULT_ALPHA,
    help="Factor for calculating n_buckets (n_buckets = alpha * sqrt(N)).",
)
parser.add_argument(
    "--reduced-dim",
    type=int,
    default=None,
    help="Target dimensionality for TSVD. If None, original dimension is used. (Tasks 2 & 3)",
)

parser.add_argument(
    "--k", type=int, default=DEFAULT_K, help="Number of nearest neighbors to find."
)
parser.add_argument(
    "--nprobes",
    nargs="+",
    type=int,
    default=None,
    help="List of nprobe values to test. If not set, defaults based on task.",
)
parser.add_argument(
    "--rerank",
    action="store_true",
    help="Enable reranking with full-dimension vectors (Task 2).",
)
parser.add_argument(
    "--ncandidates-rerank",
    type=int,
    default=1000,
    help="Number of candidates from stage 1 to consider for reranking (Task 2).",
)

args = parser.parse_args()

if args.nprobes is None:
    args.nprobes = DEFAULT_NPROBES

LMI.init_logging()
logger.info(f"Torch version: {torch.__version__}")
logger.info(f"Running with arguments: {vars(args)}")

dataset_file = f"laion2B-en-clip768v2-n={args.dataset_size}.h5"
dataset_path = Path(args.dataset_base_path) / dataset_file

if not dataset_path.exists():
    logger.error(f"Dataset not found: {dataset_path}")
    exit(1)

logger.info(f"Loading dataset info from: {dataset_path}")
n_data, data_dim_original = utils.get_dataset_shape(dataset_path)
logger.info(f"Dataset: N={n_data}, Original D={data_dim_original}")

n_buckets = int(args.alpha * sqrt(n_data))
logger.info(f"Calculated n_buckets: {n_buckets} (alpha={args.alpha}, N={n_data})")

logger.info("Creating LMI index (Rust backend)...")
lmi: LMI

start = time.time()
lmi, kmeanstime, trainmodeltime, modelingtime, encdatabasetime = LMI.create(
    dataset=dataset_path,
    epochs=args.epochs,
    lr=args.lr,
    sample_size=min(args.sample_size, n_data),
    n_buckets=n_buckets,
    chunk_size=args.chunk_size_build,
    reduced_dim=args.reduced_dim,
    SEED=SEED,
    return_time=True,
)
buildtime = time.time() - start
logger.success(f"LMI index created. Total build time: {buildtime:.2f}s")
logger.info(
    f"Detailed build times: KMeansTime={kmeanstime:.2f}s, TrainModelTime={trainmodeltime:.2f}s, ModelingTime={modelingtime:.2f}s, EncDatabaseTime={encdatabasetime:.2f}s"
)


logger.info(f"Loading queries...")
queries_original_dim = utils.load_queries().to(torch.float32)
logger.info(
    f"Loaded {queries_original_dim.shape[0]} queries, D={queries_original_dim.shape[1]}"
)

output_task_path = Path(args.output_dir) / f"task{args.task}" / args.dataset_size
output_task_path.mkdir(parents=True, exist_ok=True)
logger.info(f"Results will be stored in: {output_task_path}")

for nprobe_val in args.nprobes:
    logger.info(f"Starting search: nprobe={nprobe_val}, k={args.k}")
    search_start_time = time.time()

    actual_k_to_search = min(args.k, n_data)

    D_results_np = np.full(
        (queries_original_dim.shape[0], actual_k_to_search), -1.0, dtype=np.float32
    )
    I_results_np = np.full(
        (queries_original_dim.shape[0], actual_k_to_search), -1, dtype=np.int32
    )

    if args.rerank:
        logger.info(
            f"Performing search with reranking (ncandidates={args.ncandidates_rerank})..."
        )
        (indices_tensor, distances_tensor), encqueriestime = lmi.search_with_reranking(
            full_dim_queries=queries_original_dim,
            original_dataset_path_str=str(dataset_path),
            final_k=actual_k_to_search,
            nprobe_stage1=nprobe_val,
            num_candidates_for_rerank=args.ncandidates_rerank,
            return_time=True,
        )
        D_results_np = distances_tensor.cpu().numpy()
        I_results_np = indices_tensor.cpu().numpy()

    else:
        (indices_tensor, distances_tensor), encqueriestime = lmi.search(
            full_dim_queries=queries_original_dim,
            k=actual_k_to_search,
            nprobe=nprobe_val,
            return_time=True,
        )
        D_results_np = distances_tensor.cpu().numpy()
        I_results_np = indices_tensor.cpu().numpy()

    querytime = time.time() - search_start_time
    logger.success(
        f"Search completed for nprobe={nprobe_val} in {querytime:.2f} seconds."
    )

    param_list = [
        f"task={args.task}",
        f"ds={args.dataset_size}",
        f"ep={args.epochs}",
        f"lr={args.lr}",
        f"sample={args.sample_size}",
        f"alpha={args.alpha}",
        f"build_chunk={args.chunk_size_build}",
        f"nprobe={nprobe_val}",
        f"k={args.k}",
    ]
    if args.reduced_dim is not None:
        param_list.append(f"reduced_dim={args.reduced_dim}")
    if args.rerank:
        param_list.append("rerank=True")
        param_list.append(f"ncand_rerank={args.ncandidates_rerank}")

    identifier_str = f"rust-lmi-" + "-".join(param_list)
    output_file = output_task_path / f"{identifier_str}.h5"

    logger.info(f"Storing results to: {output_file}")
    utils.store_results(
        dst=output_file,
        algo=f"rust-lmi-task{args.task}",
        D=D_results_np,
        I=I_results_np + 1,
        kmeanstime=kmeanstime,
        trainmodeltime=trainmodeltime,
        modelingtime=modelingtime,
        encdatabasetime=encdatabasetime,
        encqueriestime=encqueriestime,
        buildtime=buildtime,
        querytime=querytime,
        params=identifier_str,
        size=args.dataset_size,
    )

logger.info("All tests finished.")
