from __future__ import annotations

import functools
import gc
import time
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import h5py
import numpy
import torch
import torch.utils
from loguru import logger
from torch import Tensor

if TYPE_CHECKING:
    import numpy as np

SEED = 42
torch.manual_seed(SEED)


def measure_runtime(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper_measure_runtime(*args, **kwargs) -> Any:  # noqa: ANN401, ANN002, ANN003
        start = time.time()
        result = func(*args, **kwargs)
        stop = time.time()

        logger.debug(f'Execution of {func.__name__} took {stop - start:.5}s.')

        return result

    return wrapper_measure_runtime


def store_results(  # noqa: PLR0913
    dst: Path,
    algo: str,
    D: np.ndarray,
    I: np.ndarray,
    modelingtime: float,
    encdatabasetime: float,
    encqueriestime: float,
    buildtime: float,
    inserttime: float,
    querytime: float,
    params: str,
    candidates: int,
    size: str,
) -> None:
    Path.mkdir(dst.parent, parents=True, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['modelingtime'] = modelingtime
    f.attrs['encdatabasetime'] = encdatabasetime
    f.attrs['encqueriestime'] = encqueriestime
    f.attrs['buildtime'] = buildtime
    f.attrs['inserttime'] = inserttime
    f.attrs['querytime'] = querytime
    f.attrs['size'] = size
    f.attrs['params'] = params
    f.attrs['candidates'] = candidates
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()


def get_dataset_shape(dataset: Path) -> tuple[int, int]:
    return h5py.File(dataset, 'r')['emb'].shape  # type: ignore


def get_dataset_size(dataset: Path) -> int:
    return get_dataset_shape(dataset)[0]


def ensure_float32(data: Tensor) -> Tensor:
    return data.to(torch.float32)  # type: ignore


def load_dataset(dataset: Path) -> Tensor:
    return torch.from_numpy(h5py.File(dataset, 'r')['emb'][:])


def load_queries() -> Tensor:
    queries_path = Path('data2024/public-queries-2024-laion2B-en-clip768v2-n=10k.h5')
    return ensure_float32(torch.from_numpy(h5py.File(queries_path, 'r')['emb'][:]))  # type: ignore


def load_ground_truth(dataset_size: str, k: int = 30) -> Tensor:
    ground_truth_path = Path(f'data2024/gold-standard-dbsize={dataset_size}--public-queries-2024-laion2B-en-clip768v2-n=10k.h5')
    return ensure_float32(torch.from_numpy(h5py.File(ground_truth_path, 'r')['knns'][:, :k]))  # type: ignore


def load_indices(dataset: Path, n_data: int, dim: int, indices: Tensor, chunk_size: int) -> Tensor:
    n_chunks = ceil(n_data / chunk_size)

    X = torch.empty((len(indices), dim))

    offset = 0
    for chunk_i in range(n_chunks):
        start, stop = chunk_i * chunk_size, (chunk_i + 1) * chunk_size

        chunk_indices = indices[(start <= indices) & (stop > indices)] - start
        if len(chunk_indices) == 0:
            continue
        chunk, _ = load_chunk(dataset, start, stop)

        X[offset : offset + len(chunk_indices)] = chunk[chunk_indices]
        del chunk
        offset += len(chunk_indices)
    return X


@measure_runtime
def sample_train_subset(dataset: Path, n_data: int, dim: int, n_sample: int, chunk_size: int) -> Tensor:
    n_chunks = ceil(n_data / chunk_size)
    sample_indices = torch.randint(0, n_data, (n_sample,))

    X = torch.empty((n_sample, dim), dtype=torch.float16)

    offset = 0
    for chunk_i in range(n_chunks):
        start, stop = chunk_i * chunk_size, (chunk_i + 1) * chunk_size

        chunk_sample_indices = sample_indices[(start <= sample_indices) & (stop > sample_indices)] - start

        chunk = load_chunk(dataset, start, stop)
        X[offset : offset + len(chunk_sample_indices)] = chunk[chunk_sample_indices]

        del chunk

        offset += len(chunk_sample_indices)
    gc.collect()
    return X


@measure_runtime
def load_chunk(data: Path, start: int, stop: int) -> Tensor:
    return torch.from_numpy(h5py.File(data, 'r')['emb'][start:stop])  # type: ignore


@measure_runtime
def load_real_data(dataset: Path, indices: Tensor) -> Tensor:
    return torch.from_numpy(h5py.File(dataset, 'r')['emb'][indices])  # type: ignore


def compute_class_mean(X: Tensor) -> Tensor:
    return X.mean(dim=0)


def herd_from(X: Tensor, size: int) -> list[int]:
    """
    An implementation of herding algorithm described in Eq.4 in the article
    Class-Incremental Learning: A Survey (https://arxiv.org/pdf/2302.03648).
    Modified for herding from one class and to return references instead of
    actual data.
    """
    class_mean = X.mean(dim=0)
    selected_indices = []
    running_sum = torch.zeros_like(class_mean)

    available_mask = torch.ones(len(X), dtype=torch.bool, device=X.device)

    for k in range(1, min(size, len(X)) + 1):
        available_samples = X[available_mask]
        candidate_means = (available_samples + running_sum) / k
        distances = torch.linalg.vector_norm(class_mean - candidate_means, dim=1)

        closest_idx_local = torch.argmin(distances).item()  # Index in available data
        global_idx = torch.where(available_mask)[0][closest_idx_local].item()  # Local index converted to index in original data

        selected_indices.append(global_idx)
        running_sum += X[global_idx]
        available_mask[global_idx] = False

    return selected_indices


def create_task_h5py(dataset: Path, nth_task: int, n_tasks: int) -> Path:
    assert nth_task <= n_tasks - 1, "nth_task must be at most n_tasks - 1"

    n_data = get_dataset_size(dataset)

    task_size = n_data // n_tasks
    remainder = n_data % n_tasks

    start = nth_task * task_size
    stop = start + task_size + (remainder if nth_task == n_tasks - 1 else 0)  # Ensure the remainder is added to the last task

    logger.debug(f"Creating h5py file for task {nth_task}")

    with h5py.File(dataset, 'r') as input_file:
        embeddings = input_file['emb'][start:stop]

    output_path = Path("data2024") / f"tmp-task-{nth_task}.h5"
    with h5py.File(output_path, 'w') as output_file:
        output_file.create_dataset('emb', data=embeddings, dtype=numpy.float16)

    logger.debug(f"Temporary dataset for task {nth_task} saved to {output_path}")
    return output_path


def delete_task_h5py(task_dataset: Path) -> None:
    task_dataset.unlink()
    logger.debug(f"Temporary dataset {task_dataset} was removed")
