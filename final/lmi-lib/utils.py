from __future__ import annotations

import functools
import gc
import time
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import h5py
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
    querytime: float,
    params: str,
    size: str,
) -> None:
    Path.mkdir(dst.parent, parents=True, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['modelingtime'] = modelingtime
    f.attrs['encdatabasetime'] = encdatabasetime
    f.attrs['encqueriestime'] = encqueriestime
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['size'] = size
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()


def get_dataset_shape(dataset: Path) -> tuple[int, int]:
    return h5py.File(dataset, 'r')['emb'].shape  # type: ignore


def get_dataset_size(dataset: Path) -> int:
    return get_dataset_shape(dataset)[0]


def load_queries() -> Tensor:
    queries_path = Path('../../data2024/public-queries-2024-laion2B-en-clip768v2-n=10k.h5')
    return torch.from_numpy(h5py.File(queries_path, 'r')['emb'][:]).to(torch.float32)  # type: ignore


def load_ground_truth(dataset_size: str, k: int = 30) -> Tensor:
    ground_truth_path = Path(f'../../data2024/gold-standard-dbsize={dataset_size}--public-queries-2024-laion2B-en-clip768v2-n=10k.h5')
    return torch.from_numpy(h5py.File(ground_truth_path, 'r')['knns'][:, :k]).to(torch.float32)  # type: ignore


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
