from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import h5py
import torch
import torch.utils
from loguru import logger
from torch import Tensor

if TYPE_CHECKING:
    import numpy as np


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
    buildtime: float,
    querytime: float,
    params: str,
    database_size: int,
    database_dim: int,
    n_queries: int,
) -> None:
    Path.mkdir(dst.parent, parents=True, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['database_size'] = database_size
    f.attrs['database_dim'] = database_dim
    f.attrs['n_queries'] = n_queries
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()


def load_queries(dataset_name: str) -> Tensor:
    queries_path = Path(f'data/{dataset_name}.hdf5')
    return torch.from_numpy(h5py.File(queries_path, 'r')['test'][:]).to(torch.float32)  # type: ignore


def load_ground_truth(dataset_name: str, k: int) -> Tensor:
    ground_truth_path = Path(f'data/{dataset_name}.hdf5')
    return torch.from_numpy(h5py.File(ground_truth_path, 'r')['neighbors'][:, :k]).to(torch.float32)  # type: ignore


@measure_runtime
def load_chunk(data: Path, start: int, stop: int) -> Tensor:
    return torch.from_numpy(h5py.File(data, 'r')['train'][start:stop])  # type: ignore


@measure_runtime
def load_all_data(data: Path) -> Tensor:
    return torch.from_numpy(h5py.File(data, 'r')['train'][:])  # type: ignore
