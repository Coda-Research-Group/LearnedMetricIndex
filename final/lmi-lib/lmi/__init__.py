from __future__ import annotations

from .lmi import LMI as LMIBase
from .helpers import extract_model_config
import utils

import torch
from torch.nn import Sequential, Linear, ReLU

import gc
from pathlib import Path
from typing import Optional
from loguru import logger
import time
from sklearn.decomposition import TruncatedSVD

import h5py


class LMI:
    def __init__(self, model, *args, **kwargs):
        model_config = extract_model_config(model)
        self._inner = LMIBase(model_config, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @utils.measure_runtime
    @staticmethod
    def _run_kmeans(
        n_buckets: int, dimensionality: int, X: torch.Tensor
    ) -> torch.Tensor:
        return LMIBase._run_kmeans(n_buckets, dimensionality, X)

    @utils.measure_runtime
    def _train_model(self, X: torch.Tensor, y: torch.Tensor, epochs: int, lr: float):
        return self._inner._train_model(X, y, epochs, lr)

    @utils.measure_runtime
    def _create_buckets(self, X: torch.Tensor):
        return self._inner._create_buckets(X)

    @utils.measure_runtime
    def _create_buckets_scalable(self, dataset: Path, n_data: int, chunk_size: int):
        return self._inner._create_buckets_scalable(dataset, n_data, chunk_size)

    @utils.measure_runtime
    def search_raw_multiple_nprobe(
        self, queries: torch.Tensor, k: int, nprobe: int
    ) -> torch.Tensor:
        return self._inner.search_raw_multiple_nprobe(queries, k, nprobe)

    @staticmethod
    def init_logging():
        LMIBase.init_logging()

    @staticmethod
    def create(
        dataset: Path,
        epochs: int,
        lr: float,
        sample_size: int,
        n_buckets: int,
        chunk_size: int,
        model: Optional[Sequential] = None,
        reduced_dim: Optional[int] = None,
        SEED: int = 42,
    ) -> LMI:
        logger.debug("Creating LMI instance...")

        n_data, data_dim = utils.get_dataset_shape(dataset)
        X_train = utils.sample_train_subset(
            dataset, n_data, data_dim, sample_size, chunk_size
        ).to(torch.float32)

        logger.debug(f"Training on {X_train.shape[0]} subset from {n_data} dataset")

        y = LMI._run_kmeans(n_buckets, data_dim, X_train)

        if model is None:
            model = Sequential(
                Linear(data_dim, 512),
                ReLU(),
                Linear(512, n_buckets),
            )

        lmi = LMI(model, n_buckets, data_dim)
        lmi._train_model(X_train, y, epochs, lr)

        if reduced_dim is not None:
            tsvd = TruncatedSVD(reduced_dim, random_state=SEED)
            start = time.time()
            tsvd.fit(X_train)

        del X_train
        gc.collect()

        lmi._create_buckets_scalable(str(dataset), n_data, chunk_size)

        return lmi
