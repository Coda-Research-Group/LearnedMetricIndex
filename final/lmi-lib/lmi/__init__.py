from __future__ import annotations

import os
import torch

# Set the LD_LIBRARY_PATH dynamically
libtorch_path = os.path.join(torch.__path__[0], 'lib')
if libtorch_path not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = libtorch_path + ":" + os.environ.get('LD_LIBRARY_PATH', '')


from .lmi import LMI as LMIBase
from .helpers import extract_model_config
from .utils import measure_runtime, get_dataset_shape, sample_train_subset

import torch
from torch.nn import Sequential, Linear, ReLU

import gc
from pathlib import Path
from typing import Optional
from loguru import logger
import time

class LMI:
    def __init__(self, model, *args, **kwargs):
        model_config = extract_model_config(model)
        self._inner = LMIBase(model_config, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @staticmethod
    @measure_runtime
    def _run_kmeans(
        n_buckets: int, dimensionality: int, X: torch.Tensor
    ) -> torch.Tensor:
        return LMIBase._run_kmeans(n_buckets, dimensionality, X)

    @measure_runtime
    def _train_model(self, X: torch.Tensor, y: torch.Tensor, epochs: int, lr: float):
        return self._inner._train_model(X, y, epochs, lr)

    @measure_runtime
    def _create_buckets(self, X: torch.Tensor):
        return self._inner._create_buckets(X)

    @measure_runtime
    def _create_buckets_scalable(
        self, dataset: Path, n_data: int, chunk_size: int
    ) -> float:
        return self._inner._create_buckets_scalable(dataset, n_data, chunk_size)

    @staticmethod
    def init_logging():
        LMIBase.init_logging()

    def _encode(self, X: torch.Tensor) -> tuple[torch.Tensor, float]:
        logger.info(
            f"Reducing query dimensionality to {self._inner.dimensionality} using LMI's TSVD..."
        )
        query_encode_start_time = time.time()
        transformed_queries = self._inner.transform_tsvd(X)
        encqueriestime = time.time() - query_encode_start_time
        logger.success(
            f"Queries transformed to D={transformed_queries.shape[1]} in {encqueriestime:.2f}s."
        )

        return transformed_queries, encqueriestime

    def search(
        self,
        full_dim_queries: torch.Tensor,
        k: int,
        nprobe: int,
        return_time: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[tuple[torch.Tensor, torch.Tensor], float]
    ):
        transformed_queries, encqueriestime = None, 0.0
        if self._inner.dimensionality != full_dim_queries.shape[1]:
            transformed_queries, encqueriestime = self._encode(full_dim_queries)
        result = self._inner.search(full_dim_queries, k, nprobe, transformed_queries)

        if return_time:
            return result, encqueriestime
        return result

    def search_with_reranking(
        self,
        full_dim_queries: torch.Tensor,
        original_dataset_path_str: str,
        final_k: int,
        nprobe_stage1: int,
        num_candidates_for_rerank: int,
        return_time: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[tuple[torch.Tensor, torch.Tensor], float]
    ):
        transformed_queries, encqueriestime = None, 0.0
        if self._inner.dimensionality != full_dim_queries.shape[1]:
            transformed_queries, encqueriestime = self._encode(full_dim_queries)
        result = self._inner.search_with_reranking(
            full_dim_queries,
            original_dataset_path_str,
            final_k,
            nprobe_stage1,
            num_candidates_for_rerank,
            transformed_queries,
        )

        if return_time:
            return result, encqueriestime
        return result

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
        return_time: bool = False,
    ) -> tuple[LMI, float, float]:
        logger.debug("Creating LMI (Rust backend) instance...")
        torch.manual_seed(SEED)

        n_data, data_dim_original = get_dataset_shape(dataset)

        logger.info(f"Sampling training subset (sample_size={sample_size})...")
        X_train = sample_train_subset(
            dataset, n_data, data_dim_original, sample_size, chunk_size
        ).to(torch.float32)
        logger.success(f"Training subset sampled: {X_train.shape}")

        logger.info(f"Running K-Means (n_buckets={n_buckets})...")
        start = time.time()
        # kmeans = faiss.Kmeans(
        #     d=data_dim_original,
        #     k=n_buckets,
        #     verbose=True,
        #     seed=SEED,
        #     spherical=True,
        #     max_points_per_centroid=1000000,
        # )
        # kmeans.train(X_train)
        # y_train = torch.from_numpy(kmeans.index.search(X_train, 1)[1].T[0])  # type: ignore
        y_train = LMI._run_kmeans(n_buckets, data_dim_original, X_train)
        kmeanstime = time.time() - start
        logger.success(f"K-Means completed in {kmeanstime:.2f} seconds. Labels shape: {y_train.shape}")

        if model is None:
            logger.info(f"Defining default model for input dim: {data_dim_original}")
            model = Sequential(
                Linear(data_dim_original, 512),
                ReLU(),
                Linear(512, n_buckets),
            )

        lmi = LMI(model, n_buckets, data_dim_original)

        logger.info(f"Training LMI model (epochs={epochs}, lr={lr})...")
        start = time.time()
        lmi._train_model(X_train, y_train, epochs, lr)
        trainmodeltime = time.time() - start
        logger.success(f"LMI model training completed in {trainmodeltime:.2f} seconds.")

        modelingtime = 0.0
        if reduced_dim is not None and reduced_dim > 0:
            logger.info(
                f"Fitting TSVD from {data_dim_original} to {reduced_dim} dimensions..."
            )
            modelingtime = time.time()
            lmi._fit_tsvd(X_train, reduced_dim)
            modelingtime = time.time() - modelingtime
            logger.success(
                f"TSVD fitting completed in {modelingtime:.2f} seconds. New LMI dim: {lmi._inner.dimensionality}"
            )

        del X_train, y_train
        gc.collect()

        logger.info("Creating LMI buckets by processing the full dataset...")
        encdatabasetime = lmi._create_buckets_scalable(str(dataset), n_data, chunk_size)
        logger.success(f"LMI buckets created.")

        if return_time:
            return lmi, kmeanstime, trainmodeltime, modelingtime, encdatabasetime
        return lmi
