from __future__ import annotations

import os
import torch

# Set the LD_LIBRARY_PATH dynamically
libtorch_path = os.path.join(torch.__path__[0], "lib")
if libtorch_path not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = (
        libtorch_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )


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
from tqdm import tqdm

import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple


class LMI:
    def __init__(self, model, *args, **kwargs):
        model_config = extract_model_config(model)
        self._inner = LMIBase(model_config, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @staticmethod
    @measure_runtime
    def _run_kmeans(
        n_buckets: int, dimensionality: int, X: torch.Tensor, n_iter_kmeans: int
    ) -> torch.Tensor:
        return LMIBase._run_kmeans(n_buckets, dimensionality, X, n_iter_kmeans)

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
        total_counts = self._inner._count_bucket_sizes(dataset, n_data, chunk_size)
        gc.collect()
        return self._inner._create_buckets_scalable(
            dataset, n_data, chunk_size, total_counts
        )

    @staticmethod
    def init_logging():
        LMIBase.init_logging()

    @staticmethod
    def _dot_product_scalar(q: torch.Tensor, d: torch.Tensor) -> float:
        return LMIBase.dot_product_scalar(q, d)

    @staticmethod
    def _dot_product_avx2(q: torch.Tensor, d: torch.Tensor) -> float:
        return LMIBase.dot_product_avx2(q, d)

    @staticmethod
    def _dot_product_avx2_fma(q: torch.Tensor, d: torch.Tensor) -> float:
        return LMIBase.dot_product_avx2_fma(q, d)

    @staticmethod
    def _dot_product_avx2_reg_sum(q: torch.Tensor, d: torch.Tensor) -> float:
        return LMIBase.dot_product_avx2_reg_sum(q, d)

    @staticmethod
    def _dot_product_avx2_fma_reg_sum(q: torch.Tensor, d: torch.Tensor) -> float:
        return LMIBase.dot_product_avx2_fma_reg_sum(q, d)

    @staticmethod
    def _dot_product_avx512(q: torch.Tensor, d: torch.Tensor) -> float:
        return LMIBase.dot_product_avx512(q, d)

    @staticmethod
    def _dot_product_f32_f16_avx2(q: torch.Tensor, d: torch.Tensor) -> float:
        return LMIBase.dot_product_f32_f16_avx2(q, d)

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

    def _rerank_single_query(self, args_tuple):
        (
            query_idx,
            full_dim_query_vector,
            candidate_indices_for_query,
            candidate_dists_for_query,
            original_dataset_path_str,
            final_k,
            num_candidates_for_rerank,
        ) = args_tuple

        if (candidate_dists_for_query > 0).sum() == 0:
            return (
                query_idx,
                torch.full((final_k,), -1, dtype=torch.int64),
                torch.full((final_k,), float("-inf"), dtype=torch.float32),
            )

        candidate_dists_topk, candidate_idx_topk = torch.topk(
            candidate_dists_for_query, num_candidates_for_rerank
        )
        valid_mask = candidate_dists_topk > 0
        filtered_original_indices = candidate_indices_for_query[
            candidate_idx_topk[valid_mask]
        ]

        if len(filtered_original_indices) == 0:
            return (
                query_idx,
                torch.full((final_k,), -1, dtype=torch.int64),
                torch.full((final_k,), float("-inf"), dtype=torch.float32),
            )

        candidates_to_load = torch.sort(filtered_original_indices)[0]

        real_data = utils.load_real_data(
            original_dataset_path_str, candidates_to_load
        ).reshape(-1, full_dim_query_vector.shape[0])

        if real_data.shape[0] == 0:
            return (
                query_idx,
                torch.full((final_k,), -1, dtype=torch.int64),
                torch.full((final_k,), float("-inf"), dtype=torch.float32),
            )

        reranked_local_indices, reranked_distances = self._inner.search_batch(
            real_data, full_dim_query_vector, final_k
        )

        valid_mask_rerank = reranked_local_indices != -1
        valid_local_indices = reranked_local_indices[valid_mask_rerank]

        final_indices = candidates_to_load[valid_local_indices]

        final_distances = reranked_distances[valid_mask_rerank]

        num_found = final_indices.shape[0]
        if num_found < final_k:
            padding_indices = torch.full(
                (final_k - num_found,),
                -1,
                dtype=torch.int64,
                device=final_indices.device,
            )
            padding_dists = torch.full(
                (final_k - num_found,),
                float("-inf"),
                dtype=torch.float32,
                device=final_distances.device,
            )
            final_indices = torch.cat((final_indices, padding_indices))
            final_distances = torch.cat((final_distances, padding_dists))

        return query_idx, final_indices, final_distances

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

        all_candidate_indices, all_candidate_dists = self._inner.search(
            full_dim_queries,
            num_candidates_for_rerank,
            nprobe_stage1,
            transformed_queries,
        )
        print(
            f"Stage 1 candidate shapes: indices={all_candidate_indices.shape}, dists={all_candidate_dists.shape}"
        )

        n_queries = full_dim_queries.shape[0]
        final_indices_all = torch.zeros((n_queries, final_k), dtype=torch.int64)
        final_distances_all = torch.zeros((n_queries, final_k), dtype=torch.float32)

        tasks = []
        for i in range(n_queries):
            actual_num_cand_for_query = min(
                num_candidates_for_rerank,
                (all_candidate_dists[i] > float("-inf")).sum().item(),
            )
            if actual_num_cand_for_query == 0:
                final_indices_all[i].fill_(-1)
                final_distances_all[i].fill_(float("-inf"))
                continue

            tasks.append(
                (
                    i,
                    full_dim_queries[i],
                    all_candidate_indices[i, :actual_num_cand_for_query],
                    all_candidate_dists[i, :actual_num_cand_for_query],
                    original_dataset_path_str,
                    final_k,
                    actual_num_cand_for_query,
                )
            )

        with ThreadPoolExecutor(max_workers=None) as executor:
            results = list(
                tqdm(
                    executor.map(self._rerank_single_query, tasks),
                    total=len(tasks),
                    desc="Reranking queries",
                )
            )

        for query_idx, f_indices, f_dists in results:
            final_indices_all[query_idx] = f_indices
            final_distances_all[query_idx] = f_dists

        if return_time:
            return (final_indices_all, final_distances_all), encqueriestime
        return final_indices_all, final_distances_all

    @staticmethod
    def create(
        dataset: Path,
        epochs: int,
        lr: float,
        sample_size: int,
        n_buckets: int,
        chunk_size: int,
        n_iter_kmeans: int = 25,
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
        y_train = LMI._run_kmeans(n_buckets, data_dim_original, X_train, n_iter_kmeans)
        kmeanstime = time.time() - start
        logger.success(
            f"K-Means completed in {kmeanstime:.2f} seconds. Labels shape: {y_train.shape}"
        )

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
        gc.collect()

        if return_time:
            return lmi, kmeanstime, trainmodeltime, modelingtime, encdatabasetime
        return lmi
