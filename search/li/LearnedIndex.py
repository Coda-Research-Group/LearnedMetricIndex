import time
from collections import defaultdict
from logging import DEBUG, INFO
from typing import Dict, List, Tuple, Optional

import faiss
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.utils.data
from li.Logger import Logger
from li.model import NeuralNetwork, data_X_to_torch
from li.PriorityQueue import EMPTY_VALUE, PriorityQueue
from li.utils import filter_path_idxs, log_runtime
from li.Bucket import Bucket, NaiveBucket, IVFBucket, IVFBucketFaiss, SketchBucket
from tqdm import tqdm

torch.manual_seed(2023)
np.random.seed(2023)


class LearnedIndex(Logger):
    root_model: NeuralNetwork
    internal_models: Dict[Tuple, NeuralNetwork]
    bucket_paths: List[Tuple]
    bucket_models: Optional[Dict[Tuple, Bucket]]

    def __init__(
        self,
        root_model: NeuralNetwork,
        internal_models: Dict[Tuple, NeuralNetwork],
        bucket_paths: List[Tuple],
    ):
        self.root_model = root_model
        """The rood model of the index."""

        self.internal_models = internal_models
        """
        Dictionary mapping the path to the internal model.
        A path is padded with `EMPTY_VALUE` to the right to match the length of the longest path.
        """

        self.bucket_paths = bucket_paths
        """List of paths to the buckets."""

        self.bucket_models = None

    def _create_buckets(
        self,
        data_navigation: pd.DataFrame,
        data_search: pd.DataFrame,
        data_prediction: pd.DataFrame,
        n_levels: int,
        bucket_type_name: str,
        mode: str,
        **kwargs,
    ) -> float:
        bucket_type = None
        if bucket_type_name == "naive":
            bucket_type = NaiveBucket
        elif bucket_type_name == "IVF":
            bucket_type = IVFBucket
        elif bucket_type_name == "IVFFaiss":
            bucket_type = IVFBucketFaiss
        elif bucket_type_name == "sketch":
            bucket_type = SketchBucket
        assert bucket_type is not None

        for level_to_search in range(1, n_levels + 1):
            data_navigation[f"category_L{level_to_search}"] = data_prediction[
                :, (level_to_search - 1)
            ]

        possible_bucket_paths = [
            f"category_L{level_to_search}" for level_to_search in range(1, n_levels + 1)
        ]

        bucket_models = {} if mode in ["build", "train"] else self.bucket_models
        assert bucket_models is not None

        total_time = 0.0

        for path, g in data_navigation.groupby(possible_bucket_paths):
            bucket_obj_indexes = g.index.to_numpy()
            data_for_this_bucket = data_search.loc[bucket_obj_indexes].to_numpy()
            kwargs["sketches"] = (
                None
                if "bucket_sketches" not in kwargs
                else kwargs["bucket_sketches"].loc[bucket_obj_indexes].to_numpy()
            )

            if mode == "build":
                bucket = bucket_type()
                total_time += bucket.build(
                    data_for_this_bucket, bucket_obj_indexes, **kwargs
                )
                bucket_models[path] = bucket

            elif mode == "train":
                bucket = bucket_type()
                total_time += bucket.train(data_for_this_bucket, **kwargs)
                bucket_models[path] = bucket

            elif mode == "add":
                bucket = bucket_models[path]
                total_time += bucket.add(
                    data_for_this_bucket, bucket_obj_indexes, **kwargs
                )

            else:
                assert False

        self.bucket_models = bucket_models

        if "sketches" in kwargs:
            kwargs.pop("sketches")

        return total_time

    @log_runtime(INFO, "Built buckets in: {}")
    def build_buckets(
        self,
        data_navigation: pd.DataFrame,
        data_search: pd.DataFrame,
        data_prediction: pd.DataFrame,
        n_levels: int,
        bucket_type_name: str = "naive",
        **kwargs,
    ) -> float:
        return self._create_buckets(
            data_navigation,
            data_search,
            data_prediction,
            n_levels,
            bucket_type_name,
            "build",
            **kwargs,
        )

    @log_runtime(INFO, "Trained buckets in: {}")
    def train_buckets(
        self,
        data_navigation: pd.DataFrame,
        data_search: pd.DataFrame,
        data_prediction: pd.DataFrame,
        n_levels: int,
        bucket_type_name: str = "naive",
        **kwargs,
    ) -> float:
        return self._create_buckets(
            data_navigation,
            data_search,
            data_prediction,
            n_levels,
            bucket_type_name,
            "train",
            **kwargs,
        )

    @log_runtime(INFO, "Added to buckets in: {}")
    def add_to_buckets(
        self,
        data_navigation: pd.DataFrame,
        data_search: pd.DataFrame,
        data_prediction: pd.DataFrame,
        n_levels: int,
        bucket_type_name: str = "naive",
        **kwargs,
    ) -> float:
        return self._create_buckets(
            data_navigation,
            data_search,
            data_prediction,
            n_levels,
            bucket_type_name,
            "add",
            **kwargs,
        )

    def search_with_buckets(
        self,
        queries_navigation: npt.NDArray[np.float32],
        queries_search: npt.NDArray[np.float32],
        n_categories: List[int],
        n_buckets: int = 1,
        k: int = 10,
        naive_order: bool = True,
        **kwargs,
    ) -> Tuple[npt.NDArray, npt.NDArray[np.uint32], Dict[str, float]]:
        """Searches for `k` nearest neighbors for each query in `queries`.
        Buckets must have previously been build with `build_buckets`

        Implementation details:
        - The search is done in two steps:
            1. The order in which the queries visit the buckets is precomputed.
            2. The queries are then searched in the `n_buckets` most similar buckets.

        Parameters
        ----------
        queries_navigation : npt.NDArray[np.float32]
            Queries used for navigation.
        queries_search : npt.NDArray[np.float32]
            Queries used for the sequential search.
        n_categories : List[int]
            Number of categories for each level of the index.
        n_buckets : int, optional
            Number of most similar buckets to search in, by default 1
        k : int, optional
            Number of nearest neighbors to search for, by default 10

        Returns
        -------
        Tuple[npt.NDArray, npt.NDArray[np.uint32], Dict[str, float]]
            Array of shape (queries_search.shape[0], k) with distances to nearest neighbors for each query,
            array of shape (queries_search.shape[0], k) with nearest neighbors for each query,
            dictionary with measured times.
        """
        assert self.bucket_models is not None

        measured_time = defaultdict(float)

        s = time.time()

        anns_final = None
        dists_final = None

        self.logger.debug("Precomputing bucket order")
        bucket_order, bucket_probability, measured_time["inference"] = (
            self._precompute_bucket_order(
                queries_navigation=queries_navigation,
                n_buckets=n_buckets,
                n_categories=n_categories,
                naive_order=naive_order,
            )
        )

        # Search in the `n_buckets` most similar buckets
        for bucket_order_idx in range(n_buckets):
            self.logger.debug(
                f"Searching in bucket {bucket_order_idx + 1} out of {n_buckets}"
            )
            (dists, anns, t_all, t_seq_search, t_sort, n_dis) = (
                self._search_single_bucket_built(
                    queries_search=queries_search,
                    bucket_path=bucket_order[:, bucket_order_idx, :],
                    k=k,
                    **kwargs,
                )
            )

            measured_time["search_within_buckets"] += t_all
            measured_time["seq_search"] += t_seq_search
            measured_time["sort"] += t_sort
            measured_time["distance_computations"] += n_dis

            self.logger.debug("Sorting the results")
            t = time.time()
            if anns_final is None:
                anns_final = anns
                dists_final = dists
            else:
                # stacks the results from the previous sorted anns and dists
                # *_final arrays now have shape (queries.shape[0], k*2)
                anns_final = np.hstack((anns_final, anns))
                dists_final = np.hstack((dists_final, dists))
                # gets the sorted indices of the stacked dists
                idx_sorted = dists_final.argsort(kind="stable", axis=1)[:, :k]
                # indexes the final arrays with the sorted indices
                # *_final arrays now have shape (queries.shape[0], k)
                idx = np.ogrid[tuple(map(slice, dists_final.shape))]
                idx[1] = idx_sorted
                dists_final = dists_final[tuple(idx)]
                anns_final = anns_final[tuple(idx)]

                assert (
                    anns_final.shape
                    == dists_final.shape
                    == (queries_search.shape[0], k)
                )
            self.logger.debug(f"Sorted the results in: {time.time() - t}")

        assert dists_final is not None
        assert anns_final is not None

        measured_time["search"] = time.time() - s

        return dists_final, anns_final, measured_time

    def search(
        self,
        data_navigation: pd.DataFrame,
        queries_navigation: npt.NDArray[np.float32],
        data_search: pd.DataFrame,
        queries_search: npt.NDArray[np.float32],
        data_prediction: npt.NDArray[np.int64],
        n_categories: List[int],
        n_buckets: int = 1,
        k: int = 10,
    ) -> Tuple[npt.NDArray, npt.NDArray[np.uint32], Dict[str, float]]:
        """Searches for `k` nearest neighbors for each query in `queries`.

        Implementation details:
        - The search is done in two steps:
            1. The order in which the queries visit the buckets is precomputed.
            2. The queries are then searched in the `n_buckets` most similar buckets.

        Parameters
        ----------
        data_navigation : pd.DataFrame
            Data used for navigation.
        queries_navigation : npt.NDArray[np.float32]
            Queries used for navigation.
        data_search : pd.DataFrame
            Data used for the sequential search.
        queries_search : npt.NDArray[np.float32]
            Queries used for the sequential search.
        data_prediction : npt.NDArray[np.int64]
            Predicted paths for each data point.
        n_categories : List[int]
            Number of categories for each level of the index.
        n_buckets : int, optional
            Number of most similar buckets to search in, by default 1
        k : int, optional
            Number of nearest neighbors to search for, by default 10

        Returns
        -------
        Tuple[npt.NDArray, npt.NDArray[np.uint32], Dict[str, float]]
            Array of shape (queries_search.shape[0], k) with distances to nearest neighbors for each query,
            array of shape (queries_search.shape[0], k) with nearest neighbors for each query,
            dictionary with measured times.
        """
        measured_time = defaultdict(float)

        s = time.time()
        n_levels = len(n_categories)

        anns_final = None
        dists_final = None

        self.logger.debug("Precomputing bucket order")
        bucket_order, bucket_probability, measured_time["inference"] = (
            self._precompute_bucket_order(
                queries_navigation=queries_navigation,
                n_buckets=n_buckets,
                n_categories=n_categories,
            )
        )

        # Add bucket location to each object as searching is done sequentially per bucket
        for level_to_search in range(1, n_levels + 1):
            data_navigation[f"category_L{level_to_search}"] = data_prediction[
                :, (level_to_search - 1)
            ]

        # Search in the `n_buckets` most similar buckets
        for bucket_order_idx in range(n_buckets):
            self.logger.debug(
                f"Searching in bucket {bucket_order_idx + 1} out of {n_buckets}"
            )
            (dists, anns, t_all, t_seq_search, t_sort, n_dis) = (
                self._search_single_bucket(
                    data_navigation=data_navigation,
                    data_search=data_search,
                    queries_search=queries_search,
                    bucket_path=bucket_order[:, bucket_order_idx, :],
                    n_levels=n_levels,
                )
            )

            measured_time["search_within_buckets"] += t_all
            measured_time["seq_search"] += t_seq_search
            measured_time["sort"] += t_sort
            measured_time["distance_computations"] += n_dis

            self.logger.debug("Sorting the results")
            t = time.time()
            if anns_final is None:
                anns_final = anns
                dists_final = dists
            else:
                # stacks the results from the previous sorted anns and dists
                # *_final arrays now have shape (queries.shape[0], k*2)
                anns_final = np.hstack((anns_final, anns))
                dists_final = np.hstack((dists_final, dists))
                # gets the sorted indices of the stacked dists
                idx_sorted = dists_final.argsort(kind="stable", axis=1)[:, :k]
                # indexes the final arrays with the sorted indices
                # *_final arrays now have shape (queries.shape[0], k)
                idx = np.ogrid[tuple(map(slice, dists_final.shape))]
                idx[1] = idx_sorted
                dists_final = dists_final[tuple(idx)]
                anns_final = anns_final[tuple(idx)]

                assert (
                    anns_final.shape
                    == dists_final.shape
                    == (queries_search.shape[0], k)
                )
            self.logger.debug(f"Sorted the results in: {time.time() - t}")

        assert dists_final is not None
        assert anns_final is not None

        # Cleanup the placement of the objects
        for level_idx_to_search in range(n_levels):
            column_name = f"category_L{level_idx_to_search + 1}"

            if column_name in data_navigation.columns:
                data_navigation.drop(column_name, axis=1, inplace=True)

        measured_time["search"] = time.time() - s

        return dists_final, anns_final, measured_time

    @log_runtime(INFO, "Precomputed bucket order time: {}")
    def _precompute_bucket_order(
        self,
        queries_navigation: npt.NDArray[np.float32],
        n_buckets: int,
        n_categories: List[int],
        naive_order: bool = True,
    ) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32], float]:
        """
        Precomputes the order in which the queries visit the buckets.

        Implementation details:
        - When visiting an internal node, the paths to the nodes/buckets under that node
        are added to the priority queue.
        - When visiting a bucket, the path to the bucket is stored in `bucket_order`.
        - The priority queue is then sorted by the probability of the next node/bucket to visit.
        - The computation is done until `n_buckets` buckets are visited for each query.

        Parameters
        ----------
        queries_navigation : np.ndarray
            Queries used for navigation.
        n_buckets : int
            Number of most similar buckets to precompute the order for.
        n_categories : List[int]
            Number of categories for each level of the index.

        Returns
        -------
        Tuple[npt.NDArray[np.int32], float]
            Array of shape (queries_navigation.shape[0], n_buckets, len(n_categories))
            with the order in which the queries visit the buckets,
            total inference time.
        """

        n_queries = queries_navigation.shape[0]
        n_levels = len(n_categories)
        assert self.root_model is not None, "Model is not trained, call `build` first."

        total_inference_t = 0.0

        s = time.time()
        pred_l1_prob, pred_l1_paths = self.root_model.predict_proba(
            data_X_to_torch(queries_navigation)
        )
        total_inference_t += time.time() - s

        if n_levels == 1:
            bucket_order = np.full(
                (n_queries, n_buckets, n_levels), fill_value=EMPTY_VALUE, dtype=np.int32
            )
            bucket_order[:, :n_buckets, 0] = pred_l1_paths[:, :n_buckets]
            return bucket_order, total_inference_t

        queue_length_upper_bound = int(np.prod(n_categories))
        pq = PriorityQueue(n_queries, queue_length_upper_bound, n_levels)

        # Populates the priority queue with the first level of the index
        # * Relies on the fact that the pred_l1_categories and pred_l1_probs are sorted,
        # * therefore the priority queue does not need to be sorted after this for loop
        for l1_idx in reversed(range(n_categories[0])):
            l1_paths = np.full(
                (n_queries, n_levels), fill_value=EMPTY_VALUE, dtype=np.int32
            )
            l1_paths[:, 0] = pred_l1_paths[:, l1_idx]
            pq.add(np.arange(n_queries), l1_paths, pred_l1_prob[:, l1_idx])

        bucket_order = np.full(
            (n_queries, n_buckets, n_levels), fill_value=EMPTY_VALUE, dtype=np.int32
        )
        bucket_order_length = np.zeros(n_queries, dtype=np.int32)
        bucket_probabilities = np.full(
            (n_queries, n_buckets), fill_value=np.NINF, dtype=np.float32
        )

        while not np.all(bucket_order_length == n_buckets):
            query_idxs = np.where(bucket_order_length < n_buckets)[0]
            path_to_visit, probability = pq.pop(query_idxs)

            inference_t = self._visit_internal_nodes(
                queries_navigation,
                query_idxs,
                pq,
                path_to_visit,
                probability,
                n_levels,
                naive_order=naive_order,
            )
            self._visit_buckets(
                query_idxs,
                path_to_visit,
                probability,
                bucket_order,
                bucket_order_length,
                bucket_probabilities,
            )

            total_inference_t += inference_t

            pq.sort()

        return bucket_order, bucket_probabilities, total_inference_t

    def _visit_internal_nodes(
        self,
        queries_navigation: npt.NDArray[np.float32],
        all_query_idxs: npt.NDArray[np.int32],
        pq: PriorityQueue,
        path_to_visit: npt.NDArray[np.int32],
        probability: npt.NDArray[np.float32],
        n_levels: int,
        naive_order: bool = True,
    ) -> float:
        """
        Visits the internal nodes specified by `paths`.
        Paths to the buckets under a specific internal node are then added into `pq`.
        Done for each possible path to an internal node separately.
        """
        total_inference_t = 0.0

        for path in self.internal_models.keys():
            path_idxs = filter_path_idxs(path_to_visit, path)
            query_idxs = all_query_idxs[path_idxs]
            if query_idxs.shape[0] == 0:
                continue

            model = self.internal_models[path]

            s = time.time()
            probabilities, categories = model.predict_proba(
                data_X_to_torch(queries_navigation[query_idxs])
            )
            total_inference_t += time.time() - s

            level = len(path) - path.count(EMPTY_VALUE)
            n_model_categories = categories.shape[1]

            for child_idx in range(n_model_categories):
                child_paths = np.full(
                    (query_idxs.shape[0], n_levels),
                    fill_value=EMPTY_VALUE,
                    dtype=np.int32,
                )
                child_paths[:] = np.array(path)
                child_paths[:, level] = categories[:, child_idx]

                probabilites_to_add = (
                    probabilities[:, child_idx]
                    if naive_order
                    else probabilities[:, child_idx] * probability[path_idxs]
                )

                pq.add(
                    query_idxs,
                    child_paths,
                    probabilites_to_add,
                )

        return total_inference_t

    def _visit_buckets(
        self,
        all_query_idxs: npt.NDArray[np.int32],
        path_to_visit: npt.NDArray[np.int32],
        probability: npt.NDArray[np.float32],
        bucket_order: npt.NDArray[np.int32],
        bucket_order_length: npt.NDArray[np.int32],
        bucket_probabilities: npt.NDArray[np.float32],
    ) -> None:
        """
        Visits the buckets specified by `paths`.
        The path to the bucket relevant to each query is then stored in `bucket_order`.
        The probability of each bucket is then stored in `bucket_probabilities`.
        Done for each possible bucket separately.
        """
        for path in self.bucket_paths:
            path_idxs = filter_path_idxs(path_to_visit, path)
            query_idxs = all_query_idxs[path_idxs]

            if query_idxs.shape[0] == 0:
                continue

            bucket_order[query_idxs, bucket_order_length[query_idxs], :] = np.array(
                path
            )
            bucket_probabilities[query_idxs, bucket_order_length[query_idxs]] = (
                np.array(probability[path_idxs])
            )
            bucket_order_length[query_idxs] += 1

    @log_runtime(DEBUG, "Searched the buckets in: {}")
    def _search_single_bucket(
        self,
        data_navigation: pd.DataFrame,
        data_search: pd.DataFrame,
        queries_search: npt.NDArray[np.float32],
        bucket_path: npt.NDArray[np.int32],
        k: int = 10,
        n_levels: int = 1,
    ) -> Tuple[npt.NDArray, npt.NDArray[np.uint32], float, float, float, int]:
        s_all = time.time()

        n_queries = queries_search.shape[0]
        nns = np.zeros((n_queries, k), dtype=np.uint32)
        dists = np.full((n_queries, k), fill_value=float("inf"), dtype=float)

        possible_bucket_paths = [
            f"category_L{level_to_search}" for level_to_search in range(1, n_levels + 1)
        ]

        t_seq_search = 0.0
        t_sort = 0.0
        n_dis_total = 0

        for path, g in tqdm(data_navigation.groupby(possible_bucket_paths)):
            bucket_obj_indexes = g.index

            relevant_query_idxs = filter_path_idxs(bucket_path, path)

            if bucket_obj_indexes.shape[0] != 0 and relevant_query_idxs.shape[0] != 0:
                queries_for_this_bucket = queries_search[relevant_query_idxs]
                data_in_this_bucket = data_search.loc[bucket_obj_indexes].to_numpy()

                s = time.time()
                similarity, indices = faiss.knn(
                    queries_for_this_bucket,
                    data_in_this_bucket,
                    k,
                    metric=faiss.METRIC_INNER_PRODUCT,
                )
                t_seq_search += time.time() - s

                distances = 1 - similarity

                n_dis_total += len(queries_for_this_bucket) * len(data_in_this_bucket)

                nns[relevant_query_idxs] = bucket_obj_indexes.to_numpy()[indices]
                dists[relevant_query_idxs] = distances

        return dists, nns, time.time() - s_all, t_seq_search, t_sort, n_dis_total

    @log_runtime(DEBUG, "Searched the buckets in: {}")
    def _search_single_bucket_built(
        self,
        queries_search: npt.NDArray[np.float32],
        bucket_path: npt.NDArray[np.int32],
        k: int = 10,
        **kwargs,
    ) -> Tuple[npt.NDArray, npt.NDArray[np.uint32], float, float, float, int]:
        assert self.bucket_models is not None
        s_all = time.time()

        n_queries = queries_search.shape[0]
        nns = np.zeros((n_queries, k), dtype=np.uint32)
        dists = np.full((n_queries, k), fill_value=float("inf"), dtype=float)

        t_seq_search = 0.0
        t_sort = 0.0
        n_dis_total = 0

        queries_sketch = kwargs.get("queries_sketch", None)

        for path, bucket in tqdm(self.bucket_models.items()):
            relevant_query_idxs = filter_path_idxs(bucket_path, path)

            if relevant_query_idxs.shape[0] == 0:
                continue

            queries_for_this_bucket = queries_search[relevant_query_idxs]
            relevant_sketches = (
                None if queries_sketch is None else queries_sketch[relevant_query_idxs]
            )

            indices, distances, t_search, n_dis = bucket.search(
                queries_for_this_bucket,
                k,
                **kwargs,
                sketches=relevant_sketches,
            )
            t_seq_search += t_search
            n_dis_total += n_dis

            # return indexes to original dataset
            nns[relevant_query_idxs] = bucket.original_idxs[indices]
            dists[relevant_query_idxs] = distances

        return dists, nns, time.time() - s_all, t_seq_search, t_sort, n_dis_total
