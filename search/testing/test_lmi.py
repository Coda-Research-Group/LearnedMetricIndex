from li.utils import load_from_pickle
from typing import Tuple, List, Optional
from li.BuildConfiguration import BuildConfiguration
from li.LearnedIndex import LearnedIndex
import numpy as np
import numpy.typing as npt
import pandas as pd
from dataclasses import dataclass
from testing.prepare_data import (
    get_dataset,
    get_dataset_normalized,
    get_queries,
    get_queries_normalized,
    get_groundtruth_idxs,
)
import os
import argparse

TEST_RESULTS_DIR = "./test_results"


@dataclass
class Data:
    navigation: npt.ArrayLike
    search: npt.ArrayLike
    sketch: Optional[npt.ArrayLike]


@dataclass
class Tester:
    li: LearnedIndex
    config: BuildConfiguration
    data_prediction: npt.ArrayLike
    n_buckets_in_index: int
    n_buckets: List[int]
    k: int
    naive_order: bool
    dynamic: bool

    def __call__(
        self, data: Data, queries: Data, groundtruth: npt.ArrayLike
    ) -> pd.DataFrame:
        raise NotImplementedError()


@dataclass
class IVFTester(Tester):
    nlist: List[int]
    nprobe: List[int]
    count_distance_computations: bool

    @property
    def bucket_type(self) -> str:
        return "IVF" if self.count_distance_computations else "IVFFaiss"

    def __call__(
        self, data: Data, queries: Data, groundtruth: npt.ArrayLike
    ) -> pd.DataFrame:
        # indexing from 1
        data_navigation = pd.DataFrame(data.navigation)
        data_navigation.index += 1

        data_search = pd.DataFrame(data.search)
        data_search.index += 1

        results = []

        for nlist in self.nlist:
            build_t = self.li.build_buckets(
                data_navigation,
                data_search,
                self.data_prediction,
                len(self.config.n_categories),
                self.bucket_type,
                nlist=nlist,
            )

            for nprobe in self.nprobe:
                if nprobe > nlist:
                    continue

                for n_buckets in self.n_buckets:
                    distances, nns, measured_time = self.li.search_with_buckets(
                        queries.navigation,
                        queries.search,
                        self.config.n_categories,
                        n_buckets,
                        self.k,
                        self.naive_order,
                        self.dynamic,
                        nprobe=nprobe,
                    )
                    recall = get_recall(nns, groundtruth, self.k)
                    results.append(
                        [
                            n_buckets,
                            nlist,
                            nprobe,
                            build_t,
                            *measured_time.values(),
                            recall,
                        ]
                    )

        return pd.DataFrame(
            results,
            columns=[
                "n_buckets",
                "nlist",
                "nprobe",
                "bucket_build_t",
                "inference",
                "search_within_buckets",
                "seq_search",
                "sort",
                "distance_computations",
                "search",
                "recall",
            ],
        )


@dataclass
class NaiveTester(Tester):
    def __call__(
        self, data: Data, queries: Data, groundtruth: npt.ArrayLike
    ) -> pd.DataFrame:
        # indexing from 1
        data_navigation = pd.DataFrame(data.navigation)
        data_navigation.index += 1

        data_search = pd.DataFrame(data.search)
        data_search.index += 1

        results = []

        for n_buckets in self.n_buckets:
            build_t = self.li.build_buckets(
                data_navigation,
                data_search,
                self.data_prediction,
                len(self.config.n_categories),
                "naive",
            )
            distances, nns, measured_time = self.li.search_with_buckets(
                queries.navigation,
                queries.search,
                self.config.n_categories,
                n_buckets,
                self.k,
                self.naive_order,
            )
            recall = get_recall(nns, groundtruth, self.k)
            results.append(
                [
                    n_buckets,
                    build_t,
                    *measured_time.values(),
                    recall,
                ]
            )

        return pd.DataFrame(
            results,
            columns=[
                "n_buckets",
                "build_t",
                "inference",
                "search_within_buckets",
                "seq_search",
                "sort",
                "distance_computations",
                "search",
                "recall",
            ],
        )

@dataclass
class NoBucketsTester(Tester):
    def __call__(
        self, data: Data, queries: Data, groundtruth: npt.ArrayLike
    ) -> pd.DataFrame:
        # indexing from 1
        data_navigation = pd.DataFrame(data.navigation)
        data_navigation.index += 1

        data_search = pd.DataFrame(data.search)
        data_search.index += 1

        results = []

        for n_buckets in self.n_buckets:
            distances, nns, measured_time = self.li.search(
                data_navigation,
                queries.navigation,
                data_search,
                queries.search,
                self.data_prediction,
                self.config.n_categories,
                n_buckets,
                self.k,
            )
            recall = get_recall(nns, groundtruth, self.k)
            results.append(
                [
                    n_buckets,
                    *measured_time.values(),
                    recall,
                ]
            )

        return pd.DataFrame(
            results,
            columns=[
                "n_buckets",
                "inference",
                "search_within_buckets",
                "seq_search",
                "sort",
                "distance_computations",
                "search",
                "recall",
            ],
        )


@dataclass
class SketchTester(Tester):
    c: List[int]

    def __call__(
        self,
        data: Data,
        queries: Data,
        groundtruth: npt.ArrayLike,
    ) -> pd.DataFrame:
        assert data.sketch is not None and queries.sketch is not None

        # indexing from 1
        data_navigation = pd.DataFrame(data.navigation)
        data_navigation.index += 1

        data_search = pd.DataFrame(data.search)
        data_search.index += 1

        data_sketch = pd.DataFrame(data.sketch)
        data_sketch.index += 1

        results = []

        build_t = self.li.build_buckets(
            data_navigation,
            data_search,
            self.data_prediction,
            len(self.config.n_categories),
            "sketch",
            bucket_sketches=data_sketch,
        )
        for n_buckets in self.n_buckets:
            for c in self.c:
                distances, nns, measured_time = self.li.search_with_buckets(
                    queries.navigation,
                    queries.search,
                    self.config.n_categories,
                    n_buckets,
                    self.k,
                    self.naive_order,
                    self.dynamic,
                    c=c,
                    queries_sketch=queries.sketch,
                )
                recall = get_recall(nns, groundtruth, self.k)
                results.append(
                    [
                        n_buckets,
                        c,
                        build_t,
                        *measured_time.values(),
                        recall,
                    ]
                )

        return pd.DataFrame(
            results,
            columns=[
                "n_buckets",
                "c",
                "build_t",
                "inference",
                "search_within_buckets",
                "seq_search",
                "sort",
                "distance_computations",
                "search",
                "recall",
            ],
        )


def load_lmi(
    path: str,
) -> Tuple[BuildConfiguration, LearnedIndex, npt.NDArray[np.int64], int]:
    return load_from_pickle(path)


def get_recall(I, gt, k):
    assert k <= I.shape[1]
    assert len(I) == len(gt)

    n = len(I)
    recall = 0
    for i in range(n):
        recall += len(set(I[i, :k]) & set(gt[i, :k]))
    return recall / (n * k)


def format_filename(
    type_navigation: str,
    type_search: str,
    size: str,
    bucket_type: str,
    k: int,
    naive_priority_queue: bool,
    dynamic: bool,
) -> str:
    return (
        f"nav={type_navigation}_"
        f"search={type_search}_"
        f"size={size}_"
        f"bucket={bucket_type}_"
        f"k={k}_"
        f"naive-pq={naive_priority_queue}_"
        f"dynamic={dynamic}"
        ".csv"
    )


def main(
    path: str,
    bucket_type: str,
    n_buckets: List[int],
    k: int,
    dataset: str,
    type_navigation: str,
    type_search: str,
    size: str,
    naive_priority_queue: bool,
    dynamic: bool,
    append_results: bool,
    **kwargs,
) -> None:
    assert os.path.exists(TEST_RESULTS_DIR)

    config, li, data_prediction, n_buckets_in_index = load_lmi(path)

    tester = None
    if bucket_type == "naive":
        tester = NaiveTester(
            li,
            config,
            data_prediction,
            n_buckets_in_index,
            n_buckets,
            k,
            naive_priority_queue,
            dynamic,
        )
    if bucket_type == "none":
        tester = NoBucketsTester(
            li,
            config,
            data_prediction,
            n_buckets_in_index,
            n_buckets,
            k,
            naive_priority_queue,
            dynamic,
        )
    elif bucket_type in ["IVF", "IVFFaiss"]:
        count_dc = bucket_type == "IVF"
        tester = IVFTester(
            li,
            config,
            data_prediction,
            n_buckets_in_index,
            n_buckets,
            k,
            naive_priority_queue,
            dynamic,
            kwargs["nlist"],
            kwargs["nprobe"],
            count_dc,
        )

    elif bucket_type == "sketch":
        tester = SketchTester(
            li,
            config,
            data_prediction,
            n_buckets_in_index,
            n_buckets,
            k,
            naive_priority_queue,
            dynamic,
            kwargs["c"],
        )
    else:
        assert False
    assert tester is not None

    data = Data(
        get_dataset(dataset, type_navigation, size),
        get_dataset_normalized(dataset, type_search, size),
        get_dataset(dataset, "hammingv2", size) if dataset != "proteins" else None,
    )
    queries = Data(
        get_queries(dataset, type_navigation),
        get_queries_normalized(dataset, type_search),
        get_queries(dataset, "hammingv2") if dataset != "proteins" else None,
    )
    groundtruth = get_groundtruth_idxs(dataset, size)

    result = tester(data, queries, groundtruth)
    result_path = os.path.join(
        TEST_RESULTS_DIR,
        format_filename(
            type_navigation,
            type_search,
            size,
            bucket_type,
            k,
            naive_priority_queue,
            dynamic,
        ),
    )

    if append_results and os.path.exists(result_path):
        existing = pd.read_csv(result_path)
        result = pd.concat([existing, result])

    result.to_csv(result_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type-navigation", default="pca32v2")
    parser.add_argument("--type-search", default="clip768v2")
    parser.add_argument(
        "--size", default="100K", choices=["100K", "300K", "10M", "30M", "70M", "100M"]
    )
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--dataset", default="sisap23")
    parser.add_argument(
        "--load-path",
        default="./models/pca32v2-100K-ep=100,100-lr=0.01,0.01-cat=10,10-model=MLP,MLP-clustering_algorithm=faiss_kmeans,faiss_kmeans.pkl",
    )
    parser.add_argument("--naive-priority-queue", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--bucket-type", default="IVF")
    parser.add_argument("--n-buckets", nargs="+", default=[50], type=int)
    parser.add_argument("--nlist", nargs="+", default=[50], type=int)
    parser.add_argument("--nprobe", nargs="+", default=[20], type=int)
    parser.add_argument("--c", nargs="+", default=[1000], type=int)
    parser.add_argument("--append-results", action="store_true")

    args = parser.parse_args()

    main(
        args.load_path,
        args.bucket_type,
        args.n_buckets,
        args.k,
        args.dataset,
        args.type_navigation,
        args.type_search,
        args.size,
        args.naive_priority_queue,
        args.dynamic,
        args.append_results,
        nlist=args.nlist,
        nprobe=args.nprobe,
        c=args.c,
    )
