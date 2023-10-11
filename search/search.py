import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import urlretrieve

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from li.Baseline import Baseline
from li.BuildConfiguration import BuildConfiguration
from li.clustering import algorithms
from li.LearnedIndexBuilder import LearnedIndexBuilder
from li.utils import save_as_pickle, serialize
from sklearn import preprocessing

np.random.seed(2023)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5.5s][%(name)-.20s] %(message)s",
)
LOG = logging.getLogger(__name__)

MODELS_DIR_NAME = "models"


def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        LOG.info("downloading %s -> %s..." % (src, dst))
        urlretrieve(src, dst)


def prepare(kind, size):
    url = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge"
    task = {
        "query": f"{url}/public-queries-10k-{kind}.h5",
        "dataset": f"{url}/laion2B-en-{kind}-n={size}.h5",
    }

    for version, url in task.items():
        target_path = os.path.join("data", kind, size, f"{version}.h5")
        download(url, target_path)
        assert os.path.exists(target_path), f"Failed to download {url}"


def store_results(dst, algo, kind, dists, anns, buildtime, querytime, params, size):
    os.makedirs(Path(dst).parent, exist_ok=True)
    LOG.info(f"Storing results in {dst}")
    f = h5py.File(dst, "w")
    f.attrs["algo"] = algo
    f.attrs["data"] = kind
    f.attrs["buildtime"] = buildtime
    f.attrs["querytime"] = querytime
    f.attrs["size"] = size
    f.attrs["params"] = params
    f.create_dataset("knns", anns.shape, dtype=anns.dtype)[:] = anns
    f.create_dataset("dists", dists.shape, dtype=dists.dtype)[:] = dists
    f.close()


def format_identifier(
    bucket: int,
    kind: str,
    config: BuildConfiguration,
    clustering_algorithms: List[str],
    short_identifier: str,
    size: str,
):
    return (
        f"{short_identifier}"
        f"-{kind}"
        f"-{size}"
        f"-ep={serialize(config.epochs)}"
        f"-lr={serialize(config.lrs)}"
        f"-cat={serialize(config.n_categories)}"
        f"-model={serialize(config.model_types)}"
        f"-buck={bucket}"
        f"-clustering_algorithm={serialize(clustering_algorithms)}"
        f"-{os.environ['PBS_JOBID']}"
    )


def format_models_filename(
    kind: str,
    config: BuildConfiguration,
    clustering_algorithms: List[str],
    preprocess: bool,
    size: str,
):
    return (
        f"./{MODELS_DIR_NAME}/{kind}"
        f"-{size}"
        f"-ep={serialize(config.epochs)}"
        f"-lr={serialize(config.lrs)}"
        f"-cat={serialize(config.n_categories)}"
        f"-model={serialize(config.model_types)}"
        f"-prep={preprocess}"
        f"-clustering_algorithm={serialize(clustering_algorithms)}"
        f"-{os.environ['PBS_JOBID']}"
    )


def run(
    kind: str,
    key: str,
    size: str,
    k: int,
    index_type: str,
    n_buckets_perc: List[int],
    n_categories: List[int],
    epochs: List[int],
    model_types: List[str],
    lr: List[float],
    preprocess: bool,
    save: bool,
    clustering_algorithms: List[str],
):
    assert index_type in {
        "baseline",
        "learned-index",
    }, f"Unknown index type: {index_type}"

    LOG.info(
        f"Running with: kind={kind}, key={key}, size={size}, n_buckets_perc={n_buckets_perc},"
        f" n_categories={n_categories}, clustering_algorithms={clustering_algorithms},"
        f" epochs={epochs}, lr={lr}, model_types={model_types}, preprocess={preprocess}, save={save}"
    )

    prepare(kind, size)

    data: npt.NDArray[np.float32] = np.array(
        h5py.File(os.path.join("data", kind, size, "dataset.h5"), "r")[key]
    )
    queries: npt.NDArray[np.float32] = np.array(
        h5py.File(os.path.join("data", kind, size, "query.h5"), "r")[key]
    )
    if preprocess:
        data = preprocessing.normalize(data)  # type: ignore
        queries = preprocessing.normalize(queries)  # type: ignore

    n, d = data.shape
    LOG.info(f"Loaded downloaded data, shape: n={n}, d={d}")
    LOG.info(f"Loaded downloaded queries, shape: queries={queries.shape}")

    if index_type == "baseline":
        baseline = Baseline()
        build_t = baseline.build(data)
        LOG.info(f"Build time: {build_t}")
        dists, nns, search_t = baseline.search(queries=queries, data=data, k=k)
    elif index_type == "learned-index":
        evaluate_learned_index(
            data,
            clustering_algorithms,
            epochs,
            model_types,
            lr,
            k,
            kind,
            n_buckets_perc,
            n_categories,
            preprocess,
            queries,
            save,
            size,
        )


def evaluate_learned_index(
    data: npt.NDArray[np.float32],
    clustering_algorithms: List[str],
    epochs: List[int],
    model_type: List[str],
    lr: List[float],
    k: int,
    kind: str,
    n_buckets_perc: List[int],
    n_categories: List[int],
    preprocess: bool,
    queries: npt.NDArray[np.float32],
    save: bool,
    size: str,
):
    s = time.time()
    # ---- data to pd.DataFrame ---- #
    data_pd = pd.DataFrame(data)
    data_pd.index += 1
    kind_search = "clip768v2"
    key_search = "emb"
    if kind != kind_search:
        LOG.info("Loading data to be used in search")
        prepare(kind_search, size)
        # ---- data_search to pd.DataFrame ---- #
        data_search = pd.DataFrame(
            np.array(
                h5py.File(os.path.join("data", kind_search, size, "dataset.h5"), "r")[
                    key_search
                ]
            )
        )
        data_search.index += 1
        queries_search = np.array(
            h5py.File(os.path.join("data", kind_search, size, "query.h5"), "r")[
                key_search
            ]
        )
        n, d = data_search.shape
        LOG.info(f"Loaded downloaded data, shape: n={n}, d={d}")
        LOG.info(f"Loaded downloaded queries, shape: queries={queries_search.shape}")
    else:
        data_search = data_pd
        queries_search = queries
    # ---- prepare index configuration ---- #
    config = BuildConfiguration(
        [algorithms[algo] for algo in clustering_algorithms],
        epochs,
        model_type,
        lr,
        n_categories,
    )
    # ---- instantiate the index builder ---- #
    builder = LearnedIndexBuilder(data_pd, config)
    # ---- build the index ---- #
    li, data_prediction, n_buckets_in_index, build_t, cluster_t = builder.build()
    LOG.info(f"Total number of buckets in the index: {n_buckets_in_index}")
    LOG.info(f"Cluster time: {cluster_t}")
    LOG.info(f"Pure build time: {build_t}")
    LOG.info(f"Overall build time: {time.time() - s}")

    if save:
        if not os.path.isdir(MODELS_DIR_NAME):
            os.mkdir(MODELS_DIR_NAME)
        save_filename = format_models_filename(
            kind, config, clustering_algorithms, preprocess, size
        )
        LOG.info(f"Saving as {save_filename}")
        save_as_pickle(f"{save_filename}.pkl", li)

    n_buckets = [int((p / 100) * n_buckets_in_index) for p in n_buckets_perc]
    n_buckets = list(set([b for b in n_buckets if b > 0]))
    LOG.info(f"Number of buckets to search in: {n_buckets}")

    for bucket in n_buckets:
        LOG.info(f"Searching with {bucket} buckets")

        dists, nns, measured_time = li.search(
            data_navigation=data_pd,
            queries_navigation=queries,
            data_search=data_search,
            queries_search=queries_search,
            data_prediction=data_prediction,
            n_categories=n_categories,
            n_buckets=bucket,
            k=k,
        )

        LOG.info(f"Inference time: {measured_time['inference']}")
        LOG.info(f"Search time: {measured_time['search']}")
        LOG.info(
            f"Search within buckets time: {measured_time['search_within_buckets']}"
        )
        LOG.info(f"Sequential search time: {measured_time['seq_search']}")
        LOG.info(f"Sort time: {measured_time['sort']}")

        short_identifier = "learned-index"
        identifier = format_identifier(
            bucket, kind, config, clustering_algorithms, short_identifier, size
        )
        store_results(
            os.path.join("result/", kind, size, f"{identifier}.h5"),
            short_identifier.capitalize(),
            kind,
            dists,
            nns,
            build_t,
            measured_time["search"],
            identifier,
            size,
        )


def expand(array: List[Any], size: int):
    """Expands an array of size one to the given size."""
    assert len(array) == 1
    return [array[0]] * size


def validate_and_expand_per_level_arguments(args: Dict[str, Any]):
    """
    Validates that the arguments that are per-level are either of size 1
    and expanded to the size of n_categories or of the same size as n_categories.
    """
    arguments = ["clustering_algorithm", "model_type", "epochs", "lr", "n_categories"]

    for arg in arguments:
        if len(args[arg]) == 1:
            args[arg] = expand(args[arg], len(args["n_categories"]))
        else:
            assert len(args[arg]) == len(args["n_categories"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pca96v2")
    parser.add_argument("--emb", default="pca96")
    parser.add_argument(
        "--size", default="100K", choices=["100K", "300K", "10M", "30M", "100M"]
    )
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--n-categories", nargs="+", default=[123], type=int)
    parser.add_argument("--epochs", nargs="+", default=[200], type=int)
    parser.add_argument("--model-type", nargs="+", default=["MLP"])
    parser.add_argument("--lr", nargs="+", default=[0.01], type=float)
    parser.add_argument("-b", "--n-buckets", nargs="+", default=[2, 3, 4], type=int)
    parser.add_argument("-bp", "--buckets-perc", nargs="+", default=[4], type=int)
    parser.add_argument("--preprocess", default=True, type=bool)
    parser.add_argument("--save", default=True, type=bool)
    parser.add_argument(
        "--clustering-algorithm",
        nargs="+",
        default=["faiss_kmeans"],
        choices=algorithms.keys(),
    )
    args = parser.parse_args()

    validate_and_expand_per_level_arguments(vars(args))

    if args.save and "PBS_JOBID" not in os.environ:
        os.environ["PBS_JOBID"] = "unknown"

    run(
        args.dataset,
        args.emb,
        args.size,
        args.k,
        "learned-index",
        args.buckets_perc,
        args.n_categories,
        args.epochs,
        args.model_type,
        args.lr,
        args.preprocess,
        args.save,
        args.clustering_algorithm,
    )
