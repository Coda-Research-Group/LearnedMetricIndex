from li.LearnedIndexBuilder import LearnedIndexBuilder, BuildConfiguration
from li.LearnedIndex import LearnedIndex
import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Dict, Any
from li.utils import save_as_pickle, serialize
import os
from testing.prepare_data import get_dataset_normalized
from li.clustering import algorithms
import argparse
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5.5s][%(name)-.20s] %(message)s",
)
LOG = logging.getLogger(__name__)

MODELS_DIR = "./models"


def build_lmi(
    data: npt.NDArray[np.float32], config: BuildConfiguration
) -> Tuple[LearnedIndex, npt.NDArray[np.int64], int, float, float]:
    data_pd = pd.DataFrame(data)
    data_pd.index += 1
    builder = LearnedIndexBuilder(data_pd, config)
    return builder.build()


def format_models_filename(
    kind: str,
    config: BuildConfiguration,
    clustering_algorithms: List[str],
    size: str,
):
    return (
        f"{kind}"
        f"-{size}"
        f"-ep={serialize(config.epochs)}"
        f"-lr={serialize(config.lrs)}"
        f"-cat={serialize(config.n_categories)}"
        f"-model={serialize(config.model_types)}"
        f"-clustering_algorithm={serialize(clustering_algorithms)}"
        ".pkl"
    )


def save_lmi(
    path: str,
    config: BuildConfiguration,
    li: LearnedIndex,
    data_prediction: npt.NDArray[np.int64],
    n_buckets_in_index: int,
) -> None:
    save_as_pickle(path, (config, li, data_prediction, n_buckets_in_index))


def main(
    type: str,
    size: str,
    dataset: str,
    clustering_algorithms: List[str],
    epochs: List[int],
    model_type: List[str],
    lr: List[float],
    n_categories: List[int],
):
    LOG.setLevel(logging.DEBUG)
    LOG.info("Loading data")
    data = get_dataset_normalized(dataset, type, size)
    n, d = data.shape
    LOG.info(f"Loaded data, shape: n={n}, d={d}")

    config = BuildConfiguration(
        [algorithms[algo] for algo in clustering_algorithms],
        epochs,
        model_type,
        lr,
        n_categories,
    )

    LOG.info("Building LMI")
    li, data_prediction, n_buckets_in_index, build_t, cluster_t = build_lmi(
        data, config
    )
    LOG.info(f"Total number of buckets in the index: {n_buckets_in_index}")
    LOG.info(f"Cluster time: {cluster_t}")
    LOG.info(f"Pure build time: {build_t}")

    filename = format_models_filename(type, config, clustering_algorithms, size)
    LOG.info(f"Saving as {filename}")
    save_lmi(
        os.path.join(MODELS_DIR, filename),
        config,
        li,
        data_prediction,
        n_buckets_in_index,
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
    parser.add_argument("--type", default="pca32v2")
    parser.add_argument(
        "--size", default="100K", choices=["100K", "300K", "10M", "30M", "70M", "100M"]
    )
    parser.add_argument("--dataset", default="sisap23")
    parser.add_argument("--epochs", nargs="+", default=[100], type=int)
    parser.add_argument("--model-type", nargs="+", default=["MLP"])
    parser.add_argument("--lr", nargs="+", default=[0.01], type=float)
    parser.add_argument("--n-categories", nargs="+", default=[10, 10], type=int)
    parser.add_argument(
        "--clustering-algorithm",
        nargs="+",
        default=["faiss_kmeans"],
        choices=algorithms.keys(),
    )
    args = parser.parse_args()

    validate_and_expand_per_level_arguments(vars(args))

    main(
        args.type,
        args.size,
        args.dataset,
        args.clustering_algorithm,
        args.epochs,
        args.model_type,
        args.lr,
        args.n_categories,
    )
