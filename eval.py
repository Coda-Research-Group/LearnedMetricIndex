# Extended evaluator for LAION + AGNEWS
# Based on SISAP 2023 evaluation script

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Generator

import h5py
import numpy as np

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def get_laion_groundtruth(size: str, k: int = 30) -> np.ndarray:
    gt_path = Path(
        f"data2024/gold-standard-dbsize={size}"
        "--public-queries-2024-laion2B-en-clip768v2-n=10k.h5"
    )
    with h5py.File(gt_path, "r") as f:
        return np.array(f["knns"][:, :k])


def get_agnews_groundtruth(dataset_path: Path, k: int = 30) -> np.ndarray:
    with h5py.File(dataset_path, "r") as f:
        return np.array(f["neighbors"][:, :k])


def get_all_results(dirname: str) -> Generator[h5py.File, None, None]:
    masks = [
        dirname + "/*.h5",
        dirname + "/*/*.h5",
        dirname + "/*/*/*.h5",
        dirname + "/*/*/*/*.h5",
    ]
    print("Searching for result files:")
    for m in masks:
        print(" ", m)
        for fn in glob.iglob(m):
            try:
                f = h5py.File(fn, "r")
                if "knns" not in f:
                    f.close()
                    continue
                yield f
                f.close()
            except Exception as e:
                print("Skipping", fn, e)



def get_recall(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    assert I.shape[0] == gt.shape[0]
    assert k <= I.shape[1]

    hits = 0
    n = I.shape[0]

    for i in range(n):
        hits += len(set(I[i, :k]) & set(gt[i, :k]))

    return hits / (n * k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        help="Directory with result .h5 files",
        default="result",
    )
    parser.add_argument(
        "--dataset",
        choices=["laion", "agnews"],
        required=True,
        help="Dataset used for evaluation",
    )
    parser.add_argument(
        "--agnews-dataset",
        help="Path to agnews-mxbai HDF5 file",
        default="data2024/agnews-mxbai-1024-euclidean.hdf5",
    )
    parser.add_argument(
        "--size",
        help="Size of the dataset",
        default="1M",
    )
    parser.add_argument("csvfile")
    args = parser.parse_args()

    columns = [
        "size",
        "algo",
        "modelingtime",
        "encdatabasetime",
        "encqueriestime",
        "buildtime",
        "querytime",
        "params",
        "recall",
    ]

    gt_cache = {}

    with Path.open(args.csvfile, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()

        for res in get_all_results(args.results):
            attrs = dict(res.attrs)

            if "knns" not in res:
                continue

            knns = np.array(res["knns"])

            if args.dataset == "agnews":
                size="769K"
                if "agnews" not in gt_cache:
                    gt_cache["agnews"] = get_agnews_groundtruth(
                        Path(args.agnews_dataset)
                    )
                gt = gt_cache["agnews"]

            else:  # LAION
                size = args.size
                if size not in {"300K", "1M", "10M", "100M"}:
                    continue

                if size not in gt_cache:
                    gt_cache[size] = get_laion_groundtruth(size)
                gt = gt_cache[size]

            recall = get_recall(knns, gt, k=30)

            row = {k: attrs.get(k, 0) for k in columns}
            row["recall"] = recall
            row["size"] = size

            print(attrs.get("algo"), attrs.get("params"), "=>", recall)
            writer.writerow(row)
