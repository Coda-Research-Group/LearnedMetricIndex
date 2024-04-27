import os
import re
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

GRAPHS_DIR = "./graphs"
TEST_RESULT_DIR = "./test_results"


@dataclass
class Grapher:
    data: pd.DataFrame
    data_navigation: str
    data_search: str
    size: str
    bucket_t: str
    k: int
    naive_pq: bool
    dynamic: bool

    figsize = (10, 6)

    def save_figure(self, figure: Figure, graph_type: str) -> None:
        filename = (
            f"{graph_type}_size={self.size}_nav={self.data_navigation}_search={self.data_search}"
            f"bucket-type={self.bucket_t}_k={self.k}"
            f"naive-pq={self.naive_pq}_dynamic={self.dynamic}"
        )
        path = os.path.join(GRAPHS_DIR, filename)
        figure.savefig(path)

    def inference_time_to_recall(self) -> Figure:
        figure = plt.figure(figsize=self.figsize)
        plt.scatter(self.data["inference"], self.data["recall"])

        plt.xlabel("Inference time")
        plt.ylabel("Recall")
        plt.title(
            "Inference time vs. Recall\n"
            f"bucket: {self.bucket_t}\n"
            f"({self.data_navigation}, {self.size}, k={self.k}"
            f"naive priority queue: {self.naive_pq}, dynamic search: {self.dynamic}"
        )
        plt.legend()
        plt.grid(True)

        return figure

    def search_time_to_recall(self) -> Figure:
        figure = plt.figure(figsize=self.figsize)
        plt.scatter(self.data["search"], self.data["recall"])

        plt.xlabel("Search time")
        plt.ylabel("Recall")
        plt.title(
            "Search time vs. Recall\n"
            f"bucket: {self.bucket_t}\n"
            f"({self.data_navigation}, {self.size}, k={self.k}"
            f"naive priority queue: {self.naive_pq}, dynamic search: {self.dynamic}"
        )
        plt.legend()
        plt.grid(True)

        return figure

    def distance_computations_to_recall(self) -> Figure:
        figure = plt.figure(figsize=self.figsize)
        plt.scatter(self.data["distance_computations"], self.data["recall"])

        plt.xlabel("Distance computations")
        plt.ylabel("Recall")
        plt.title(
            "Distance computations vs. Recall\n"
            f"bucket: {self.bucket_t}\n"
            f"({self.data_navigation}, {self.size}, k={self.k}"
            f"naive priority queue: {self.naive_pq}, dynamic search: {self.dynamic}"
        )
        plt.legend()
        plt.grid(True)

        return figure


def get_grapher(filename: str) -> Grapher:
    mtch = re.match(
        r"nav=(.+)_search=(.+)_size=(.+)_bucket=(.+)_k=(.+)_naive-pq=(.+)_dynamic=(.+).csv",
        filename,
    )
    assert mtch is not None

    data_navigation, data_search, size, bucket_t, k, naive_pq, dynamic = (
        mtch.groups()
    )

    k = int(k)
    naive_pq = naive_pq == "True"
    dynamic = dynamic == "True"

    path = os.path.join(TEST_RESULT_DIR, filename)
    data = pd.read_csv(path)

    return Grapher(
        data=data,
        data_navigation=data_navigation,
        data_search=data_search,
        size=size,
        bucket_t=bucket_t,
        k=k,
        naive_pq=naive_pq,
        dynamic=dynamic,
    )


FILENAMES = ["nav=pca32v2_search=clip768v2_size=100K_bucket=sketch_k=10_naive-pq=False_dynamic=False.csv"]
GRAPHS = [
    "inference_time_to_recall",
    "search_time_to_recall",
    "distance_computations_to_recall",
]

if __name__ == "__main__":
    for filename in FILENAMES:
        grapher = get_grapher(filename)

        for graph_type in GRAPHS:
            try:
                graph = getattr(grapher, graph_type)()
            except AttributeError:
                continue

            grapher.save_figure(graph, graph_type.replace('_', '-'))
