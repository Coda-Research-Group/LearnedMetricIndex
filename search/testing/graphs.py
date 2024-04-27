import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

GRAPHS_DIR = "./graphs"
TEST_RESULT_DIR = "./test_results"


@dataclass
class MetricName:
    col_name: str
    full_name: str


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

    group_by_col_name: Optional[MetricName]

    figsize = (10, 6)

    def save_figure(self, figure: Figure, graph_type: str) -> None:
        filename = (
            f"{graph_type}_size={self.size}_nav={self.data_navigation}_search={self.data_search}"
            f"bucket-type={self.bucket_t}_k={self.k}"
            f"naive-pq={self.naive_pq}_dynamic={self.dynamic}"
        )
        path = os.path.join(GRAPHS_DIR, filename)
        figure.savefig(path)

    def _scatterplot(
        self,
        x_name: MetricName,
        y_name: MetricName,
        group_by_name: Optional[MetricName] = None,
        filter: Optional[Dict[str, List]] = None,
    ) -> Figure:
        figure = plt.figure(figsize=self.figsize)

        filtered_data = self.data
        if filter is not None:
            for col, values in filter.items():
                filtered_data = filtered_data[filtered_data[col].isin(values)]

        if group_by_name is not None:
            for group_by_value, group_by_data in filtered_data.groupby(
                group_by_name.col_name
            ):
                plt.scatter(
                    group_by_data[x_name.col_name],
                    group_by_data[y_name.col_name],
                    label=f"{group_by_name.full_name} = {group_by_value}",
                )
        else:
            plt.scatter(filtered_data[x_name.col_name], filtered_data[y_name.col_name])

        plt.xlabel(x_name.full_name)
        plt.ylabel(y_name.full_name)
        plt.title(
            f"{x_name.full_name} vs. {y_name.full_name}\n"
            f"bucket type: {self.bucket_t}\n"
            f"{self.data_navigation}, {self.size}, k: {self.k}, "
            f"naive priority queue: {self.naive_pq}, dynamic search: {self.dynamic}"
        )
        plt.legend()
        plt.grid(True)

        return figure

    def inference_time_to_recall(
        self, filter: Optional[Dict[str, List]] = None
    ) -> Figure:
        return self._scatterplot(
            MetricName("inference", "Inference time"),
            MetricName("recall", "Recall"),
            self.group_by_col_name,
            filter,
        )

    def search_time_to_recall(self, filter: Optional[Dict[str, List]] = None) -> Figure:
        return self._scatterplot(
            MetricName("search", "Search time"),
            MetricName("recall", "Recall"),
            self.group_by_col_name,
            filter,
        )

    def distance_computations_to_recall(
        self, filter: Optional[Dict[str, List]] = None
    ) -> Figure:
        return self._scatterplot(
            MetricName("distance_computations", "Distance computations"),
            MetricName("recall", "Recall"),
            self.group_by_col_name,
            filter,
        )


def get_grapher(filename: str) -> Grapher:
    mtch = re.match(
        r"nav=(.+)_search=(.+)_size=(.+)_bucket=(.+)_k=(.+)_naive-pq=(.+)_dynamic=(.+).csv",
        filename,
    )
    assert mtch is not None

    data_navigation, data_search, size, bucket_t, k, naive_pq, dynamic = mtch.groups()

    k = int(k)
    naive_pq = naive_pq == "True"
    dynamic = dynamic == "True"

    path = os.path.join(TEST_RESULT_DIR, filename)
    data = pd.read_csv(path)

    group_by_col_name = None
    if bucket_t == "naive":
        group_by_col_name = None
    elif bucket_t in ["IVF", "IVFFaiss"]:
        group_by_col_name = MetricName("nprobe", "nprobe")
    elif bucket_t == "sketch":
        group_by_col_name = MetricName("c", "c")
    else:
        assert False
    assert group_by_col_name is not None

    return Grapher(
        data=data,
        data_navigation=data_navigation,
        data_search=data_search,
        size=size,
        bucket_t=bucket_t,
        k=k,
        naive_pq=naive_pq,
        dynamic=dynamic,
        group_by_col_name=group_by_col_name,
    )


FILENAMES = [
    "nav=pca32v2_search=clip768v2_size=100K_bucket=IVFFaiss_k=10_naive-pq=False_dynamic=True.csv"
]
GRAPHS = [
    "inference_time_to_recall",
    "search_time_to_recall",
    "distance_computations_to_recall",
]
FILTER = {"nprobe": [1, 5, 9]}

if __name__ == "__main__":
    for filename in FILENAMES:
        grapher = get_grapher(filename)

        for graph_type in GRAPHS:
            try:
                graph = getattr(grapher, graph_type)(FILTER)
            except AttributeError:
                continue

            grapher.save_figure(graph, graph_type.replace("_", "-"))
