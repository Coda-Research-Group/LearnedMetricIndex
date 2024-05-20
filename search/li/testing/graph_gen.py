import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from typing import List, Optional, Tuple, Dict, NamedTuple
import re

DATA_DIR = "test_results"
OUTPUT_DIR = "graphs/final/laion"
sns.set_theme(style="whitegrid", font_scale=1.2)


class Interval(NamedTuple):
    lower: Optional[float]
    upper: Optional[float]

    def __contains__(self, x: float) -> bool:
        if self.lower is None and self.upper is None:
            return True
        if self.lower is None:
            return x <= self.upper
        if self.upper is None:
            return x >= self.lower
        return self.lower <= x <= self.upper


def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, filename))


def plot(
    data: pd.DataFrame,
    x: str,
    x_label: str,
    y: str,
    y_label: str,
    title: str,
    filename: str,
    hue: Optional[str] = None,
) -> None:
    sns.lineplot(data, x=x, y=y, marker="o", hue=hue, palette=sns.color_palette())
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), transparent=True)
    plt.clf()


def plot_dc_recall(
    data: pd.DataFrame, filename: str, hue: Optional[str] = None
) -> None:
    plot(
        data,
        x="distance_computations",
        x_label="Distance computations performed",
        y="recall",
        y_label="Recall",
        title="LMI distance computations vs. recall\nPROTEINS 30M subset",
        filename=filename,
        hue=hue,
    )


def plot_search_recall(
    data: pd.DataFrame, filename: str, hue: Optional[str] = None
) -> None:
    plot(
        data,
        x="search",
        x_label="Search time (seconds)",
        y="recall",
        y_label="Recall",
        title="LMI distance computations vs. recall\nPROTEINS 30M subset",
        filename=filename,
        hue=hue,
    )


def transplant(
    data_from: pd.DataFrame,
    data_to: pd.DataFrame,
    columns: List[str],
    sort_by: List[str],
) -> pd.DataFrame:
    assert len(data_from) == len(data_to)

    data_from_cp = data_from.copy()
    data_to_cp = data_to.copy()
    data_from_cp.sort_values(by=sort_by, inplace=True, ignore_index=True)
    data_to_cp.sort_values(by=sort_by, inplace=True, ignore_index=True)
    data_to_cp[columns] = data_from_cp[columns]
    return data_to_cp


def add_percent_searched(data: pd.DataFrame) -> None:
    assert "nprobe" in data.columns and "nlist" in data.columns
    data["percent_searched"] = data["nprobe"] / data["nlist"]


def parse_filename(filename: str) -> Tuple[bool, bool, bool]:
    mtch = re.match(r".+naive-pq=(.+)_dynamic=(.+)_overflow=(.+).csv", filename)
    assert mtch is not None

    return tuple(map(lambda x: True if x == "True" else False, mtch.groups()))


def filter_values(data: pd.DataFrame, filter: Dict[str, Interval] = {}) -> pd.DataFrame:
    for col, interval in filter.items():
        for item in data[col].unique():
            if item in interval:
                continue
            data = data[data[col] != item]

    return data


def type_plot(
    data: pd.DataFrame,
    type_name: str,
    filter: Dict[str, Interval] = {},
    hue: Optional[str] = None,
) -> None:
    data = filter_values(data, filter)

    plot_dc_recall(data, f"{type_name}_dc_recall.svg", hue)
    plot_search_recall(data, f"{type_name}_search_recall.svg", hue)


def none_plot(filename: str, filter: Dict[str, Interval] = {}) -> pd.DataFrame:
    data = load_data(filename)
    type_plot(data, "none", filter)
    return data


def ivf_consolidate(dataIVF: pd.DataFrame, dataIVFFaiss: pd.DataFrame) -> pd.DataFrame:
    data = transplant(
        dataIVF,
        dataIVFFaiss,
        ["distance_computations"],
        ["n_buckets", "nlist", "nprobe"],
    )
    add_percent_searched(data)
    return data


def ivf_plot(
    filenameIVF: str, filenameIVFFaiss: str, filter: Dict[str, Interval] = {}
) -> pd.DataFrame:
    naive, dynamic, overflow = parse_filename(filenameIVF)
    _data = load_data(filenameIVF)
    _data_faiss = load_data(filenameIVFFaiss)

    data = ivf_consolidate(_data, _data_faiss)

    type_plot(
        data,
        f"ivf{'_dynamic' if dynamic else ''}{'_overflow' if overflow else ''}{'_nonaive' if not naive else ''}",
        filter,
        "percent_searched",
    )
    return data


def with_type(data: pd.DataFrame, type: str) -> pd.DataFrame:
    data["type"] = type
    return data


def sketch_plot(filename: str, filter: Dict[str, Interval] = {}) -> pd.DataFrame:
    naive, dynamic, overflow = parse_filename(filename)
    data = load_data(filename)

    type_plot(
        data,
        f"sketch{'_dynamic' if dynamic else ''}{'_overflow' if overflow else ''}{'_nonaive' if not naive else ''}",
        filter,
        "c",
    )
    return data


def best_plot(
    dict: Dict[str, Tuple[pd.DataFrame, Dict[str, Interval]]],
    filter: Dict[str, Interval] = {},
    type_name: str = "best",
) -> None:
    best = pd.concat(
        [
            with_type(filter_values(df, filter), type)
            for (type, (df, filter)) in dict.items()
        ]
    )
    best = filter_values(best, filter)
    print(best)
    type_plot(best, type_name, hue="type")


best_dict: Dict[str, Tuple[pd.DataFrame, Dict[str, Interval]]] = {}

## NO BUCKETS
DATA_NONE = "nav=pca96v2_search=clip768v2_size=10M_bucket=none_k=10_naive-pq=True_dynamic=False_overflow=False.csv"

data_none = none_plot(DATA_NONE)
best_dict["baseline"] = (data_none, {})

## INVERTED FILE
DATA_IVF = "nav=pca96v2_search=clip768v2_size=10M_bucket=IVF_k=10_naive-pq=True_dynamic=False_overflow=False.csv"
DATA_IVF_FAISS = "nav=pca96v2_search=clip768v2_size=10M_bucket=IVFFaiss_k=10_naive-pq=True_dynamic=False_overflow=False.csv"

data_ivf = ivf_plot(
    DATA_IVF,
    DATA_IVF_FAISS,
    {
        "nprobe": Interval(10, 30),
        "recall": Interval(0.86, 0.94),
    #     "distance_computations": Interval(None, 2_000_000_000.0),
    },
)
best_dict["ivf"] = (data_ivf, {"nprobe": Interval(25, 25)})


## INVERTED FILE -- DYNAMIC, OVERFLOW, NONAIVEPQ
DATA_IVF_DYNAMIC_OVERFLOW_NONAIVEPQ = "nav=pca96v2_search=clip768v2_size=10M_bucket=IVF_k=10_naive-pq=False_dynamic=True_overflow=True.csv"
DATA_IVF_FAISS_DYNAMIC_OVERFLOW_NONAIVEPQ = "nav=pca96v2_search=clip768v2_size=10M_bucket=IVFFaiss_k=10_naive-pq=False_dynamic=True_overflow=True.csv"

data_ivf_dynamic_overflow_nonaive = ivf_plot(
    DATA_IVF_DYNAMIC_OVERFLOW_NONAIVEPQ,
    DATA_IVF_FAISS_DYNAMIC_OVERFLOW_NONAIVEPQ,
    {
        "nprobe": Interval(10, 30),
        "recall": Interval(0.86, 0.94),
        # "distance_computations": Interval(None, 2_000_000_000.0),
    },
)
best_dict["ivf (a-u-n)"] = (
    data_ivf_dynamic_overflow_nonaive,
    {"nprobe": Interval(20, 20)},
)


## INVERTED FILE -- DYNAMIC, NONAIVEPQ
DATA_IVF_DYNAMIC_NONAIVEPQ = "nav=pca96v2_search=clip768v2_size=10M_bucket=IVF_k=10_naive-pq=False_dynamic=True_overflow=False.csv"
DATA_IVF_FAISS_DYNAMIC_NONAIVEPQ = "nav=pca96v2_search=clip768v2_size=10M_bucket=IVFFaiss_k=10_naive-pq=False_dynamic=True_overflow=False.csv"

data_ivf_dynamic_nonaive = ivf_plot(
    DATA_IVF_DYNAMIC_NONAIVEPQ,
    DATA_IVF_FAISS_DYNAMIC_NONAIVEPQ,
    {
        "nprobe": Interval(10, 30),
        # "recall": Interval(0.84, None),
        # "distance_computations": Interval(None, 2_000_000_000.0),
    },
)
best_dict["ivf (a-u-d)"] = (data_ivf_dynamic_nonaive, {"nprobe": Interval(15, 15)})

## INVERTED FILE -- DYNAMIC, OVERFLOW
DATA_IVF_DYNAMIC_OVERFLOW = "nav=pca96v2_search=clip768v2_size=10M_bucket=IVF_k=10_naive-pq=True_dynamic=True_overflow=True.csv"
DATA_IVF_FAISS_DYNAMIC_OVERFLOW = "nav=pca96v2_search=clip768v2_size=10M_bucket=IVFFaiss_k=10_naive-pq=True_dynamic=True_overflow=True.csv"

data_ivf_dynamic_overflow = ivf_plot(
    DATA_IVF_DYNAMIC_OVERFLOW,
    DATA_IVF_FAISS_DYNAMIC_OVERFLOW,
    {
        "nprobe": Interval(10, 30),
        "recall": Interval(0.86, 0.94),
        # "distance_computations": Interval(None, 2_000_000_000.0),
    },
)
best_dict["ivf (a-i-n)"] = (data_ivf_dynamic_overflow, {"nprobe": Interval(10, 10)})


## INVERTED FILE -- DYNAMIC
DATA_IVF_DYNAMIC = "nav=pca96v2_search=clip768v2_size=10M_bucket=IVF_k=10_naive-pq=True_dynamic=True_overflow=False.csv"
DATA_IVF_FAISS_DYNAMIC = "nav=pca96v2_search=clip768v2_size=10M_bucket=IVFFaiss_k=10_naive-pq=True_dynamic=True_overflow=False.csv"

data_ivf_dynamic = ivf_plot(
    DATA_IVF_DYNAMIC,
    DATA_IVF_FAISS_DYNAMIC,
    {
        "nprobe": Interval(5, 20),
        "recall": Interval(0.86, 0.94),
        # "distance_computations": Interval(None, 2_000_000_000.0),
    },
)
best_dict["ivf (a-i-d)"] = (data_ivf_dynamic_overflow, {"nprobe": Interval(8, 8)})

## SKETCH
DATA_SKETCH = "nav=pca96v2_search=clip768v2_size=10M_bucket=sketch_k=10_naive-pq=True_dynamic=False_overflow=False.csv"

data_sketch = sketch_plot(
    DATA_SKETCH,
    {
        "c": Interval(400, 800),
        "recall": Interval(0.88, 0.92),
        "distance_computations": Interval(None, 1_000_000_000),
    },
)
best_dict["sketch"] = (data_sketch, {"c": Interval(600, 600)})


## SKETCH, DYNAMIC, OVERFLOW, NONAIVEPQ
DATA_SKETCH_DYNAMIC_OVERFLOW_NONAIVEPQ = "nav=pca96v2_search=clip768v2_size=10M_bucket=sketch_k=10_naive-pq=False_dynamic=True_overflow=True.csv"

data_sketch_dynamic_overflow_nonaive = sketch_plot(
    DATA_SKETCH_DYNAMIC_OVERFLOW_NONAIVEPQ,
    {
        "c": Interval(200, 600),
        "recall": Interval(0.86, 0.94),
        # "distance_computations": Interval(None, 200_000),
    },
)
best_dict["sketch (a-u-n)"] = (
    data_sketch_dynamic_overflow_nonaive,
    {"c": Interval(200, 200)},
)

## SKETCH, DYNAMIC, NONAIVEPQ
DATA_SKETCH_DYNAMIC_NONAIVEPQ = "nav=pca96v2_search=clip768v2_size=10M_bucket=sketch_k=10_naive-pq=False_dynamic=True_overflow=False.csv"

data_sketch_dynamic_nonaive = sketch_plot(
    DATA_SKETCH_DYNAMIC_NONAIVEPQ,
    {
        "c": Interval(100, 500),
        "recall": Interval(0.86, 0.94),
        # "distance_computations": Interval(None, 200_000),
        # "search": Interval(None, 1200),
    },
)
best_dict["sketch (a-u-d)"] = (data_sketch_dynamic_nonaive, {"c": Interval(200, 200)})

## SKETCH, DYNAMIC, OVERFLOW
DATA_SKETCH_DYNAMIC_OVERFLOW = "nav=pca96v2_search=clip768v2_size=10M_bucket=sketch_k=10_naive-pq=True_dynamic=True_overflow=True.csv"

data_sketch_dynamic_overflow = sketch_plot(
    DATA_SKETCH_DYNAMIC_OVERFLOW,
    {
        "c": Interval(200, 600),
        "recall": Interval(0.86, 0.94),
        # # "distance_computations": Interval(None, 200_000),
        # "search": Interval(None, 1200),
    },
)
best_dict["sketch (a-i-n)"] = (data_sketch_dynamic_overflow, {"c": Interval(300, 300)})

## SKETCH, DYNAMIC
DATA_SKETCH_DYNAMIC = "nav=pca96v2_search=clip768v2_size=10M_bucket=sketch_k=10_naive-pq=True_dynamic=True_overflow=False.csv"

data_sketch_dynamic = sketch_plot(
    DATA_SKETCH_DYNAMIC,
    {
        "c": Interval(200, 600),
        "recall": Interval(0.86, 0.94),
        # # "distance_computations": Interval(None, 200_000),
        # "search": Interval(None, 1200),
    },
)
best_dict["sketch (a-i-d)"] = (data_sketch_dynamic, {"c": Interval(400, 400)})


## BEST
compare_ivf = ["ivf", "ivf (a-u-n)", "ivf (a-u-d)", "ivf (a-i-n)", "ivf (a-i-d)"]
compare_sketches = [
    "sketch",
    "sketch (a-u-n)",
    "sketch (a-u-d)",
    "sketch (a-i-n)",
    "sketch (a-i-d)",
]
best_final = ["ivf", "ivf (a-u-n)", "sketch", "sketch (a-u-n)", "baseline"]

best_plot({key: best_dict[key] for key in compare_ivf}, {}, type_name="best_ivf")
best_plot(
    {key: best_dict[key] for key in compare_sketches}, {}, type_name="best_sketch"
)
best_plot(
    {key: best_dict[key] for key in best_final},
    {"distance_computations": Interval(None, 10_000_000_000)},
)
