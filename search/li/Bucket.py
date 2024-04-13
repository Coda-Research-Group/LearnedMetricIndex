import numpy as np
import numpy.typing as npt
import pandas as pd
import time

from li.inverted_file import InvertedFileIndex
from typing import Tuple, Any, Optional
import faiss


class Bucket:
    data: Any
    original_idxs: npt.ArrayLike

    def train(self, data: pd.DataFrame, **kwargs) -> float:
        raise NotImplementedError()

    def add(self, data: pd.DataFrame, original_idxs: npt.ArrayLike) -> float:
        raise NotImplementedError()

    def build(self, data: pd.DataFrame, original_idxs: npt.ArrayLike, **kwargs) -> float:
        return self.train(data, **kwargs) + self.add(data, original_idxs)

    def reset(self) -> None:
        raise NotImplementedError()

    def search(
        self, queries: npt.NDArray[np.float32], k: int, **kwargs
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], float, int]:
        raise NotImplementedError()


class NaiveBucket(Bucket):
    data: Optional[pd.DataFrame]

    def train(self, data: pd.DataFrame, **kwargs) -> float:
        return 0.0

    def add(self, data: pd.DataFrame, original_idxs: npt.ArrayLike) -> float:
        assert len(data) == len(original_idxs)

        self.original_idxs = original_idxs
        t_start = time.time()
        self.data = data
        return time.time() - t_start

    def reset(self) -> None:
        self.data = None

    def search(
        self, queries: npt.NDArray[np.float32], k: int, **kwargs
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], float, int]:
        assert self.data is not None

        s = time.time()
        similarity, indices = faiss.knn(
            queries,
            self.data,
            k,
            metric=faiss.METRIC_INNER_PRODUCT,
        )
        t_search = time.time() - s

        distances = 1 - similarity  # similarity is not distance

        return indices, distances, t_search, len(queries) * len(self.data)


class IVFBucket(Bucket):
    data: Optional[InvertedFileIndex] = None

    def train(self, data: pd.DataFrame, **kwargs) -> float:
        nlist = kwargs.get('nlist', 5)
        nlist = min(nlist, len(data))

        no, d = data.shape
        s = time.time()
        self.data = InvertedFileIndex(d, nlist)
        self.data.train(data)

        return time.time() - s

    def add(self, data: pd.DataFrame, original_idxs: npt.ArrayLike) -> float:
        assert self.data is not None and self.data.trained
        assert len(data) == len(original_idxs)

        self.original_idxs = original_idxs
        s = time.time()
        self.data.add(data)

        return time.time() - s

    def reset(self) -> None:
        self.data.reset()

    def search(
        self, queries: npt.NDArray[np.float32], k: int, **kwargs
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], float, int]:
        assert self.data is not None and self.data.trained
        nprobe = kwargs.get('nprobe', 5)
        nprobe = min(nprobe, self.data.nlist)

        self.data.nprobe = nprobe

        s = time.time()
        dc = self.data.distance_computer.distance_computations
        vectors, distances = self.data.search(
            queries,
            k,
        )
        t_search = time.time() - s

        return (
            vectors,
            distances,
            t_search,
            self.data.distance_computer.distance_computations - dc,
        )


class IVFBucketFaiss(Bucket):
    data: Optional[faiss.IndexIVFFlat] = None

    def train(self, data: pd.DataFrame, **kwargs) -> float:
        nlist = kwargs.get('nlist', 5)
        nlist = min(nlist, len(data))

        no, d = data.shape
        s = time.time()
        quantizer = faiss.IndexFlatIP(d)
        self.data = faiss.IndexIVFFlat(quantizer, d, nlist)
        self.data.train(data)

        return time.time() - s

    def add(self, data: pd.DataFrame, original_idxs: npt.ArrayLike) -> float:
        self.original_idxs = original_idxs

        s = time.time()
        self.data.add(data)

        return time.time() - s

    def reset(self) -> None:
        self.data.reset()

    def search(
        self, queries: npt.NDArray[np.float32], k: int, **kwargs
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], float, int]:
        assert self.data is not None
        nprobe = kwargs.get('nprobe', 5)
        nprobe = min(nprobe, self.data.nlist)

        self.data.nprobe = nprobe

        s = time.time()
        distances, indices = self.data.search(
            queries,
            k,
        )
        t_search = time.time() - s

        return indices, distances, t_search, np.NINF
