from li.clustering.scikit_kmeans import cluster
import numpy.typing as npt
from typing import Optional, Tuple, List
import numpy as np
from sklearn.cluster import KMeans
from logging import DEBUG
from li.Logger import Logger
from li.utils import log_runtime

Vector = npt.NDArray[np.float32]
VectorMatrix = npt.NDArray[np.float32]
DistanceMatrix = npt.NDArray[np.float32]
IndexArray = npt.NDArray[np.int64]

VectorMatrixMatrix = npt.NDArray[np.float32]
DistanceMatrixMatrix = npt.NDArray[np.float32]
IndexArrayArray = npt.NDArray[np.int64]


class DistanceComputer:
    distance_computations: int = 0

    def reset(self) -> None:
        self.distance_computations = 0

    def __call__(self, left: VectorMatrix, right: VectorMatrix):
        raise NotImplementedError()


class InnerProductComputer(DistanceComputer):
    def __call__(self, left: VectorMatrix, right: VectorMatrix):
        """Computes complement to cosine similarity normalized to (0, 1)."""
        self.distance_computations += len(left) * len(right)
        return 1 - (np.inner(left, right) + 1) / 2


class InvertedFileIndex(Logger):
    d: int
    nlist: int
    distance_computer: DistanceComputer
    nprobe: int = 5
    kmeans: Optional[KMeans] = None
    data: Optional[VectorMatrix] = None
    cells: Optional[List[IndexArray]] = None
    centroids: Optional[VectorMatrix] = None
    trained: bool = False

    def __init__(
        self,
        d: int,
        nlist: int,
        distance_computer: DistanceComputer = InnerProductComputer(),
    ) -> None:
        self.d = d
        self.nlist = nlist
        self.distance_computer = distance_computer

    def train(self, data: VectorMatrix) -> None:
        self.kmeans, _ = cluster(
            data,
            self.nlist,
            parameters={
                "verbose": 0,
                "random_state": None,
                "init": "random",
                "max_iter": 25,
                "n_init": 1,
            },
        )
        self.centroids = self.kmeans.cluster_centers_
        self.trained = True

    def add(self, data: VectorMatrix) -> None:
        labels = self.kmeans.predict(data)
        cells = [np.arange(len(data))[labels == label] for label in range(self.nlist)]
        self.cells = cells
        self.data = data

    def reset(self) -> None:
        self.data = None
        self.cells = None

    @log_runtime(DEBUG, "Computed distance from centroids in: {}")
    def _compute_distance_centroids(self, queries: VectorMatrix) -> VectorMatrix:
        assert self.centroids is not None and self.centroids.shape == (
            self.nlist,
            self.d,
        )
        assert queries.shape[1] == self.d

        return self.distance_computer(queries, self.centroids)

    def _compute_distance_vector_cell(
        self, query: Vector, cell_no: int
    ) -> VectorMatrix:
        assert self.cells is not None

        return self.distance_computer(
            np.expand_dims(query, axis=0), self.data[self.cells[cell_no]]
        )[0]

    @log_runtime(DEBUG, "Searching done in: {}")
    def _search_cells(
        self, queries: VectorMatrix, cell_nos: IndexArray, k: int
    ) -> Tuple[List[VectorMatrix], List[DistanceMatrix]]:
        vecs: List[IndexArray] = []
        diss: List[DistanceMatrix] = []

        for query, cell_no in zip(queries, cell_nos):
            distances = self._compute_distance_vector_cell(query, cell_no)

            l = min(k, len(distances))
            first_l = distances.argsort()[:l]

            vecs.append(self.cells[cell_no][first_l])
            diss.append(distances[first_l])

        return vecs, diss

    def _consolidate(
        self,
        vectors_a: List[IndexArray],
        distances_a: List[DistanceMatrix],
        vectors_b: List[IndexArray],
        distances_b: List[DistanceMatrix],
        k: int,
    ) -> Tuple[List[IndexArray], List[DistanceMatrix]]:
        vecs: List[IndexArray] = []
        diss: List[DistanceMatrix] = []

        for vecs_a, diss_a, vecs_b, diss_b in zip(
            vectors_a, distances_a, vectors_b, distances_b
        ):
            vecs_comb = np.concatenate((vecs_a, vecs_b), axis=0)
            diss_comb = np.concatenate((diss_a, diss_b), axis=0)

            l = min(k, len(diss_comb))
            first_l = diss_comb.argsort()[:l]

            vecs.append(vecs_comb[first_l])
            diss.append(diss_comb[first_l])

        return vecs, diss

    def _padd(
        self, vecs: List[IndexArray], diss: List[DistanceMatrix], k: int
    ) -> Tuple[IndexArrayArray, DistanceMatrixMatrix]:
        max_len = k

        vecs_padded = np.array(
            [
                np.concatenate(
                    (idxs, np.full(((max_len - len(idxs)),), -1))
                )
                for idxs in vecs
            ]
        )
        diss_padded = np.array(
            [
                np.concatenate((dists, np.full(((max_len - len(dists)),), np.float32('inf'))))
                for dists in diss
            ]
        )

        return vecs_padded, diss_padded

    def search(
        self, queries: VectorMatrix, k: int
    ) -> Tuple[IndexArrayArray, DistanceMatrixMatrix]:
        vecs: List[IndexArray] = [np.empty((0,), dtype=np.int64) for _ in queries]
        diss: List[DistanceMatrix] = [np.empty((0,), dtype=np.float32) for _ in queries]

        self.logger.debug("Computing distance from centroids.")
        distances_to_centroids = self._compute_distance_centroids(queries)
        order_to_search = np.argsort(distances_to_centroids)

        for cell_nos in order_to_search[:, : self.nprobe].T:
            self.logger.debug("Searching one cell for each query.")
            cell_nns, cell_diss = self._search_cells(queries, cell_nos, k)
            vecs, diss = self._consolidate(vecs, diss, cell_nns, cell_diss, k)

        return self._padd(vecs, diss, k)
