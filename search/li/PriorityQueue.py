import numpy as np
import numpy.typing as npt
from typing import Tuple

EMPTY_VALUE = -1


class PriorityQueue:
    """
    A priority queue storing probabilities and paths for the next nodes to visit for each query.

    The priority queue is realized by three numpy arrays:
    - `probability`: stores the probability of the next node/bucket to visit
    - `path`: stores the path to the next node/bucket
    - `length`: stores the current length of the queue for each query
    - `should_sort`: stores whether the queue associated with this query should be sorted
    """

    def __init__(self, n_queries: int, queue_length_upper_bound: int, n_levels: int):
        self.probability: npt.NDArray[np.float32] = np.full(
            (n_queries, queue_length_upper_bound),
            fill_value=EMPTY_VALUE,
            dtype=np.float32,
        )
        self.path: npt.NDArray[np.int32] = np.full(
            (n_queries, queue_length_upper_bound, n_levels),
            fill_value=EMPTY_VALUE,
            dtype=np.int32,
        )
        self.length: npt.NDArray[np.int32] = np.zeros(n_queries, dtype=np.int32)

        self.should_sort: npt.NDArray[np.bool_] = np.full(
            n_queries, fill_value=np.False_, dtype=np.bool_
        )
        self.n_levels = n_levels

    def add(
        self,
        indices: npt.NDArray[np.int32],
        path: npt.NDArray[np.int32],
        probabilities: npt.NDArray[np.float32],
    ) -> None:
        """
        Adds a new node/bucket path to visit to the priority queue
        but only for queries specified by `indices`.
        """
        self.probability[indices, self.length[indices]] = probabilities
        self.path[indices, self.length[indices], :] = path
        self.should_sort[indices] = np.True_

        self.length[indices] += 1

    def pop(
        self, indices: npt.NDArray[np.int32]
    ) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        """
        Pops the next node/bucket path to visit for each query
        along with their corresponding probability.
        """
        self.length[indices] -= 1

        return self.path[indices, self.length[indices], :], self.probability[indices, self.length[indices]]

    def sort(self) -> None:
        """
        Sorts the queues by the probability.
        A particular queue is sorted only if `should_sort` is `True`.

        Implementation details:
        Firstly, we obtain the indexes of the sorted probabilities.
        Then, we use these indexes to sort the probabilities and paths.
        Sorting of paths is done for each level separately.
        The whole process is repeated for each queue length separately.
        """
        for queue_length in np.unique(self.length):
            if queue_length in {0, 1}:
                continue

            idxs_to_sort = np.where(
                np.logical_and(
                    self.length == queue_length,
                    self.should_sort == np.True_,
                )
            )[0]

            sorted_idxs = self.probability[idxs_to_sort, :queue_length].argsort()

            self.probability[idxs_to_sort, :queue_length] = np.take_along_axis(
                self.probability[idxs_to_sort, :queue_length],
                sorted_idxs,
                axis=1,
            )
            for level_idx in range(self.n_levels):
                self.path[idxs_to_sort, :queue_length, level_idx] = np.take_along_axis(
                    self.path[idxs_to_sort, :queue_length, level_idx],
                    sorted_idxs,
                    axis=1,
                )

            self.should_sort[idxs_to_sort] = np.False_
