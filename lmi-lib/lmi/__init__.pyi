import torch

class LMI:
    """
    Learned Metric Index (LMI) for efficient similarity search made.

    This class implements a learned index structure for approximate nearest neighbor search.
    It uses a neural network to learn a mapping from the data space to bucket IDs,
    and then for each query, performs exact search within the buckets predicted by the model.

    Attributes:
        n_buckets: Number of buckets to partition the data into
        dimensionality: Dimensionality of the input data vectors
    """

    def __init__(self, n_buckets: int, data_dimensionality: int) -> None: ...
    def run_tests(self) -> None: ...
    def _run_kmeans(self, X: torch.Tensor) -> torch.Tensor: ...
    def _train_model(
        self, X: torch.Tensor, y: torch.Tensor, epochs: int, lr: float
    ) -> None: ...
    def _create_buckets(self, X: torch.Tensor) -> None: ...
    def build(self, X: torch.Tensor, y: torch.Tensor, epochs: int, lr: float) -> None: ...
    def search_raw_parallel(self, query: torch.Tensor, k: int) -> torch.Tensor: ...

