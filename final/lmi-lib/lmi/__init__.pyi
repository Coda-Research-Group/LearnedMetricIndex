import torch

class LMI:
    """
    Learned Metric Index (LMI) for efficient similarity search.

    This class implements a learned index structure for approximate nearest neighbor search.
    It uses a neural network to learn a mapping from the data space to bucket IDs,
    and then for each query, performs exact search within the buckets predicted by the model.

    Attributes:
        n_buckets: Number of buckets to partition the data into
        dimensionality: Dimensionality of the input data vectors
    """

    def __init__(self, n_buckets: int, data_dimensionality: int) -> None: ...
    def run_tests(self) -> None: ...
    def _run_kmeans(self, X: torch.Tensor) -> torch.Tensor:
        """
        Run k-means clustering on the input data.

        Args:
            X: Input data tensor of shape (n_samples, dimensionality)

        Returns:
            Cluster assignments for each data point as a tensor of shape (n_samples,)
        """
        ...

    def _train_model(
        self, X: torch.Tensor, y: torch.Tensor, epochs: int, lr: float
    ) -> None:
        """
        Train the neural network model to predict bucket assignments.

        Args:
            X: Input data tensor of shape (n_samples, dimensionality)
            y: Target bucket assignments from k-means of shape (n_samples,)
            epochs: Number of training epochs
            lr: Learning rate
        """
        ...

    def _create_buckets(self, X: torch.Tensor) -> None:
        """
        Create buckets by assigning data points to buckets using the trained model.

        Args:
            X: Input data tensor of shape (n_samples, dimensionality)
        """
        ...

    def build(self, X: torch.Tensor, epochs: int, lr: float) -> None:
        """
        Build the complete LMI by internally calling _run_kmeans, _train_model and _create_buckets.

        Args:
            X: Input data tensor of shape (n_samples, dimensionality)
            epochs: Number of training epochs
            lr: Learning rate
        """
        ...

    def search(self, query: torch.Tensor, k: int) -> torch.Tensor: ...
    def search_multiple(self, queries: torch.Tensor, k: int) -> torch.Tensor: ...
    def search_raw(self, query: torch.Tensor, k: int) -> torch.Tensor: ...
    def search_raw_multiple(self, queries: torch.Tensor, k: int) -> torch.Tensor: ...
