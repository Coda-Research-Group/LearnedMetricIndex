from dataclasses import dataclass, field
from typing import Any, List, Union, Dict

from chromadb.li_index.search.li.clustering import ClusteringAlgorithm
from chromadb.li_index.search.li.model import ModelParameters


@dataclass
class BuildConfiguration:
    """
    BuildConfiguration specifies parameters of models on each level of the hierarchy.

    Expectes the arguments to be either a single value or a list of values of the same size
    as n_categories. If a single value is provided, it is expanded to the size of n_categories.
    The n_categories argument must be a list of integers specifying the number of categories
    at each level of the hierarchy.

    Example:
    ```
    config1 = BuildConfiguration(
        clustering.algorithms["faiss_kmeans"],
        40,
        "MLP-3",
        0.001,
        [5, 10],
    )

    config2 = BuildConfiguration(
        [clustering.algorithms["faiss_kmeans"]],
        [40],
        ["MLP-3"],
        [0.001],
        [5, 10],
    )
    ```

    In both cases, the arguments are expanded to the following (identical) configuration:
    ```
    config3 = BuildConfiguration(
        [clustering.algorithms["faiss_kmeans"], clustering.algorithms["faiss_kmeans"]],
        [40, 40],
        ["MLP-3", "MLP-3"],
        [0.001, 0.001],
        [5, 10],
    )
    ```
    """

    clustering_algorithms: List[ClusteringAlgorithm]
    epochs: List[int]
    model_types: List[str]
    lrs: List[float]
    n_categories: List[int]

    level_configurations: List[ModelParameters] = field(init=False)
    n_levels: int = field(init=False)
    kmeans: Dict = None

    def __init__(
        self,
        clustering_algorithms: Union[List[ClusteringAlgorithm], ClusteringAlgorithm],
        epochs: Union[List[int], int],
        model_types: Union[List[str], str],
        lrs: Union[List[float], float],
        n_categories: List[int],
        kmeans: Dict = None,
    ):
        BuildConfiguration._validate(
            clustering_algorithms, epochs, model_types, lrs, n_categories
        )

        # Expand the arguments to the size of n_categories
        self.clustering_algorithms = BuildConfiguration._expand(
            clustering_algorithms, len(n_categories)
        )
        self.epochs = BuildConfiguration._expand(epochs, len(n_categories))
        self.model_types = BuildConfiguration._expand(model_types, len(n_categories))
        self.lrs = BuildConfiguration._expand(lrs, len(n_categories))
        self.n_categories = n_categories
        self.kmeans = kmeans

        # Fields that are populated mainly for convenience and readability
        self.level_configurations = [
            ModelParameters(
                clustering_algorithm=self.clustering_algorithms[i],
                model_type=self.model_types[i],
                epochs=self.epochs[i],
                lr=self.lrs[i],
                n_categories=self.n_categories[i],
            )
            for i in range(len(self.n_categories))
        ]
        self.n_levels = len(self.n_categories)

    @staticmethod
    def _validate(
        clustering_algorithms: Union[List[ClusteringAlgorithm], ClusteringAlgorithm],
        epochs: Union[List[int], int],
        model_types: Union[List[str], str],
        lrs: Union[List[float], float],
        n_categories: List[int],
    ) -> None:
        """
        Validate the arguments to BuildConfiguration and raise an AssertionError if they are invalid.
        """
        assert len(n_categories) > 0, "n_categories must specify at least one level"

        arguments = [clustering_algorithms, epochs, model_types, lrs]

        arguments_are_lists = all([isinstance(arg, list) for arg in arguments])
        arguments_are_scalars = all(
            [
                callable(clustering_algorithms),
                isinstance(epochs, int),
                isinstance(model_types, str),
                isinstance(lrs, float),
            ]
        )

        assert (
            arguments_are_lists or arguments_are_scalars
        ), "clustering_algorithms, epochs, model_types, and lrs must be lists or single values"

        for arg in arguments:
            if isinstance(arg, list):
                assert len(arg) == 1 or len(arg) == len(n_categories), (
                    "clustering_algorithms, epochs, model_types, and lrs must "
                    "be lists of size 1 or the same size as n_categories"
                )

    @staticmethod
    def _expand(arg: Union[List[Any], Any], n_categories: int) -> List[Any]:
        """
        Expects arg to be either a single value, a list with a single value or a list
        of the same size as n_categories. If single value inputs are provided,
        they are expanded to the size of n_categories and returned as lists.
        """
        if isinstance(arg, list):
            if len(arg) == 1:
                return [arg[0]] * n_categories
            else:
                return arg
        else:
            return [arg] * n_categories
