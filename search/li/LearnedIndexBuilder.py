import time
from itertools import product, takewhile
from logging import DEBUG
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.utils.data
from li.BuildConfiguration import BuildConfiguration
from li.clustering import ClusteringAlgorithm
from li.LearnedIndex import LearnedIndex
from li.Logger import Logger
from li.model import LIDataset, ModelParameters, NeuralNetwork, data_X_to_torch
from li.PriorityQueue import EMPTY_VALUE
from li.utils import filter_path_idxs, log_runtime
from tqdm import tqdm


class LearnedIndexBuilder(Logger):
    def __init__(self, data: pd.DataFrame, config: BuildConfiguration):
        # Attributes used during the build
        self.data = data
        """Data to build the index on."""

        self.config = config
        """Model configurations for each level of the index."""

        # Attributes passed to the index after the build
        self.root_model: Optional[NeuralNetwork] = None
        """The root model of the index."""

        self.internal_models: Dict[Tuple, NeuralNetwork] = {}
        """
        Dictionary mapping the path to the internal model.
        A path is padded with `EMPTY_VALUE` to the right to match the length of the longest path.
        """

        self.bucket_paths: List[Tuple] = []
        """List of paths to the buckets."""

    def build(
        self,
    ) -> Tuple[LearnedIndex, npt.NDArray[np.int64], int, float, float]:
        """
        Builds the index.

        Parameters
        ----------
        data : pd.DataFrame
            Data to build the index on.
        config : BuildConfiguration
            Configuration for the training.

        Returns
        -------
        Tuple[npt.NDArray[np.int64], int, float, float]
            An array of shape (data.shape[0], len(config.n_levels)) with predicted paths for each data point,
            number of buckets, time it took to build the index, time it took to cluster the data.
        """
        s = time.time()

        n_levels = self.config.n_levels

        # Where should the training data be placed with respect to each level
        data_prediction: npt.NDArray[np.int64] = np.full(
            (self.data.shape[0], n_levels), fill_value=EMPTY_VALUE, dtype=np.int64
        )

        self.logger.debug("Training the root model.")
        self.root_model, root_cluster_t = self._train_model(
            self.data,
            self.config.level_configurations[0],
        )
        data_prediction[:, 0] = self.root_model.predict(data_X_to_torch(self.data))

        if n_levels == 1:
            for bucket_index in range(len(np.unique(data_prediction[:, 0]))):
                self.bucket_paths.append((bucket_index,))

            return (
                self._create_index(),
                data_prediction,
                len(self.bucket_paths),
                time.time() - s,
                root_cluster_t,
            )

        self.logger.debug(f"Training {self.config.n_categories[:-1]} internal models.")
        s_internal = time.time()
        internal_cluster_t = self._train_internal_models(
            self.data,
            data_prediction,
            self.config,
        )
        self.logger.debug(
            f"Trained {self.config.n_categories[:-1]} internal models in {time.time()-s_internal:.2f}s."
        )

        return (
            self._create_index(),
            data_prediction,
            len(self.bucket_paths),
            time.time() - s,
            root_cluster_t + internal_cluster_t,
        )

    def _create_index(self) -> LearnedIndex:
        """Creates the index from the trained models."""
        assert self.root_model is not None, "The root model is not trained."

        return LearnedIndex(
            self.root_model,
            self.internal_models,
            self.bucket_paths,
        )

    @log_runtime(DEBUG, "Trained the model in: {}")
    def _train_model(
        self,
        data: pd.DataFrame,
        model_parameters: ModelParameters,
    ) -> Tuple[NeuralNetwork, float]:
        """
        Trains a single model.
        The model is trained until it predicts the correct number of categories.
        The same number of epochs is used for each training iteration.

        Parameters
        ----------
        data : pd.DataFrame
            Data to train the model on.
        clustering_algorithm : ClusteringAlgorithm
            Clustering algorithm to use.
        model_type : str
            Type of the model.
        epochs : int
            The minimal number of epochs to train the model for.
        lr : float
            Learning rate for the model.
        n_categories : int
            Number of categories to predict.

        Returns
        -------
        Tuple[NeuralNetwork, float]
            Trained model, time it took to cluster the data.

        Raises
        ------
        RuntimeError
            If the model does not converge after 1000 iterations
            (after training `epochs` epochs 1000 times).
        """
        clustering_algorithm, model_type, epochs, lr, n_categories = model_parameters

        _, labels, cluster_t = self._cluster(data, clustering_algorithm, n_categories)
        n_clusters = len(np.unique(labels))

        if n_clusters != n_categories:
            self.logger.debug(
                "Clustering algorithm did not return %d clusters, got %d.",
                n_categories,
                n_clusters,
            )
            self.logger.debug("Setting n_categories to %d.", n_clusters)
            n_categories = n_clusters

        train_loader = torch.utils.data.DataLoader(
            dataset=LIDataset(data, labels),
            batch_size=256,
            sampler=torch.utils.data.SubsetRandomSampler(data.index.values.tolist()),
        )
        torch_data = data_X_to_torch(data)

        model = NeuralNetwork(
            input_dim=data.shape[1],
            output_dim=n_categories,
            lr=lr,
            model_type=model_type,
        )
        is_trained = False

        iters = 0
        while not is_trained:
            model.train_batch(train_loader, epochs=epochs, logger=self.logger)
            predictions = model.predict(torch_data)
            iters += 1

            if iters > 1_000:
                raise RuntimeError("The model did not converge after 1000 iterations.")

            is_trained = len(np.unique(predictions)) == n_categories

        if iters > 1:
            self.logger.debug(
                f"Trained for {iters * epochs} epochs instead of {epochs}."
            )

        return model, cluster_t

    def _train_internal_models(
        self,
        data: pd.DataFrame,
        data_prediction: npt.NDArray[np.int64],
        config: BuildConfiguration,
    ) -> float:
        """
        Trains the internal models.

        ! The `data_prediction` array is modified in-place.

        Parameters
        ----------
        data : pd.DataFrame
            Data to train the models on.
        data_prediction : npt.NDArray[np.int64]
            Predicted paths for each data point.
        config : BuildConfiguration
            Configuration for the training.

        Returns
        -------
        float
            Time it took to cluster the data.
        """
        assert (
            self.root_model is not None
        ), "The root model is not trained, call `_train_root_model` first."

        overall_cluster_t = 0.0

        for level in range(1, config.n_levels):
            internal_node_paths = self._generate_internal_node_paths(
                level, config.n_levels, config.n_categories
            )
            self.logger.debug(f"Training level {level}.")

            for path in tqdm(internal_node_paths):
                self.logger.debug(f"Training model on path {path}.")

                data_idxs = filter_path_idxs(data_prediction, path)
                assert (
                    data_idxs.shape[0] != 0
                ), "There are no data points associated with the given path."

                # +1 as the data is indexed from 1
                training_data = data.loc[data_idxs + 1]

                # The subset needs to be reindexed; otherwise, the object accesses are invalid.
                original_pd_indices = training_data.index.values
                training_data = training_data.set_index(
                    pd.Index(range(1, training_data.shape[0] + 1))
                )

                model, cluster_t = self._train_model(
                    training_data,
                    config.level_configurations[level],
                )
                self.internal_models[path] = model

                overall_cluster_t += cluster_t

                # Restore back to the original indices
                training_data = training_data.set_index(
                    pd.Index(original_pd_indices.tolist())
                )

                predictions = model.predict(data_X_to_torch(training_data))

                # original_pd_indices-1 as data is indexed from 1
                # level as we are predicting the next level but the indexing is 0-based
                data_prediction[original_pd_indices - 1, level] = predictions

                if level == config.n_levels - 1:
                    for bucket_index in range(len(np.unique(predictions))):
                        self.bucket_paths.append(path[:-1] + (bucket_index,))

        return overall_cluster_t

    def _cluster(
        self,
        data: pd.DataFrame,
        clustering_algorithm: ClusteringAlgorithm,
        n_clusters: int,
    ) -> Tuple[Optional[Any], npt.NDArray[np.int32], float]:
        s = time.time()

        if data.shape[0] < 2:
            return None, np.array([0] * data.shape[0]), time.time() - s

        if data.shape[0] < n_clusters:
            n_clusters = data.shape[0] // 5
            if n_clusters < 2:
                n_clusters = 2

        clustering_object, labels = clustering_algorithm(
            np.array(data),
            n_clusters,
            None,
        )

        return clustering_object, labels, time.time() - s

    def _serialize_path(self, path: Tuple) -> str:
        """
        Serializes the path to a string.

        Example:
        >>> self._serialize_path((1, 2, -1, -1))
        "1.2"
        """
        valid_path = takewhile(lambda x: x != EMPTY_VALUE, path)

        return ".".join(map(str, valid_path))

    def _deserialize_path(self, path: str, n_levels: int) -> Tuple:
        """
        Deserializes the path from a string.

        Example:
        >>> self._deserialize_path("1.2", 4)
        (1, 2, -1, -1)
        """
        levels = path.split(".")

        return tuple(list(map(int, levels)) + [EMPTY_VALUE] * (n_levels - len(levels)))

    def _generate_internal_node_paths(
        self, level: int, n_levels: int, n_categories: List[int]
    ) -> List[Tuple]:
        """Generates all possible paths to internal nodes at the given `level`.

        Parameters
        ----------
        level : int
            Desired level of the internal nodes.
        n_levels : int
            Total number of levels in the index.
        n_categories : List[int]
            Number of categories for each level of the index.

        Returns
        -------
        List[Tuple]
            List of all possible paths to internal nodes at the given `level`.
        """
        path_combinations = [range(n_categories[lvl]) for lvl in range(level)]
        padding = [[EMPTY_VALUE]] * (n_levels - level)

        return list(product(*path_combinations, *padding))
