from __future__ import annotations

from pathlib import Path

import h5py
import torch
from faiss import METRIC_L2, Kmeans, knn
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, ReLU, Sequential
from torch.nn.functional import softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

SEED = 42
torch.manual_seed(SEED)


# Encapsulates the dataset and the labels for training
class LMIDataset(Dataset):
    def __init__(self, X: Tensor, y: Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.X[index], self.y[index]


class LMI:
    def __init__(self, n_buckets: int, data_dimensionality: int):
        self.n_buckets: int = n_buckets
        """Number of buckets."""
        self.dimensionality: int = data_dimensionality
        """Dimensionality of the data."""
        self.bucket_data: dict[int, Tensor] = {}
        """Mapping from bucket ID to the data in the bucket."""
        self.bucket_data_ids: dict[int, Tensor] = {}
        """Mapping from bucket ID to the indices of the data in the bucket."""

        # Create a model
        self.model = Sequential(
            Linear(data_dimensionality, 512),
            ReLU(),
            Linear(512, 384),
            ReLU(),
            Linear(384, n_buckets),
        )

        # Model's hyperparameters
        self.epochs = 10
        self.lr = 0.001
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(params=self.model.parameters(), lr=self.lr)

    def train(self, X: Tensor) -> None:
        assert self.dimensionality == X.shape[1]

        # Run k-means to obtain training labels
        kmeans = Kmeans(
            d=self.dimensionality,
            k=self.n_buckets,
            verbose=True,
            seed=SEED,
        )
        kmeans.train(X)
        y = torch.from_numpy(kmeans.index.search(X, 1)[1].T[0])  # type: ignore

        # Prepare the data loader for training
        train_loader = DataLoader(
            dataset=LMIDataset(X, y), batch_size=256, shuffle=True
        )

        # Train the model
        self.model.train()

        step = max(1, self.epochs // 10)
        print(f"Epochs: {self.epochs}, step: {step}")

        for epoch in range(self.epochs):
            for X_batch, y_batch in train_loader:
                loss = self.loss_fn(self.model(X_batch), y_batch)

                # Do the backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if epoch % step == 0 and epoch != 0:
                print(f"Epoch {epoch} | Loss {loss.item():.5f}")  # type: ignore

        # Predict to which bucket each vector belongs
        classes = self._predict(X, 1)[1].reshape(-1)

        # Store the vectors and their IDs in the corresponding buckets
        self.bucket_data = {i: X[classes == i] for i in range(self.n_buckets)}
        self.bucket_data_ids = {
            i: torch.where(classes == i)[0] for i in range(self.n_buckets)
        }

    def search(self, query: Tensor, k: int) -> Tensor:
        # Bucket in which we should search for the nearest neighbors
        bucket_id = int(self._predict(query, 1)[1].item())

        # Retrieve the data and their IDs from the bucket
        bucket_data = self.bucket_data[bucket_id]
        bucket_data_ids = self.bucket_data_ids[bucket_id]

        # Compute the k nearest neighbors in a brute-force way
        # ! Beware: these are not global nearest neighbors IDs, but only the IDs of nearest neighbors within the bucket
        nearest_neighbors = knn(query, bucket_data, k, metric=METRIC_L2)[1][0]

        # Convert the indices back to the global IDs
        return bucket_data_ids[nearest_neighbors]

    def _predict(self, X: Tensor, top_k: int) -> tuple[Tensor, Tensor]:
        assert self.model is not None, "Model is not trained yet."

        self.model.eval()

        # Disable gradient computation
        with torch.no_grad():
            # Evaluate the model
            logits = self.model(X)

        # Compute probabilities from logits
        probs = softmax(logits, dim=1)
        # Select the top k most probable classes and their probabilities
        probabilities, classes = probs.topk(top_k)

        return probabilities, classes


if __name__ == "__main__":
    # Load the dataset
    dataset_path = Path("laion2B-en-clip768v2-n=100K.h5")
    X = torch.from_numpy(h5py.File(dataset_path, "r")["emb"][:]).to(torch.float32)  # type: ignore
    n, d = X.shape

    # Create an instance of the LMI
    lmi = LMI(n_buckets=320, data_dimensionality=d)
    lmi.train(X)

    # Obtain a query from the user -- here we sample a random query from the dataset
    query = X[torch.randint(0, n, (1,))]
    # Number of neighbors to look for
    k = 10
    # Search for the k nearest neighbors
    nearest_neighbors = lmi.search(query, k)

    # Evaluate the accuracy of the LMI's result

    # Calculate the ground truth for the query over the whole dataset
    ground_truth = torch.argsort(torch.cdist(query, X)).reshape(-1)[:k]

    # Calculate the recall -- closer to 1 is better
    recall = (
        len(set(nearest_neighbors.tolist()).intersection(set(ground_truth.tolist())))
        / k
    )
    print(f"Recall: {recall}")
