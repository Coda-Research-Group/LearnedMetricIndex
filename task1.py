from __future__ import annotations

import os

from matplotlib import pyplot as plt

os.environ['MKL_NUM_THREADS'] = '27'
os.environ['OMP_NUM_THREADS'] = '27'
os.environ['OMP_DYNAMIC'] = 'FALSE'
os.environ['MKL_DYNAMIC'] = 'FALSE'

import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
from math import ceil, sqrt
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.utils
from loguru import logger
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Module, ReLU, Sequential
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import eval
import utils

torch.set_num_threads(27)
faiss.omp_set_num_threads(27)
SEED = 42
torch.manual_seed(SEED)

Offsets = dict[int, dict[int, int]]


class MLP(Module):
    def __init__(self, in_features: int, out_features: int):
        super(MLP, self).__init__()
        self.layers = Sequential(
            Linear(in_features, 512),
            ReLU(),
            Linear(512, out_features),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.layers(inputs)
        return outputs

    def expand_to(self, n_buckets: int) -> None:
        old_classifier: Linear = self.layers[-1]
        current_classes = old_classifier.out_features

        if n_buckets <= current_classes:
            return

        new_classifier = Linear(old_classifier.in_features, n_buckets)
        with torch.no_grad():
            new_classifier.weight[:current_classes] = old_classifier.weight[:current_classes]
            new_classifier.bias[:current_classes] = old_classifier.bias[:current_classes]
        self.layers[-1] = new_classifier


class LMIDataset(Dataset):
    def __init__(self, X: Tensor, y: Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.X[index], self.y[index]


class LMI:
    def __init__(self, n_buckets: int, data_dimensionality: int, model: MLP):
        self.n_buckets: int = n_buckets
        """Number of buckets."""
        self.dimensionality: int = data_dimensionality
        """Dimensionality of the data."""
        self.model = model
        """Model."""
        self.bucket_data: dict[int, Tensor] = {}
        """Mapping from bucket ID to the data in the bucket."""
        self.bucket_data_ids: dict[int, Tensor] = {}
        """Mapping from bucket ID to the indices of the data in the bucket."""
        self._next_id: int = 0
        """Keeps track of the next available unique data ID to be inserted."""
        self._bucket_threshold: int = 0
        """Maximum possible number of samples in a bucket before split."""

    @utils.measure_runtime
    @staticmethod
    def _train_model(
        model: MLP,
        X: Tensor,
        y: Tensor,
        epochs: int,
        lr: float,
    ) -> None:
        train_loader = DataLoader(dataset=LMIDataset(X, y), batch_size=256, shuffle=True)
        loss_fn = CrossEntropyLoss()
        optimizer = Adam(params=model.parameters(), lr=lr)

        model.train()

        logger.debug(f'Epochs: {epochs}')

        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                loss = loss_fn(model(X_batch.to(torch.float32)), y_batch)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            logger.debug(f'Epoch {epoch} | Loss {loss.item()}')  # type: ignore

        logger.debug('Finished training')

    def _visit_bucket(self, bucket: int, query: Tensor, k: int) -> tuple[Tensor, Tensor]:
        if len(self.bucket_data[bucket]) == 0:
            return torch.full((k,), float('-inf'), dtype=torch.float32), torch.full((k,), -1)

        bucket_data = self.bucket_data[bucket].to(torch.float32)
        D, I = faiss.knn(query, bucket_data, k, metric=faiss.METRIC_INNER_PRODUCT)
        del bucket_data

        temp_dist = torch.from_numpy(D[0])
        temp_answer = self.bucket_data_ids[bucket][I[0]]

        return temp_dist, temp_answer

    def _visit_buckets(
        self,
        k: int,
        predicted_buckets: Tensor,
        query: Tensor,
        query_idx: int,
        nprobe: int,
    ) -> tuple[Tensor, Tensor, int]:
        Is = torch.empty((k * nprobe,), dtype=torch.int32)
        Ds = torch.empty((k * nprobe,))

        for nth_bucket in range(nprobe):
            D, I = self._visit_bucket(int(predicted_buckets[nth_bucket].item()), query, k)

            start, stop = nth_bucket * k, (nth_bucket + 1) * k
            Ds[start:stop], Is[start:stop] = D, I

        dists, indices_to_keep = torch.topk(Ds, k)

        return dists, Is[indices_to_keep], query_idx

    def _split_bucket(self, bucket: int) -> None:
        logger.warning(f"Splitting bucket {bucket} with {len(self.bucket_data[bucket])} samples")

        data = self.bucket_data[bucket]
        ids = self.bucket_data_ids[bucket]

        y = self._run_kmeans(2, self.dimensionality, data)

        new_bucket = self.n_buckets

        split_existing = (y == 0)
        split_new = ~split_existing

        self.bucket_data[bucket] = data[split_existing]
        self.bucket_data_ids[bucket] = ids[split_existing]
        self.bucket_data[new_bucket] = data[split_new]
        self.bucket_data_ids[new_bucket] = ids[split_new]

        self.n_buckets += 1
        self.model.expand_to(self.n_buckets)

    @utils.measure_runtime
    def search(self, queries: Tensor, k: int, nprobe: int = 100) -> tuple[np.ndarray, np.ndarray]:
        predicted_bucket_ids = self._predict(queries, nprobe)
        n_queries = queries.shape[0]
        D = np.empty((n_queries, k), dtype=np.float32)
        I = np.empty((n_queries, k), dtype=np.int32)

        torch.set_num_threads(3)
        faiss.omp_set_num_threads(3)

        with ThreadPoolExecutor(max_workers=9) as executor:
            results = executor.map(
                lambda i: self._visit_buckets(k, predicted_bucket_ids[i], queries[i : i + 1], i, nprobe),
                range(n_queries),
            )
            for dists, nns, query_id in tqdm(results, total=n_queries):
                D[query_id, :] = dists
                I[query_id, :] = nns

        return D, I

    @utils.measure_runtime
    def _predict(self, X: Tensor, top_k: int) -> Tensor:
        assert self.model is not None, 'Model is not trained yet.'

        self.model.eval()

        with torch.no_grad():
            logits = self.model(X)

        return logits.topk(top_k)[1]

    def _bucket_init(self, classes: Tensor, bucket: int) -> None:
        indices = torch.where(classes == bucket)[0]
        self.bucket_data_ids[bucket] = indices
        self.bucket_data[bucket] = torch.empty((len(indices), self.dimensionality), dtype=torch.float16)

    def _sort_data(self, data: Tensor, classes: Tensor, bucket: int, start: int, stop: int) -> None:
        self.bucket_data[bucket][start:stop] = data[classes == bucket]

    @utils.measure_runtime
    def _label_data(self, dataset: Path, chunk_i: int, chunk_size: int) -> tuple[Tensor, int, int]:
        start, stop = chunk_i * chunk_size, (chunk_i + 1) * chunk_size

        chunk = utils.load_chunk(dataset, start, stop)
        predicted_bucket_ids = self._predict(chunk.to(torch.float32), 1).reshape(-1)
        del chunk

        return predicted_bucket_ids, start, stop

    def _chunk_sort(self, dataset: Path, classes: Tensor, chunk_i: int, offsets: Offsets, chunk_size: int) -> None:
        start, stop = chunk_i * chunk_size, (chunk_i + 1) * chunk_size
        classes_chunk = classes[start:stop]

        chunk = utils.load_chunk(dataset, start, stop)

        with ThreadPoolExecutor() as executor:
            executor.map(
                lambda x: self._sort_data(
                    chunk,  # noqa: F821
                    classes_chunk,
                    x,
                    offsets[x][chunk_i],
                    offsets[x][chunk_i + 1],
                ),
                range(self.n_buckets),
            )

        del chunk

    def _create_offsets(self, classes: Tensor, n_chunks: int, chunk_size: int) -> Offsets:
        offsets = {i: {0: 0} for i in range(self.n_buckets)}

        for chunk_i in range(n_chunks):
            start, stop = chunk_i * chunk_size, (chunk_i + 1) * chunk_size

            counts = torch.bincount(classes[start:stop], minlength=self.n_buckets)
            for i in range(self.n_buckets):
                offsets[i][chunk_i + 1] = offsets[i][chunk_i] + int(counts[i])

        return offsets

    def _set_threshold(self) -> None:
        total = 0
        for bucket in range(self.n_buckets):
            total += int(self.bucket_data_ids[bucket].shape[0])
        self._bucket_threshold = total // self.n_buckets * 2

    @utils.measure_runtime
    def _create_buckets(self, dataset: Path, n_data: int, chunk_size: int) -> None:
        logger.debug('Started bucket creation')

        n_chunks = ceil(n_data / chunk_size)
        classes = torch.empty((n_data,), dtype=torch.int32)

        for i in range(n_chunks):
            result = self._label_data(dataset, i, chunk_size)
            classes[result[1] : result[2]] = result[0]
        gc.collect()

        offsets = self._create_offsets(classes, n_chunks, chunk_size)

        with ThreadPoolExecutor() as executor:
            executor.map(lambda x: self._bucket_init(classes, x), range(self.n_buckets))

        logger.debug('First part done')

        for i in range(n_chunks):
            self._chunk_sort(dataset, classes, i, offsets, chunk_size)
        gc.collect()

        # After creating and filling up all the buckets, set the next available id and threshold
        self._next_id = sum(len(ids) for ids in self.bucket_data_ids.values())
        self._set_threshold()

    @utils.measure_runtime
    @staticmethod
    def _run_kmeans(n_buckets: int, data_dim: int, X_train: Tensor) -> Tensor:
        kmeans = faiss.Kmeans(
            d=data_dim,
            k=n_buckets,
            verbose=False,
            seed=SEED,
            spherical=True,
        )
        kmeans.train(X_train)
        return torch.from_numpy(kmeans.index.search(X_train, 1)[1].T[0])  # type: ignore

    @utils.measure_runtime
    @staticmethod
    def create(
        dataset: Path,
        epochs: int,
        lr: float,
        sample_size: int,
        n_buckets: int,
        chunk_size: int,
    ) -> LMI:
        n_data, data_dim = utils.get_dataset_shape(dataset)
        X_train = utils.sample_train_subset(dataset, n_data, data_dim, sample_size, chunk_size)

        logger.debug(f'Training on {X_train.shape[0]} subset from {n_data} dataset')

        y = LMI._run_kmeans(n_buckets, data_dim, X_train)

        nn = MLP(data_dim, n_buckets)
        LMI._train_model(nn, X_train, y, epochs, lr)

        del X_train
        gc.collect()

        lmi = LMI(n_buckets, data_dim, nn)

        # Store the vectors and their IDs in the corresponding buckets
        lmi._create_buckets(dataset, n_data, chunk_size)

        return lmi

    @utils.measure_runtime
    def naive_insert(self, data: Tensor, stats: BucketInsertionStats) -> None:
        # Ensure batched input in case of a single data point
        if data.dim() == 1:
            data = data.unsqueeze(0)

        data = utils.ensure_float32(data).cpu()
        predicted_bucket_ids = self._predict(data, top_k=1).reshape(-1)

        n_data = data.shape[0]
        new_ids = torch.arange(self._next_id, self._next_id + n_data, dtype=torch.int32)
        self._next_id += n_data

        for i in range(n_data):
            vector = data[i : i + 1].to(torch.float16)
            bucket = int(predicted_bucket_ids[i].item())
            stats.insert_into(bucket)

            self.bucket_data[bucket] = torch.cat([self.bucket_data[bucket], vector], dim=0)
            self.bucket_data_ids[bucket] = torch.cat([self.bucket_data_ids[bucket], new_ids[i : i + 1]], dim=0)

    def get_bucket_sizes(self) -> str:
        result = str()
        for bucket in range(self.n_buckets):
            result += f'Bucket {bucket}: {int(self.bucket_data_ids[bucket].shape[0])} samples\n'
        return result

class BucketInsertionStats:
    def __init__(self, n_buckets: int):
        self.stats = {bucket: 0 for bucket in range(n_buckets)}

    def insert_into(self, bucket: int) -> None:
        logger.debug(f'Inserting into bucket {bucket}')
        self.stats[bucket] += 1

    def __str__(self) -> str:
        result = str()
        for bucket, inserted in self.stats.items():
            if inserted >= 1:
                result += f'Bucket {bucket}: {inserted} insertions\n'
        return result


def plot_recalls(recalls, nprobe, plot_every=1):
    x, y = zip(*recalls)
    plt.plot(x[::plot_every], y[::plot_every])
    plt.axhline(0.9, linestyle='--', color='red', label='90% Recall Target')
    plt.xlabel('Number of Insertions')
    plt.ylabel(f'Recall After nprobe={nprobe}')
    plt.title('Recall During Naive Inserts')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def task1(
    dataset_size: str,
    epochs: int,
    lr: float,
    sample_size: int,
    alpha: float,
    nprobe: int,
    chunk_size: int,
) -> None:
    dataset = Path(f'data2024/laion2B-en-clip768v2-n={dataset_size}.h5')

    n_buckets = int(alpha * sqrt(utils.get_dataset_size(dataset)))

    lmi = LMI.create(dataset, epochs, lr, sample_size, n_buckets, chunk_size)
    logger.debug(f'Bucket sizes before insert:\n{lmi.get_bucket_sizes()}')

    queries = utils.load_queries()

    # Naive insertion - Experiment
    true_I = eval.get_groundtruth(size='300K')
    recalls = []
    search_every = 10
    k = 10

    insertion_stats = BucketInsertionStats(n_buckets=n_buckets)

    for i in range(queries.shape[0]):
        lmi.naive_insert(queries[i], insertion_stats)  # insert one vector at a time

        if (i + 1) % search_every == 0:
            # Run search on all queries seen so far
            q = queries[: i + 1]
            _, I = lmi.search(q, k=k, nprobe=nprobe)

            # Compare predicted indices to ground truth
            gt = true_I[: i + 1, :k]

            recall = eval.get_recall(I + 1, gt, k=k)
            recalls.append((i + 1, recall))

    logger.debug(f'Bucket insertions:\n{insertion_stats}')
    logger.debug(f'Bucket sizes after insert:\n{lmi.get_bucket_sizes()}')

    # Plot recalls after insertions
    # plot_every = queries.shape[0] // search_every // 25
    plot_recalls(recalls, nprobe)


# python task1.py --dataset-size 300K --sample-size 100000 --chunk-size 100000 --nprobe 1 &>task1.log
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.00098)
    parser.add_argument('--sample-size', type=int, default=1_000_000)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--nprobe', type=int, default=5)
    parser.add_argument('--dataset-size', type=str, default='100M')
    parser.add_argument('--chunk-size', type=int, default=1_000_000)
    args = parser.parse_args()

    task1(**vars(args))
