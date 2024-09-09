from __future__ import annotations

import os
os.environ['MKL_NUM_THREADS'] = '27'
os.environ['OMP_NUM_THREADS'] = '27'
os.environ['OMP_DYNAMIC'] = 'FALSE'
os.environ['MKL_DYNAMIC'] = 'FALSE'

import argparse
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from math import ceil, sqrt
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.utils
from loguru import logger
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, ReLU, Sequential
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import utils

torch.set_num_threads(27)
faiss.omp_set_num_threads(27)
SEED = 42
torch.manual_seed(SEED)

Offsets = dict[int, dict[int, int]]


class LMIDataset(Dataset):
    def __init__(self, X: Tensor, y: Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.X[index], self.y[index]


class LMI:
    def __init__(self, n_buckets: int, data_dimensionality: int, model: Sequential):
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

    @utils.measure_runtime
    @staticmethod
    def _train_model(
        model: Sequential,
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

    @utils.measure_runtime
    def search(self, queries: Tensor, k: int, nprobe: int = 100) -> tuple[np.ndarray, np.ndarray]:
        predicted_bucket_ids = self._predict(queries, nprobe)
        n_queries = queries.shape[0]
        D = np.empty((n_queries, k), dtype=np.float16)
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

        nn = Sequential(
            Linear(data_dim, 512),
            ReLU(),
            Linear(512, n_buckets),
        )

        LMI._train_model(nn, X_train, y, epochs, lr)

        del X_train
        gc.collect()

        lmi = LMI(n_buckets, data_dim, nn)

        # Store the vectors and their IDs in the corresponding buckets
        lmi._create_buckets(dataset, n_data, chunk_size)

        return lmi


def task1(
    dataset_size: str,
    epochs: int,
    lr: float,
    sample_size: int,
    alpha: float,
    # nprobe: int,
    chunk_size: int,
) -> None:
    dataset = Path(f'data2024/laion2B-en-clip768v2-n={dataset_size}.h5')

    n_buckets = int(alpha * sqrt(utils.get_dataset_size(dataset)))

    start = time.time()
    lmi = LMI.create(dataset, epochs, lr, sample_size, n_buckets, chunk_size)
    buildtime = time.time() - start

    queries = utils.load_queries()

    k = 30

    nprobes = range(1, 30 + 1)
    if dataset_size == '300K':
        nprobes = [1]

    for nprobe in nprobes:
        start = time.time()
        D, I = lmi.search(queries, k, nprobe)
        searchtime = time.time() - start

        identifier = f't1-{dataset_size}-epochs={epochs}-lr={lr}-sample={sample_size}-alpha={alpha}-chunk_size={chunk_size}-nprobe={nprobe}'
        modelingtime, encdatabasetime, encqueriestime = 0.0, 0.0, 0.0

        utils.store_results(
            Path('result/') / 'task1' / dataset_size / f'{identifier}.h5',
            'lmi',
            D,
            I + 1,
            modelingtime,
            encdatabasetime,
            encqueriestime,
            buildtime,
            searchtime,
            identifier,
            dataset_size,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.00098)
    parser.add_argument('--sample-size', type=int, default=1_000_000)
    parser.add_argument('--alpha', type=float, default=1.0)
    # parser.add_argument('--nprobe', type=int, default=5)
    parser.add_argument('--dataset-size', type=str, default='100M')
    parser.add_argument('--chunk-size', type=int, default=1_000_000)
    args = parser.parse_args()

    task1(**vars(args))
