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
from torch.nn import CrossEntropyLoss, Linear, Module, ReLU, Sequential
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

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
        n_current_buckets = old_classifier.out_features

        if n_buckets <= n_current_buckets:
            return

        logger.debug(f'Expanding classifier to {n_buckets} buckets')

        new_classifier = Linear(old_classifier.in_features, n_buckets)
        with torch.no_grad():
            new_classifier.weight[:n_current_buckets] = old_classifier.weight[:n_current_buckets]
            new_classifier.bias[:n_current_buckets] = old_classifier.bias[:n_current_buckets]

        self.layers[-1] = new_classifier


class LMIReferenceDataset(Dataset):
    def __init__(self, buckets: dict[int, Tensor], references: list[tuple[int, int]]):
        self.buckets = buckets
        self.references = references  # [(bucket_id, data_idx), ...]
        self.labels = torch.tensor([b for b, _ in references], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        bucket_id, data_index = self.references[index]
        return self.buckets[bucket_id][data_index], self.labels[index]


class LMIDataset(Dataset):
    def __init__(self, X: Tensor, y: Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.X[index], self.y[index]


class DynamicLMI:
    def __init__(self, alpha: float, n_buckets: int, data_dimensionality: int, model: MLP, replay_size: int):
        self._alpha: float = alpha
        """Multiplier used in the calculation of the number of buckets."""
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
        self._replay_size: int = replay_size
        """Maximum amount of old data to use when retraining the model."""

    @utils.measure_runtime
    @staticmethod
    def _train_model(
        model: MLP,
        train_loader: DataLoader,
        epochs: int,
        lr: float,
    ) -> None:
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

    @utils.measure_runtime
    def _retrain_model(self, affected_buckets: list[int]) -> None:
        assert self.model is not None, 'Model is not trained yet.'

        # Load all the data from affected buckets (to account for new bucket and modified split bucket)
        affected_references = list()
        for bucket_id in affected_buckets:
            n_samples = self.bucket_data[bucket_id].shape[0]
            affected_references.extend([(bucket_id, i) for i in range(n_samples)])

        per_bucket = self._replay_size // max(1, (self.n_buckets - len(affected_buckets)))

        replay_references = list()
        for bucket_id in range(self.n_buckets):
            if bucket_id in affected_buckets:
                continue
            data = self.bucket_data[bucket_id]
            if data.shape[0] == 0:
                continue

            selected_indices = utils.herd_from(data, per_bucket)
            replay_references.extend([(bucket_id, i) for i in selected_indices])

        logger.debug(f'Retraining model with affected buckets {affected_buckets}')

        affected_dataset = LMIReferenceDataset(self.bucket_data, affected_references)
        replay_dataset = LMIReferenceDataset(self.bucket_data, replay_references)
        train_dataset = ConcatDataset([affected_dataset, replay_dataset])

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=256,
            shuffle=True,
        )

        DynamicLMI._train_model(
            model=self.model,
            train_loader=train_loader,
            epochs=5,
            lr=0.00098,
        )

        del affected_dataset, replay_dataset, train_dataset, train_loader
        gc.collect()

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
        predicted_buckets: list[int],
        query: Tensor,
        query_idx: int,
    ) -> tuple[Tensor, Tensor, int]:
        n_buckets = len(predicted_buckets)
        Is = torch.empty((k * n_buckets,), dtype=torch.int32)
        Ds = torch.empty((k * n_buckets,))

        for nth_bucket in range(n_buckets):
            D, I = self._visit_bucket(predicted_buckets[nth_bucket], query, k)

            start, stop = nth_bucket * k, (nth_bucket + 1) * k
            Ds[start:stop], Is[start:stop] = D, I

        dists, indices_to_keep = torch.topk(Ds, k)

        return dists, Is[indices_to_keep], query_idx

    def _split_bucket(self, bucket: int) -> int:
        data = self.bucket_data[bucket]
        ids = self.bucket_data_ids[bucket]

        y = DynamicLMI._run_kmeans(2, self.dimensionality, data)

        new_bucket = self.n_buckets

        split_existing = (y == 0)
        split_new = ~split_existing

        self.bucket_data[bucket] = data[split_existing]
        self.bucket_data_ids[bucket] = ids[split_existing]
        self.bucket_data[new_bucket] = data[split_new]
        self.bucket_data_ids[new_bucket] = ids[split_new]

        self.n_buckets += 1

        return new_bucket

    @utils.measure_runtime
    def search(self, queries: Tensor, k: int, n_candidates: int = 1_000) -> tuple[np.ndarray, np.ndarray]:
        predicted_bucket_ids = self._predict(queries, n_candidates)
        n_queries = queries.shape[0]
        D = np.empty((n_queries, k), dtype=np.float32)
        I = np.empty((n_queries, k), dtype=np.int32)

        torch.set_num_threads(3)
        faiss.omp_set_num_threads(3)

        with ThreadPoolExecutor(max_workers=9) as executor:
            results = executor.map(
                lambda i: self._visit_buckets(k, predicted_bucket_ids[i], queries[i : i + 1], i),
                range(n_queries),
            )
            for dists, nns, query_id in tqdm(results, total=n_queries):
                D[query_id, :] = dists
                I[query_id, :] = nns

        return D, I

    @utils.measure_runtime
    def _predict_top_bucket(self, X: Tensor) -> Tensor:
        assert self.model is not None, 'Model is not trained yet.'

        self.model.eval()

        with torch.no_grad():
            logits = self.model(X)

        return logits.topk(1)[1]

    @utils.measure_runtime
    def _predict(self, X: Tensor, n_candidates: int) -> list[list[int]]:
        assert self.model is not None, 'Model is not trained yet.'

        self.model.eval()

        with torch.no_grad():
            logits = self.model(X)
            predicted_buckets: list[list[int]] = list()

            for row in logits:
                _, bucket_indices = torch.sort(row, descending=True)

                buckets = list()
                candidates = 0

                for nth_bucket in range(len(bucket_indices)):
                    bucket = int(bucket_indices[nth_bucket].item())

                    buckets.append(bucket)
                    candidates += self.bucket_data_ids[bucket].shape[0]

                    if candidates >= n_candidates:
                        break

                predicted_buckets.append(buckets)

        return predicted_buckets

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
        predicted_bucket_ids = self._predict_top_bucket(chunk.to(torch.float32)).reshape(-1)
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

        # After creating and filling up all the buckets, set the next available id
        self._next_id = sum(len(ids) for ids in self.bucket_data_ids.values())

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
        alpha: float,
        n_buckets: int,
        chunk_size: int,
        replay_size: int,
    ) -> DynamicLMI:
        n_data, data_dim = utils.get_dataset_shape(dataset)
        X_train = utils.sample_train_subset(dataset, n_data, data_dim, sample_size, chunk_size)

        logger.debug(f'Training on {X_train.shape[0]} subset from {n_data} dataset')

        y = DynamicLMI._run_kmeans(n_buckets, data_dim, X_train)

        nn = MLP(data_dim, n_buckets)
        train_loader = DataLoader(dataset=LMIDataset(X_train, y), batch_size=256, shuffle=True)
        DynamicLMI._train_model(nn, train_loader, epochs, lr)

        del X_train
        gc.collect()

        lmi = DynamicLMI(alpha, n_buckets, data_dim, nn, replay_size)

        # Store the vectors and their IDs in the corresponding buckets
        lmi._create_buckets(dataset, n_data, chunk_size)

        return lmi

    def _get_largest_bucket(self) -> int:
        # Bucket data ids are always expected to have at least one entry
        return max(self.bucket_data_ids.items(), key=lambda item: len(item[1]))[0]

    @utils.measure_runtime
    def insert(
        self,
        data: Tensor,
        # stats: BucketInsertionStats,
    ) -> None:
        # Ensure batched input in case of a single data point
        if data.dim() == 1:
            data = data.unsqueeze(0)

        data = utils.ensure_float32(data).cpu()

        # [[3], [1], [5]] -> [3, 1, 5]
        predicted_bucket_ids = self._predict_top_bucket(data).reshape(-1)

        n_data = data.shape[0]
        new_ids = torch.arange(self._next_id, self._next_id + n_data, dtype=torch.int32)
        self._next_id += n_data

        target_buckets = int(self._alpha * sqrt(self._next_id))
        splits_needed = target_buckets - self.n_buckets
        assert splits_needed >= 0, "Splits needed cannot be negative."

        for i in range(n_data):
            vector = data[i : i + 1].to(torch.float16)
            bucket = int(predicted_bucket_ids[i].item())
            # stats.insert_into(bucket)

            self.bucket_data[bucket] = torch.cat([self.bucket_data[bucket], vector], dim=0)
            self.bucket_data_ids[bucket] = torch.cat([self.bucket_data_ids[bucket], new_ids[i : i + 1]], dim=0)

        affected_buckets = set()  # Some buckets may split multiple times

        for _ in range(splits_needed):
            bucket = self._get_largest_bucket()
            new_bucket = self._split_bucket(bucket)
            affected_buckets.update([bucket, new_bucket])
            # stats.extend_with(new_bucket)

        logger.debug(f"Performed split on buckets {affected_buckets}")

        # Retrain once, after all the splits were done
        if splits_needed > 0:
            self.model.expand_to(self.n_buckets)
            self._retrain_model(affected_buckets=list(affected_buckets))

    # def get_bucket_sizes(self) -> str:
    #     result = str()
    #     for bucket in range(self.n_buckets):
    #         result += f'Bucket {bucket}: {int(self.bucket_data_ids[bucket].shape[0])} samples\n'
    #     return result


# class BucketInsertionStats:
#     def __init__(self, n_buckets: int):
#         self.stats = {bucket: 0 for bucket in range(n_buckets)}
#
#     def insert_into(self, bucket: int) -> None:
#         self.stats[bucket] += 1
#
#     def extend_with(self, bucket: int):
#         if bucket not in self.stats:
#             self.stats[bucket] = 0
#
#     def __str__(self) -> str:
#         result = str()
#         for bucket, inserted in self.stats.items():
#             if inserted >= 1:
#                 result += f'Bucket {bucket}: {inserted} insertions\n'
#         return result


def insert_experiment(
    dataset_size: str,
    epochs: int,
    lr: float,
    sample_size: int,
    n_tasks: int,
    alpha: float,
    n_candidates: int,
    chunk_size: int,
    replay_size: int,
) -> None:
    dataset = Path(f'data2024/laion2B-en-clip768v2-n={dataset_size}.h5')
    initial_task_dataset = utils.create_task_h5py(dataset, 0, n_tasks)

    n_buckets = int(alpha * sqrt(utils.get_dataset_size(initial_task_dataset)))

    start = time.time()
    lmi = DynamicLMI.create(initial_task_dataset, epochs, lr, sample_size, alpha, n_buckets, chunk_size, replay_size)
    utils.delete_task_h5py(initial_task_dataset)
    buildtime = time.time() - start

    # logger.debug(f'Bucket sizes before insert:\n{lmi.get_bucket_sizes()}')
    # insertion_stats = BucketInsertionStats(n_buckets=n_buckets)

    start = time.time()
    for nth_task in range(1, n_tasks):
        task_dataset = utils.create_task_h5py(dataset, nth_task, n_tasks)
        lmi.insert(
            utils.load_dataset(task_dataset),
            # insertion_stats,
        )  # TODO: Rework to insert in chunks
        utils.delete_task_h5py(task_dataset)
    inserttime = time.time() - start

    # logger.debug(f'Bucket insertion stats:\n{insertion_stats}')
    # logger.debug(f'Bucket sizes after insert:\n{lmi.get_bucket_sizes()}')

    queries = utils.load_queries()

    k = 30
    assert k <= n_candidates, 'Number of k neighbors is larger than the candidates to search for.'

    candidates_increments = [(i + 1) * n_candidates for i in range(10)]

    for candidates in candidates_increments:
        start = time.time()
        D, I = lmi.search(queries, k, candidates)
        searchtime = time.time() - start

        identifier = f't1-{dataset_size}-epochs={epochs}-lr={lr}-sample={sample_size}-alpha={alpha}-n_tasks={n_tasks}-chunk_size={chunk_size}-n_candidates={candidates}-replay_size={replay_size}'
        modelingtime, encdatabasetime, encqueriestime = 0.0, 0.0, 0.0

        utils.store_results(
            Path('result/') / 'insert_experiment' / dataset_size / f'{identifier}.h5',
            'dlmi',
            D,
            I + 1,
            modelingtime,
            encdatabasetime,
            encqueriestime,
            buildtime,
            inserttime,
            searchtime,
            identifier,
            candidates,
            dataset_size,
        )


def static_experiment(
    dataset_size: str,
    epochs: int,
    lr: float,
    sample_size: int,
    n_tasks: int,
    alpha: float,
    n_candidates: int,
    chunk_size: int,
    replay_size: int,
) -> None:
    dataset = Path(f'data2024/laion2B-en-clip768v2-n={dataset_size}.h5')

    n_buckets = int(alpha * sqrt(utils.get_dataset_size(dataset)))

    start = time.time()
    lmi = DynamicLMI.create(dataset, epochs, lr, sample_size, alpha, n_buckets, chunk_size, replay_size)
    buildtime = time.time() - start

    queries = utils.load_queries()

    k = 30
    assert k <= n_candidates, 'Number of k neighbors is larger than the candidates to search for.'

    candidates_increments = [(i + 1) * n_candidates for i in range(10)]

    for candidates in candidates_increments:
        start = time.time()
        D, I = lmi.search(queries, k, candidates)
        searchtime = time.time() - start

        identifier = f't1-{dataset_size}-epochs={epochs}-lr={lr}-sample={sample_size}-alpha={alpha}-chunk_size={chunk_size}-n_candidates={candidates}'
        modelingtime, encdatabasetime, encqueriestime = 0.0, 0.0, 0.0

        utils.store_results(
            Path('result/') / 'static_experiment' / dataset_size / f'{identifier}.h5',
            'lmi',
            D,
            I + 1,
            modelingtime,
            encdatabasetime,
            encqueriestime,
            buildtime,
            0.0,
            searchtime,
            identifier,
            candidates,
            dataset_size,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.00098)
    parser.add_argument('--sample-size', type=int, default=1_000_000)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--n-tasks', type=int, default=100)
    parser.add_argument('--n-candidates', type=int, default=1_000)  # minimum candidates to load during search
    parser.add_argument('--dataset-size', type=str, default='100M')
    parser.add_argument('--chunk-size', type=int, default=1_000_000)
    parser.add_argument('--replay-size', type=int, default=5_000)
    args = parser.parse_args()

    # python insert_experiment.py --dataset-size 300K --sample-size 100000 --chunk-size 100000 --n-candidates 500 --n-tasks 2 &>insert_experiment.log
    insert_experiment(**vars(args))

    # python insert_experiment.py --dataset-size 300K --sample-size 100000 --chunk-size 100000 --n-candidates 500 &>static_experiment.log
    # static_experiment(**vars(args))
