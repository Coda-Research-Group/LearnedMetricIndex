# Adapted from https://github.com/sisap-challenges/sisap23-laion-challenge-evaluation/blob/0a6f90debe73365abee210d3950efc07223c846d/eval.py

import argparse
import csv
import glob
import os
from collections.abc import Generator
from pathlib import Path

import numpy as np

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # Solves: Errno 121

import h5py


def get_groundtruth(dataset_name: str) -> np.ndarray:
    out_fn = Path(f'data/{dataset_name}.hdf5')
    gt_f = h5py.File(out_fn, 'r')
    true_I = np.array(gt_f['neighbors'])
    gt_f.close()
    return true_I


def get_all_results(dirname: str) -> Generator[tuple[h5py.File, str], None, None]:
    mask = dirname + '/*.hdf5'
    print(f'search for results matching: {mask}')
    for fn in glob.iglob(mask):  # noqa: PTH207
        print(fn)
        f = h5py.File(fn, 'r')
        dataset_name = fn.split('/')[-1].split('-epochs=')[0]
        if 'knns' not in f:
            print('Ignoring ' + fn)
            f.close()
            continue
        yield f, dataset_name
        f.close()


def get_recall(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    assert k <= I.shape[1]
    assert len(I) == len(gt)

    n = len(I)
    recall = 0
    for i in range(n):
        recall += len(set(I[i, :k]) & set(gt[i, :k]))
    return recall / (n * k)


def return_h5_str(f: h5py.File, param: str) -> str:
    if param not in f:
        return '0'
    x = f[param][()]  # type: ignore
    if isinstance(x, np.bytes_):
        return x.decode()
    return x  # type: ignore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, help='directory in which results are stored', default='result')
    parser.add_argument('csvfile')
    args = parser.parse_args()
    true_I_cache: dict[str, np.ndarray] = {}  # noqa: N816

    k = 10

    columns = [
        'algo',
        'dataset_name',
        'buildtime',
        'querytime',
        'database_size',
        'database_dim',
        'n_queries',
        'params',
        'recall',
    ]

    with Path.open(args.csvfile, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for res, dataset_name in get_all_results(args.results):
            print(dataset_name)
            try:
                d = dict(res.attrs)
            except Exception as e:  # noqa: BLE001
                print(f'Error: {e}')
                d = {k: return_h5_str(res, k) for k in columns}
            if dataset_name not in true_I_cache:
                true_I_cache[dataset_name] = get_groundtruth(dataset_name)
            recall = get_recall(np.array(res['knns']), true_I_cache[dataset_name], k)
            d['dataset_name'] = dataset_name  # type: ignore
            d['recall'] = recall  # type: ignore
            print(d['algo'], d['params'], '=>', recall)
            writer.writerow(d)
