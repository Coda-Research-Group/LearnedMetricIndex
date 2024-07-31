# Adapted from https://github.com/sisap-challenges/sisap23-laion-challenge-evaluation/blob/0a6f90debe73365abee210d3950efc07223c846d/eval.py

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Generator

import numpy as np

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # Solves: Errno 121

import h5py


def get_groundtruth(size: str = '300K'):
    out_fn = Path(f'data2024/gold-standard-dbsize={size}--public-queries-2024-laion2B-en-clip768v2-n=10k.h5')
    gt_f = h5py.File(out_fn, 'r')
    true_I = np.array(gt_f['knns'])
    gt_f.close()
    return true_I


def get_all_results(dirname: str) -> Generator[h5py.File, None, None]:
    mask = [dirname + '/*/*/*.h5', dirname + '/*/result/*/*/*.h5', dirname + '/*/result/*/*/*/*.h5']
    print('search for results matching:')
    print('\n'.join(mask))
    for m in mask:
        for fn in glob.iglob(m):
            print(fn)
            f = h5py.File(fn, 'r')
            if 'knns' not in f or not ('size' in f or 'size' in f.attrs):
                print('Ignoring ' + fn)
                f.close()
                continue
            yield f
            f.close()


def get_recall(I, gt, k: int) -> float:
    assert k <= I.shape[1]
    assert len(I) == len(gt)

    n = len(I)
    recall = 0
    for i in range(n):
        recall += len(set(I[i, :k]) & set(gt[i, :k]))
    return recall / (n * k)


def return_h5_str(f, param):
    if param not in f:
        return 0
    x = f[param][()]
    if type(x) == np.bytes_:
        return x.decode()
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='directory in which results are stored', default='results')
    parser.add_argument('csvfile')
    args = parser.parse_args()
    true_I_cache = {}  # noqa: N816
    test_sizes = ['300K', '10M', '100M']

    columns = [
        'size',
        'algo',
        'modelingtime',
        'encdatabasetime',
        'encqueriestime',
        'buildtime',
        'querytime',
        'params',
        'recall',
    ]

    with Path.open(args.csvfile, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for res in get_all_results(args.results):
            try:
                size = res.attrs['size']
                d = dict(res.attrs)
            except:
                size = res['size'][()].decode()
                d = {k: return_h5_str(res, k) for k in columns}
            if size not in test_sizes:
                continue
            if size not in true_I_cache:
                true_I_cache[size] = get_groundtruth(size)
            recall = get_recall(np.array(res['knns']), true_I_cache[size], 30)
            d['recall'] = recall
            print(d['algo'], d['params'], '=>', recall)
            writer.writerow(d)
