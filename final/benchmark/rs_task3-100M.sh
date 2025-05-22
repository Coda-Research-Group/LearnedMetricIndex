#!/bin/bash
#PBS -l select=1:ncpus=32:mem=400gb:cluster=elmu1
#PBS -l walltime=24:00:00

export OMP_NUM_THREADS=$PBS_NUM_PPN

module add mambaforge || exit 2
mamba activate /storage/brno12-cerit/home/lickomar/mamba/rustenv || exit 3

python3 /storage/brno12-cerit/home/lickomar/sisap24-rs/run.py \
    --task 3 \
    --dataset-size "100M" \
    --output-dir "/storage/brno12-cerit/home/lickomar/sisap24-rs/frfinal_results" \
    --epochs 15 \
    --lr 0.00098 \
    --sample-size 1000000 \
    --chunk-size-build 500000 \
    --alpha 1.0 \
    --reduced-dim 240 \
    --k 30 \
    --nprobes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
    &>/storage/brno12-cerit/home/lickomar/sisap24-rs/logs/final-task3-100M.log