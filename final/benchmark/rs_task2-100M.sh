#!/bin/bash
#PBS -l select=1:ncpus=8:mem=36gb:cluster=elmu1:scratch_ssd=160gb
#PBS -l walltime=24:00:00

cp "/storage/brno12-cerit/home/prochazka/datasets/sisap24/laion2B-en-clip768v2-n=100M.h5" "${SCRATCHDIR}/laion2B-en-clip768v2-n=100M.h5" || {
	echo >&2 "Could not copy dataset"
	exit 1
}

export OMP_NUM_THREADS=$PBS_NUM_PPN

module add mambaforge || exit 2
mamba activate /storage/brno12-cerit/home/lickomar/mamba/rustenv || exit 3

python3 /storage/brno12-cerit/home/lickomar/sisap24-rs/run.py \
    --task 2 \
    --dataset-size "100M" \
    --dataset-base-path "${SCRATCHDIR}" \
    --output-dir "/storage/brno12-cerit/home/lickomar/sisap24-rs/fr_final_results" \
    --epochs 15 \
    --lr 0.00098 \
    --sample-size 1000000 \
    --chunk-size-build 100000 \
    --alpha 1.0 \
    --reduced-dim 135 \
    --k 30 \
    --rerank \
    --ncandidates-rerank 1000 \
    --nprobes 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 80 90 100 \
    &>/storage/brno12-cerit/home/lickomar/sisap24-rs/logs/final-task2-100M.log