#!/bin/bash
#PBS -q large_mem@pbs-m1.metacentrum.cz
#PBS -l walltime=72:0:0
#PBS -l select=1:ncpus=32:mem=1000gb:scratch_local=800gb:cluster=eltu
#PBS -N task1_1B

export OMP_NUM_THREADS=$PBS_NUM_PPN

cd $SCRATCHDIR || exit 1
module add mambaforge || exit 2
mamba activate /storage/brno12-cerit/home/cernansky-jozef/.conda/envs/learnedmetricindex-bp || exit 3
cp /storage/brno12-cerit/home/cernansky-jozef/LearnedMetricIndex/experiments/task1-1B.py main.py || exit 4
cp /storage/brno12-cerit/home/cernansky-jozef/LearnedMetricIndex/utils.py utils.py || exit 5
cp '/storage/brno12-cerit/home/cernansky-jozef/datasets/DEEP/base.1B.fbin' './base.1B.fbin' || exit 6

python3 main.py &>/storage/brno12-cerit/home/cernansky-jozef/logs/task1-1B-5M-inner.log
CODE=$?

rm -r ./*
exit $CODE
