#!/bin/bash
#PBS -q large_mem@pbs-m1.metacentrum.cz
#PBS -l walltime=72:0:0
#PBS -l select=1:ncpus=28:mem=512gb:scratch_local=200gb:cluster=elwe
#PBS -N task1_experiments

export OMP_NUM_THREADS=$PBS_NUM_PPN

cd $SCRATCHDIR || exit 1
module add mambaforge || exit 2
mamba activate /storage/brno12-cerit/home/cernansky-jozef/.conda/envs/learnedmetricindex-bp || exit 3
cp /storage/brno12-cerit/home/cernansky-jozef/LearnedMetricIndex/task1-inmemory.py inmemory.py || exit 4
cp /storage/brno12-cerit/home/cernansky-jozef/LearnedMetricIndex/task1.py base.py || exit 4
cp /storage/brno12-cerit/home/cernansky-jozef/LearnedMetricIndex/task1-compile.py compile.py || exit 4
cp /storage/brno12-cerit/home/cernansky-jozef/LearnedMetricIndex/task1-quant.py quant.py || exit 4
cp /storage/brno12-cerit/home/cernansky-jozef/LearnedMetricIndex/utils.py utils.py || exit 5
cp '/storage/brno2/home/cernansky-jozef/datasets/laion2B-en-clip768v2-n=100M.h5' './laion2B-en-clip768v2-n=100M.h5' || exit 6

python3 quant.py &>/storage/brno12-cerit/home/cernansky-jozef/logs/quant.log
python3 compile.py &>/storage/brno12-cerit/home/cernansky-jozef/logs/compile.log
python3 inmemory.py &>/storage/brno12-cerit/home/cernansky-jozef/logs/inmemory.log
python3 base.py &>/storage/brno12-cerit/home/cernansky-jozef/logs/inmemory-base.log
CODE=$?

rm -r ./*
exit $CODE
