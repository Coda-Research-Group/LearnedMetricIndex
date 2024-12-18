#!/bin/bash
#PBS -q large_mem@pbs-m1.metacentrum.cz
#PBS -l walltime=48:0:0
#PBS -l select=1:ncpus=32:mem=1000gb:scratch_local=20gb:cluster=eltu
#PBS -N LMI23vsLMI24

export OMP_NUM_THREADS=$PBS_NUM_PPN

cd $SCRATCHDIR || exit 1
module add mambaforge || exit 2
mamba activate /storage/brno12-cerit/home/cernansky-jozef/.conda/envs/learnedmetricindex-bp || exit 3
cp /storage/brno12-cerit/home/cernansky-jozef/LearnedMetricIndex/task1.py main.py || exit 4
cp /storage/brno12-cerit/home/cernansky-jozef/LearnedMetricIndex/utils.py utils.py || exit 5
cp /storage/brno12-cerit/home/cernansky-jozef/sisap23-laion-challenge-learned-index/search/search.py ./search.py || exit 6
cp -r /storage/brno12-cerit/home/cernansky-jozef/sisap23-laion-challenge-learned-index/search/li ./li || exit 7
cp '/storage/brno2/home/cernansky-jozef/datasets/laion2B-en-clip768v2-n=10M.h5' './laion2B-en-clip768v2-n=10M.h5' || exit 8

python3 main.py --dataset-size 10M &>/storage/brno12-cerit/home/cernansky-jozef/logs/LMI24-10M-profile.log
mamba activate /storage/brno12-cerit/home/cernansky-jozef/.conda/envs/sisap23 || exit 9
python3 search.py --size 10M &>/storage/brno12-cerit/home/cernansky-jozef/logs/LMI23-10M-profile.log
CODE=$?

rm -r ./*
exit $CODE
