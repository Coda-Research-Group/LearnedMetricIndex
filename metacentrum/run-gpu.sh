#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=48:0:0
#PBS -l select=1:ngpus=1:ncpus=28:mem=512gb:scratch_local=200gb
#PBS -N task1_gpu

export OMP_NUM_THREADS=$PBS_NUM_PPN

cd $SCRATCHDIR || exit 1
module add mambaforge || exit 2
cp /storage/brno12-cerit/home/cernansky-jozef/LearnedMetricIndex/task1-gpu.py gpu.py || exit 4
cp /storage/brno12-cerit/home/cernansky-jozef/LearnedMetricIndex/task1.py base.py || exit 4
cp /storage/brno12-cerit/home/cernansky-jozef/LearnedMetricIndex/utils.py utils.py || exit 5
cp '/storage/brno2/home/cernansky-jozef/datasets/laion2B-en-clip768v2-n=100M.h5' './laion2B-en-clip768v2-n=100M.h5' || exit 6

mamba activate /storage/brno12-cerit/home/cernansky-jozef/.conda/envs/lmi-gpu || exit 3
python3 gpu.py &>/storage/brno12-cerit/home/cernansky-jozef/logs/gpu.log
mamba activate /storage/brno12-cerit/home/cernansky-jozef/.conda/envs/learnedmetricindex-bp || exit 3
python3 base.py &>/storage/brno12-cerit/home/cernansky-jozef/logs/gpu-base.log

CODE=$?

rm -r ./*
exit $CODE