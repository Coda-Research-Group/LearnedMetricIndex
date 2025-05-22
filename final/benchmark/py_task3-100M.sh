#!/bin/bash
#PBS -l select=1:ncpus=8:mem=40gb:cluster=elmu1:scratch_ssd=160gb
#PBS -l walltime=24:00:00

cp "/storage/brno12-cerit/home/prochazka/datasets/sisap24/laion2B-en-clip768v2-n=100M.h5" "${SCRATCHDIR}/laion2B-en-clip768v2-n=100M.h5" || {
	echo >&2 "Could not copy dataset"
	exit 1
}
 
export OMP_NUM_THREADS=$PBS_NUM_PPN
 
# cd /storage/brno12-cerit/home/prochazka/projects/learnedmetricindex-bp/ || exit 1
module add mambaforge || exit 2
mamba activate /storage/brno12-cerit/home/prochazka/.conda/envs/learnedmetricindex-bp || exit 3
 
python3 /storage/brno12-cerit/home/lickomar/sisap24-python/task2.py --dataset-size 100M --dataset-path "${SCRATCHDIR}/laion2B-en-clip768v2-n=100M.h5" &>/storage/brno12-cerit/home/lickomar/sisap24-python/logs/t2-100M-attempt2.log