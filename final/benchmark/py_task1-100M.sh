#!/bin/bash
#PBS -l select=1:ncpus=32:mem=200gb:cluster=elmu1
#PBS -l walltime=24:00:00
 
export OMP_NUM_THREADS=$PBS_NUM_PPN
 
module add mambaforge || exit 2
mamba activate /storage/brno12-cerit/home/prochazka/.conda/envs/learnedmetricindex-bp || exit 3
 
python3 /storage/brno12-cerit/home/lickomar/sisap24-python/task1.py --dataset-size 100M &>/storage/brno12-cerit/home/lickomar/sisap24-python/logs/t1-100M.log
