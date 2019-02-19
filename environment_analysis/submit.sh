#!/bin/bash

#SBATCH --job-name=test_job
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:0:0
#SBATCH --mem=32Gb

cd $SLURM_SUBMIT_DIR
echo Time is `date`

python analyse_squalane.py
