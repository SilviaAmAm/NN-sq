#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=15:0:0
#SBATCH --mem=32gb

module load languages/anaconda2/5.0.1.tensorflow-1.6.0
module unload languages/anaconda2/5.0.1.tensorflow-1.6.0
module load languages/intel/2017.01

export LD_LIBRARY_PATH=/mnt/storage/software/libraries/nvidia/cudnn-9.0/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/mnt/storage/software/libraries/nvidia/cuda-9.0/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
cd $SLURM_SUBMIT_DIR
echo Time is `date`

python train.py
