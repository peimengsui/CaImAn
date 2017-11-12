#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=myPythonJobGPU
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:2
#SBATCH --mail-type=END
#SBATCH --mail-user=ps3336@nyu.edu
#SBATCH --output=slurm_%j.out


RUNDIR=/mnt/home/speimeng/dev/CaImAn/use_cases/edge-cutter/neuron_regressor

cd $RUNDIR

python main.py --lr 0.03 

