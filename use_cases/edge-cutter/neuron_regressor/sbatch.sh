#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=myPythonJobGPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=100:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=ps3336@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load gcc/7.2.0
module load python3/3.6.2
module load cuda/8.0.61
pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl 
pip3 install torchvision

RUNDIR=/mnt/home/speimeng/dev/CaImAn/use_cases/edge-cutter/neuron_regressor
mkdir $RUNDIR

DATADIR=$SCRATCH/my_project/data

cd $RUNDIR

python main.py

