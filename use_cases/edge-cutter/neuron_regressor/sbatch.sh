#!/bin/bash
#
#SBATCH --job-name=myPythonJobGPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=100:00:00
#SBATCH --mem=10GB
#SBATCH --mail-type=END
#SBATCH --mail-user=rf1711@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load python/intel/2.7.12

python movie_based.py

##python main.py --data ./data/gutenberg --cuda --log-interval 200 --epochs 25 --shuffle True --suffix _shuffle_gu --save model_shuffle_gu.pt --k 10000
##python main.py --cuda --log-interval 200 --epochs 1 --suffix _shuffle --shuffle True --save model_shuffle.pt --k 10000
##python dcGAN.py --dataset cifar10 --dataroot . --cuda