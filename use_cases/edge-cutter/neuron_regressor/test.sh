#!/bin/bash
#SBATCH -N3 --exclusive --ntasks-per-node=5
# Start from an "empty" module collection.
module purge
# Load in what we need to execute mpirun.
module load slurm gcc openmpi
mpirun ./mpi_hello_world
