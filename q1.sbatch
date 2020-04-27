#!/bin/bash

#SBATCH --job-name=q1
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --ntasks=120
#SBATCH --mem-per-cpu=2G
#SBATCH --output=q1.out

# Load the default mpi4py/Anaconda module.
 module load mpi4py/3.0.1a0_py3
 
# Run the python program with mpirun, using & to run jobs at the same time
 
# mpirun python ./health_index.py
 
 mpirun -n 1 python ./health_index.py
 mpirun -n 5 python ./health_index.py
 mpirun -n 10 python ./health_index.py
  
#  for i in {1..100}
#  do
#    mpirun -n $i python ./health_index.py
#  done
 
# Wait until all simultaneous mpiruns are done
  wait
