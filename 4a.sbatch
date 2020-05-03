
#!/bin/bash

#SBATCH --job-name=q4a
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --mem-per-cpu=2G
#SBATCH --account=macs30123
#SBATCH --output=q4a.out

# Load the default mpi4py/Anaconda module.
 module load mpi4py/3.0.1a0_py3
 
# Run the python program with mpirun, using & to run jobs at the same time
 
 mpirun -n 20 python ./4a.py
