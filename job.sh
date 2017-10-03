#!/bin/bash
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 100 # memory pool for all cores
#
#SBATCH -t 0-2:00:00 # time (D-HH:MM:SS)
#
#SBATCH -J my_name # Name
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

python $1 $2 $3

# Usage: parallel ./job ::: experiment.py ::: a b c ::: d e f