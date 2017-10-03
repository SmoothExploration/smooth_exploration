#!/bin/bash
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 100 # memory pool for all cores
#
#SBATCH -t 0-2:00:00 # time (D-HH:MM:SS)
#
#SBATCH -J my_name # Name
#SBATCH -o slurm.%N.%A_%a.out # STDOUT
#SBATCH -e slurm.%N.%A_%a.err # STDERR
#
#SBATCH --array=1-10%1 # 10 job array, 1 job at a time

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------

if test -e state.cpt; then
     # There is a checkpoint file, restart;
else
     # There is no checkpoint file, start a new simulation.
fi

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------

# Usage: parallel ./job ::: experiment.py ::: a b c ::: d e f