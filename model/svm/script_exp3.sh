#!/bin/bash
#SBATCH --job-name=AbuseDetectionExperiment3
#SBATCH --partition=vulture
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=2
#SBATCH --time=07:00:00
#SBATCH --array=1-1
#SBATCH --output=prints/experiment3/slurm-%A_%a.out

module purge
module load Python
module list

INPUTFILE=exp3_parameters.in

ARGS="$(cat $INPUTFILE | head -n ${SLURM_ARRAY_TASK_ID} | tail -n 1)"

python3 modelSVM.py $ARGS
