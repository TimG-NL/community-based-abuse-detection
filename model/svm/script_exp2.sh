#!/bin/bash
#SBATCH --job-name=AbuseDetectionExperiment2
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --array=1-12
#SBATCH --output=prints/experiment2/slurm-%A_%a.out

module purge
module load Python
module list

INPUTFILE=exp2_parameters.in

ARGS="$(cat $INPUTFILE | head -n ${SLURM_ARRAY_TASK_ID} | tail -n 1)"

python3 modelSVM.py $ARGS
