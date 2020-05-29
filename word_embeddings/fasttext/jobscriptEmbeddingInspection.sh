#!/bin/bash
#SBATCH --job-name=InspectEmbeddingsScript
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --array=1-12

module purge
module load Python
module list

INPUTFILE=parameters.in

ARGS="$(cat $INPUTFILE | head -n ${SLURM_ARRAY_TASK_ID} | tail -n 1)"

python3 inspectEmbeddings.py $ARGS
