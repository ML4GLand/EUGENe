#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --partition carter-compute
#SBATCH --mem=20G


date
echo -e "Job ID: $SLURM_JOB_ID\n"

# Define arguments
testseqs=$1
modelname=$2
output=$3

# Predict on data
echo -e gkmpredict $testseqs $modelname".model.txt" $modelname.$output".predict.txt" -T $SLURM_CPUS_PER_TASK
gkmpredict $testseqs $modelname".model.txt" $modelname.$output".predict.txt" -T $SLURM_CPUS_PER_TASK

date