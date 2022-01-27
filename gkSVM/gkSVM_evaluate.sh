#!/bin/bash
#SBATCH --job-name=gkSVM_evaluate
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH -o ./out/%A.%x.out
#SBATCH -e ./err/%A.%x.err
#SBATCH --partition carter-compute
#SBATCH --mem=20G

name=$1
testseqs=$2
modelname=$3
resultdir="Result_"$3

'''
Program: gkmpredict (lsgkm program for scoring sequences using a trained model)
Version: v0.1.1

Usage: gkmpredict [options] <test_seqfile> <model_file> <output_file>

 score test sequences using trained gkm-SVM

Arguments:
 test_seqfile: sequence file for test (fasta format)
 model_file: output of gkmtrain
 output_file: name of output file

Options:
 -v <0|1|2|3|4>  set the level of verbosity (default: 2)
                   0 -- error msgs only (ERROR)
                   1 -- warning msgs (WARN)
                   2 -- progress msgs at coarse-grained level (INFO)
                   3 -- progress msgs at fine-grained level (DEBUG)
                   4 -- progress msgs at finer-grained level (TRACE)
-T <1|4|16>      set the number of threads for parallel calculation, 1, 4, or 16
                 (default: 1)

'''

# Predict on other held-out data
echo -e gkmpredict $testseqs $modelname".model.txt" $resultdir/$name.predict.txt -T 16
gkmpredict $testseqs $modelname".model.txt" $resultdir/$name.predict.txt -T 16