#!/bin/bash
#SBATCH --job-name=gkSVM_explain
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
Usage: gkmexplain [options] <test_seqfile> <model_file> <output_file>

 explain prediction on test sequences using trained gkm-SVM

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
 -m <0|1>  set the explanation mode (default: 0)
                   0 -- importance scores
                   1 -- hypothetical importance scores (considering lmers with d mismatches)
                   2 -- hypothetical importance scores (considering d+1 mismatches)
                   3 -- perturbation effect estimation (considering lmers with d mismatches)
                   4 -- perturbation effect estimation (considering d+1 mismatches)
                   5 -- score perturbations for only the central position in the region

'''

# Explain these predictions please
echo -e gkmexplain $testseqs $modelname".model.txt.2" $resultdir/$name.explanations.txt
gkmexplain $testseqs $modelname".model.txt.2" $resultdir/$name.explanations.txt
