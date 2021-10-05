#!/bin/bash
#SBATCH --job-name=gkSVM
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aklie@eng.ucsd.edu
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH -o ./out/%A.%x.out
#SBATCH -e ./err/%A.%x.err
#SBATCH --partition carter-compute
#SBATCH --mem=20G

inputdir="./"
trainposseqs=$inputdir$1
trainnegseqs=$inputdir$2
testposseqs=$inputdir$3
testnegseqs=$inputdir$4
modelname=$5

resultdir="Result_"$5
[ ! -d $resultdir ] && mkdir $resultdir

echo -e gkmtrain $trainposseqs $trainnegseqs $modelname -v 2 -T 16
gkmtrain $trainposseqs $trainnegseqs $modelname -v 2 -T 16
echo -e "\n"

echo -e gkmpredict $testposseqs $modelname".model.txt" $resultdir/$5".predict.txt"
gkmpredict $testposseqs $modelname".model.txt" $resultdir/$5".predict.txt"
echo -e "\n"

echo -e gkmpredict $testnegseqs $modelname".model.txt" $resultdir/$5".neg.predict.txt"
gkmpredict $testnegseqs $modelname".model.txt" $resultdir/$5".neg.predict.txt"