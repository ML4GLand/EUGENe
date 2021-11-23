#!/bin/bash
#SBATCH --job-name=gkSVM_param
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aklie@eng.ucsd.edu
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH -o ./out/%A.%x.out
#SBATCH -e ./err/%A.%x.err
#SBATCH --partition carter-compute
#SBATCH --mem=20G

inputdir="/cellar/users/aklie/projects/EUGENE/data/2021_OLS_Library/fasta/"
trainposseqs=$inputdir$1
trainnegseqs=$inputdir$2
testposseqs=$inputdir$3
testnegseqs=$inputdir$4
modelname=$5  #should be data_model (e.g., 2021-OLS-X-fasta_0.18-0.4_baseline)

resultdir="Result_"$5
[ ! -d $resultdir ] && mkdir $resultdir

'''
Options:
 -t <0 ~ 5>   set kernel function (default: 2 gkm)
              NOTE: RBF kernels (3 and 5) work best with -c 10 -g 2
                0 -- gapped-kmer
                1 -- estimated l-mer with full filter
                2 -- estimated l-mer with truncated filter (gkm)
                3 -- gkm + RBF (gkmrbf)
                4 -- gkm + center weighted (wgkm)
                     [weight = max(M, floor(M*exp(-ln(2)*D/H)+1))]
                5 -- gkm + center weighted + RBF (wgkmrbf)
 -l <int>     set word length, 3<=l<=12 (default: 11)
 -k <int>     set number of informative column, k<=l (default: 7)
 -d <int>     set maximum number of mismatches to consider, d<=4 (default: 3)
 -g <float>   set gamma for RBF kernel. -t 3 or 5 only (default: 1.0)
 -M <int>     set the initial value (M) of the exponential decay function
              for wgkm-kernels. max=255, -t 4 or 5 only (default: 50)
 -H <float>   set the half-life parameter (H) that is the distance (D) required
              to fall to half of its initial value in the exponential decay
              function for wgkm-kernels. -t 4 or 5 only (default: 50)
 -R           if set, reverse-complement is not considered as the same feature
 -c <float>   set the regularization parameter SVM-C (default: 1.0)
 -e <float>   set the precision parameter epsilon (default: 0.001)
 -w <float>   set the parameter SVM-C to w*C for the positive set (default: 1.0)
 -m <float>   set cache memory size in MB (default: 100.0)
              NOTE: Large cache signifcantly reduces runtime. >4Gb is recommended
 -s           if set, use the shrinking heuristics
 -x <int>     set N-fold cross validation mode (default: no cross validation)
 -i <int>     run i-th cross validation only 1<=i<=ncv (default: all)
 -r <int>     set random seed for shuffling in cross validation mode (default: 1)
 -v <0 ~ 4>   set the level of verbosity (default: 2)
                0 -- error msgs only (ERROR)
                1 -- warning msgs (WARN)
                2 -- progress msgs at coarse-grained level (INFO)
                3 -- progress msgs at fine-grained level (DEBUG)
                4 -- progress msgs at finer-grained level (TRACE)
-T <1|4|16>   set the number of threads for parallel calculation, 1, 4, or 16
                 (default: 1)
'''

# Train the model
echo -e gkmtrain $trainposseqs $trainnegseqs $modelname -c 0.5 -m 8000.0 -v 2 -T 16
gkmtrain $trainposseqs $trainnegseqs $modelname -c 0.5 -m 8000.0 -v 2 -T 16
echo -e "\n"

# Predict on test positive seqs
echo -e gkmpredict $testposseqs $modelname".model.txt" $resultdir/$5".predict.txt"
gkmpredict $testposseqs $modelname".model.txt" $resultdir/$5".predict.txt"
echo -e "\n"

# Predict on test negative seqs
echo -e gkmpredict $testnegseqs $modelname".model.txt" $resultdir/$5".neg.predict.txt"
gkmpredict $testnegseqs $modelname".model.txt" $resultdir/$5".neg.predict.txt"
echo -e "\n"

# Predict on train positive seqs
echo -e gkmpredict $trainposseqs $modelname".model.txt" $resultdir/$5".tr.predict.txt"
gkmpredict $trainposseqs $modelname".model.txt" $resultdir/$5".tr.predict.txt"
echo -e "\n"

# Predict on train negative seqs
echo -e gkmpredict $trainnegseqs $modelname".model.txt" $resultdir/$5".neg.tr.predict.txt"
gkmpredict $trainnegseqs $modelname".model.txt" $resultdir/$5".neg.tr.predict.txt"