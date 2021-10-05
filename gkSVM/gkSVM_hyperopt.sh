#!/bin/bash
#SBATCH --job-name=gkSVM_hyperopt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aklie@eng.ucsd.edu
#SBATCH --array=1-23%23
#SBATCH -o ./out/%A.%x.%a.out
#SBATCH -e ./err/%A.%x.%a.err
#SBATCH --partition carter-compute

#TODO
#SETS OF PARAMS IN A FILE AND READ IN LINE BY LINE