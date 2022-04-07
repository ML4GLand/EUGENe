#! /bin/bash
#SBATCH --account carter-gpu
#SBATCH --partition carter-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --error ./err/%x.%A.err
#SBATCH --output ./out/%x.%A.out

date
echo -e "Job ID: $SLURM_JOB_ID"

source activate /cellar/users/aklie/opt/miniconda3/envs/pytorch_dev
echo -e "python ../eugene/dsEUGENE_hyperopt.py fit --seed_everything 13 --config $1"
python ../eugene/dsEUGENE_hyperopt.py --seed_everything 13 --config $1

date
