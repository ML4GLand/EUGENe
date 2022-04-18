#! /bin/bash
#SBATCH --account carter-gpu
#SBATCH --partition carter-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --error ./err/%x.%A.err
#SBATCH --output ./out/%x.%A.out

date
echo -e "Job ID: $SLURM_JOB_ID"

source activate /cellar/users/aklie/opt/miniconda3/envs/pytorch_dev
echo -e "python $1 fit --seed_everything 13 --trainer $2 --model $3 --data $4"
python $1 fit --seed_everything 13 --config $2 --config $3 --config $4

date
