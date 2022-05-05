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

ckt_path=$1
model_type=$2
data=$3
output=$4


source activate /cellar/users/aklie/opt/miniconda3/envs/pytorch_dev
echo -e "$ckt_path"
echo -e "$model_type"
echo -e "$data"
echo -e "$output"

if [ $model_type == "rnn" ] || [ $model_type == "fcn" ]
    then
        python interpret.py score \
            --model $ckt_path \
            --model_type $model_type \
            --data $data \
            --out $output
    else
        python interpret.py score \
            --model $ckt_path \
            --model_type $model_type \
            --data $data \
            --out $output
            
        python interpret.py pwm \
            --model $ckt_path \
            --model_type $model_type \
            --out $output
fi
date
