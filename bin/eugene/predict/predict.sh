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

model=$1
model_config=$2
ckt_path=$3
data_config=$4
output=$5

source activate /cellar/users/aklie/opt/miniconda3/envs/pytorch_dev
echo -e "python $model predict \
    --seed_everything 13 \
    --config $model_path \
    --ckpt_path $ckt_path \
    --config $data_config \
    --trainer.callbacks=PredictionWriter \
    --trainer.callbacks.output_dir $output \
    --trainer.logger False \
    --traner.gpus 1"

python $model predict \
    --seed_everything 13 \
    --config $model_config \
    --ckpt_path $ckt_path \
    --config $data_config \
    --trainer.callbacks=PredictionWriter \
    --trainer.callbacks.output_dir $output \
    --trainer.logger False \
    --trainer.gpus 1

date
