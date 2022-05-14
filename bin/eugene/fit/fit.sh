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

model="/cellar/users/aklie/projects/EUGENE/eugene/models/$1.py"
model_config=$2
data_config=$3
output_dir=$4
output_name=$5
output_version=$6

source activate /cellar/users/aklie/opt/miniconda3/envs/pytorch_dev
echo -e "python $model fit
    --seed_everything 13
    --config $model_config
    --config $data_config
    --trainer.callbacks=EarlyStopping
    --trainer.callbacks.patience 5
    --trainer.callbacks.monitor "val_loss"
    --trainer.logger.class_path pytorch_lightning.loggers.TensorBoardLogger
    --trainer.logger.init_args.save_dir "$output_dir"
    --trainer.logger.init_args.name "$output_name"
    --trainer.logger.init_args.version "$output_version"
    --trainer.max_epochs 100
    --trainer.gpus 1"
    
python $model fit \
    --seed_everything 13 \
    --config $model_config \
    --config $data_config \
    --trainer.callbacks=EarlyStopping \
    --trainer.callbacks.patience 5 \
    --trainer.callbacks.monitor "val_loss" \
    --trainer.logger.class_path pytorch_lightning.loggers.TensorBoardLogger \
    --trainer.logger.init_args.save_dir "$output_dir" \
    --trainer.logger.init_args.name "$output_name" \
    --trainer.logger.init_args.version "$output_version" \
    --trainer.max_epochs 100 \
    --trainer.gpus 1

date