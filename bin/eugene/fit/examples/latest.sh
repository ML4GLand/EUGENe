python /cellar/users/aklie/projects/EUGENE/eugene/models/hybrid.py fit \
    --seed_everything 13 \
    --config /cellar/users/aklie/projects/EUGENE/config/models/simple/binary_classification/sshybrid_bin-clf.yaml \
    --config /cellar/users/aklie/projects/EUGENE/config/data/train/binary_classification/2021_OLS_Library_Training_NPY-T_bin-clf.yaml \
    --trainer.callbacks=EarlyStopping \
    --trainer.callbacks.patience 5 \
    --trainer.callbacks.monitor val_loss \
    --trainer.logger.class_path pytorch_lightning.loggers.TensorBoardLogger \
    --trainer.logger.init_args.save_dir /cellar/users/aklie/projects/EUGENE/results/simple/binary_classification \
    --trainer.logger.init_args.name sshybrid \
    --trainer.logger.init_args.version 2022_05_16_NPY_Full_Test \
    --trainer.max_epochs 100 \
    --trainer.gpus 1