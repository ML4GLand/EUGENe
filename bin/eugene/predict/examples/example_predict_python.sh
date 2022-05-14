python /cellar/users/aklie/projects/EUGENE/eugene/models/hybrid.py predict \
    --seed_everything 13 \
    --config /cellar/users/aklie/projects/EUGENE/config/models/simple/regression/sshybrid_reg.yaml \
    --ckpt_path /cellar/users/aklie/projects/EUGENE/results/simple/regression/sshybrid/2022_05_14_NPY_Full_Test/checkpoints/epoch=24-step=10124.ckpt \
    --config /cellar/users/aklie/projects/EUGENE/config/data/test/2021_OLS_Library_Training_NPY-T_reg.yaml \
    --trainer.callbacks=PredictionWriter \
    --trainer.callbacks.output_dir /cellar/users/aklie/projects/EUGENE/results/simple/regression/sshybrid/2022_05_14_NPY_Full_Test/predictions/2021_OLS_Library_Training_ \
    --trainer.logger False \
    --trainer.gpus 1