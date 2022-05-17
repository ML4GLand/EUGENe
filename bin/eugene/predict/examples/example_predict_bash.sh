sbatch --job-name=predict_reg_sshybrid predict.sh \
    /cellar/users/aklie/projects/EUGENE/results/simple/regression/sshybrid/2022_05_14_NPY_Full_Test/checkpoints/epoch=24-step=10124.ckpt \
    hybrid \
    /cellar/users/aklie/projects/EUGENE/config/data/test/2021_OLS_Library_Training_NPY-T_reg.yaml \
    /cellar/users/aklie/projects/EUGENE/results/simple/regression/sshybrid/2022_05_14_NPY_Full_Test/interpretations/2021_OLS_Library_Training