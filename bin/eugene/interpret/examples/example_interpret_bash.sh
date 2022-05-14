sbatch --job-name=predict_reg_sshybrid fit.sh \
    /cellar/users/aklie/projects/EUGENE/eugene/models/hybrid.py \
    /cellar/users/aklie/projects/EUGENE/config/models/simple/regression/sshybrid_reg.yaml \
    /cellar/users/aklie/projects/EUGENE/results/simple/regression/sshybrid/2022_05_14_NPY_Full_Test/checkpoints/epoch=24-step=10124.ckpt \
    /cellar/users/aklie/projects/EUGENE/config/data/test/2021_OLS_Library_Training_NPY-T_reg.yaml \
    /cellar/users/aklie/projects/EUGENE/results/simple/regression/sshybrid/2022_05_14_NPY_Full_Test/predictions/2021_OLS_Library_Training_ 