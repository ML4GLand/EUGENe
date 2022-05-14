python /cellar/users/aklie/projects/EUGENE/eugene/interpret/interpret.py score \
    --model /cellar/users/aklie/projects/EUGENE/results/simple/regression/sshybrid/2022_05_14_NPY_Full_Test/checkpoints/epoch=24-step=10124.ckpt \
    --model_type hybrid \
    --data /cellar/users/aklie/projects/EUGENE/config/data/test/2021_OLS_Library_Training_NPY-T_reg.yaml \
    --out /cellar/users/aklie/projects/EUGENE/results/simple/regression/sshybrid/2022_05_14_NPY_Full_Test/interpretations/2021_OLS_Library_Training
    
python /cellar/users/aklie/projects/EUGENE/eugene/interpret/interpret.py pwm \
    --model /cellar/users/aklie/projects/EUGENE/results/simple/regression/sshybrid/2022_05_14_NPY_Full_Test/checkpoints/epoch=24-step=10124.ckpt \
    --model_type hybrid \
    --out /cellar/users/aklie/projects/EUGENE/results/simple/regression/sshybrid/2022_05_14_NPY_Full_Test/interpretations/2021_OLS_Library_Training