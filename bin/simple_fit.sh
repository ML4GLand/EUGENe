model=/cellar/users/aklie/projects/EUGENE/eugene/models/CNN.py
data=/cellar/users/aklie/projects/EUGENE/config/data/2021_OLS_Library_NPY_Baseline_Classification.yaml
configs=/cellar/users/aklie/projects/EUGENE/config

# ssCNN
sbatch --job-name=simple_ssCNN fit.sh $model $data $configs/trainers/Trainer_ssCNN_Baseline_Classification.yaml $configs/models/Model_ssCNN_Baseline_Classification.yaml

# dsCNN
sbatch --job-name=simple_dsCNN fit.sh $model $data $configs/trainers/Trainer_dsCNN_Baseline_Classification.yaml $configs/models/Model_dsCNN_Baseline_Classification.yaml

# ssRNN

# dsRNN