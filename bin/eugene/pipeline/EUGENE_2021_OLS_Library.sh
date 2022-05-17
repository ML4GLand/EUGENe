################
### FIT
################

# Set-up all directories
task=bin-clf
strand=ss
model=hybrid
scripts_dir=/cellar/users/aklie/projects/EUGENE/bin/eugene
train_data=/cellar/users/aklie/projects/EUGENE/config/data/train/$task/2021_OLS_Library_Training_NPY-T_bin-clf.yaml
results_dir=/cellar/users/aklie/projects/EUGENE/results/simple/$task
model_cfg=/cellar/users/aklie/projects/EUGENE/config/models/simple/$task/${strand}${model}_${task}.yaml
version=2022_05_17_Pipeline_Test

# Train the model specified above
sbatch --job-name=fit_bin-clf_${strand}${model} --wait $scripts_dir/fit/fit.sh \
    $model \
    $model_cfg \
    $train_data \
    $results_dir \
    ${strand}${model} \
    $version
    
################
### PREDICT
################

# Grab the results directory and model checkpoint
model_results=$results_dir/${strand}${model}/$version
[ -d $model_results/predictions ] && echo "Directory $model_results exists." || echo "Error: Directory $model_results does not exists."
[ ! -d $model_results/predictions ] && mkdir -p $model_results/predictions
model_ckp=$model_results/checkpoints/*
    
# Grab predictions on all the 2021 OLS Sequences 
sbatch --job-name=predict_bin-clf_${strand}${model} $scripts_dir/predict/predict.sh \
    $model \
    $model_cfg \
    $model_ckp \
    /cellar/users/aklie/projects/EUGENE/config/data/test/binary_classification/All_2021_OLS_Library_Sequences_NPY-T_bin-clf.yaml \
    $model_results/predictions/All_2021_OLS_Library_Sequences_
    
# Get predictions on genomic enhancers
sbatch --job-name=predict_bin-clf_${strand}${model} $scripts_dir/predict/predict.sh \
    $model \
    $model_cfg \
    $model_ckp \
    /cellar/users/aklie/projects/EUGENE/config/data/test/binary_classification/All_Genomic_Sequences_TSV_bin-clf.yaml \
    $model_results/predictions/All_Genomic_Sequences_
    
################
### INTERPRET
################

# Make the intepretations dir
[ -d $model_results/interpretations ] && echo "Directory $model_results/interpretations exists." || echo "Error: Directory $model_results/interpretations does not exists."
[ ! -d $model_results/interpretations ] && mkdir -p $model_results/interpretations

# Get interpretations on genomic enhancers
sbatch --job-name=intepret_bin-clf_${strand}${model} $scripts_dir/interpret/interpret.sh \
    $model_ckp \
    $model \
    /cellar/users/aklie/projects/EUGENE/config/data/test/binary_classification/All_2021_OLS_Library_Sequences_NPY-T_bin-clf.yaml \
    $model_results/interpretations/All_2021_OLS_Library_Sequences

# Get interpretations on genomic enhancers
sbatch --job-name=intepret_bin-clf_${strand}${model} $scripts_dir/interpret/interpret.sh \
    $model_ckp \
    $model \
    /cellar/users/aklie/projects/EUGENE/config/data/test/binary_classification/All_Genomic_Sequences_TSV_bin-clf.yaml \
    $model_results/interpretations/All_Genomic_Sequences