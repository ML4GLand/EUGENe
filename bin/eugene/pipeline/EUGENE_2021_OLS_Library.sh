################
### SET-UP
################
task=$1  # bin-clf
strand=$2  # ss
model=$3  # hybrid
type=$4  # benchmark
subtype=$5  # benchmark2
version=$6  # 2022_05_19_Pipeline_Test
scripts_dir=/cellar/users/aklie/projects/EUGENE/bin/eugene
results_dir=/cellar/users/aklie/projects/EUGENE/results/$type/$subtype/$task

# Example run: bash EUGENE_2021_OLS_Library.sh bin-clf ss hybrid benchmark benchmark2 2022_05_19_Pipleine_Test

################
### FIT
################
traindata_cfg=/cellar/users/aklie/projects/EUGENE/config/data/2021_OLS_Library_Training_OHE-T_${task}_train.yaml
model_cfg=/cellar/users/aklie/projects/EUGENE/config/models/$subtype/${strand}${model}_${task}.yaml

# Train the model specified above
fit_cmd="sbatch --job-name=fit_${task}_${strand}${model}_2021_OLS_Library_Training --wait $scripts_dir/fit/fit.sh \
    $model \
    $model_cfg \
    $traindata_cfg \
    $results_dir \
    ${strand}${model} \
    $version"
echo $fit_cmd
$fit_cmd
    
################
### PREDICT
################

# Grab the results directory and model checkpoint
model_results=$results_dir/${strand}${model}/$version
[ -d $model_results/predictions ] && echo "Directory $model_results exists." || echo "Error: Directory $model_results does not exists. making directory"
[ ! -d $model_results/predictions ] && mkdir -p $model_results/predictions
model_ckpt=$model_results/checkpoints/*
    
# Grab predictions on all the all 2021 OLS Sequences 
predict_cmd="sbatch --job-name=predict_${task}_${strand}${model}_2021_OLS_Library_All $scripts_dir/predict/predict.sh \
    $model \
    $model_cfg \
    $model_ckpt \
    /cellar/users/aklie/projects/EUGENE/config/data/2021_OLS_Library_All_OHE-T_${task}_test.yaml \
    $model_results/predictions/2021_OLS_Library_All_"
echo $predict_cmd
$predict_cmd
    
# Get predictions on genomic enhancers
predict_cmd="sbatch --job-name=predict_${task}_${strand}${model}_All_Genomic_Sequences $scripts_dir/predict/predict.sh \
    $model \
    $model_cfg \
    $model_ckpt \
    /cellar/users/aklie/projects/EUGENE/config/data/All_Genomic_Sequences_TSV_${task}_test.yaml \
    $model_results/predictions/All_Genomic_Sequences_"
echo $predict_cmd
$predict_cmd
    
################
### INTERPRET
################

# Make the intepretations dir
#[ -d $model_results/interpretations ] && echo "Directory $model_results/interpretations exists." || echo "Error: Directory $model_results/interpretations does not exists."
#[ ! -d $model_results/interpretations ] && mkdir -p $model_results/interpretations

# Get interpretations on genomic enhancers
#sbatch --job-name=intepret_bin-clf_${strand}${model} $scripts_dir/interpret/interpret.sh \
#    $model_ckp \
#    $model \
#    /cellar/users/aklie/projects/EUGENE/config/data/test/binary_classification/All_2021_OLS_Library_Sequences_NPY-T_bin-clf.yaml \
#    $model_results/interpretations/All_2021_OLS_Library_Sequences

# Get interpretations on genomic enhancers
#sbatch --job-name=intepret_bin-clf_${strand}${model} $scripts_dir/interpret/interpret.sh \
#    $model_ckp \
#    $model \
#    /cellar/users/aklie/projects/EUGENE/config/data/test/binary_classification/All_Genomic_Sequences_TSV_bin-clf.yaml \
#    $model_results/interpretations/All_Genomic_Sequences