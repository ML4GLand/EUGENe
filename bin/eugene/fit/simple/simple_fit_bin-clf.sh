data=/cellar/users/aklie/projects/EUGENE/config/data/train/2021_OLS_Library_Training_NPY-T_bin-clf.yaml
rnn_data=/cellar/users/aklie/projects/EUGENE/config/data/train/2021_OLS_Library_Training_NPY_bin-clf.yaml
model_configs=/cellar/users/aklie/projects/EUGENE/config/models/simple/binary_classification
strands=("ss" "ds" "ts")
models=("fcn" "cnn" "rnn" "hybrid")
result_dir=/cellar/users/aklie/projects/EUGENE/results/simple/binary_classification
version=2022_05_07_NPY_Baseline

for strand in "${strands[@]}"
do
    for model in "${models[@]}"
    do
        name=$strand$model
        model_yml=$model_configs/${name}_bin-clf.yaml
        if [ $model == "rnn" ]
        then
            CMD="sbatch --job-name=simple_fit_bin-clf_$name fit.sh $model $model_yml $rnn_data $result_dir $name $version"
        else
            CMD="sbatch --job-name=simple_fit_bin-clf_$name fit.sh $model $model_yml $data $result_dir $name $version"     
        fi
        echo $CMD
        $CMD
    done
done
