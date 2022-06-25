data=/cellar/users/aklie/projects/EUGENE/config/data/train/2021_OLS_Library_Training_NPY-T_reg.yaml
rnn_data=/cellar/users/aklie/projects/EUGENE/config/data/train/2021_OLS_Library_Training_NPY_reg.yaml
model_configs=/cellar/users/aklie/projects/EUGENE/config/models/simple/regression
strands=("ss" "ds" "ts")
models=("fcn" "cnn" "rnn" "hybrid")
result_dir=/cellar/users/aklie/projects/EUGENE/results/simple/regression
version=2022_05_07_NPY_Baseline

for strand in "${strands[@]}"
do
    for model in "${models[@]}"
    do
        name=$strand$model
        model_yml=$model_configs/${name}_reg.yaml
        if [ $model == "rnn" ]
        then
            CMD="sbatch --job-name=simple_fit_reg_$name fit.sh $model $model_yml $rnn_data $result_dir $name $version"
        else
            CMD="sbatch --job-name=simple_fit_reg_$name fit.sh $model $model_yml $data $result_dir $name $version"
        fi
        echo $CMD
        $CMD
    done
done
