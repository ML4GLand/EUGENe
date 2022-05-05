model_dir=/cellar/users/aklie/projects/EUGENE/eugene/models
data=/cellar/users/aklie/projects/EUGENE/config/data/2021_OLS_Library_NPY-T_bin_clf.yaml
rnn_data=/cellar/users/aklie/projects/EUGENE/config/data/2021_OLS_Library_NPY_bin_clf.yaml
configs=/cellar/users/aklie/projects/EUGENE/config
strands=("ss" "ds" "ts")
models=("fcn" "cnn" "rnn" "hybrid")

for strand in "${strands[@]}"
do
    for model in "${models[@]}"
    do
        yml_file=${strand}_bin-clf_${model}.yaml
        if [ $model == "rnn" ]
        then
            CMD="sbatch --job-name=simple_fit_$strand$model fit.sh $model_dir/$model.py $rnn_data $configs/trainers/$yml_file $configs/models/simple/$yml_file"
        else
            CMD="sbatch --job-name=simple_fit_$strand$model fit.sh $model_dir/$model.py $data $configs/trainers/$yml_file $configs/models/simple/$yml_file"
            
        fi
        echo $CMD
        $CMD
    done
done