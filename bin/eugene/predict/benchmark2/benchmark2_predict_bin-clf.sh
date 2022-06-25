model_dir=/cellar/users/aklie/projects/EUGENE/eugene/models
configs=/cellar/users/aklie/projects/EUGENE/config
outputs=/cellar/users/aklie/projects/EUGENE/results/simple/classification
strands=("ss" "ds" "ts")
models=("fcn" "cnn" "rnn" "hybrid")

for strand in "${strands[@]}"
do
    for model in "${models[@]}"
    do
        model_yml=${strand}_bin-clf_${model}.yaml
        model_cp=$outputs/${strand}${model}/2022_04_23_NPY_Baseline/checkpoints/*

        # 2021 OLS Training Sequences
        output=$outputs/${strand}${model}/2022_04_23_NPY_Baseline/predictions/2021_OLS_Training_
        if [ $model == "rnn" ]
        then
            data_config=$configs/data/test/2021_OLS_Library_Training_NPY_bin_clf.yaml
        else
            data_config=$configs/data/test/2021_OLS_Library_Training_NPY-T_bin_clf.yaml
        fi
        CMD="sbatch --job-name=simple_predict_${strand}${model}_2021_OLS_Library_Training predict.sh \
            $model_dir/$model.py \
            $configs/models/simple/$model_yml \
            $model_cp \
            $data_config \
            $output"
        echo $CMD
        $CMD

        # 2021 OLS Holdout Sequences
        output=$outputs/${strand}${model}/2022_04_23_NPY_Baseline/predictions/2021_OLS_Holdout_
        if [ $model == "rnn" ]
        then
            data_config=$configs/data/test/2021_OLS_Library_Holdout_NPY_bin_clf.yaml
        else
            data_config=$configs/data/test/2021_OLS_Library_Holdout_NPY-T_bin_clf.yaml
        fi
        CMD="sbatch --job-name=simple_predict_${strand}${model}_2021_OLS_Library_Holdout predict.sh \
            $model_dir/$model.py \
            $configs/models/simple/$model_yml \
            $model_cp \
            $data_config \
            $output"
        echo $CMD
        $CMD

        # TODO: All Genomic Sequences
    done
done
