configs=/cellar/users/aklie/projects/EUGENE/config
outputs=/cellar/users/aklie/projects/EUGENE/results/simple/classification
strands=("ss" "ds" "ts")
models=("fcn" "cnn" "rnn" "hybrid")

for strand in "${strands[@]}"
do
    for model in "${models[@]}"
    do
        model_cp=$outputs/${strand}${model}/2022_04_23_NPY_Baseline/checkpoints/*
        output=$outputs/${strand}${model}/2022_04_23_NPY_Baseline/interpretations/2021_OLS_Libarary_Holdout
        if [ $model == "rnn" ]
        then
            data_config=$configs/data/test/2021_OLS_Library_Holdout_NPY_bin_clf.yaml
        else
            data_config=$configs/data/test/2021_OLS_Library_Holdout_NPY-T_bin_clf.yaml
        fi
        CMD="sbatch --job-name simple_interpret_${strand}${model}_2021_OLS_Library_Holdout interpret.sh \
            $model_cp \
            $model \
            $data_config \
            $output"
        echo $CMD
        $CMD
    done
done
