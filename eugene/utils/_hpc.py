import os
from os import PathLike

def gkmsvm_slurm_train_script(
    input_dir: PathLike,
    pos_seqs: str,
    neg_seqs: str,
    val_seqs: str,
    result_dir: PathLike,
    hyperparams: str,
    preprocess: str,
    features: str = "fasta",
    architecture: str = "gkmSVM",
):
    """
    Generate a slurm script for training a gkmSVM model.

    Parameters
    ----------
    input_dir : PathLike
        Path to directory containing input data.
    pos_seqs : str
        Path to file containing positive sequences.
    neg_seqs : str
        Path to file containing negative sequences.
    val_seqs : str
        Path to file containing validation sequences.
    result_dir : PathLike
        Path to directory to store results.
    hyperparams : str
        Path to file containing hyperparameters.
    preprocess : str
        Preprocessing to perform on input sequences.
    features : str
        Features to use for training.
    architecture : str
        Architecture to use for training.
    """
    if not os.path.exists(result_dir):
        print("{} does not exist, making dir".format(result_dir))
        os.makedirs(result_dir)

    # Set up model name
    task = "clf" if hyperparams.split("-")[0] == "2" else "reg"
    model = "{}_{}_{}-{}_{}".format(
        preprocess, features, architecture, task, hyperparams
    )
    model_name = os.path.join(result_dir, model)

    # Set up hyperparams
    hyperparams = hyperparams.split("-")
    if hyperparams[5] == "True":
        hyperparams.remove("True")
        hyperparams = "-y {} -t {} -l {} -k {} -d {} -R -c {} -w {}".format(
            *hyperparams
        )
    else:
        hyperparams.remove("False")
        hyperparams = "-y {} -t {} -l {} -k {} -d {} -c {} -w {}".format(*hyperparams)

    # Set up file pointers
    output = [
        "#!/bin/bash",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --time=48:00:00",
        "#SBATCH --partition carter-compute\n",
    ]
    output += ['date\necho -e "Job ID: $SLURM_JOB_ID\\n"\n']
    output += [
        "trainposseqs={}".format(os.path.join(input_dir, pos_seqs)),
        "trainnegseqs={}".format(os.path.join(input_dir, neg_seqs)),
        "valseqs={}".format(os.path.join(input_dir, val_seqs)),
        "resultdir={}".format(result_dir),
        "modelname={}".format(model_name),
    ]
    output += ["[ ! -d $resultdir ] && mkdir $resultdir\n"]

    # Set-up training command
    train_command = "gkmtrain $trainposseqs $trainnegseqs $modelname {} -v 2 -T $SLURM_CPUS_PER_TASK -m 8000.0".format(
        hyperparams
    )
    output += ["echo -e {}".format(train_command)]
    output += [train_command]
    output += ['echo -e "\\n"\n']

    # Set up positive train seq predict
    predict_pos_train_command = 'gkmpredict $trainposseqs $modelname".model.txt" $modelname".train-pos.predict.txt"'
    output += ["echo -e {}".format(predict_pos_train_command)]
    output += [predict_pos_train_command]
    output += ['echo -e "\\n"\n']

    if hyperparams[1] == "2":
        # Set up negative train seq predict
        predict_neg_train_command = 'gkmpredict $trainnegseqs $modelname".model.txt" $modelname".train-neg.predict.txt"'
        output += ["echo -e {}".format(predict_neg_train_command)]
        output += [predict_neg_train_command]
        output += ['echo -e "\\n"\n']

    # Set up val seq predict
    predict_val_command = (
        'gkmpredict $valseqs $modelname".model.txt" $modelname".test.predict.txt"'
    )
    output += ["echo -e {}".format(predict_val_command)]
    output += [predict_val_command]
    output += ['echo -e "\\n"\n']

    output += ["date\n"]

    # Bash command to edit
    usage = "Usage: sbatch --job-name=train_{0} -o {1}/train_{0}.out -e {1}/train_{0}.err --mem=20G train_{0}.sh".format(
        model, result_dir
    )
    print(usage)
    output += [usage]

    # Write to script
    with open("{}/train_{}.sh".format(result_dir, model), "w") as f:
        f.write("\n".join(output))
        print("Successfully generated {}/train_{}.sh".format(result_dir, model))
