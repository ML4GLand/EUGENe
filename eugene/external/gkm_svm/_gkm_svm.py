import os
import threading
import math
import subprocess
import numpy as np
from ... import settings

param_dict = {
    "binary_classification": 0, 
    "regression": 3
}  # gkm-SVM parameter for task type

mode_dict = {
    "importances": 0
}

def to_fasta(
    sdata, 
    target_key=None,
    train_key=None,
    task="binary_classification",
    out_dir=None,
    file_name="seqs"
):
    """Utility function to generate a fasta file from an sdata object
    
    Useful for running gkm-SVM. If the task is binary classfiction, the seqs are
    split into two files, one for positive label and one for negative label. If the task 
    is regression, the seqs are written to a single file and the target values are written
    to a single file.

    Parameters
    ----------
    sdata : SeqData
        SeqData object containing sequences and target values
    target_key : str
        Key for the target values in the sdata object
    train_key : str, optional
        Key for the train/test split in the sdata object, by default None
    task : str, optional
        Task type, by default "binary_classification"
    file_name : str, optional
        Name of the fasta file, by default "seqs"
    
    Returns
    -------
    None
    """
    out_dir = out_dir if out_dir is not None else settings.dataset_dir
    if target_key is None:
        file_names = [f"{file_name}.fasta"]
        seqs_to_save = [sdata.seqs]
        ids_to_save = [sdata.seqs_annot.index]
        targets_to_save = None
    else:
        if train_key is not None:
            train = sdata[sdata[train_key] == True]
            test = sdata[sdata[train_key] == False]
        if task == "binary_classification":
            if train_key is not None:
                train_pos = train[train[target_key] == 1]
                train_neg = train[train[target_key] == 0]
                test_pos = test[test[target_key] == 1]
                test_neg = test[test[target_key] == 0]
                seqs_to_save = [train_pos.seqs, train_neg.seqs, test_pos.seqs, test_neg.seqs]
                ids_to_save = [train_pos.seqs_annot.index, train_neg.seqs_annot.index, test_pos.seqs_annot.index, test_neg.seqs_annot.index]
                file_names = [f"{file_name}_train_pos.fasta", f"{file_name}_train_neg.fasta", f"{file_name}_val_pos.fasta", f"{file_name}_val_neg.fasta"]
            else:
                pos = sdata[sdata[target_key] == 1]
                neg = sdata[sdata[target_key] == 0]
                seqs_to_save = [pos.seqs, neg.seqs]
                ids_to_save = [pos.seqs_annot.index, neg.seqs_annot.index]
                file_names = [f"{file_name}_pos.fasta", f"{file_name}_neg.fasta"]
            targets_to_save = None
        elif task == "regression":
            if train_key is not None:
                seqs_to_save = [train.seqs, test.seqs]
                ids_to_save = [train.seqs_annot.index, test.seqs_annot.index]
                targets_to_save = [train[target_key].values, test[target_key].values]
                file_names = [f"{file_name}_train.fasta", f"{file_name}_val.fasta"]
            else:
                seqs_to_save = [sdata.seqs]
                ids_to_save = [sdata.seqs_annot.index]
                file_names = [f"{file_name}.fasta"]
                targets_to_save = [sdata[target_key].values]
    for seqs, ids, file_name in zip(seqs_to_save, ids_to_save, file_names):
        file_name = os.path.join(out_dir, file_name)
        with open(file_name, "w") as f:
            for j, seq in enumerate(seqs):
                f.write(">" + ids[j] + "\n" + seqs[j] + "\n")
    if targets_to_save is not None:
        for targets, file_name in zip(targets_to_save, file_names):
            file_name = os.path.join(out_dir, file_name)
            with open(file_name.replace(".fasta", ".targets"), "w") as f:
                for target in targets:
                    f.write(str(target) + "\n")

    
def fit(
    sdata,
    target_key,
    train_key=None,
    task="binary_classification",
    kernel_type="gkm",
    word_length=11,
    informative_columns=7,
    mismatches=3,
    gamma=1.0,
    decay_M=50,
    half_life=50,
    reverse_complement=False,
    regularization_strength=1.0,
    epsilon=0.001,
    positive_weight=1.0,
    cached_mem_size=8000,
    threads=None,
    verbose=2,
    data_dir=None,
    log_dir=None,
    name="seqs",
    prefix="",
    suffix=""
):
    data_dir = data_dir if data_dir is not None else settings.dataset_dir
    log_dir = log_dir if log_dir is not None else settings.logging_dir
    os.makedirs(log_dir) if not os.path.exists(log_dir) else None
    threads = threads if threads is not None else 4**(math.floor(math.log(threading.active_count(), 4)))
    if task == "binary_classification":
        if train_key is not None:
            file1_name = os.path.join(data_dir, f"{name}_train_pos.fasta")
            file2_name = os.path.join(data_dir, f"{name}_train_neg.fasta")  
        else:
            file1_name = os.path.join(data_dir, f"{name}_pos.fasta")
            file2_name = os.path.join(data_dir, f"{name}_neg.fasta")
    elif task == "regression":
        if train_key is not None:
            file1_name = os.path.join(data_dir, f"{name}_train.fasta")
            file2_name = os.path.join(data_dir, f"{name}_train.targets") 

        else:
            file1_name = os.path.join(data_dir, f"{name}.fasta")
            file2_name = os.path.join(data_dir, f"{name}.targets")
    if os.path.exists(file1_name) and os.path.exists(file2_name):
        print("Train files already exist, skipping generation")
    else:
        print(file1_name, file2_name)
        to_fasta(sdata, target_key, train_key=train_key, task=task, file_name=name)   
    print("Fitting model")
    log_file = open(os.path.join(log_dir, f"{prefix}{name}_fit{suffix}.log"), "w")
    process = subprocess.Popen(
        [
            'gkmtrain', 
            file1_name, # positive training file
            file2_name, # negative training file
            os.path.join(log_dir, prefix+name+suffix), # output file name
            '-y', str(param_dict[task]), # task type
            '-t', kernel_type, # kernel type
            '-l', str(word_length), # word length
            '-k', str(informative_columns), # number of informative columns
            '-d', str(mismatches), # number of mismatches
            '-g', str(gamma), # gamma
            '-M', str(decay_M), # decay M
            '-H', str(half_life), # half life
            #'-R', str(reverse_complement), # reverse complement
            '-c', str(regularization_strength), # regularization strength
            '-e', str(epsilon), # epsilon
            '-w', str(positive_weight), # positive weight
            #'-m', str(cached_mem_size), # cached memory size
            '-T', str(threads), # number of threads
            '-v', str(verbose), # verbose
        ],
                    stdout=log_file,
                    stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if stderr.decode("utf-8") != "":
        err_file = open(os.path.join(log_dir, f"{prefix}{name}_fit{suffix}.err"), "w")
        err_file.write(stderr.decode("utf-8"))
        raise Exception("Error in gkmtrain, check error file: " + os.path.join(log_dir, f"{prefix}{name}_fit{suffix}.err"))
    print("Model fit, log file saved to", os.path.join(log_dir, f"{prefix}{name}_fit{suffix}.log"))
    log_file.close()
    return


def predict(
    model,
    sdata=None,
    file_names=None,
    data_dir=None,
    log_dir=None,
    out_dir=None,
    file_label=None,
    threads=None,
    prefix="",
    suffix="",
    verbose=2
):  
    data_dir = data_dir if data_dir is not None else settings.dataset_dir
    log_dir = log_dir if log_dir is not None else settings.logging_dir
    out_dir = out_dir if out_dir is not None else settings.output_dir
    os.makedirs(out_dir) if not os.path.exists(out_dir) else None
    file_label = file_label if file_label is not None else "sdata"
    threads = threads if threads is not None else 4**(math.floor(math.log(threading.active_count(), 4)))
    
    # Check if model exists
    print(os.path.join(log_dir, prefix + model + suffix+".model.txt"))
    if not os.path.exists(model):
        if os.path.exists(os.path.join(log_dir, model + ".model.txt")):
            model = os.path.join(log_dir, model + ".model.txt")
        elif os.path.exists(os.path.join(log_dir, prefix + model + suffix+".model.txt")):
            model = os.path.join(log_dir, prefix+model+suffix + ".model.txt")
        else:
            raise Exception("Model file does not exist")
    if sdata is not None:
        file_name = os.path.join(data_dir, prefix+file_label+suffix+".fasta")
        print("here")
        if not os.path.exists(file_name):
            print("not_here")
            seqs = sdata.seqs
            ids = sdata.seqs_annot.index
            with open(file_name, "w") as f:
                for j, seq in enumerate(seqs):
                    f.write(">" + ids[j] + "\n" + seqs[j] + "\n")
        file_names = [file_name]
    else:
        assert file_names is not None, "Either sdata or file_names must be provided"
        if isinstance(file_names, str):
            file_names = [file_names]
    for file_name in file_names:
        if os.path.exists(os.path.join(data_dir, file_name)):
            file_name = os.path.join(data_dir, file_name)
        if not os.path.exists(file_name):
            raise Exception("Data file does not exist: " + file_name)
        log_file = open(os.path.join(out_dir, f"{prefix}{file_label}_predict{suffix}.log"), "w")
        process = subprocess.Popen(
            [
                'gkmpredict', 
                file_name, # seqs to test
                model, # model file
                os.path.join(out_dir, prefix+file_label+suffix+"_predictions.txt"), # output file name
                '-T', str(threads), # number of threads
                '-v', str(verbose), # verbose
            ],
                        stdout=log_file,
                        stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        if stderr.decode("utf-8") != "":
            err_file = open(os.path.join(out_dir, f"{prefix}{file_label}_predict{suffix}.err"), "w")
            err_file.write(stderr.decode("utf-8"))
            raise Exception("Error in gkmpredict, check error file: " + os.path.join(log_dir, f"{prefix}{file_name}predict{suffix}.err"))
        print("Predictions saved to", os.path.join(out_dir, prefix+file_label+suffix+"_predictions.txt"))
        log_file.close()

    if sdata is not None:
        f = open(os.path.join(out_dir, prefix+file_label+suffix+"_predictions.txt"), "r")
        d = [float(x.strip().split('\t')[1]) for x in f]
        f.close()
        sdata.seqs_annot[f"{file_label}_gkm_svm_predictions"] = d
    return
        

def explain(
    model,
    sdata=None,
    explanation_mode="importances",
    file_names=None,
    data_dir=None,
    log_dir=None,
    out_dir=None,
    file_label=None,
    threads=None,
    prefix="",
    suffix="",
    verbose=2
):  
    data_dir = data_dir if data_dir is not None else settings.dataset_dir
    log_dir = log_dir if log_dir is not None else settings.logging_dir
    out_dir = out_dir if out_dir is not None else settings.output_dir
    os.makedirs(out_dir) if not os.path.exists(out_dir) else None
    file_label = file_label if file_label is not None else "sdata"
    threads = threads if threads is not None else 4**(math.floor(math.log(threading.active_count(), 4)))
    
    # Check if model exists
    print(os.path.join(log_dir, prefix + model + suffix+".model.txt"))
    if not os.path.exists(model):
        if os.path.exists(os.path.join(log_dir, model + ".model.txt")):
            model = os.path.join(log_dir, model + ".model.txt")
        elif os.path.exists(os.path.join(log_dir, prefix + model + suffix+".model.txt")):
            model = os.path.join(log_dir, prefix+model+suffix + ".model.txt")
        else:
            raise Exception("Model file does not exist")

    if sdata is not None:
        file_name = os.path.join(data_dir, prefix+file_label+suffix+".fasta")
        if not os.path.exists(file_name):
            seqs = sdata.seqs
            ids = sdata.seqs_annot.index
            with open(prefix+file_label+suffix+".fasta", "w") as f:
                for j, seq in enumerate(seqs):
                    f.write(">" + ids[j] + "\n" + seqs[j] + "\n")
        file_names = [file_name]
    else:
        assert file_names is not None, "Either sdata or file_names must be provided"
        if isinstance(file_names, str):
            file_names = [file_names]
    for file_name in file_names:
        if os.path.exists(os.path.join(data_dir, file_name)):
            file_name = os.path.join(data_dir, file_name)
        if not os.path.exists(file_name):
            raise Exception("Data file does not exist: " + file_name)
        log_file = open(os.path.join(out_dir, f"{prefix}{file_label}_explain{suffix}.log"), "w")
        print("Running gkmexplain on", file_name, "with model", model, "and mode", mode_dict[explanation_mode])
        print(" ".join([
                'gkmexplain', 
                file_name, # seqs to test
                model, # model file
                os.path.join(out_dir, prefix+file_label+suffix+"_explanations.txt"), # output file name
                '-m', str(mode_dict[explanation_mode]), # explanation mode
                '-T', str(threads), # number of threads
                '-v', str(verbose), # verbose
            ]))
        process = subprocess.Popen(
            [
                'gkmexplain', 
                file_name, # seqs to test
                model, # model file
                os.path.join(out_dir, prefix+file_label+suffix+"_explanations.txt"), # output file name
                '-m', str(mode_dict[explanation_mode]), # explanation mode
                '-v', str(verbose), # verbose
            ],
                        stdout=log_file,
                        stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        if stderr.decode("utf-8") != "":
            err_file = open(os.path.join(out_dir, f"{prefix}{file_label}_explain{suffix}.err"), "w")
            err_file.write(stderr.decode("utf-8"))
            raise Exception("Error in gkmexplain, check error file: " + os.path.join(log_dir, f"{prefix}{file_name}explain{suffix}.err"))
        print("Predictions saved to", os.path.join(out_dir, prefix+file_label+suffix+"_explanations.txt"))
        log_file.close()

    if sdata is not None:
        file = os.path.join(out_dir, prefix+file_label+suffix+"_explanations.txt")
        impscores = np.array([np.array([[float(z) for z in y.split(",")] for y in x.rstrip().split("\t")[2].split(";")]).transpose() for x in open(file, "r")]) # list of np arrays of shape (L, 4) for each sequence
        sdata.uns[f"{file_label}_imps"] = impscores
    return 