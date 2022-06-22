# >>> lsgkm helper functions >>>

# Function to generate a gkSVM slurm script from passed in set of file names hyperparameters
# See 
def generate_slurm_train_script(input_dir,
                                pos_seqs,
                                neg_seqs,
                                val_seqs,
                                result_dir,
                                hyperparams,
                                preprocess,
                                features="fasta",
                                architecture="gkSVM"):
    if not os.path.exists(result_dir):
        print("{} does not exist, making dir".format(result_dir))
        os.makedirs(result_dir)
           
    # Set up model name
    model = "{}_{}_{}-clf_{}".format(preprocess, features, architecture, hyperparams)
    model_name = os.path.join(result_dir, model)
    
    # Set up hyperparams
    hyperparams = hyperparams.split("-")
    if hyperparams[5]:
        hyperparams.remove("True")
        hyperparams = "-t {} -l {} -k {} -d {} -R -c {} -w {}".format(*hyperparams)
    else:
        hyperparams.remove("False")
        hyperparams = "-t {} -l {} -k {} -d -c {} -w {}".format(*hyperparams)
        
    # Set up file pointers
    output = ["#!/bin/bash", "#SBATCH --cpus-per-task=16", "#SBATCH --time=48:00:00",
              "#SBATCH --partition carter-compute\n"]
    output += ['date\necho -e "Job ID: $SLURM_JOB_ID\\n"\n']
    output += ["trainposseqs={}".format(os.path.join(input_dir, pos_seqs)),
               "trainnegseqs={}".format(os.path.join(input_dir, neg_seqs)),
               "valseqs={}".format(os.path.join(input_dir, val_seqs)),
               "resultdir={}".format(result_dir),
               "modelname={}".format(model_name)]
    output += ["[ ! -d $resultdir ] && mkdir $resultdir\n"]
    
    # Set-up training command
    train_command = "gkmtrain $trainposseqs $trainnegseqs $modelname {} -v 2 -T $SLURM_CPUS_PER_TASK -m 8000.0".format(hyperparams)
    output += ["echo -e {}".format(train_command)]
    output += [train_command]
    output += ['echo -e "\\n"\n']
    
    # Set up positive train seq predict
    predict_pos_train_command = 'gkmpredict $trainposseqs $modelname".model.txt" $modelname".train-pos.predict.txt"'
    output += ["echo -e {}".format(predict_pos_train_command)]
    output += [predict_pos_train_command]
    output += ['echo -e "\\n"\n']
    
    # Set up negative train seq predict
    predict_neg_train_command = 'gkmpredict $trainnegseqs $modelname".model.txt" $modelname".train-neg.predict.txt"'
    output += ["echo -e {}".format(predict_neg_train_command)]
    output += [predict_neg_train_command]
    output += ['echo -e "\\n"\n']
    
    # Set up val seq predict
    predict_val_command = 'gkmpredict $valseqs $modelname".model.txt" $modelname".test.predict.txt"'
    output += ["echo -e {}".format(predict_val_command)]
    output += [predict_val_command]
    output += ['echo -e "\\n"\n']
    
    output += ["date"]
    
    # Write to script
    with open("{}/train_{}.sh".format(result_dir, model), "w") as f:
        f.write("\n".join(output))
        print("Successfully generated {}/train_{}.sh".format(result_dir, model))
        
    # Bash command to edit
    print("Usage: sbatch train_{0}.sh --job-name=train_{0} -o {1}/train_{0}.out -e {1}/train_{0}.err --mem=20G".format(model, result_dir))

# Function to grab scores from output of gkmtest
# name => filepath to read from
def get_scores(fname):
    f = open(fname)
    d = [float(x.strip().split('\t')[1]) for x in f]
    f.close()
    return d

# Function to score test predictions made on the positive and negative classes
# pos_file => ls-gkm scores for positive test seqs
# neg_file => ls-gkm scores for negative test seqs
# thresh => float threshold for accuracy scoring
def score(pos_file, neg_file, thresh):
    pos_scores = get_scores(pos_file)
    neg_scores = get_scores(neg_file)
    labels = [1]*len(pos_scores) + [0]*len(neg_scores)

    labels_shuf = deepcopy(labels)
    shuffle(labels_shuf)

    auprc = average_precision_score(labels, pos_scores+neg_scores)
    auroc = roc_auc_score(labels, pos_scores+neg_scores)
    auprc_shuf = average_precision_score(labels_shuf, pos_scores+neg_scores)
    auroc_shuf = roc_auc_score(labels_shuf, pos_scores+neg_scores)
    acc_thresh0 = sum([x==int(y>thresh) for x,y in zip(labels, pos_scores+neg_scores)])/len(labels)
    acc_thresh0_shuf = sum([x==int(y>thresh) for x,y in zip(labels_shuf, pos_scores+neg_scores)])/len(labels)

    print("Metric\tValue\tRandomised")
    print("Accuracy_at_threshold_{}\t{:.4f}\t{:.4f}".format(thresh, acc_thresh0, acc_thresh0_shuf))
    print("AUROC\t{:.4f}\t{:.4f}".format(auroc, auroc_shuf))
    print("AUPRC\t{:.4f}\t{:.4f}".format(auprc, auprc_shuf))

# Get all the needed information for viz sequence of gkmexplain result. Returns importance 
# scores per position along with the sequences, IDs and one-hot sequences
def get_gksvm_explain_data(explain_file, fasta_file):
    impscores = [np.array( [[float(z) for z in y.split(",")] for y in x.rstrip().split("\t")[2].split(";")]) for x in open(explain_file)]
    fasta_seqs = [x.rstrip() for (i,x) in enumerate(open(fasta_file)) if i%2==1]
    fasta_ids = [x.rstrip().replace(">", "") for (i,x) in enumerate(open(fasta_file)) if i%2==0]
    onehot_data = np.array([one_hot_encode_along_channel_axis(x) for x in fasta_seqs])
    return impscores, fasta_seqs, fasta_ids, onehot_data


# Save a list of sequences to separate pos and neg fa files. Must supply target 0 or 1 labels
def gkmSeq2Fasta(seqs, IDs, ys, name="seqs"):
    neg_mask = (ys==0)
    
    neg_seqs, neg_ys, neg_IDs = seqs[neg_mask], ys[neg_mask], IDs[neg_mask]
    neg_file = open("{}-neg.fa".format(name), "w")
    for i in range(len(neg_seqs)):
        neg_file.write(">" + neg_IDs[i] + "\n" + neg_seqs[i] + "\n")
    neg_file.close()
    
    pos_seqs, pos_ys, pos_IDs = seqs[~neg_mask], ys[~neg_mask], IDs[~neg_mask]
    pos_file = open("{}-pos.fa".format(name), "w")
    for i in range(len(pos_seqs)):
        pos_file.write(">" + pos_IDs[i] + "\n" + pos_seqs[i] + "\n")
    pos_file.close()
    
    

# <<< lsgkm helper functions <<<
