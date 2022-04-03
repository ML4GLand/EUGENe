#!/usr/bin/python
import pandas as pd
import numpy as np
import sys
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import optuna
import joblib

def hyperopt(trial, train_X, test_X, train_y, test_y):
    
    print("Starting trial number {}\n".format(trial.number))
    
    # Define search space
    n_estimators = trial.suggest_int("n_estimators", 100, 2000) # default is 100
    max_depth = trial.suggest_int("max_depth", 2, 32) # default is 2
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)  # minimum num samples to split an internal node; 2 is default value
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)  # default is 1
    bootstrap = trial.suggest_categorical("bootstrap", [True, False]) # default is True
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None, 13])  # default is auto
    
    # Instantiate a model
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split, 
                                 min_samples_leaf=min_samples_leaf,
                                 bootstrap=bootstrap,
                                 max_features=max_features,
                                 random_state=13
                                )
    
    # Fit the model and predict
    clf.fit(train_X, train_y)
    probs_y = clf.predict_proba(test_X)[:, 1]
    
    # Evaluate AUC and return
    fprs, tprs, threshs = roc_curve(y_true=test_y, y_score=probs_y)
    roc =  auc(fprs, tprs)
    
    print('Trial number {} complete\n'.format(trial.number))
    return roc

def main(args):
    
    # Define params
    DATASET = os.path.join("/cellar/users/aklie/projects/EUGENE/data", args.dataset)  # Which dataset to look at
    PREPROCESS = args.preprocess # Preprocessing steps, separated by "-"
    FEATURES = args.features  # What features to use to train the model
    LABELS = args.labels    
    SPLIT = args.split
    OUTDIR = args.out
    NUM_TRIALS = args.num_trials
    
    if not os.path.exists(OUTDIR):
        print("{} does not exist, making dir".format(os.makedirs(OUTDIR)))
    
    # Load train and val
    print('Loading data from {}'.format(DATASET))
    X_train = np.load('{0}/{1}/{2}_X-train-{3}_{4}.npy'.format(DATASET, FEATURES.replace("-", "_"), PREPROCESS, SPLIT, FEATURES))
    X_test = np.load('{0}/{1}/{2}_X-test-{3}_{4}.npy'.format(DATASET, FEATURES.replace("-", "_"), PREPROCESS, round(1-SPLIT, 1), FEATURES))
    y_train = np.loadtxt('{0}/{1}/{2}_y-train-{3}_{1}.txt'.format(DATASET, LABELS, PREPROCESS, SPLIT), dtype=int)
    y_test = np.loadtxt('{0}/{1}/{2}_y-test-{3}_{1}.txt'.format(DATASET, LABELS, PREPROCESS, round(1-SPLIT, 1)), dtype=int)
    
    # For testing
    if args.test:
        X_train, y_train = X_train[:100], y_train[:100]
        X_test, y_test = X_test[:10], y_test[:10]
    print("Dataset sizes:\n\tTrain:{}, {}\n\tTest:{}, {}\n".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    
    # Make the study
    name = "{1}_{2}_RF-study_{3}".format(OUTDIR, PREPROCESS, FEATURES, str(args.num_trials) + "-trial-optuna")
    print('Starting study: {}'.format(name))
    study = optuna.create_study(direction="maximize", study_name=name)  # Create an optuna study --> maximize b/c want maximum AUC
    
    try:
        study.optimize(lambda trial: hyperopt(trial, X_train, X_test, y_train, y_test), n_trials=args.num_trials, timeout=None, n_jobs=32) 
    except KeyboardInterrupt:
        pass

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("\tNumber of finished trials: ", len(study.trials))
    print("\tNumber of pruned trials: ", len(pruned_trials))
    print("\tNumber of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("\tValue: ", trial.value)

    print("\tParams: ")
    for key, value in trial.params.items():
        print("\t\t{}: {}".format(key, value))

    joblib.dump(study, os.path.join(OUTDIR, "{}.pickle".format(name)))
    study.trials_dataframe().to_csv(os.path.join(OUTDIR, "{}.tsv".format(name)), sep='\t', index=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="2021_OLS_Library", help="Name of folder in 'data'")
    parser.add_argument("--preprocess", type=str, help="Preprocessing done on dataset (e.g. 0.18-0.4)")
    parser.add_argument("--features", type=str, help="Features to train with (e.g. mixed-1.0)")
    parser.add_argument("--labels", type=str, default="binary", help="Folder within datast with labels (e.g binary)")
    parser.add_argument("--split", type=float, default=0.9, help="Train/test split desired")   
    parser.add_argument("--out", type=str, default="./", help="Output directory for Optuna files")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials to complete")
    parser.add_argument("--test", action='store_true', help="Whether to test on 100 seqs")
    args = parser.parse_args()
    main(args)