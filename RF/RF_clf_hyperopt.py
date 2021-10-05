#!/usr/bin/python
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import optuna
import joblib

def hyperopt(trial, train_X, test_X, train_y, test_y):
    
    print("Starting trial number {}\n".format(trial.number))
    
    # Define search space
    n_estimators = trial.suggest_int("n_estimators", 800, 2000) # default is 100
    max_depth = trial.suggest_int("max_depth", 2, 32) # default is 2
    min_samples_split = trial.suggest_int("min_samples_split", 4, 10)  # minimum num samples to split an internal node; 2 is default value
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 10)  # default is 1
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


if __name__ == "__main__":
    
    print('Loading data...', end='')
    X, y = np.loadtxt('../data/2021_OLS_Library/mixed_2.0/X_mixed_2.0_0.18-0.4.txt', delimiter=' ', dtype=float), np.loadtxt('../data/2021_OLS_Library/y_binary_0.18-0.4.txt', delimiter=' ', dtype=int)
    
    # For testing
    if len(sys.argv) > 2:
        if sys.argv[2] == 'test':
            X, y = X[:100], y[:100]
    
    print("Dataset size: {}\n".format(len(X)))
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=13, shuffle=True)
    
    name = sys.argv[1]
    print('Starting study: {}'.format(name))
    study = optuna.create_study(direction="maximize", study_name=name)  # Create an optuna study --> maximize b/c want maximum AUC
    
    try:
        study.optimize(lambda trial: hyperopt(trial, X_train, X_test, y_train, y_test), n_trials=500, timeout=None, n_jobs=32) 
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

    joblib.dump(study, "{}_RF-clf-hyperopt.pkl".format(name))
    study.trials_dataframe().to_csv("{}_RF-clf-hyperopt.tsv".format(name), sep='\t', index=False)