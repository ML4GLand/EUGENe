# -*- coding: utf-8 -*-


"""
Python script for evaluating EUGENE project models
TODO: 
    1. 
"""


# Built-in/Generic Imports
import glob
import os
import argparse
import warnings
import sys


# Libs
import tqdm
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, auc
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from copy import deepcopy
from random import shuffle
from mdutils.mdutils import MdUtils
from mdutils import Html
import torch
import torch.nn as nn
from livelossplot import PlotLosses
sys.path.append('/cellar/users/aklie/bin/make_it_train/')
from early_stop import EarlyStop


# Tags
__author__ = "Adam Klie"
__data__ = "09/28/2021"


# <<< Exploratory data analysis helper functions <<<
# Function to print useful stats
def contigency_table_stats(data, col, label):
    cont_table = pd.crosstab(data[label], data[col])
    if (not(cont_table.sum(axis=1).values == data["label"].value_counts().values).all()) or (not (np.array(sorted(cont_table.sum(axis=0).values)) == np.array(sorted(data[col].value_counts().values))).all()):
        print("Something is rotten in the state of Denmark")
    table = sm.stats.Table(cont_table)
    diff = table.table_orig - table.fittedvalues
    resid_pearson = diff / (table.fittedvalues ** 0.5)
    rslt = table.test_nominal_association()
    return cont_table, diff, resid_pearson, rslt


# Function to compute odds-rations for a specific column of a dataframe with labels
def odds_ratios(data, col, label, alpha=0.05):
    cont_table = pd.crosstab(data[label], data[col])
    # Defnie a CI and an empty dataframe to hold odds-ratio info
    odds_df = pd.DataFrame(columns=["OR", 
                                    "OR " + str(alpha * 100) + "%",
                                    "OR " + str((1 - alpha) * 100) + "%"],
                           index=cont_table.columns)

    for col in cont_table.columns:
        tab = pd.concat([cont_table[col], cont_table.drop(col, axis=1).sum(axis=1)], axis=1)
        tab = tab.loc[[1, 0]]
        table = sm.stats.Table2x2(tab.values)
        if not (
            (
                table.table_orig.sum(axis=1)
                == data["label"].value_counts().loc[[1, 0]].values
            ).all()
        ):
            print("Something is rotten in the state of Denmark")
        odds_df.loc[col]["OR"] = table.oddsratio
        (
            odds_df.loc[col]["OR " + str(alpha * 100) + "%"],
            odds_df.loc[col]["OR " + str((1 - alpha) * 100) + "%"],
        ) = table.oddsratio_confint(0.05)
    return odds_df


# Function to plot those odds-ratios with CI
def plot_odds_ratios(odds_df, col, savefig=None):
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axvline(1, ls="--", linewidth=1, color="black")
    n = 0
    for index, i in odds_df.iterrows():
        x = [i["OR 5.0%"], i["OR 95.0%"]]
        y = [n, n]
        ax.plot(x, y, 
                "|", markersize=25, markeredgewidth=3,linewidth=3, color=sns.color_palette("muted")[n])
        ax.plot(x, y,
            "-", markersize=25, markeredgewidth=3, linewidth=3, color=sns.color_palette("muted")[n])
        y = [n]
        x = [i["OR"]]
        ax.plot(x, y, "o", color=sns.color_palette("muted")[n], markersize=15)
        n += 1
    ax.set_ylabel("{0} in position {1}".format(col.split("_")[0], col.split("_")[1]), fontsize=30)
    ax.set_yticks(range(0, n))
    ax.set_yticklabels(odds_df.index, fontsize=24)
    ax.set_xlabel("Odds Ratio w/ 95% CI", fontsize=30)
    ax.tick_params(axis="x", labelsize=24)
    #ax.text(
    #    odds_df["OR 95.0%"].max() + 0.05,
    #    n - 1.5,
    #    r"$\chi^{2} =$" + str(round(chi2, 2)),
    #    fontsize=36,
    #)
    #ax.text(odds_df["OR 95.0%"].max() + 0.05, n - 2.5, "p = {}".format(p), fontsize=36)
    if savefig != None:
        plt.savefig(savefig)
# >>> Exploratory data analysis helper functions >>>


# <<< Data preprocessing helper functions <<<
# Function to perform train test splitting
def split_train_test(X_data, y_data, split=0.8, subset=None, rand_state=13, shuf=True):
    train_X, test_X, train_y, test_y = train_test_split(X_data, y_data, train_size=split, random_state=rand_state, shuffle=shuf)
    if subset != None:
        num_train = int(len(train_X)*subset)
        num_test = int(len(test_X)*subset)
        train_X, test_X, train_y, test_y = train_X[:num_train, :], test_X[:num_test, :], train_y[:num_train], test_y[:num_test]
    return train_X, test_X, train_y, test_y


# Function to standardize features based on passed in indeces and optionally save stats
def standardize_features(train_X, test_X, indeces=None, stats_file=None):
    if type(indeces) == None:
        indeces = range(train_X.shape[1])
    if len(indeces) == 0:
        return train_X, test_X
    
    means = train_X[:, indeces].mean(axis=0)
    stds = train_X[:, indeces].std(axis=0)
    
    train_X_scaled = train_X.copy()
    train_X_scaled[:, indeces] = train_X[:, indeces] - means
    train_X_scaled[:, indeces] = train_X_scaled[:, indeces] / stds
    
    test_X_scaled = test_X.copy()
    test_X_scaled[:, indeces] = test_X[:, indeces] - means
    test_X_scaled[:, indeces] = test_X_scaled[:, indeces] / stds
    
    if stats_file != None:
        stats_dict = {"indeces": indeces, "means": means, "stds": stds}
        with open(stats_file, 'wb') as handle:
            pickle.dump(stats_dict, handle)
            
    return train_X_scaled, test_X_scaled


# Function to one-hot encode a set of sequences (array-like)
def ohe_seqs(seqs):
    # Define encoders
    integer_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder(categories=[np.array([0, 1, 2, 3])], handle_unknown="ignore")
    
    X_features = []  # will hold one hot encoded sequence
    for i, seq in enumerate(tqdm.tqdm(seqs)):
        integer_encoded = integer_encoder.fit_transform(list(seq))  # convert to integer
        integer_encoded = np.array(integer_encoded).reshape(-1, 1)
        one_hot_encoder.fit(integer_encoded)  # convert to one hot
        one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
        X_features.append(one_hot_encoded.toarray())
        
    print("Encoded {} seqs".format(len(X_features)))
        
    # convert to numpy array
    X_ohe_seq = np.array(X_features)
    
    # Sanity check encoding for randomly chosens sequences  
    l = len(X_features)
    if l < 1000:
        print("Checking all {} seqs for proper encoding".format(l))
        indeces = np.random.choice(len(X_features), size=len(X_features))
    else:
        indeces = np.random.choice(len(X_features), size=1000)
        print("Checking 1000 random seqs for proper encoding")
    bad_encoding = False
    for j, ind in enumerate(indeces):
        seq = seqs[ind]
        one_hot_seq = X_features[ind]
        for i, bp in enumerate(seq):
            if bp == "A":
                if (one_hot_seq[i] != [1.0, 0.0, 0.0, 0.0]).all():
                    print("You one hot encoded wrong dummy!")
                    print(seq, one_hot_seq)
                    bad_encoding = True
            elif bp == "C":
                if (one_hot_seq[i] != [0.0, 1.0, 0.0, 0.0]).all():
                    print("You one hot encoded wrong dummy!")
                    print(seq, one_hot_seq)
                    bad_encoding = True
            elif bp == "G":
                if (one_hot_seq[i] != [0.0, 0.0, 1.0, 0.0]).all():
                    print("You one hot encoded wrong dummy!")
                    print(seq, one_hot_seq)
                    bad_encoding = True
            elif bp == "T":
                if (one_hot_seq[i] != [0.0, 0.0, 0.0, 1.0]).all():
                    print("You one hot encoded wrong dummy!")
                    print(seq, one_hot_seq)
                    bad_encoding = True
            elif bp == "N":
                if (one_hot_seq[i] != [0.0, 0.0, 0.0, 0.0]).all():
                    print("You one hot encoded wrong dummy!")
                    print(seq, one_hot_seq)
                    bad_encoding = True
            else:
                print(bp)
    if bad_encoding:
        print("Something is amiss in the state of Denmark, try encoding again")
    else:
        print("Sequence encoding was great success")
    return X_ohe_seq 
# >>> Data preprocessing helper functions >>>


# >>> Classification metric helper functions >>>
# Confusion matrix function
def train_test_confusion_matrix(train_y, train_y_preds, test_y, test_y_preds, figsize=(16,8), savefile=None):
    fig, ax = plt.subplots(1,2,figsize=figsize)

    rc = {'font.size': 20}
    with plt.rc_context(rc):
        sns.heatmap(confusion_matrix(train_y, train_y_preds), annot=True, fmt='d', cmap='viridis', ax=ax[0])
        ax[0].set_xlabel('predicted')
        ax[0].set_ylabel('true label')
        ax[0].set_title('Train Set')

        sns.heatmap(confusion_matrix(test_y, test_y_preds), annot=True, fmt='d', cmap='viridis', ax=ax[1])
        ax[1].set_xlabel('predicted')
        ax[1].set_ylabel('true label')
        ax[1].set_title('Test Set')

        plt.tight_layout()
    
    if savefile != None:
        plt.savefig(savefile)
        plt.close()

        
def cf_plot_from_df(data, label_col="FXN_LABEL", pred_col="PREDS", title="Sequences", xlab="Predicted Activity", ylab="True Activity", threshold=0.5):
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    rc = {"font.size": 16}
    with plt.rc_context(rc):
        cf_names = ["True Neg","False Pos", "False Neg","True Pos"]
        cf_mtx = confusion_matrix(data[label_col], data[pred_col])
        cf_pcts = ["{0:.2%}".format(value) for value in (cf_mtx/cf_mtx.sum(axis=1)[:,None]).flatten()]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(cf_mtx.flatten(),cf_pcts, cf_names)]
        labels = np.asarray(labels).reshape(2,2)
        sns.heatmap(cf_mtx, annot=labels, fmt='s', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel(xlab, fontsize=20)
        ax.set_ylabel(ylab, fontsize=20)
        ax.set_title(title, fontsize=24)
        ax.set_yticklabels(["Inactive", "Active"], fontsize=16)
        ax.set_xticklabels(["Inactive (Score<{})".format(str(threshold)), "Active (Score>{})".format(str(threshold))], fontsize=16)
        plt.tight_layout();

        
        
# Wrapper function around sklearn accuracy_score to print accuracy score for train and test
def train_test_metrics(train_y, train_y_preds, test_y, test_y_preds):
    output= ["Metric", "Train", "Test"]
    print("\t".join(output))
    train_acc, test_acc = accuracy_score(y_true=train_y, y_pred=train_y_preds), accuracy_score(y_true=test_y, y_pred=test_y_preds)
    train_prec, test_prec = precision_score(y_true=train_y, y_pred=train_y_preds), precision_score(y_true=test_y, y_pred=test_y_preds)
    train_recall, test_recall = recall_score(y_true=train_y, y_pred=train_y_preds), recall_score(y_true=test_y, y_pred=test_y_preds)
    
    def fbeta_score(pr, rec, beta):
        return (1+(beta**2))*((pr*rec)/(((beta**2)*pr)+rec))
    fbeta_train_scores, fbeta_test_scores = [], []
    
    for b in [0.1, 0.5, 1, 2, 10]:
        fbeta_train_scores.append(fbeta_score(train_prec, train_recall, b))
        fbeta_test_scores.append(fbeta_score(test_prec, test_recall, b))
    output.extend(["Accuracy", train_acc, test_acc])
    output.extend(["Precision", train_prec, test_prec])
    output.extend(["Recall", train_recall, test_recall])
    print("{:s}\t{:.4f}\t{:.4f}".format("Accuracy", train_acc, test_acc))
    print("{:s}\t{:.4f}\t{:.4f}".format("Precision", train_prec, test_prec))
    print("{:s}\t{:.4f}\t{:.4f}".format("Recall", train_recall, test_recall))
    for i, b in enumerate([0.1, 0.5, 1, 2, 10]):
        print("F{:s}-Score\t{:.4f}\t{:.4f}".format(str(b), fbeta_train_scores[i], fbeta_test_scores[i]))
        output.extend(["F" + str(b), fbeta_train_scores[i], fbeta_test_scores[i]])
    return output
        
        
def train_test_pr_curve(train_y, train_y_probs, test_y, test_y_probs, savefile=None):
    precs_train, recs_train, threshs_train = precision_recall_curve(y_true=train_y, probas_pred=train_y_probs)
    avg_prec_train = average_precision_score(y_true=train_y, y_score=train_y_probs)
    
    precs_test, recs_test, threshs_test = precision_recall_curve(y_true=test_y, probas_pred=test_y_probs)
    avg_prec_test = average_precision_score(y_true=test_y, y_score=test_y_probs)
    
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.step(recs_train, precs_train, where='post', lw=3, alpha=0.4, label='Training auPRC = %0.4f' % (avg_prec_train))
    ax.step(recs_test, precs_test, where='post', lw=3, alpha=0.4, label='Testing auPRC = %0.4f' % (avg_prec_test))
    ax.axhline(len(test_y[test_y==1])/len(test_y), linestyle='--', lw=3, color='k', label='No skill', alpha=.8)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Recall', fontsize=18)
    ax.set_ylabel('Precision', fontsize=18)
    ax.set_title('Precision recall', fontsize=18)
    ax.legend(loc="lower right", fontsize=16)
    if savefile != None:
        plt.savefig(savefile)
        plt.close()
    
    
def train_test_roc_curve(train_y, train_y_probs, test_y, test_y_probs, savefile=None):
    fprs_train, tprs_train, threshs_train = roc_curve(y_true=train_y, y_score=train_y_probs)
    roc_auc_train = auc(fprs_train, tprs_train)

    fprs_test, tprs_test, threshs_test = roc_curve(y_true=test_y, y_score=test_y_probs)
    roc_auc_test = auc(fprs_test, tprs_test)

    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.plot(fprs_train, tprs_train, lw=3, alpha=0.4, label='Training (auROC = %0.4f)' % (roc_auc_train))
    ax.plot(fprs_test, tprs_test, lw=3, alpha=0.4, label='Test (auROC= %0.4f)' % (roc_auc_test))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=3, color='k', label='No skill', alpha=.8)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=18)
    ax.set_ylabel('True Positive Rate', fontsize=18)
    ax.set_title('Receiver operating characteristic', fontsize=18)
    ax.legend(loc="lower right", fontsize=16)
    if savefile != None:
        plt.savefig(savefile)
        plt.close()
# <<< Classification metric helper functions <<<


# <<< Classification report function <<<
# NEED TO GENERALIZE TO ANY NUMBER OF PROVIDED SETS
def classification_report(out_path,
                          train_X, test_X, 
                          train_y, test_y,
                          train_preds=None, test_preds=None,
                          train_probs=None, test_probs=None,
                          predict=False, title=None, iters_trained=None):
    
    # Quick set-up
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if title == None:
        title = out_path + " Classifier Report"
        
    # Generate predictions if needed
    if predict:
        train_preds = clf.predict(train_X)
        test_preds = clf.predict(test_X)
        train_probs = clf.predict_proba(train_X)[:, 1]
        test_probs = clf.predict_proba(test_X)[:, 1]
    else:
        print("Predictions provided, skipping them")
        
    print("Generating confusion matrix")
    train_test_confusion_matrix(train_y=train_y, 
                                train_y_preds=train_preds, 
                                test_y=test_y, 
                                test_y_preds=test_preds,
                                savefile="{}/confusion.png".format(out_path))
    
    print("Calculating classification metrics")
    clf_metrics = train_test_metrics(train_y=train_y, 
                                     train_y_preds=train_preds, 
                                     test_y=test_y, 
                                     test_y_preds=test_preds)
    
    print("Plotting PR Curve")
    train_test_pr_curve(train_y=train_y, 
                        train_y_probs=train_probs, 
                        test_y=test_y, 
                        test_y_probs=test_probs,
                        savefile="{}/pr_curve.png".format(out_path))
    
    print("Plotting ROC Curve")
    train_test_roc_curve(train_y=train_y, 
                         train_y_probs=train_probs, 
                         test_y=test_y, 
                         test_y_probs=test_probs,
                         savefile="{}/roc_curve.png".format(out_path))
    
    print("Generating report")
    if iters_trained != None:
        mdFile = MdUtils(file_name="{}/classification-report_{}-iters.md".format(out_path, iters_trained), title=title)
        mdFile.new_line(text="Model was trained for a total of {} iterations".format(iters_trained))
    else:
        mdFile = MdUtils(file_name="{}/classification-report.md".format(out_path), title=title)
    
    mdFile.new_header(level=1, title='Confusion Matrices')
    mdFile.new_line(mdFile.new_inline_image(text="confusion_matrices", path="confusion.png"))
    mdFile.new_line()

    mdFile.new_header(level=1, title='Classification Metrics')
    mdFile.new_table(columns=3, rows=9, text=clf_metrics, text_align='center')

    mdFile.new_header(level=1, title='Precision-Recall Curve')
    mdFile.new_line(mdFile.new_inline_image(text="pr_curve", path="pr_curve.png"))
    mdFile.new_line()

    mdFile.new_header(level=1, title='Reciever Operator Characteristic')
    mdFile.new_line(mdFile.new_inline_image(text="roc_curve", path="roc_curve.png"))
    mdFile.new_line()

    mdFile.new_table_of_contents(table_title='Contents', depth=1)

    mdFile.create_md_file()
    
    
def threshold_plot(data, label_col="FXN_LABEL", score_col="SCORES", threshold=0.5):
    tns, fps, fns, tps = [], [], [], []
    threshs = np.arange(data[score_col].min(), data[score_col].max(), 0.1)
    for i, thresh in tqdm.tqdm(enumerate(threshs)):
        data["curr_pred"] = (data[score_col] >= thresh).astype(int)
        data["curr_class"] = ["-".join(list(value)) for value in data[["MPRA_FXN", "curr_pred"]].values.astype(str)]
        data["curr_class"] = data["curr_class"].replace({"0.0-0.0": "TN", "1.0-0.0": "FN", "0.0-1.0": "FP", "1.0-1.0": "TP"})

        class_results = data["curr_class"].value_counts()
        if "TN" in class_results:
            tns.append(class_results["TN"])
        else:
            tns.append(0)

        if "FP" in class_results:
            fps.append(class_results["FP"])
        else:
            fps.append(0)

        if "FN" in class_results:
            fns.append(class_results["FN"])
        else:
            fns.append(0)

        if "TP" in class_results:
            tps.append(class_results["TP"])
        else:
            tps.append(0)

    nb_negs = (data[label_col] == 0).sum()
    nb_pos = (data[label_col] == 1).sum()
    if (not (np.array([tns, fps, fns, tps]).sum(axis=0) == len(data)).all()) or (not (len(tns) == len(fps) == len(fns) == len(tps) == len(threshs))):
        print("Something rotten in Denmark")

    fig, ax = plt.subplots(1,1, figsize=(8,8))
    plt.plot(threshs, tns/nb_negs, color="lightgreen", label="True Negative Rate")
    plt.plot(threshs, fns/nb_pos, color="lightcoral", label="False Negative Rate")
    plt.plot(threshs, fps/nb_negs, color="darkred", label="False Positive Rate")
    plt.plot(threshs, tps/nb_pos, color="darkgreen", label="True Positive Rate")
    plt.vlines(threshold, 0, 1, color="orange", linestyle="dashed", label="Threshold")
    plt.legend(bbox_to_anchor=(1,1), fontsize=16)
    plt.xlabel("Score Threshold")
    plt.ylabel("Classification Rate")
    
    
def coefficient_plot(classifier, features, xlab="Feature", ylab="Coefficient", title="Model Coefficients"):
    rc = {"font.size": 14}
    with plt.rc_context(rc):
        coefs = pd.DataFrame(classifier.coef_[0], columns=["Coefficients"], index=features)
        coefs.plot(kind="bar", figsize=(16, 8), legend=None)
        plt.title(title, fontsize=18)
        plt.axhline(y=0, color=".5")
        plt.subplots_adjust(left=0.3)
        plt.xlabel(xlab, fontsize=16)
        plt.ylabel(ylab, fontsize=16)
# <<< Classification report function <<<


# >>> lsgkm helper functions >>>
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
    
    
# Function to generate a gkSVM slurm script
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
    if hyperparams[4]:
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
# <<< lsgkm helper functions <<<


# >>> Neural network functions >>>
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        
        
# Function to calcuate accuracy given raw values from classifier and labels
def accuracy(raw, labels):
    predictions = torch.round(torch.sigmoid(raw))
    return sum(predictions.eq(labels)).item()


# Current livelossplot compatible training script
def train_binary_classifier(model, 
                          dataloaders, 
                          double_stranded=False,
                          device="cpu",
                          criterion=torch.nn.BCEWithLogitsLoss(reduction='sum'), 
                          optimizer=None, 
                          num_epoch=50, 
                          early_stop=False, 
                          patience=3,
                          plot_frequency=10):
    liveloss = PlotLosses()
    loss_history, acc_history = {}, {}
    if early_stop:
        print("Using early stopping with a patience of {}".format(patience))
        stop = EarlyStop(patience)
        e_stop = False
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(num_epoch):
        logs = {}
        for phase in ['train', 'validation']:

            if phase == 'train' and epoch > 0:
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_acc = 0.0
            for inputs, targets in dataloaders[phase]:
                if double_stranded:
                    input_forward = inputs[:, :, :, 0].to(device)
                    input_reverse = inputs[:, :, :, 1].to(device)
                    outputs = model(input_forward.float(), input_reverse.float())
                else:
                    inputs = inputs.to(device)
                    outputs = model(inputs.float())
                targets = targets.unsqueeze(dim=1).to(device)
                loss = criterion(outputs, targets.float())
                if phase == 'train' and epoch > 0:
                    optimizer.zero_grad()
                    loss.backward()                
                    optimizer.step()          
                
                running_loss += loss.item()
                running_acc += accuracy(outputs, targets)
            
            len_dataset = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / len_dataset
            epoch_acc = running_acc / len_dataset
                
            prefix = ''
            if phase == 'validation':
                prefix = 'val_'
                if early_stop:
                    e_stop, best_model = stop(epoch_loss, model)
                    
            logs[prefix + 'loss'] = epoch_loss
            logs[prefix + 'acc'] = epoch_acc
            
            loss_history.setdefault(phase, []).append(epoch_loss)
            acc_history.setdefault(phase, []).append(epoch_acc)
            
        liveloss.update(logs)
        if epoch % plot_frequency == 0:
            liveloss.send()
        if early_stop:
            if e_stop:
                print("Early stopping occured at epoch {}".format(epoch))
                break
        best_model = model
            
    return best_model, epoch, loss_history, acc_history, liveloss
# <<< Neural network functions <<<


# >>> Seq utils functions >>>
# One hot encode a sequence with a loop. From gkmexplain
def one_hot_encode_along_channel_axis(sequence):
    to_return = np.zeros((len(sequence),4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                 sequence=sequence, one_hot_axis=1)
    return to_return


# One hot encode a sequence with a loop. From gkmexplain
def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis==0 or one_hot_axis==1
    if (one_hot_axis==0):
        assert zeros_array.shape[1] == len(sequence)
    elif (one_hot_axis==1): 
        assert zeros_array.shape[0] == len(sequence)
    #will mutate zeros_array
    for (i,char) in enumerate(sequence):
        if (char=="A" or char=="a"):
            char_idx = 0
        elif (char=="C" or char=="c"):
            char_idx = 1
        elif (char=="G" or char=="g"):
            char_idx = 2
        elif (char=="T" or char=="t"):
            char_idx = 3
        elif (char=="N" or char=="n"):
            continue #leave that pos as all 0's
        else:
            raise RuntimeError("Unsupported character: "+str(char))
        if (one_hot_axis==0):
            zeros_array[char_idx,i] = 1
        elif (one_hot_axis==1):
            zeros_array[i,char_idx] = 1
            

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
    

# Save a list of sequences to fasta
def seq2Fasta(seqs, IDs, name="seqs"):
    file = open("{}.fa".format(name), "w")
    for i in range(len(seqs)):
        file.write(">" + IDs[i] + "\n" + seqs[i] + "\n")
    file.close()
# >>> Seq utils functions >>>
