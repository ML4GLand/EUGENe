# -*- coding: utf-8 -*-


"""
Python script for evaluating EUGENE project models
TODO: 
    1. Add more to report -- TBD
    2. Add interpretation functions?
    3. 
"""


# Built-in/Generic Imports
import glob
import os
import argparse
import warnings


# Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, auc
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from copy import deepcopy
from random import shuffle
from mdutils.mdutils import MdUtils
from mdutils import Html
import torch
import torch.nn as nn


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


# Function to standardize features based on passed in indeces
def standardize_features(train_X, test_X, indeces=None):
    if type(indeces) == None:
        indeces = range(train_X.shape[1])
    if len(indeces) == 0:
        return train_X, test_X
    means = train_X[:, indeces].mean(axis=0)
    stds = train_X[:, indeces].std(axis=0)
    train_X[:, indeces] -= means
    train_X[:, indeces] /= stds
    test_X[:, indeces] -= means
    test_X[:, indeces] /= stds
    return train_X, test_X
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
# <<< Classification metric helper functions <<<


# <<< Classification report function <<<
def classification_report(filename, md_title, clf, train_X, test_X, train_y, test_y):
    train_y_preds = clf.predict(train_X)
    test_y_preds = clf.predict(test_X)
    train_y_probs = clf.predict_proba(train_X)[:, 1]
    test_y_probs = clf.predict_proba(test_X)[:, 1]
    print("Generating confusion matrix")
    train_test_confusion_matrix(train_y=train_y, 
                                train_y_preds=train_y_preds, 
                                test_y=test_y, 
                                test_y_preds=test_y_preds,
                                savefile="tmp.confusion.png")
    
    print("Calculating classification metrics")
    clf_metrics = train_test_metrics(train_y=train_y, 
                                     train_y_preds=train_y_preds, 
                                     test_y=test_y, 
                                     test_y_preds=test_y_preds)
    
    print("Plotting PR Curve")
    train_test_pr_curve(train_y=train_y, 
                        train_y_probs=train_y_probs, 
                        test_y=test_y, 
                        test_y_probs=test_y_probs,
                        savefile="tmp.pr_curve.png")
    
    print("Plotting ROC Curve")
    train_test_roc_curve(train_y=train_y, 
                         train_y_probs=train_y_probs, 
                         test_y=test_y, 
                         test_y_probs=test_y_probs,
                         savefile="tmp.roc_curve.png")
    
    print("Generating report")
    mdFile = MdUtils(file_name=filename, title=md_title)
    
    mdFile.new_header(level=1, title='Confusion Matrices')
    mdFile.new_line(mdFile.new_inline_image(text="confusion_matrices", path="tmp.confusion.png"))
    mdFile.new_line()

    mdFile.new_header(level=1, title='Classification Metrics')
    mdFile.new_table(columns=3, rows=9, text=clf_metrics, text_align='center')

    mdFile.new_header(level=1, title='Precision-Recall Curve')
    mdFile.new_line(mdFile.new_inline_image(text="pr_curve", path="tmp.pr_curve.png"))
    mdFile.new_line()

    mdFile.new_header(level=1, title='Reciever Operator Characteristic')
    mdFile.new_line(mdFile.new_inline_image(text="roc_curve", path="tmp.roc_curve.png"))
    mdFile.new_line()

    mdFile.new_table_of_contents(table_title='Contents', depth=2)

    mdFile.create_md_file()
    #os.remove("tmp.confusion.png")
    #os.remove("tmp.pr_curve.png")
    #os.remove("tmp.roc_curve.png")
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
# <<< lsgkm helper functions <<<


# >>> Neural network functions >>>
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        
        
# Function to calcuate accuracy given raw values from classifier and labels
def accuracy(raw, labels):
    predictions = torch.round(torch.sigmoid(raw))
    return sum(predictions.eq(labels)).item()
# <<< Neural network functions <<<


# >>> Seq utils functions >>>
def one_hot_encode_along_channel_axis(sequence):
    to_return = np.zeros((len(sequence),4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                 sequence=sequence, one_hot_axis=1)
    return to_return

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
            

def get_gksvm_explain_data(explain_file, fasta_file):
    impscores = [np.array( [[float(z) for z in y.split(",")] for y in x.rstrip().split("\t")[2].split(";")]) for x in open(explain_file)]
    fasta_seqs = [x.rstrip() for (i,x) in enumerate(open(fasta_file)) if i%2==1]
    fasta_ids = [x.rstrip().replace(">", "") for (i,x) in enumerate(open(fasta_file)) if i%2==0]
    onehot_data = np.array([one_hot_encode_along_channel_axis(x) for x in fasta_seqs])
    return impscores, fasta_seqs, fasta_ids, onehot_data
# >>> Seq utils functions >>>