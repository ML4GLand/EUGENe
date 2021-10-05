# -*- coding: utf-8 -*-

"""
Python script for evaluating EUGENE project models
TODO: 
    1. 
    2. 
    3. 
"""

# Built-in/Generic Imports
import glob
import os
import argparse
import warnings


# Libs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, auc
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from copy import deepcopy
from random import shuffle
from mdutils.mdutils import MdUtils
from mdutils import Html

# Tags
__author__ = "Adam Klie"
__data__ = "09/28/2021"


# <<< Data preprocessing helper functions <<<

# Function to perform train test splitting
def split_train_test(X_data, y_data, split=0.8, test=False, rand_state=13, shuf=True):
    train_X, test_X, train_y, test_y = train_test_split(X_data, y_data, train_size=split, random_state=rand_state, shuffle=shuf)
    if test:
        num_train = int(len(train_X)/10)
        num_test = int(len(test_X)/10)
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

