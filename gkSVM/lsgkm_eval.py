from sklearn.metrics import average_precision_score, roc_auc_score
from copy import deepcopy
from random import shuffle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pos", type=str, help="gkmpredict output on positive test examples", required=True)
parser.add_argument("-n", "--neg", type=str, help="gkmpredict output on negative test examples", required=True)
parser.add_argument("-t", "--thresh", type=float, help="threshold for accuracy prediction", default=0)
args = parser.parse_args()

def get_scores(fname):
    f = open(fname)
    d = [float(x.strip().split('\t')[1]) for x in f]
    f.close()
    return d

pos_scores = get_scores(args.pos)
neg_scores = get_scores(args.neg)
labels = [1]*len(pos_scores) + [0]*len(neg_scores)

labels_shuf = deepcopy(labels)
shuffle(labels_shuf)

auprc = average_precision_score(labels, pos_scores+neg_scores)
auroc = roc_auc_score(labels, pos_scores+neg_scores)
auprc_shuf = average_precision_score(labels_shuf, pos_scores+neg_scores)
auroc_shuf = roc_auc_score(labels_shuf, pos_scores+neg_scores)
acc_thresh0 = sum([x==int(y>args.thresh) for x,y in zip(labels, pos_scores+neg_scores)])/len(labels)
acc_thresh0_shuf = sum([x==int(y>args.thresh) for x,y in zip(labels_shuf, pos_scores+neg_scores)])/len(labels)

print("Metric\tValue\tRandomised")
print("Accuracy_at_threshold_{}\t{:.4f}\t{:.4f}".format(args.thresh, acc_thresh0, acc_thresh0_shuf))
print("AUROC\t{:.4f}\t{:.4f}".format(auroc, auroc_shuf))
print("AUPRC\t{:.4f}\t{:.4f}".format(auprc, auprc_shuf))
