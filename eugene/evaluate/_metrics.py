import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from scipy import stats


def median_calc(y_true, y_score):
    """
    Calculate the median of the scores in y_score for the positive class labels (1) in y_true.

    Parameters
    ----------
    y_true : array-like
        The true binary labels. Assumed to be 1 for positive and 0 for negative.
    y_score : array-like
        The scores predicted by a model
    
    Returns
    -------
    median : float
        The median of the scores in y_score for the positive class labels in y_true.
    """
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    nan_mask = ~np.isnan(y_score)
    y_score = y_score[nan_mask]
    y_true = y_true[nan_mask]
    indeces_1 = np.where(y_true == 1)[0]
    return np.median(y_score[indeces_1])


def auc_calc(y_true, y_score):
    """
    Calculate the area under the curve for a binary y_true against scores in y_score.

    Parameters
    ----------
    y_true : array-like
        The true binary labels. Assumed to be 1 for positive and 0 for negative.
    y_score : array-like
        The scores predicted by a model
    """
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    nan_mask = ~np.isnan(y_score)
    y_true = y_true[nan_mask]
    y_score = y_score[nan_mask]
    return roc_auc_score(y_true, y_score)


def escore(y_true, y_score):
    """
    Calculate the E-score for a binary y_true against scores in y_score.
    The E-score is 

    Parameters
    ----------
    y_true : array-like
        The true binary labels. Assumed to be 1 for positive and 0 for negative.
    y_score : array-like
        The scores predicted by a model
    """
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    nan_mask = ~np.isnan(y_score)
    y_score = y_score[nan_mask]
    y_true = y_true[nan_mask]
    l_0 = np.where(y_true == 0)[0]
    l_1 = np.where(y_true == 1)[0]
    y_0 = y_score[l_0]
    y_1 = y_score[l_1]
    indeces_y_0, indeces_y_1 = np.argsort(y_0)[::-1], np.argsort(y_1)[::-1]
    sorted_y_0, sorted_y_1 = np.sort(y_0)[::-1], np.sort(y_1)[::-1]
    indeces_y_0_top = indeces_y_0[: int(len(sorted_y_0) / 2)]
    indeces_y_1_top = indeces_y_1[: int(len(sorted_y_1) / 2)]
    sorted_y_0_top = sorted_y_0[: int(len(sorted_y_0) / 2)]
    sorted_y_1_top = sorted_y_1[: int(len(sorted_y_1) / 2)]
    l_0_top = l_0[indeces_y_0_top]
    l_1_top = l_1[indeces_y_1_top]
    l_top = np.concatenate([l_0_top, l_1_top])
    return auc_calc(y_true[l_top], y_score[l_top])


def evaluate_model(y_test, pred, task, verbose=True):
    if task == 'regression': #isinstance(pl_model.criterion, torch.nn.modules.loss.MSELoss):
        mse = calculate_mse(y_test, pred)
        pearsonr = calculate_pearsonr(y_test, pred)
        spearmanr = calculate_spearmanr(y_test, pred)
        if verbose:
            print("Test MSE       : %.4f +/- %.4f"%(np.nanmean(mse), np.nanstd(mse)))
            print("Test Pearson r : %.4f +/- %.4f"%(np.nanmean(pearsonr), np.nanstd(pearsonr)))
            print("Test Spearman r: %.4f +/- %.4f"%(np.nanmean(spearmanr), np.nanstd(spearmanr)))
        return mse, pearsonr, spearmanr

    else: 
        auroc = calculate_auroc(y_test, pred) 
        aupr = calculate_aupr(y_test, pred) 
        if verbose:
            print("Test AUROC: %.4f +/- %.4f"%(np.nanmean(auroc), np.nanstd(auroc)))
            print("Test AUPR : %.4f +/- %.4f"%(np.nanmean(aupr), np.nanstd(aupr)))
        return auroc, aupr


def calculate_auroc(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append( roc_auc_score(y_true[:,class_index], y_score[:,class_index]) )    
    return np.array(vals)

def calculate_aupr(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append( average_precision_score(y_true[:,class_index], y_score[:,class_index]) )    
    return np.array(vals)

def calculate_mse(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append( mean_squared_error(y_true[:,class_index], y_score[:,class_index]) )    
    return np.array(vals)

def calculate_pearsonr(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append( stats.pearsonr(y_true[:,class_index], y_score[:,class_index])[0] )    
    return np.array(vals)
    
def calculate_spearmanr(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append( stats.spearmanr(y_true[:,class_index], y_score[:,class_index])[0] )    
    return np.array(vals)