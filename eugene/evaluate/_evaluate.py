import numpy as np
from .metrics._multiclass_classification import calculate_auroc, calculate_aupr
from .metrics._regression import calculate_mse, calculate_pearsonr, calculate_spearmanr


def evaluate_model(y_test, pred, task, verbose=True):
    if (
        task == "regression"
    ):  # isinstance(pl_model.criterion, torch.nn.modules.loss.MSELoss):
        mse = calculate_mse(y_test, pred)
        pearsonr = calculate_pearsonr(y_test, pred)
        spearmanr = calculate_spearmanr(y_test, pred)
        if verbose:
            print("Test MSE       : %.4f +/- %.4f" % (np.nanmean(mse), np.nanstd(mse)))
            print(
                "Test Pearson r : %.4f +/- %.4f"
                % (np.nanmean(pearsonr), np.nanstd(pearsonr))
            )
            print(
                "Test Spearman r: %.4f +/- %.4f"
                % (np.nanmean(spearmanr), np.nanstd(spearmanr))
            )
        return mse, pearsonr, spearmanr
    else:
        auroc = calculate_auroc(y_test, pred)
        aupr = calculate_aupr(y_test, pred)
        if verbose:
            print("Test AUROC: %.4f +/- %.4f" % (np.nanmean(auroc), np.nanstd(auroc)))
            print("Test AUPR : %.4f +/- %.4f" % (np.nanmean(aupr), np.nanstd(aupr)))
        return auroc, aupr
