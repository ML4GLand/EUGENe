from os import PathLike
from typing import Union
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from scipy.stats import spearmanr, pearsonr, kendalltau
from ._utils import _plot_seaborn, _save_fig
from .. import settings


metric_dict = {
    "r2": r2_score,
    "mse": mean_squared_error,
    "spearman": spearmanr,
    "pearson": pearsonr,
    "kendall": kendalltau,
    "accuracy": accuracy_score,
    "roc_auc": roc_auc_score,
    "average_precision": average_precision_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score
}


def _model_performances_across_groups(
    sdataframe: pd.DataFrame,
    target_key: str,
    prediction_keys: list = None,
    prediction_groups: list = None,
    groupby: str = None,
    metrics: str = "r2",
    clf_thresh: float = 0,
    **kwargs
):
    if isinstance(metrics, str):
        metrics = [metrics]
    prediction_keys = (
        sdataframe.columns[sdataframe.columns.str.contains("predictions")]
        if prediction_keys is None
        else prediction_keys 
    )
    conc = pd.DataFrame()
    for group, data in sdataframe.groupby(groupby):
        predicts = data[prediction_keys]
        bin_predicts = (predicts >= clf_thresh).astype(int)
        true = data[target_key]
        scores = pd.DataFrame()
        for metric in metrics:
            func = metric_dict[metric]
            if metric in ["r2", "mse"]:
                scores = pd.concat(
                    [scores, predicts.apply(lambda x: func(true, x), axis=0).to_frame(name=metric)],
                    axis=1,
                )
            elif metric in ["spearman", "pearson", "kendall"]:
                scores = pd.concat(
                    [scores, predicts.apply(lambda x: func(true, x)[0], axis=0).to_frame(name=metric)],
                    axis=1,
                )
            elif metric in ["accuracy", "precision", "recall"]:
                scores = pd.concat(
                    [scores, bin_predicts.apply(lambda x: func(true, x), axis=0).to_frame(name=metric)],
                    axis=1,
                )
            elif metric in ["roc_auc", "average_precision"]:
                scores = pd.concat(
                    [scores, predicts.apply(lambda x: func(true, x), axis=0).to_frame(name=metric)],
                    axis=1,
                )
        scores[groupby] = group
        if prediction_groups is not None:
            scores["prediction_groups"] = prediction_groups
        conc = pd.concat([conc, scores])
    return conc


def _model_performances(
    sdataframe: pd.DataFrame, 
    target_key: str,
    prediction_keys: list = None, 
    prediction_groups: list = None, 
    metrics: str = "r2", 
    clf_thresh: float = 0
):
    if isinstance(metrics, str):
        metrics = [metrics]
    true = sdataframe[target_key]
    prediction_keys = (
        sdataframe.columns[sdataframe.columns.str.contains("predictions")]
        if prediction_keys is None
        else prediction_keys 
    )
    predicts = sdataframe[prediction_keys]
    bin_predicts = (predicts >= clf_thresh).astype(int)
    scores = pd.DataFrame()
    for metric in metrics:
        func = metric_dict[metric]
        if metric in ["r2", "mse"]:
            scores = pd.concat(
                [scores, predicts.apply(lambda x: func(true, x), axis=0).to_frame(name=metric)],
                axis=1,
            )
        elif metric in ["spearman", "pearson", "kendall"]:
            scores = pd.concat([scores, predicts.apply(lambda x: func(true, x)[0], axis=0).to_frame(name=metric)],
                axis=1,
            )
        elif metric in ["accuracy", "precision", "recall", "f1"]:
            scores = pd.concat([scores, bin_predicts.apply(lambda x: func(true, x), axis=0).to_frame(name=metric)],
                axis=1,
            )
        elif metric in ["roc_auc", "average_precision"]:
            scores = pd.concat(
                [scores, predicts.apply(lambda x: func(true, x), axis=0).to_frame(name=metric)],
                axis=1,
            )
    if prediction_groups is not None:
        scores["prediction_groups"] = prediction_groups
    return scores


def performance_summary(
    sdata,
    target_key: str,
    prediction_keys: list = None,
    prediction_groups: list = None,
    groupby: str = None,
    add_swarm: bool = False,
    size: int = 5,
    metrics: Union[str, list] = "r2",
    orient: str = "v",
    rc_context=settings.rc_context,
    return_axes: bool = False,
    save: PathLike = None,
    **kwargs
):
    sdataframe = sdata.seqs_annot
    if groupby is None:
        scores = _model_performances(
            sdataframe, 
            target_key, 
            prediction_keys, 
            prediction_groups, 
            metrics
        )
    else:
        scores = _model_performances_across_groups(
            sdataframe, 
            target_key, 
            prediction_keys, 
            prediction_groups, 
            groupby, 
            metrics
        )
    with plt.rc_context(rc_context):
        ax = _plot_seaborn(
            scores,
            keys=metrics,
            func=sns.boxplot,
            groupby="prediction_groups" if groupby is None else groupby,
            orient=orient,
            **kwargs
        )
        if add_swarm:
            _plot_seaborn(
                scores,
                keys=metrics,
                func=sns.swarmplot,
                groupby="prediction_groups" if groupby is None else groupby,
                orient=orient,
                size=size,
                edgecolor="black",
                linewidth=2,
                ax=ax,
                **kwargs
            )
    if save is not None:
        _save_fig(save)
    if return_axes:
        return ax
    else:
        return scores
