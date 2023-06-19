from os import PathLike
from typing import Union
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
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
    "f1": f1_score,
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
    """
    Calculate model performance for a metric or set of metrics across groups.

    Compares a target column to a set of prediction column in sdataframe
    and calculates the performance of the model for each group in groupby.

    Parameters
    ----------
    sdataframe : pd.DataFrame
        A dataframe containing the target and prediction columns in target_key and prediction_keys respectively.
    target_key : str
        The name of the column in sdataframe containing the target values.
    prediction_keys : list, optional
        A list of the names of the columns in sdataframe containing the prediction values.
        If None, all columns containing "predictions" in their name will be used.
    prediction_groups : list, optional
        A list of the names of the groups for each prediction column.
    groupby : str, optional
        The name of the column in sdataframe to group by. If None, the prediction_groups will be used.
    metrics : str, optional
        The name of the metric to calculate. If None, all metrics will be calculated.
    clf_thresh : float, optional
        The threshold to use for binary classification. Default is 0.
    **kwargs : dict, optional

    Returns
    -------
    pd.DataFrame
        A dataframe containing the calculated metrics for each group.
    """
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
                    [
                        scores,
                        predicts.apply(lambda x: func(true, x), axis=0).to_frame(
                            name=metric
                        ),
                    ],
                    axis=1,
                )
            elif metric in ["spearman", "pearson", "kendall"]:
                scores = pd.concat(
                    [
                        scores,
                        predicts.apply(lambda x: func(true, x)[0], axis=0).to_frame(
                            name=metric
                        ),
                    ],
                    axis=1,
                )
            elif metric in ["accuracy", "precision", "recall"]:
                scores = pd.concat(
                    [
                        scores,
                        bin_predicts.apply(lambda x: func(true, x), axis=0).to_frame(
                            name=metric
                        ),
                    ],
                    axis=1,
                )
            elif metric in ["roc_auc", "average_precision"]:
                scores = pd.concat(
                    [
                        scores,
                        predicts.apply(lambda x: func(true, x), axis=0).to_frame(
                            name=metric
                        ),
                    ],
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
    clf_thresh: float = 0,
):
    """
    Calculate model performance for a metric or set of metrics.

    Uses columns from a passed in dataframe to calcuate a set of metrics.

    Parameters
    ----------
    sdataframe : pd.DataFrame
        A dataframe containing the target and prediction columns in target_key and prediction_keys respectively.
    target_key : str
        The name of the column in sdataframe containing the target values.
    prediction_keys : list, optional
        A list of the names of the columns in sdataframe containing the prediction values.
        If None, all columns containing "predictions" in their name will be used.
    prediction_groups : list, optional
        A list of the names of the groups for each prediction column.
    groupby : str, optional
        The name of the column in sdataframe to group by. If None, the prediction_groups will be used.
    metrics : str, optional
        The name of the metric to calculate. If None, all metrics will be calculated.
    clf_thresh : float, optional
        The threshold to use for binary classification. Default is 0.
    Returns
    -------
    pd.DataFrame
        A dataframe containing the calculated metrics.
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    true = sdataframe[target_key]
    predicts = sdataframe[prediction_keys]
    bin_predicts = (predicts >= clf_thresh).astype(int)
    scores = pd.DataFrame()
    for metric in metrics:
        func = metric_dict[metric]
        if metric in ["r2", "mse"]:
            scores = pd.concat(
                [
                    scores,
                    predicts.apply(lambda x: func(true, x), axis=0).to_frame(
                        name=metric
                    ),
                ],
                axis=1,
            )
        elif metric in ["spearman", "pearson", "kendall"]:
            scores = pd.concat(
                [
                    scores,
                    predicts.apply(lambda x: func(true, x)[0], axis=0).to_frame(
                        name=metric
                    ),
                ],
                axis=1,
            )
        elif metric in ["accuracy", "precision", "recall", "f1"]:
            scores = pd.concat(
                [
                    scores,
                    bin_predicts.apply(lambda x: func(true, x), axis=0).to_frame(
                        name=metric
                    ),
                ],
                axis=1,
            )
        elif metric in ["roc_auc", "average_precision"]:
            scores = pd.concat(
                [
                    scores,
                    predicts.apply(lambda x: func(true, x), axis=0).to_frame(
                        name=metric
                    ),
                ],
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
    """
    Plot a performance summary across model predictions for a passed in metric

    Uses model predictions and targets to calculate a set of metrics and plot them.

    Parameters
    ----------
    sdata : pd.DataFrame
        A dataframe containing the target and prediction columns in target_key and prediction_keys respectively.
    target_key : str
        The name of the column in sdataframe containing the target values.
    prediction_keys : list, optional
        A list of the names of the columns in sdataframe containing the prediction values.
        If None, all columns containing "predictions" in their name will be used.
    prediction_groups : list, optional
        A list of the names of the groups for each prediction column.
    groupby : str, optional
        The name of the column in sdataframe to group by. If None, the prediction_groups will be used.
    add_swarm : bool, optional
        Whether to add a swarmplot to the violinplot. Default is False.
    size : int, optional
        The size of the points to plot if add_swarm is True. Default is 5.
    metrics : str, optional
        The name of the metrics to calculate.
    orient : str, optional
        The orientation of the plot. Default is "v".
    rc_context : dict, optional
        A dictionary of rcParams to pass to matplotlib. Default is settings.rc_context.
    return_axes : bool, optional
        Whether to return the axes object. Default is False.
    save : PathLike, optional
        The path to save the figure to. Default is None.
    **kwargs
        Additional keyword arguments to pass to sns.violinplot.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    prediction_keys = prediction_keys = (
        [k for k in sdata.keys() if "preds" in k]
        if prediction_keys is None
        else prediction_keys
    )
    sdataframe = (
        sdata[["id"] + [target_key] + prediction_keys].to_dataframe().set_index("id")
    )
    if groupby is None:
        scores = _model_performances(
            sdataframe, target_key, prediction_keys, prediction_groups, metrics
        )
    else:
        scores = _model_performances_across_groups(
            sdataframe, target_key, prediction_keys, prediction_groups, groupby, metrics
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
