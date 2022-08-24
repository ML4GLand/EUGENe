import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr, kendalltau
from ._utils import _create_matplotlib_axes, _label_plot, _plot_seaborn

metric_dict = {
    "r2": r2_score,
    "mse": mean_squared_error,
    "spearman": spearmanr,
    "pearson": pearsonr,
    "kendall": kendalltau,
}

default_rc_context = {
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _model_performances_across_groups(
    sdataframe,
    target,
    prediction_labels=None,
    prediction_groups=None,
    groupby=None,
    metrics="r2",
):
    if isinstance(metrics, str):
        metrics = [metrics]
    prediction_labels = (
        sdataframe.columns[sdataframe.columns.str.contains("predictions")]
        if prediction_labels is None
        else prediction_labels
    )
    conc = pd.DataFrame()
    for group, data in sdataframe.groupby(groupby):
        predicts = data[prediction_labels]
        true = data[target]
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
        scores[groupby] = group
        if prediction_groups is not None:
            scores["prediction_groups"] = prediction_groups
        conc = pd.concat([conc, scores])
    return conc


def _model_performances(
    sdataframe, target, prediction_labels=None, prediction_groups=None, metrics="r2"
):
    if isinstance(metrics, str):
        metrics = [metrics]
    true = sdataframe[target]
    prediction_labels = (
        sdataframe.columns[sdataframe.columns.str.contains("predictions")]
        if prediction_labels is None
        else prediction_labels
    )
    predicts = sdataframe[prediction_labels]
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
    if prediction_groups is not None:
        scores["prediction_groups"] = prediction_groups
    return scores


def performance_summary(
    sdata,
    target,
    prediction_labels=None,
    prediction_groups=None,
    groupby=None,
    add_swarm=False,
    size=5,
    metrics="r2",
    orient="v",
    rc_context=default_rc_context,
    return_axes=False,
    **kwargs
):
    sdataframe = sdata.seqs_annot
    if groupby is None:
        scores = _model_performances(
            sdataframe, target, prediction_labels, prediction_groups, metrics
        )
    else:
        scores = _model_performances_across_groups(
            sdataframe, target, prediction_labels, prediction_groups, groupby, metrics
        )
    print(scores.columns)
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
    if return_axes:
        return ax
    else:
        return scores
