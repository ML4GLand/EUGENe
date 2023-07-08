import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from os import PathLike
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from typing import Optional, Sequence, Union

from .. import settings
from ._utils import _create_matplotlib_axes, _save_fig


def _plot_performance_scatter(
    sdata: xr.Dataset,
    target_key: str,
    prediction_key: str,
    metrics: Union[str, Sequence[str]] = ["r2", "mse", "pearsonr", "spearmanr"],
    groupby: Optional[str] = None,
    figsize: tuple = (8, 8),
    save: Optional[PathLike] = None,
    ax: Optional[bool] = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """Plot a scatter plot of the performance of the model on a subset of the sequences.

    Classic predicted vs observed scatterplot that will be annotated with r2, mse and spearman correlation.
    If a groupby key is passed, the scatterplot will be colored according to group.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    target_key : str
        Name of the target_key variable.
    prediction_key : str
        Name of the prediction_key variable.
    metrics : str or list of str
        Metrics to plot. Should be the string name of the metric used in PL
    **kwargs

    Returns
    -------
    

    Note
    ----
    This function uses Matplotlib as opposed to Seaborn.
    """
    target = sdata[target_key].to_numpy()
    prediction = sdata[prediction_key].to_numpy()

    nan_mask = ~np.isnan(target)
    target = target[nan_mask]
    prediction = prediction[nan_mask]

    r2 = r2_score(target, prediction) if "r2" in metrics else None
    mse = mean_squared_error(target, prediction) if "mse" in metrics else None
    pearsr = pearsonr(target, prediction)[0] if "pearsonr" in metrics else None
    spearr = (
        spearmanr(target, prediction).correlation if "spearmanr" in metrics else None
    )
    if "c" in kwargs:
        if kwargs["c"] in sdata.data_vars.keys():
            kwargs["c"] = sdata[kwargs["c"]]
    ax = _create_matplotlib_axes(1, subplot_size=figsize) if ax is None else ax
    if groupby is not None:
        i = 0
        print("Group", "R2", "MSE", "Pearsonr", "Spearmanr")
        seqs_annot = sdata[[groupby, target_key, prediction_key]].to_dataframe()
        for group, data in seqs_annot.groupby(groupby):
            target = data[target_key]
            prediction = data[prediction_key]
            group_r2 = r2_score(target, prediction) if "r2" in metrics else None
            group_mse = mean_squared_error(
                target, prediction if "mse" in metrics else None
            )
            group_pearsr = (
                pearsonr(target, prediction)[0] if "pearsonr" in metrics else None
            )
            group_spearr = (
                spearmanr(target, prediction).correlation
                if "spearmanr" in metrics
                else None
            )
            im = ax.scatter(target, prediction, label=group, color="bgrcm"[i], **kwargs)
            print(group, group_r2, group_mse, group_spearr)
            i += 1
            ax.legend()
    else:
        im = ax.scatter(
            target, prediction, edgecolor="black", linewidth=0.1, s=10, **kwargs
        )
    if "c" in kwargs:
        plt.colorbar(im, location="bottom", label=kwargs["c"].name)
    ax.set_xlabel(target_key)
    ax.set_ylabel(prediction_key)
    ax.text(
        1.02, 0.95, f"$R^2$: {r2:.2f}", transform=plt.gca().transAxes, fontsize=16
    ) if r2 is not None else None
    ax.text(
        1.02, 0.90, f"MSE: {mse:.2f}", transform=plt.gca().transAxes, fontsize=16
    ) if mse is not None else None
    ax.text(
        1.02,
        0.85,
        rf"Spearman $\rho$: {spearr:.2f}",
        transform=plt.gca().transAxes,
        fontsize=16,
    ) if spearr is not None else None
    ax.text(
        1.02,
        0.80,
        rf"Pearson $r$: {pearsr:.2f}",
        transform=plt.gca().transAxes,
        fontsize=16,
    ) if pearsr is not None else None
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, color="black", linestyle="--", zorder=0)
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    if save is not None:
        _save_fig(save)
    return ax


def performance_scatter(
    sdata: xr.Dataset,
    target_vars: Union[str, Sequence[str]],
    prediction_vars: Union[str, Sequence[str]],
    seq_idx: Optional[Union[Sequence[int], np.ndarray]] = None,
    rc_context: dict = settings.rc_context,
    return_axes: bool = False,
    **kwargs,
) -> Optional[plt.Axes]:
    """Plot a scatter plot of the performance of the model on a subset of the sequences.

    Classic predicted vs observed scatterplot that will be annotated with r2, mse and spearman correlation.
    If a groupby key is passed, the scatterplot will be colored according to group.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    target_key : str
        Name of the target_key variable.
    prediction_key : str
        Name of the prediction_key variable.
    seq_idx : list of int
        List of indices of sequences to plot.
    **kwargs

    Returns
    -------
    None
    """
    if seq_idx is not None:
        sdata = sdata[seq_idx]
    if isinstance(target_vars, str) and isinstance(prediction_vars, str):
        target_vars = [target_vars]
        prediction_vars = [prediction_vars]
    if type(target_vars) is list and type(prediction_vars) is list:
        assert len(target_vars) == len(prediction_vars)
    else:
        target_vars = [target_vars]
        prediction_vars = [prediction_vars]
    with plt.rc_context(rc_context):
        for target_key, prediction_key in zip(target_vars, prediction_vars):
            targs = sdata[target_key].values
            nan_mask = xr.DataArray(np.isnan(targs), dims=["_sequence"])
            print(f"Dropping {int(nan_mask.sum().values)} sequences with NaN targets.")
            sdata = sdata.where(~nan_mask, drop=True)
            ax = _plot_performance_scatter(
                sdata, target_key=target_key, prediction_key=prediction_key, **kwargs
            )
    if return_axes:
        return ax
    