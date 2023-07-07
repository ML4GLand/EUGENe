import os
import numpy as np
import pandas as pd
from typing import Union, List, Optional
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from eugene import settings
import xarray as xr
import seqdata as sd
from ._utils import PredictionWriter


def predictions(
    model: LightningModule,
    dataloader: DataLoader,
    gpus: Optional[int] = None,
    out_dir: Optional[os.PathLike] = None,
    name: Optional[str] = None,
    version: Optional[str] = "",
    file_label: Optional[str] = "",
):
    """Predictions from a model and dataloader.
    
    Simple wrapper around pytorch_lightning.Trainer.predict() that returns a pandas dataframe.
    Makes use of a custom callback, PredictionWriter, to write predictions to disk if the out_dir
    argument is provided.

    Parameters
    ----------
    model : LightningModule
        Model to predict with.
    dataloader : DataLoader
        Dataloader to predict with.
    gpus : int, optional
        Number of GPUs to use. If None, uses settings.gpus.
    out_dir : os.PathLike, optional
        Directory to write predictions to. If None, does not write predictions to disk.
    name : str, optional
        Name of the model. If None, uses model.model_name.
    version : str, optional
        Version of the model. If None, uses "".
    file_label : str, optional
        Label to add to the file name. If None, uses "".
    
    Returns
    -------
    preds : pd.DataFrame
        Predictions from the model and dataloader.
    """
    gpus = gpus if gpus is not None else settings.gpus
    model_name = model.model_name
    name = name if name is not None else model_name
    if out_dir is not None:
        out_dir = os.path.join(out_dir, name, version)
        predictor = Trainer(
            logger=False,
            callbacks=PredictionWriter(output_dir=out_dir, file_label=file_label),
            devices=gpus,
        )
    else:
        predictor = Trainer(logger=False, devices=gpus)
    ps = np.concatenate(predictor.predict(model, dataloader), axis=0)
    num_outs = model.output_dim
    preds = pd.DataFrame(data=ps[:, 0:num_outs])
    preds.columns = [f"predictions_{i}" for i in range(num_outs)]
    return preds


def predictions_sequence_module(
    model: LightningModule,
    sdata: Optional[xr.Dataset] = None,
    seq_var: str = "ohe_seq",
    target_vars: Optional[Union[str, List[str]]] = None,
    gpus: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    transforms: Optional[dict] = None,
    prefetch_factor: Optional[int] = None,
    store_only: bool = False,
    in_memory: bool = False,
    out_dir: Optional[os.PathLike] = None,
    name: Optional[str] = None,
    version: str = "",
    file_label: str = "",
    prefix: str = "",
    suffix: str = "",
    copy: bool = False,
) -> xr.Dataset:
    """Predictions for a SequenceModule model and SeqData

    This is a wrapper around the predictions function that takes a SeqData object and
    builds a dataloader from it. It also adds the predictions to the SeqData object.

    Parameters
    ----------
    model : LightningModule
        Model to predict with.
    sdata : xr.Dataset, optional
        SeqData object to predict with. If None, uses the sdata in settings.
    seq_var : str, optional
        Key in sdata to use as the sequence. If None, uses "ohe_seq".
    target_vars : str or list of str, optional
        Key(s) in sdata to use as the target. If None, uses None.
    gpus : int, optional
        Number of GPUs to use. If None, uses settings.gpus.
    batch_size : int, optional
        Batch size to use. If None, uses settings.batch_size.
    num_workers : int, optional
        Number of workers to use. If None, uses settings.dl_num_workers.
    transforms : dict, optional 

    Returns
    -------
    sdata : xr.Dataset
        SeqData object with predictions added if copy=True. If copy=False, returns None.
    """
    sdata = sdata.copy() if copy else sdata
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    target_vars = [target_vars] if type(target_vars) == str else target_vars
    if not store_only:
        out_dir = out_dir if out_dir is not None else settings.output_dir
    if target_vars is not None:
        if isinstance(target_vars, str):
            target_vars = [target_vars]
        if len(target_vars) == 1:
            sdata["target"] = sdata[target_vars[0]]
        else:
            sdata["target"] = xr.concat(
                [sdata[target_var] for target_var in target_vars], dim="_targets"
            ).transpose("_sequence", "_targets")
    if in_memory:
        print(f"Loading {seq_var} and {target_vars} into memory")
        sdata[seq_var].load()
        sdata["target"].load()
    dataloader = sd.get_torch_dataloader(
        sdata,
        sample_dims=["_sequence"],
        variables=[seq_var, "target"],
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        transforms=transforms,
        shuffle=False,
    )
    preds = predictions(
        model=model,
        dataloader=dataloader,
        gpus=gpus,
        out_dir=out_dir,
        name=name,
        version=version,
        file_label=file_label,
    )
    pred_cols = preds.columns
    for i, target_var in enumerate(target_vars):
        # print(f"Adding {prefix}{target_var}_predictions{suffix} to sdata")
        sdata[f"{prefix}{target_var}_predictions{suffix}"] = xr.DataArray(
            preds[pred_cols[i]].values, dims=["_sequence"]
        )
    return sdata if copy else None


def train_val_predictions(
    model: LightningModule,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    train_var="train_val",
    gpus: Optional[int] = None,
    out_dir: Optional[os.PathLike] = None,
    name: Optional[str] = None,
    version: str = "",
) -> pd.DataFrame:
    """Predictions from a model and train/val dataloaders.

    Same as predictions function, but takes separate train and val dataloaders.

    Parameters
    ----------
    model : LightningModule
        Model to predict with.
    train_dataloader : DataLoader
        Dataloader to predict with.
    val_dataloader : DataLoader
        Dataloader to predict with.
    train_var : str, optional
        Key in sdata to use as the train/val variable. If None, uses "train_val".
    gpus : int, optional
        Number of GPUs to use. If None, uses settings.gpus.
    out_dir : os.PathLike, optional
        Directory to write predictions to. If None, does not write predictions to disk.
    name : str, optional
        Name of the model. If None, uses model.model_name.
    version : str, optional
        Version of the model. If None, uses "".

    Returns
    -------
    preds : pd.DataFrame
        Predictions from the model and dataloader in a pandas dataframe.
    """
    gpus = gpus if gpus is not None else settings.gpus
    model_name = model.model_name
    name = name if name is not None else model_name
    if out_dir is not None:
        out_dir = os.path.join(out_dir, name, version)
        train_predictor = Trainer(
            logger=False,
            callbacks=PredictionWriter(out_dir, file_label="train"),
            devices=gpus,
        )
        val_predictor = Trainer(
            logger=False,
            callbacks=PredictionWriter(out_dir, file_label="val"),
            devices=gpus,
        )
    else:
        train_predictor = Trainer(logger=False, devices=gpus)
        val_predictor = Trainer(logger=False, devices=gpus)

    t = np.concatenate(train_predictor.predict(model, train_dataloader), axis=0)
    v = np.concatenate(val_predictor.predict(model, val_dataloader), axis=0)
    num_outs = model.output_dim
    preds = pd.concat(
        [
            pd.DataFrame(data=t[:, 0:num_outs]),
            pd.DataFrame(data=v[:, 0:num_outs]),
        ],
        axis=0,
    ).reset_index(drop=True)
    preds.columns = [f"predictions_{i}" for i in range(num_outs)]
    preds[train_var] = [True] * len(t) + [False] * len(v)
    return preds


def train_val_predictions_sequence_module(
    model: LightningModule,
    sdata=None,
    seq_var: str = "ohe_seq",
    target_vars: Optional[Union[str, List[str]]] = None,
    train_var="train_val",
    gpus: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    transforms: Optional[dict] = None,
    prefetch_factor: Optional[int] = None,
    store_only: bool = False,
    in_memory: bool = False,
    out_dir: Optional[os.PathLike] = None,
    name: Optional[str] = None,
    version: str = "",
    prefix: str = "",
    suffix: str = "",
    copy: bool = False,
):
    """Predictions for a SequenceModule model and SeqData

    This is a wrapper around the predictions function that takes a SeqData object and
    builds a dataloader from it. It also adds the predictions to the SeqData object.

    Parameters
    ----------
    model : LightningModule
        Model to predict with.
    sdata : xr.Dataset, optional
        SeqData object to predict with. If None, uses the sdata in settings.
    seq_var : str, optional
        Key in sdata to use as the sequence. If None, uses "ohe_seq".
    target_vars : str or list of str, optional
        Key(s) in sdata to use as the target. If None, uses None.
    train_var : str, optional
        Key in sdata to use as the train/val variable. If None, uses "train_val".
    gpus : int, optional
        Number of GPUs to use. If None, uses settings.gpus.
    batch_size : int, optional
        Batch size to use. If None, uses settings.batch_size.
    num_workers : int, optional
        Number of workers to use. If None, uses settings.dl_num_workers.
    transforms : dict, optional 

    Returns
    -------
    sdata : xr.Dataset
        SeqData object with predictions added if copy=True. If copy=False, returns None.
    """
    # Set-up dataloaders
    sdata = sdata.copy() if copy else sdata
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    target_vars = [target_vars] if type(target_vars) == str else target_vars
    if not store_only:
        out_dir = out_dir if out_dir is not None else settings.output_dir
    if target_vars is not None:
        if isinstance(target_vars, str):
            target_vars = [target_vars]
        if len(target_vars) == 1:
            sdata["target"] = sdata[target_vars[0]]
        else:
            sdata["target"] = xr.concat(
                [sdata[target_var] for target_var in target_vars], dim="_targets"
            ).transpose("_sequence", "_targets")
    if in_memory:
        print(f"Loading {seq_var} and {target_vars} into memory")
        sdata[seq_var].load()
        sdata["target"].load()
    sdata[train_var].load()
    train_sdata = sdata.where(sdata[train_var] == 1, drop=True)
    val_sdata = sdata.where(sdata[train_var] == 0, drop=True)
    train_dataloader = sd.get_torch_dataloader(
        train_sdata,
        sample_dims=["_sequence"],
        variables=[seq_var, "target"],
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        transforms=transforms,
        shuffle=False,
        drop_last=False,
    )
    val_dataloader = sd.get_torch_dataloader(
        val_sdata,
        sample_dims=["_sequence"],
        variables=[seq_var, "target"],
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        transforms=transforms,
        shuffle=False,
        drop_last=False,
    )

    # Predict with train_val_predictions function
    preds = train_val_predictions(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        gpus=gpus,
        out_dir=out_dir,
        name=name,
        version=version,
    )

    # Create an empty dataframe the same size as preds
    ordered_preds = pd.DataFrame(index=preds.index, columns=preds.columns).drop(
        columns=["train_val"]
    )

    # Grab the train and val predictions and put them in the right place
    train_preds = preds[preds[train_var] == True].drop(columns=["train_val"])
    train_pred_index = np.where(sdata[train_var].values == 1)[0]
    val_preds = preds[preds[train_var] == False].drop(columns=["train_val"])
    val_pred_index = np.where(sdata[train_var].values == 0)[0]
    ordered_preds.iloc[train_pred_index] = train_preds
    ordered_preds.iloc[val_pred_index] = val_preds

    # Add the predictions to the sdata
    pred_cols = ordered_preds.columns
    for i, target_var in enumerate(target_vars):
        # print(f"Adding {prefix}{target_var}_predictions{suffix} to sdata")
        sdata[f"{prefix}{target_var}_predictions{suffix}"] = xr.DataArray(
            ordered_preds[pred_cols[i]].values, dims=["_sequence"]
        )
    return sdata if copy else None
