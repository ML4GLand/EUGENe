import os
import numpy as np
import pandas as pd
from typing import Union, List
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from eugene import settings
import xarray as xr
import seqdata as sd
from ._utils import PredictionWriter


def predictions(
    model: LightningModule,
    dataloader: DataLoader,
    gpus: int = None,
    out_dir: os.PathLike = None,
    name: str = None,
    version: str = "",
    file_label: str = "",
):
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
    sdata=None,
    seq_key: str = "ohe_seq",
    target_keys: Union[str, List[str]] = None,
    gpus: int = None,
    batch_size: int = None,
    num_workers: int = None,
    transforms: List = None,
    prefetch_factor: int = None,
    store_only: bool = False,
    in_memory: bool = False,
    out_dir: os.PathLike = None,
    name: str = None,
    version: str = "",
    file_label: str = "",
    prefix: str = "",
    suffix: str = "",
    copy: bool = False,
):
    sdata = sdata.copy() if copy else sdata
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    target_keys = [target_keys] if type(target_keys) == str else target_keys
    if not store_only:
        out_dir = out_dir if out_dir is not None else settings.output_dir
    if target_keys is not None:
        if isinstance(target_keys, str):
            target_keys = [target_keys]
        if len(target_keys) == 1:
            sdata["target"] = sdata[target_keys[0]]
        else:
            sdata["target"] = xr.concat(
                [sdata[target_key] for target_key in target_keys], dim="_targets"
            ).transpose("_sequence", "_targets")
    if in_memory:
        print(f"Loading {seq_key} and {target_keys} into memory")
        sdata[seq_key].load()
        sdata["target"].load()
    dataloader = sd.get_torch_dataloader(
        sdata,
        sample_dims=["_sequence"],
        variables=[seq_key, "target"],
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
    for i, target_key in enumerate(target_keys):
        # print(f"Adding {prefix}{target_key}_predictions{suffix} to sdata")
        sdata[f"{prefix}{target_key}_predictions{suffix}"] = xr.DataArray(
            preds[pred_cols[i]].values, dims=["_sequence"]
        )
    return sdata if copy else None


def train_val_predictions(
    model: LightningModule,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    train_key="train_val",
    gpus: int = None,
    out_dir: os.PathLike = None,
    name: str = None,
    version: str = "",
):
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
    preds[train_key] = [True] * len(t) + [False] * len(v)
    return preds


def train_val_predictions_sequence_module(
    model: LightningModule,
    sdata=None,
    seq_key: str = "ohe_seq",
    target_keys: Union[str, List[str]] = None,
    train_key="train_val",
    gpus: int = None,
    batch_size: int = None,
    num_workers: int = None,
    transforms: List = None,
    prefetch_factor: int = None,
    store_only: bool = False,
    in_memory: bool = False,
    out_dir: os.PathLike = None,
    name: str = None,
    version: str = "",
    prefix: str = "",
    suffix: str = "",
    copy: bool = False,
):
    # Set-up dataloaders
    sdata = sdata.copy() if copy else sdata
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    target_keys = [target_keys] if type(target_keys) == str else target_keys
    if not store_only:
        out_dir = out_dir if out_dir is not None else settings.output_dir
    if target_keys is not None:
        if isinstance(target_keys, str):
            target_keys = [target_keys]
        if len(target_keys) == 1:
            sdata["target"] = sdata[target_keys[0]]
        else:
            sdata["target"] = xr.concat(
                [sdata[target_key] for target_key in target_keys], dim="_targets"
            ).transpose("_sequence", "_targets")
    if in_memory:
        print(f"Loading {seq_key} and {target_keys} into memory")
        sdata[seq_key].load()
        sdata["target"].load()
    sdata[train_key].load()
    train_sdata = sdata.where(sdata[train_key] == 1, drop=True)
    val_sdata = sdata.where(sdata[train_key] == 0, drop=True)
    train_dataloader = sd.get_torch_dataloader(
        train_sdata,
        sample_dims=["_sequence"],
        variables=[seq_key, "target"],
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
        variables=[seq_key, "target"],
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
    train_preds = preds[preds[train_key] == True].drop(columns=["train_val"])
    train_pred_index = np.where(sdata[train_key].values == 1)[0]
    val_preds = preds[preds[train_key] == False].drop(columns=["train_val"])
    val_pred_index = np.where(sdata[train_key].values == 0)[0]
    ordered_preds.iloc[train_pred_index] = train_preds
    ordered_preds.iloc[val_pred_index] = val_preds

    # Add the predictions to the sdata
    pred_cols = ordered_preds.columns
    for i, target_key in enumerate(target_keys):
        # print(f"Adding {prefix}{target_key}_predictions{suffix} to sdata")
        sdata[f"{prefix}{target_key}_predictions{suffix}"] = xr.DataArray(
            ordered_preds[pred_cols[i]].values, dims=["_sequence"]
        )
    return sdata if copy else None
