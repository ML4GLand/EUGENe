import numpy as np
from tqdm.auto import tqdm
from seqexplainer import attribute
from .._settings import settings
from seqdata import get_torch_dataloader
import xarray as xr
import torch.nn as nn
from typing import Optional, Dict, Any


def attribute_sdata(
    model: nn.Module,
    sdata: xr.Dataset,
    seq_var: str = "ohe_seq",
    method: str = "InputXGradient",
    reference_type: Optional[str] = None,
    target: int = 0,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    num_workers: Optional[int] = None,
    prefetch_factor: Optional[int] = None,
    transforms: Optional[Dict[str, Any]] = None,
    prefix: str = "",
    suffix: str = "",
    copy: bool = False,
) -> Optional[xr.Dataset]:
    """Compute attributions for model and SeqData combination.

    This function wraps the `attribute` function from the `seqexplainer` package
    to compute attributions for a model and SeqData combination. The attributions
    are stored in the `sdata` object as a new variable. The attributions are
    computed in batches to avoid memory issues.

    Parameters
    ----------
    model : nn.Module
        Model to compute attributions for.
    sdata : xr.Dataset
        SeqData to compute attributions for.
    seq_var : str, optional
        Name of the sequence variable in `sdata`, by default "ohe_seq".
    method : str, optional
        Attribution method to use, by default "InputXGradient".
    reference_type : Optional[str], optional
        Reference type to use, by default None.
    target : int, optional
        Target class to compute attributions for, by default 0.
    batch_size : Optional[int], optional
        Batch size to use, by default None.
    device : Optional[str], optional
        Device to use, by default None.
    num_workers : Optional[int], optional
        Number of workers to use, by default None.
    prefetch_factor : Optional[int], optional
        Prefetch factor to use, by default None.
    transforms : Optional[Dict[str, Any]], optional
        Additional transforms to apply to the data, by default None.
    prefix : str, optional
        Prefix to add to the attribution variable name, by default "".
    suffix : str, optional
        Suffix to add to the attribution variable name, by default "".
    copy : bool, optional
        Whether to copy the data before adding the attribution variable, by default False.

    Returns
    -------
    Optional[xr.Dataset]
        The `sdata` object with the attribution variable added if `copy` is False,
    """
    # Copy the data if requested
    sdata = sdata.copy() if copy else sdata

    # Configure the device, batch size, and number of workers
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    prefetch_factor = prefetch_factor if prefetch_factor is not None else None

    # Create the dataloader
    dl = get_torch_dataloader(
        sdata,
        sample_dims=["_sequence"],
        variables=[seq_var],
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        transforms=transforms,
        shuffle=False,
        drop_last=False,
    )

    # Compute the attributions
    attrs = []
    for _, batch in tqdm(
        enumerate(dl),
        total=len(dl),
        desc=f"Computing saliency on batches of size {batch_size}",
    ):
        attr = attribute(
            model=model,
            inputs=batch[seq_var],
            method=method,
            references=reference_type,
            target=target,
            batch_size=batch_size,
            device=device,
            verbose=False,
        )
        attrs.append(attr)

    # Store the attributions
    attrs = np.concatenate(attrs)
    sdata[f"{prefix}{method}_attrs{suffix}"] = xr.DataArray(
        attrs, dims=["_sequence", "_ohe", "length"]
    )
    return sdata if copy else None
