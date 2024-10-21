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
    transforms: Optional[Dict[str, Any]] = None,
    prefetch_factor: Optional[int] = None,
    prefix: str = "",
    suffix: str = "",
    copy: bool = False,
) -> Optional[xr.Dataset]:
    """Compute attributions for SeqData sequences using a PyTorch nn.Module.

    This function wraps the `attribute` function from the `seqexplainer` package
    to compute attributions for a model and SeqData combination. The attributions
    are stored in the `sdata` object as a new variable. The attributions are
    computed in batches to avoid memory issues.

    This function will only work for targets that are 1D tensors. Support for
    multi-dimensional or multimodal targets will be added in the future releases.

    This function is mainly used to simplify the process of computing attributions
    for a model and SeqData combination. If you are familiar with the XArray
    package, you may benefit from using the `attribute` function directly as it 
    allows for more flexibility in defining references.

    Parameters
    ----------
    model : nn.Module
        PyTorch nn.Module to compute attributions for.
    sdata : xr.Dataset
        SeqData to compute attributions for.
    seq_var : str, optional
        Name of the sequence variable in `sdata` that should be used as input, 
        by default "ohe_seq".
    method : str, optional
        Attribution method to use, by default "InputXGradient". Currently supported
        methods are "NaiveISM", "InputXGradient", "DeepLift", "GradientShap", and 
        "DeepLiftShap". Use "DeepLiftShap" with caution, as it has known issues with
        certain operations (see https://github.com/pytorch/captum/issues/1085)
        and can behave in unexpected ways with certain network architectures. Future 
        releases will incorpote a DeepLiftShap attribution method similar to the bpnet-lite
        repository version (https://github.com/jmschrei/bpnet-lite/blob/master/bpnetlite/attributions.py)
    reference_type : Optional[str], optional
        Reference type to use. Only applicable to attribution methods that use
        references, by default None. Currently supported reference types are
        "zeros", "random", "shuffle", "dinuc_shuffle", "gc", and "profile".
    target : int, optional
        Index of the targets to compute attributions for, by default this is the first
        target, 0.
    batch_size : Optional[int], optional
        Batch size to use. If None, settings.batch_size is used, by default None.
    device : Optional[str], optional
        Device to use. If None, settings.gpus will be used to infer the device, by default None.
    num_workers : Optional[int], optional
        Number of workers to use, by default None.
    prefetch_factor : Optional[int], optional
        Number of samples to prefetch into a buffer to speed up dataloading. 
        If None, uses settings.dl_prefetch_factor.
    transforms : Optional[Dict[str, Any]], optional
        Dictionary of functional transforms to apply to the input sequence. If None, no
        transforms are applied.
    prefix : str, optional
        Prefix to add to the attribution variable nam in the SeqData, by default "".
    suffix : str, optional
        Suffix to add to the attribution variable name in the SeqData, by default "".
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
