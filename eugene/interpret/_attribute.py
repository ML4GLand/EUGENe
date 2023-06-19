import numpy as np
from tqdm.auto import tqdm
from seqexplainer import attribute
from .._settings import settings
from seqdata import get_torch_dataloader
import xarray as xr


def attribute_sdata(
    model,
    sdata,
    seq_key="ohe_seq",
    method="InputXGradient",
    reference_type=None,
    target=0,
    batch_size=None,
    device=None,
    num_workers=None,
    prefetch_factor=None,
    transforms={},
    prefix="",
    suffix="",
    copy=False,
):
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
        variables=[seq_key],
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
            inputs=batch[seq_key],
            method=method,
            reference_type=reference_type,
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
