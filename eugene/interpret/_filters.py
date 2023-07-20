import os
from typing import Union, Optional, List, Dict, Any, Literal, Tuple

import numpy as np
import xarray as xr
import torch.nn as nn
from tqdm.auto import tqdm

from seqexplainer import get_layer_outputs
from seqexplainer import get_activators_max_seqlets, get_activators_n_seqlets, get_pfms
from seqdata import get_torch_dataloader
from motifdata import from_kernel
from motifdata._transform import pfms_to_ppms
from motifdata import write_meme
from eugene.utils import make_dirs
from .._settings import settings


def generate_pfms_sdata(
    model: nn.Module,
    sdata: xr.Dataset,
    seq_var: str,
    layer_name: str,
    kernel_size: Optional[int] = None,
    activations: Optional[np.ndarray] = None,
    seqs: Optional[np.ndarray] = None,
    num_seqlets: int = 100,
    padding: int = 0,
    activation_threshold: Optional[float] = None,
    num_filters: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    num_workers: Optional[int] = None,
    prefetch_factor: Optional[int] = None,
    transforms: Optional[Dict[str, Any]] = None,
    prefix: str = "",
    suffix: str = "",
    copy: bool = False,
) -> Optional[xr.Dataset]:
    """Generate position frequency matrices (PFMs) for a given layer in a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to generate PFMs for.
    sdata : xr.Dataset
        The dataset to use for generating PFMs.
    seq_var : str
        The name of the sequence variable in the dataset.
    layer_name : str
        The name of the layer to generate PFMs for.
    kernel_size : int, optional
        The size of the kernel to use for generating PFMs. If not specified, the kernel size will be inferred from the layer.
    activations : torch.Tensor, optional
        The activations to use for generating PFMs. If not specified, the activations will be computed using the dataset and layer.
    seqs : List[str], optional
        The sequences to use for generating PFMs. If not specified, the sequences will be inferred from the dataset.
    num_seqlets : int, optional
        The number of sequencelets to use for generating PFMs.
    padding : int, optional
        The amount of padding to use when generating sequencelets.
    activation_threshold : float, optional
        The threshold to use for selecting sequencelets based on their activation values.
    num_filters : int, optional
        The number of filters to use for generating PFMs. If not specified, all filters will be used.
    batch_size : int, optional
        The batch size to use when generating PFMs.
    device : str, optional
        The device to use for generating PFMs.
    num_workers : int, optional
        The number of workers to use for generating PFMs.
    prefetch_factor : int, optional
        The prefetch factor to use when generating PFMs.
    transforms : Dict[str, Any], optional
        The transforms to apply to the dataset.
    prefix : str, optional
        The prefix to use for the output file.
    suffix : str, optional
        The suffix to use for the output file.
    copy : bool, optional
        Whether to copy the dataset before generating PFMs.
    
    Returns
    -------
    pfms : np.ndarray
        The position frequency matrices.
    """
    # Copy the data if requested
    sdata = sdata.copy() if copy else sdata

    # Configure the device, batch size, and number of workers
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    prefetch_factor = prefetch_factor if prefetch_factor is not None else None

    if activations is None:
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

        # Compute the acivations for each sequence
        layer_outs = []
        all_seqs = []
        for _, batch in tqdm(
            enumerate(dl),
            total=len(dl),
            desc=f"Getting activations on batches of size {batch_size}",
        ):
            batch_seqs = batch[seq_var]
            outs = get_layer_outputs(
                model=model,
                inputs=batch_seqs,
                layer_name=layer_name,
                batch_size=batch_size,
                device=device,
                verbose=False,
            )
            layer_outs.append(outs)
            all_seqs.append(batch_seqs.detach().cpu().numpy())
        layer_outs = np.concatenate(layer_outs, axis=0)
        all_seqs = np.concatenate(all_seqs, axis=0)
    else:
        layer_outs = activations
        all_seqs = seqs
        print(
            f"Using provided activations of shape {layer_outs.shape} and sequences of shape {all_seqs.shape}."
        )

    if num_filters is None:
        num_filters = layer_outs.shape[1]
        print(f"Using all {num_filters} filters.")

    # Get the maximal activators
    if activation_threshold is None:
        assert num_seqlets is not None
        activators = get_activators_n_seqlets(
            activations=layer_outs,
            sequences=all_seqs,
            kernel_size=kernel_size,
            padding=padding,
            num_seqlets=num_seqlets,
            num_filters=num_filters,
        )
    else:
        activators = get_activators_max_seqlets(
            layer_outs,
            sequences=all_seqs,
            kernel_size=kernel_size,
            activation_threshold=activation_threshold,
            num_filters=num_filters,
        )

    # Convert the activators to PFMs
    pfms = get_pfms(activators, kernel_size=kernel_size)

    # Store the PFMs in the sdata
    sdata[f"{prefix}{layer_name}_pfms{suffix}"] = xr.DataArray(
        pfms,
        dims=[
            f"_{layer_name}_{num_filters}_filters",
            f"_{layer_name}_{kernel_size}_kernel_size",
            "_ohe",
        ],
    )
    return sdata if copy else None


def filters_to_meme_sdata(
    sdata: xr.Dataset,
    filters_var: str,
    filter_inds: Optional[List[int]] = None,
    axis_order: Tuple[str, str, str] = ("_num_kernels", "_ohe", "_kernel_size"),
    output_dir: Optional[str] = None,
    filename: str = "filters.meme",
    alphabet: str = "ACGT",
    bg: Dict[str, float] = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
) -> None:
    """Convert position frequency matrices (PFMs) to a MEME motif file.

    Parameters
    ----------
    sdata : xr.Dataset
        The dataset containing the PFMs.
    filters_var : str
        The name of the variable containing the PFMs.
    filter_inds : List[int], optional
        The indices of the filters to convert to a MEME file. If not specified, all filters will be converted.
    axis_order : Tuple[str, str, str], optional
        The order of the axes in the PFMs. By default, the axes are assumed to be in the order (num_kernels, ohe, kernel_size).
    output_dir : str, optional
        The directory to write the MEME file to. By default, the MEME file will be written to the output directory specified in the settings.
    filename : str, optional
        The name of the MEME file to write.
    alphabet : str, optional
        The alphabet to use for the MEME file.
    bg : Dict[str, float], optional
        The background frequencies to use for the MEME file.
    
    Returns
    -------
    None
    """
    # Make sure the output directory exists
    output_dir = output_dir if output_dir is not None else settings.output_dir
    make_dirs(output_dir)
    outfile = os.path.join(output_dir, filename)

    # Grab the filters if they are there
    try:
        pfms = sdata[filters_var].transpose(*axis_order).to_numpy()
    except KeyError:
        print("No filters found in sdata. Run generate_pfms_sdata first.")

    # Subset down to the filters you want
    if filter_inds is None:
        filter_inds = range(pfms.shape[0])

    #
    alphabet_len = len(alphabet)
    if alphabet_len != pfms.shape[1]:
        raise ValueError(
            f"Alphabet length ({alphabet_len}) does not match second dimension of pfms: ({pfms.shape[1]})."
        )

    # Convert pfms to ppms and to a motif set
    ppms = pfms_to_ppms(pfms, pseudocount=0)
    motif_set = from_kernel(kernel=ppms, alphabet=alphabet, bg=bg)

    # Write the motif set to a meme file
    write_meme(motif_set=motif_set, filename=outfile)