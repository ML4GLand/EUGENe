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
    layer_name: str,
    kernel_size: int,
    padding: int = 0,
    num_filters: Optional[int] = None,
    seq_var: Optional[str] = "ohe_seq",
    activations: Optional[np.ndarray] = None,
    seqs: Optional[np.ndarray] = None,
    activation_threshold: Optional[float] = None,
    num_seqlets: int = 100,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    prefetch_factor: Optional[int] = None,
    transforms: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    prefix: str = "",
    suffix: str = "",
    copy: bool = False,
) -> Optional[xr.Dataset]:
    """Generate position frequency matrices (PFMs) from SeqData using a a given convolutional layer of a PyTorch nn.Module. 

    This function wraps the `get_activators_max_seqlets` and `get_pfms` functions from the `seqexplainer` package
    to generate PFMs for a given layer in a PyTorch nn.Module. The PFMs are stored in the `sdata` object as a new variable.

    You can also bypass the layer activation computation by providing the activations and sequences directly. This can be useful
    if the outputs you are interested in don't come directly from a named layer in the model. A good example of this is if you use
    functional ReLU in your model (which I don't recommend). In this case, you can use the `get_layer_outputs` function from the
    `seqexplainer` package to get the activations for a given layer then pass them throug functional ReLU yourself. You can then
    pass the activations and sequences to this function to generate the PFMs.

    >>> layer = models.get_layer(model, layer_name)
    >>> activations = F.relu(layer(seqs))

    Be careful with your padding here! The padding strategy used for the layer you are generating PFMs for will affect the
    positions of the activations relative to the input sequences. We need to crop from these input seqlets in positions
    that correspond to high acivations, and if the indeces are not aligned, you will get the wrong sequences. This is corrected for in 
    the `get_activators_*` functions in the `seqexplainer` package if you pass in the correct padding value here.


    Parameters
    ----------
    model : torch.nn.Module
        The model to generate PFMs for.
    sdata : xr.Dataset
        The SeqData to use for generating PFMs.
    layer_name : str
        The name of the layer to generate PFMs for, required regardless of whether you provide the activations and sequences directly 
        for naming the output variable in the SeqData object.
    kernel_size : int, optional
        The size of the kernels in the layer. This should be the same kernel size of the layer you are generating PFMs for.
    padding : int, optional
        The amount of padding to use when generating seqlets to be applied to the input sequences. 
        This should be the same padding value used in the layer you are generating PFMs for.
    num_filters : int, optional
        The number of filters to use for generating PFMs. Can be used if you only want to calculate PFMs for the first `num_filters` filters.
    seq_var : str
        The name of the sequence variable in the dataset for generating PFMs. By default, this is "ohe_seq".
    activations : torch.Tensor, optional
        The activations to use for generating PFMs. If not specified, the activations will be computed using the dataset and layer.
    seqs : List[str], optional
        The sequences to use for generating PFMs. If specified with activations, bypasses the activation computation.
    activation_threshold : float, optional
        Percentage of the maximum activation to use as a threshold for generating PFMs. All seqlets with activations above this threshold
        will be used for generating PFMs. If specified, num_seqlets will be ignored.
    num_seqlets : int, optional
        The number of seqlets to use for generating PFMs. Only used if `activation_threshold` is not specified.
    batch_size : int, optional
        The batch size for the dataloader created from the the SeqData. If not specified, settings.batch_size will be used.
    num_workers : int, optional
        The number of workers to use for the dataloader created from the the SeqData. If not specified, settings.dl_num_workers will be used.
    prefetch_factor : int, optional
        The prefetch factor to use for the dataloader created from the the SeqData. If not specified, settings.dl_prefetch_factor will be used.
    transforms : Dict[str, Any], optional
        The transforms to use for the dataloader created from the the SeqData. If not specified, no transforms will be used.
    device : str, optional
        The device to use for generating PFMs (if activations are not provided). 
        If not specified, settings.gpus will be used to determine whether to use "cuda" or "cpu".
    prefix : str, optional
        The prefix to use in the SeqData variable name for the PFMs.
    suffix : str, optional
        The suffix to use in the SeqData variable name for the PFMs.
    copy : bool, optional
        Whether to copy the dataset before generating PFMs.
    
    Returns
    -------
    Optional[xr.Dataset]
        The `sdata` object with the PFMs added if `copy` is False.
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
    axis_order: Tuple[str, str, str] = ("_num_kernels", "_ohe", "_kernel_size"),
    output_dir: Optional[str] = None,
    filename: str = "filters.meme",
    alphabet: str = "ACGT",
    bg: Dict[str, float] = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
) -> None:
    """Save position frequency matrices (PFMs) to a MEME motif file.

    This function wraps the `pfms_to_ppms` and `from_kernel` functions from the `motifdata` package
    by applying them to a SeqData object. The PFMs are stored in the `sdata` object as a new variable
    after running the `generate_pfms_sdata` function.

    Parameters
    ----------
    sdata : xr.Dataset
        The dataset containing the PFMs.
    filters_var : str
        The name of the variable containing the PFMs. This variable should be in the dataset after running the `generate_pfms_sdata` function.
    axis_order : Tuple[str, str, str], optional
        The order of the axes in the PFMs. By default, the axes are assumed to be in the order (num_kernels, ohe, kernel_size). This
        will likely need to be specified if you generated these using the `generate_pfms_sdata` function.
    output_dir : str, optional
        The directory to write the MEME file to. By default, the MEME file will be written to the output directory specified in the settings.
    filename : str, optional
        The name of the MEME file to write. By default, the MEME file will be named "filters.meme".
    alphabet : str, optional
        The alphabet to use for the MEME file. By default, the alphabet is "ACGT".
    bg : Dict[str, float], optional
        The background frequencies to use for the MEME file. By default, the background frequencies are uniform.
    
    Returns
    -------
    None. Writes a MEME file to the specified filename in the specified output directory.
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

    # Make sure the alphabet length matches the second dimension of the pfms
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
