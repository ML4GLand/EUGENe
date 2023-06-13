import os
import numpy as np
from tqdm.auto import tqdm
from seqexplainer import get_layer_outputs
from seqexplainer import get_activators_max_seqlets, get_activators_n_seqlets, get_pfms
from seqdata import get_torch_dataloader
import xarray as xr
from .._settings import settings
from motifdata import from_kernel
from motifdata._transform import pfms_to_ppms
from motifdata import write_meme
from eugene.utils import make_dirs


def generate_pfms_sdata(
    model,
    sdata,
    seq_key,
    layer_name,
    kernel_size=None,
    activations=None,
    seqs=None,
    num_seqlets=100,
    padding=0,
    activation_threshold=None,
    num_filters=None,
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

    if activations is None:
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

        # Compute the acivations for each sequence
        layer_outs = []
        all_seqs = []
        for _, batch in tqdm(
            enumerate(dl),
            total=len(dl),
            desc=f"Getting activations on batches of size {batch_size}",
        ):
            batch_seqs = batch[seq_key]
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
    sdata,
    filters_key: str,
    filter_inds=None,
    axis_order=("_num_kernels", "_ohe", "_kernel_size"),
    output_dir: str = None,
    filename="filters.meme",
    alphabet="ACGT",
    bg={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
):
    # Make sure the output directory exists
    output_dir = output_dir if output_dir is not None else settings.output_dir
    make_dirs(output_dir)
    outfile = os.path.join(output_dir, filename)

    # Grab the filters if they are there
    try:
        pfms = sdata[filters_key].transpose(*axis_order).to_numpy()
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
