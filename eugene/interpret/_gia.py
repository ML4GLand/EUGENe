import torch
from tqdm.auto import tqdm
from .._settings import settings
import xarray as xr
import numpy as np
from tqdm.auto import tqdm
from seqexplainer.gia._perturb import tile_pattern_seq
from seqexplainer.gia._complex_perturb import embed_deepstarr_distance_cooperativity
from seqexplainer.gia._gia import deepstarr_motif_distance_cooperativity_gia
import seqpro as sp

from typing import Optional, List, Dict, Any

def feature_implant_seq_sdata(
    model: torch.nn.Module,
    sdata: xr.Dataset,
    seq_id: str,
    feature: np.ndarray,
    seq_var: str = "ohe_seq",
    id_var: str = "id",
    feature_name: str = "feature",
    encoding: str = "onehot",
    store: bool = True,
    device: Optional[str] = None,
):
    """Implant a feature into a single sequence within in a SeqData object and return the model predictions.

    This function wraps the `tile_pattern_seq` function from `seqexplainer` to implant a feature
    at every position in a sequence. The model predictions are then computed for each implanted sequence
    and returned. The predictions can optionally be stored in the SeqData object, but are always returned.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch nn.Module to compute predictions with.
    sdata : xr.Dataset
        The dataset containing the sequence data.
    seq_id : str
        The ID of the sequence to implant the feature into. This must be an ID that is present in the
        SeqData object under the `id_var` key.
    feature : np.ndarray
        The feature to implant. Either a one-hot encoded array or a string depending on the `encoding` parameter.
    seq_var : str, optional
        The key for the sequence data in the dataset, by default "ohe_seq".
    id_var : str, optional
        The key for the sequence IDs in the dataset, by default "id".
    feature_name : str, optional
        The name of the feature, by default "feature".
    encoding : str, optional
        The encoding of the sequence data, either "onehot" or "str", by default "onehot".
    store : bool, optional
        Whether to store the predictions in the dataset, by default True.
    device : str, optional
        The device to use for predictions, inferred from `settings.gpus` if None, by default None.

    Returns
    -------
    np.ndarray
        The model predictions.
    """
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval().to(device)
    sdata[id_var].load()
    seq_xr = sdata.sel(_sequence=sdata[id_var] == seq_id)
    if encoding == "str":
        seq = seq_xr[seq_var].values.astype("U")[0]
        implanted_seqs = tile_pattern_seq(
            seq, feature, pattern_encoding="str", ohe=False
        )
        implanted_seqs = sp.ohe(implanted_seqs, sp.ALPHABETS["DNA"]).transpose(0, 2, 1)
    elif encoding == "onehot":
        seq_xr[seq_var] = seq_xr[seq_var].transpose("_sequence", "_ohe", "length")
        seq = seq_xr[seq_var].values.squeeze()
        implanted_seqs = tile_pattern_seq(
            seq, feature, pattern_encoding="onehot", ohe=True
        )
    else:
        raise ValueError("Encoding not recognized.")
    X = torch.tensor(implanted_seqs, dtype=torch.float32).to(device)
    preds = model.predict(X, verbose=False).cpu().detach().numpy().squeeze()
    if store:
        sdata[f"{seq_id}_{feature_name}_slide"] = xr.DataArray(
            preds, dims=[f"{feature_name}_slide"]
        )
    return preds


def positional_gia_sdata(
    model: torch.nn.Module,
    sdata: xr.Dataset,
    feature: np.ndarray,
    feature_name="feature",
    seq_var: str = "ohe_seq",
    id_var: str = "id",
    store_var: str = None,
    device: str = "cpu",
    encoding: str = "onehot",
):
    """Implant a feature into all sequences in a SeqData object and return the model predictions.

    This function wraps the `feature_implant_seq_sdata` function to implant a feature into all sequences
    in a SeqData object. The model predictions are then computed for each implanted sequence and returned.
    The predictions can optionally be stored in the SeqData object, but are always returned.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for predictions.
    sdata : xr.Dataset
        The dataset containing the sequence data.
    feature : np.ndarray
        The feature to implant.
    feature_name : str, optional
        The name of the feature, by default "feature".
    seq_var : str, optional
        The key for the sequence data in the dataset, by default "ohe_seq".
    id_var : str, optional
        The key for the sequence IDs in the dataset, by default "id".
    store_var : str, optional
        The key to store the predictions in the dataset, by default None.
    device : str, optional
        The device to use for predictions, by default "cpu".
    encoding : str, optional
        The encoding of the sequence data, either "onehot" or "str", by default "onehot".

    Returns
    -------
    np.ndarray
        The model predictions.

    """
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval().to(device)
    predictions = []
    for i, seq_id in tqdm(
        enumerate(sdata[id_var].values),
        desc="Implanting feature in all seqs of sdata",
        total=sdata.dims["_sequence"],
    ):
        predictions.append(
            feature_implant_seq_sdata(
                model=model,
                sdata=sdata,
                seq_id=seq_id,
                feature=feature,
                feature_name=feature_name,
                id_var=id_var,
                seq_var=seq_var,
                encoding=encoding,
                store=False,
            )
        )
    if store_var is not None:
        sdata[store_var] = xr.DataArray(
            predictions, dims=["_sequence", f"{feature_name}_test_slide"]
        )
    else:
        return np.array(predictions)


def motif_distance_dependence_gia(
    model: torch.nn.Module,
    sdata: xr.Dataset,
    feature_A: str,
    feature_B: str,
    tile_step: int = 1,
    style: str ="deAlmeida22",
    seq_var: str = "seq",
    results_var: str = "cooperativity",
    distance_var: str = "distance",
    device: str = "cpu",
    batch_size: int = 128,
):
    """Calculate the dependence of the model predictions on the distance between two motifs.

    Currently only supports a DeepSTARR style analysis in which a single motif is implanted into
    the center of a set of background sequences, and a second motif is tiled across the background
    at different distances from the first motif. A cooperativity score is calculated for the two motifs
    to quantify the dependence of the model predictions on the distance between the motifs.

    The results are stored directly in the SeqData object.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to use for predictions.
    sdata : xr.Dataset
        The dataset containing the background sequences.
    feature_A : str
        The first motif to implant in the center of the background sequences.
    feature_B : str
        The second motif to tile across the background sequences.
    tile_step : int, optional
        The step size to use to tile feature_B across the background sequences, by default 1.
    style : str, optional
        The style of the analysis, by default "deAlmeida22".
    seq_var : str, optional
        The key for the sequence data in the dataset, by default "seq".
    results_var : str, optional
        The key to store the results in the dataset, by default "cooperativity".
    distance_var : str, optional
        The key to store the distances in the dataset, by default "distance".
    device : str, optional
        The device to use for predictions, by default "cpu".
    batch_size : int, optional
        The batch size to use for predictions, by default 128.
    """

    # Make sure the backbones are compatible with the next function
    backbones = np.array([b"".join(backbone) for backbone in sdata[seq_var].values]).astype('U')

    # Do the embedding based on the passed in style
    A_seqs, B_seqs, AB_seqs, motif_b_pos, motif_b_distances = embed_deepstarr_distance_cooperativity(
        null_sequences=backbones,
        motif_a=feature_A,
        motif_b=feature_B,
        step=tile_step
    )

    # Get results that are dependent on the style
    cooperativity_results = deepstarr_motif_distance_cooperativity_gia(
        model=model,
        b_seqs=backbones,
        A_seqs=A_seqs,
        B_seqs=B_seqs,
        AB_seqs=AB_seqs,
        motif_b_distances=motif_b_distances,
        batch_size=batch_size,
        device=device
    )

    # Set up the xarray of the results
    distances = np.array(list(cooperativity_results.keys()))
    predictions = np.array(list(cooperativity_results.values()))

    # Merge the results with the original dataset
    sdata[distance_var] = xr.DataArray(distances, dims=[f"_{distance_var}"])
    sdata[results_var] = xr.DataArray(predictions, dims=[f"_{distance_var}", "_sequence", "_predictions"])
