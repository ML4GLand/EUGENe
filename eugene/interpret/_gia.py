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


def feature_implant_seq_sdata(
    model: torch.nn.Module,
    sdata,
    seq_id: str,
    feature: np.ndarray,
    seq_key: str = "ohe_seq",
    id_key: str = "id",
    feature_name: str = "feature",
    encoding: str = "onehot",
    store: bool = True,
    device: str = "cpu",
):
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval().to(device)
    sdata[id_key].load()
    seq_xr = sdata.sel(_sequence=sdata["name"] == seq_id)
    if encoding == "str":
        seq = seq_xr[seq_key].values.astype("U")[0]
        implanted_seqs = tile_pattern_seq(
            seq, feature, pattern_encoding="str", ohe=False
        )
        implanted_seqs = sp.ohe(implanted_seqs, sp.ALPHABETS["DNA"]).transpose(0, 2, 1)
    elif encoding == "onehot":
        seq_xr[seq_key] = seq_xr[seq_key].transpose("_sequence", "_ohe", "length")
        seq = seq_xr[seq_key].values.squeeze()
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
    sdata,
    feature: np.ndarray,
    feature_name="feature",
    seq_key: str = "ohe_seq",
    id_key: str = "id",
    store_key: str = None,
    device: str = "cpu",
    encoding: str = "onehot",
):
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval().to(device)
    predictions = []
    for i, seq_id in tqdm(
        enumerate(sdata[id_key].values),
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
                id_key=id_key,
                seq_key=seq_key,
                encoding=encoding,
                store=False,
            )
        )
    if store_key is not None:
        sdata[store_key] = xr.DataArray(
            predictions, dims=["_sequence", f"{feature_name}_test_slide"]
        )
    else:
        return np.array(predictions)


def motif_distance_dependence_gia(
    model,
    sdata,
    feature_A,
    feature_B,
    tile_step=1,
    style="deAlmeida22",
    seq_key: str = "seq",
    results_key: str = "cooperativity",
    distance_key: str = "distance",
    device: str = "cpu",
    batch_size: int = 128,
):

    # Make sure the backbones are compatible with the next function
    backbones = np.array([b"".join(backbone) for backbone in sdata[seq_key].values]).astype('U')

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
    sdata[distance_key] = xr.DataArray(distances, dims=[f"_{distance_key}"])
    sdata[results_key] = xr.DataArray(predictions, dims=[f"_{distance_key}", "_sequence", "_predictions"])