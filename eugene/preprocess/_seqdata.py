import numpy as np
from tqdm.auto import tqdm
from seqdata import SeqData
from ..utils._decorators import track
from ._dataset import split_train_test, binarize_values
from gaston import sanitize_seqs, ohe_seqs, reverse_complement_seqs


@track
def sanitize_seqs_sdata(sdata: SeqData, copy=False) -> SeqData:
    """
    Sanitize sequences in SeqData object.

    Parameters
    ----------
    sdata : SeqData
        SeqData object to sanitize.
    copy : bool, optional
        Whether to return a copy of the SeqData object, by default False

    Returns
    -------
    SeqData
        Sanitized SeqData object if copy is True, else the original SeqData object
        is modified in place.
    """
    sdata = sdata.copy() if copy else sdata
    sdata.seqs = sanitize_seqs(sdata.seqs) if sdata.seqs is not None else None
    sdata.rev_seqs = (
        sanitize_seqs(sdata.rev_seqs) if sdata.rev_seqs is not None else None
    )
    return sdata if copy else None

@track
def ohe_seqs_sdata(
    sdata: SeqData,
    vocab="DNA",
    seq_align="center",
    maxlen=None,
    fill_value=None,
    copy=False,
    **kwargs,
) -> SeqData:
    """
    One-hot encode sequences in SeqData object.

    Parameters
    ----------
    sdata : SeqData
        SeqData object to one-hot encode.
    vocab : str, optional
        Vocabulary to use for one-hot encoding, by default "DNA"
    seq_align : str, optional
        Alignment of sequences, by default "center"
    maxlen : int, optional
        Maximum length of sequences, by default None
    fill_value : str, optional
        Value to pad sequences with, by default None
    copy : bool, optional
        Whether to return a copy of the SeqData object, by default False

    Returns
    -------
    SeqData
        SeqData object with one-hot encoded sequences if copy is True, else the
        original SeqData object is modified in place.
    """
    sdata = sdata.copy() if copy else sdata
    if sdata.seqs is not None and sdata.ohe_seqs is None:
        sdata.ohe_seqs = ohe_seqs(
            sdata.seqs,
            vocab,
            seq_align=seq_align,
            maxlen=maxlen,
            fill_value=fill_value,
            **kwargs,
        )
    if sdata.rev_seqs is not None:
        sdata.ohe_rev_seqs = ohe_seqs(
            sdata.rev_seqs,
            vocab,
            seq_align=seq_align,
            maxlen=maxlen,
            fill_value=fill_value,
            **kwargs,
        )
    return sdata if copy else None

@track
def reverse_complement_seqs_sdata(
    sdata: SeqData, 
    vocab="DNA", 
    rc_seqs=False, 
    copy=False
) -> SeqData:
    """
    Reverse complement sequences in SeqData object.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    vocab : str, optional
        Vocabulary to use for one-hot encoding, by default "DNA"
    rc_seqs : bool, optional
        Whether to reverse complement the string sequences as well, by default False
    copy : bool, optional
        Whether to return a copy of the SeqData object, by default False
    Returns
    -------
    SeqData
        SeqData object with reverse complement sequences. If copy is True, a copy
        of the SeqData object is returned, else the original SeqData object is
        modified in place.
    """
    sdata = sdata.copy() if copy else sdata
    if sdata.ohe_seqs is not None:
        sdata.ohe_rev_seqs = reverse_complement_seqs(sdata.ohe_seqs)
    if sdata.ohe_seqs is None or rc_seqs:
        sdata.rev_seqs = reverse_complement_seqs(sdata.seqs, vocab)
    return sdata if copy else None

@track
def clean_nan_targets_sdata(
    sdata: SeqData,
    target_keys: list,
    nan_threshold=0.5,
    fill_value=np.nan,
    copy=False,
):
    """
    Remove targets with excessive NaN values in a SeqData object.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    target_keys : list
        List of target keys to clean.
    nan_threshold : int, optional
        Maximum fraction of NaN values allowed, by default 1
    fill_value : int, optional
        Value to fill NaN values with if target has NaNs but is kept, by default np.nan
    copy : bool, optional
        Whether to return a copy of the SeqData object, by default False

    Returns
    -------
    SeqData
        SeqData object with cleaned targets. If copy is True, a copy of the SeqData
        object is returned, else the original SeqData object is modified in place.
    """
    sdata = sdata.copy() if copy else sdata
    if type(target_keys) is str:
        target_keys = [target_keys]
    dropped_keys = []
    for target_key in target_keys:
        if (
            sdata[target_key].astype(float).isna().sum() / sdata[target_key].__len__()
            > nan_threshold
        ):
            sdata.seqs_annot = sdata.seqs_annot.drop(target_key, axis=1)
            dropped_keys.append(target_key)
        else:
            sdata[target_key].fillna(value=fill_value, inplace=True)
    print(f"Dropped targets: {dropped_keys}")
    return sdata if copy else None

@track
def clamp_targets_sdata(
    sdata: SeqData,
    target_keys: list,
    percentile: float = 0.995,
    train_key: str = None,
    clamp_nums: list = None,
    store_clamp_nums=False,
    suffix=False,
    copy=False,
) -> SeqData:
    """
    Clamp targets to a given percentile if they are above that percentile in a SeqData object.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    target_keys : list
        List of target keys to clamp.
    percentile : float, optional
        Percentile to clamp to, by default 0.995
    train_key : str, optional
        Key to use if you only want to calculate percentiles on training data, by default None
    clamp_nums : list, optional
        You can provide numbers to clamp to, by default None
    store_clamp_nums : bool, optional
        Whether to store the clamp numbers in the SeqData object, by default False
    copy : bool, optional
        Whether to return a copy of the SeqData object, by default False

    Returns
    -------
    SeqData
        SeqData object with clamped targets. If copy is True, a copy of the SeqData
        object is returned, else the original SeqData object is modified in place.
    """
    sdata = sdata.copy() if copy else sdata
    if type(target_keys) is str:
        target_keys = [target_keys]
    if clamp_nums is None:
        assert percentile is not None
        if train_key is not None:
            clamp_nums = (
                sdata[sdata.seqs_annot[train_key]]
                .seqs_annot[target_keys]
                .quantile(percentile)
            )
        else:
            clamp_nums = sdata.seqs_annot[target_keys].quantile(percentile)
    else:
        assert len(clamp_nums) == len(target_keys)
    sdata.seqs_annot[
        [f"{target_key}_clamped" for target_key in target_keys]
        if suffix
        else target_keys
    ] = sdata.seqs_annot[target_keys].clip(upper=clamp_nums, axis=1)
    if store_clamp_nums:
        sdata.uns["clamp_nums"] = clamp_nums
    return sdata if copy else None

@track
def scale_targets_sdata(
    sdata: SeqData,
    target_keys,
    train_key=None,
    scaler=None,
    store_scaler=True,
    suffix=False,
    copy=False,
) -> SeqData:
    """
    Scale targets to zero mean and unit variance in a SeqData object.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    target_keys : list
        List of target keys to scale.
    train_key : str, optional
        Key to use if you only want to calculate percentiles on training data, by default None
    scaler : sklearn scaler, optional
        Scaler to use if you want to pass one in, by default None
    store_scaler : bool, optional
        Whether to store the scaler in the SeqData object, by default True
    suffix : bool, optional
        Whether to add a suffix to the scaled target keys, by default True
    copy : bool, optional
        Whether to return a copy of the SeqData object, by default False

    Returns
    -------
    SeqData
        SeqData object with scaled target_keys. If copy is True, a copy of the SeqData
        object is returned, else the original SeqData object is modified in place.
    """
    sdata = sdata.copy() if copy else sdata
    from sklearn.preprocessing import StandardScaler

    if type(target_keys) is str:
        target_keys = [target_keys]
    if train_key is not None:
        scale_data = sdata[sdata[train_key]].seqs_annot[target_keys]
    else:
        scale_data = sdata.seqs_annot[target_keys]
    to_scale = sdata.seqs_annot[target_keys]
    if len(target_keys) == 1:
        scale_data = scale_data.values.reshape(-1, 1)
        to_scale.values.reshape(-1, 1)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(scale_data)
    assert scaler.n_features_in_ == len(target_keys)
    # Remove _scaled to help with fragmentation?
    sdata.seqs_annot[
        [f"{target_key}_scaled" for target_key in target_keys]
        if suffix
        else target_keys
    ] = scaler.transform(to_scale)
    if store_scaler:
        sdata.uns["scaler"] = scaler
    return sdata if copy else None

@track
def binarize_targets_sdata(
    sdata: SeqData,
    target_keys,
    upper_threshold=0.5,
    lower_threshold=None,
    suffix=None,
    copy=False,
    **kwargs,
) -> SeqData:
    """
    Binarize target values based on passed in thresholds in a SeqData object.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    target_keys : list
        List of target keys to binarize.
    upper_threshold : float, optional
        Upper threshold to binarize, by default 0.5
    lower_threshold : float, optional
        Lower threshold to binarize, by default None
    copy : bool, optional
        Whether to return a copy of the SeqData object, by default False

    Returns
    -------
    SeqData
        SeqData object with binarized targets. If copy is True, a copy of the SeqData
        object is returned, else the original SeqData object is modified in place.
    """
    sdata = sdata.copy() if copy else sdata
    if type(target_keys) is str:
        target_keys = [target_keys]
    for target_key in target_keys:
        sdata.seqs_annot[
            f"{target_key}_binarized" if suffix is not None else target_key
        ] = binarize_values(
            sdata[target_key], upper_threshold, lower_threshold, **kwargs
        )
    return sdata if copy else None

@track
def train_test_split_sdata(
    sdata: SeqData, 
    train_key="train_val", 
    chr=None, 
    copy=False, 
    **kwargs
) -> SeqData:
    """
    Train test split a SeqData object.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    train_key : str, optional
        Key to use for train/val split, by default "train_val"
    chr : str, optional
        Chromosome to use for train/val split, by default None
    copy : bool, optional
        Whether to return a copy of the SeqData object, by default False
    """
    sdata = sdata.copy() if copy else sdata
    if chr is not None:
        chr = [chr] if isinstance(chr, str) else chr
        sdata[train_key] = ~sdata["chr"].isin(chr)
        return sdata if copy else None
    else:
        train_indeces, _, _, _ = split_train_test(
            X_data=sdata.seqs_annot.index, y_data=sdata.seqs_annot.index, **kwargs
        )
        sdata[train_key] = sdata.seqs_annot.index.isin(train_indeces)
        return sdata if copy else None

@track
def add_ranges_sdata(
    sdata: SeqData, 
    chr_delim=":", 
    rng_delim="-", 
    copy=False
) -> SeqData:
    """
    Add position annotations to a SeqData object.

    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    chr_delim : str, optional
        Delimiter to use for chromosome, by default ":"
    rng_delim : str, optional
        Delimiter to use for range, by default "-"

    Returns
    -------
    SeqData
        SeqData object with position annotation.
    """
    sdata = sdata.copy() if copy else sdata
    idx = sdata.seqs_annot.index
    if chr_delim not in idx[0] or rng_delim not in idx[0]:
        raise ValueError("Invalid index format.")
    chr = [i[0] for i in idx.str.split(chr_delim)]
    rng = [i[1].split(rng_delim) for i in idx.str.split(chr_delim)]
    sdata["chr"] = chr
    sdata["start"] = [int(i[0]) for i in rng]
    sdata["end"] = [int(i[1]) for i in rng]
    return sdata if copy else None

@track
def seq_len_sdata(sdata, dummy=False, copy=False):
    sdata = sdata.copy() if copy else sdata
    sdata.seqs_annot["seq_len"] = [len(seq) for seq in sdata.seqs]
    return sdata
    
@track
def downsample_sdata(
    sdata, 
    n=None, 
    frac=None, 
):
    sdata = sdata.copy()
    if n is None and frac is None:
        raise ValueError("Must specify either n or frac")
    if n is not None and frac is not None:
        raise ValueError("Must specify either n or frac, not both")
    num_seqs = sdata.n_obs
    if n is not None:
        if n > num_seqs:
            raise ValueError("n must be less than or equal to the number of sequences")
        rand_idx = np.random.choice(num_seqs, n, replace=False)
        sdata = sdata[rand_idx]
    elif frac is not None:
        if frac > 1:
            raise ValueError("frac must be less than or equal to 1")
        rand_idx = np.random.choice(num_seqs, int(num_seqs * frac), replace=False)
        sdata = sdata[rand_idx]
    return sdata
