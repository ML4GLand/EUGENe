from tqdm.auto import tqdm
from ..dataloading import SeqData
from ._dataset_preprocess import split_train_test
from ._seq_preprocess import ohe_DNA_seqs, reverse_complement_seqs
from ..utils._decorators import track


@track
def reverse_complement_data(sdata: SeqData, copy=False) -> SeqData:
    """Reverse complement sequences.
    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    Returns
    -------
    SeqData
        SeqData object with reverse complement sequences.
    """
    sdata = sdata.copy() if copy else sdata
    sdata.rev_seqs = reverse_complement_seqs(sdata.seqs)
    return sdata if copy else None


@track
def one_hot_encode_data(sdata: SeqData, copy=False) -> SeqData:
    """One-hot encode sequences.
    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    Returns
    -------
    SeqData
        SeqData object with one-hot encoded sequences.
    """
    sdata = sdata.copy() if copy else sdata
    if sdata.seqs is not None:
        sdata.ohe_seqs = ohe_DNA_seqs(sdata.seqs)
    if sdata.rev_seqs is not None:
        sdata.ohe_rev_seqs = ohe_DNA_seqs(sdata.rev_seqs)
    return sdata if copy else None


@track
def train_test_split_data(sdata: SeqData, copy=False, **kwargs) -> SeqData:
    """Train test split.
    Parameters
    ----------
    sdata : SeqData"""
    sdata = sdata.copy() if copy else sdata
    train_indeces, _, _, _ = split_train_test(
        X_data=sdata.seqs_annot.index, y_data=sdata.seqs_annot.index, **kwargs
    )
    sdata["train"] = sdata.seqs_annot.index.isin(train_indeces)
    return sdata if copy else None


@track
def add_pos_annot(sdata: SeqData, copy=False) -> SeqData:
    """Add position annotation.
    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    Returns
    -------
    SeqData
        SeqData object with position annotation.
    """
    sdata = sdata.copy() if copy else sdata
    sdata["POS"] = sdata.seqs_annot.index.map(lambda x: x.split("_")[1])
    return sdata if copy else None


@track
def scale_targets(sdata: SeqData, target, train_key, copy=False) -> SeqData:
    """Scale targets.
    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    Returns
    -------
    SeqData
        SeqData object with scaled targets.
    """
    from sklearn.preprocessing import StandardScaler

    sdata = sdata.copy() if copy else sdata
    scaler = StandardScaler()
    scaler.fit(sdata[sdata[train_key] == True][target].values.reshape(-1, 1))
    sdata[f"{target}_scaled"] = scaler.transform(sdata[target].values.reshape(-1, 1))
    return sdata if copy else None


@track
def prepare_data(
    sdata: SeqData,
    steps=["reverse_complement", "one_hot_encode", "train_test_split"],
    copy=False,
) -> SeqData:
    """Prepare data.
    Parameters
    ----------
    sdata : SeqData
        SeqData object.
    Returns
    -------
    SeqData
        SeqData object with prepared data.
    """
    if not isinstance(steps, list):
        steps = [steps]

    steps = list(steps)
    pbar = tqdm(steps)
    for step in pbar:
        pbar.set_description(f"{step_name[step]} on SeqData")
        preprocessing_steps[step].__wrapped__(sdata)

    return sdata if copy else None


preprocessing_steps = dict(
    reverse_complement=reverse_complement_data,
    one_hot_encode=one_hot_encode_data,
    train_test_split=train_test_split_data,
)

step_name = dict(
    reverse_complement="Reverse complementing",
    one_hot_encode="One hot encoding",
    train_test_split="Train/test splitting",
)
