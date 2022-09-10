from ast import Raise
import numpy as np
import pandas as pd
import pyranges as pr
from typing import Any, Union, Optional, List
from typing import Dict, Iterable, Sequence, Mapping
from os import PathLike
from collections import OrderedDict
from functools import singledispatch
from pandas.api.types import is_string_dtype
from copy import copy, deepcopy
from ._SeqDataset import SeqDataset


Index1D = Union[slice, int, str, np.int64, np.ndarray]
try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):
            def __getitem__(cls, values):
                if not isinstance(values, tuple):
                    values = (values,)
                return type("Literal_", (Literal,), dict(__args__=values))

        class Literal(metaclass=LiteralMeta):
            pass


class SeqData:
    def __init__(
        self,
        seqs: np.ndarray = None,
        names: np.ndarray = None,
        rev_seqs: np.ndarray = None,
        seqs_annot: Optional[Union[pd.DataFrame, Mapping[str, Iterable[Any]]]] = None,
        pos_annot: Union[pr.PyRanges, Dict, str] = None,
        ohe_seqs: np.ndarray = None,
        ohe_rev_seqs: np.ndarray = None,
        seqsm: Optional[Union[np.ndarray, Mapping[str, Sequence[Any]]]] = None,
        uns: Optional[Mapping[str, Any]] = None,
        seqidx: Index1D = None,
    ):
        """Initialize SeqData object.
        Args:
            seqs: numpy array of sequences.
            names: numpy array of names.
            rev_seqs: numpy array of reverse complement sequences.
            seqs_annot: pandas DataFrame or dict of sequences annotations.
            pos_annot: PyRanges object or dict of positions annotations.
            ohe_seqs: numpy array of one-hot encoded sequences.
            ohe_rev_seqs: numpy array of one-hot encoded reverse complement sequences.
            seqsm: numpy array of sequences or dict of sequences.
            uns: dict of additional information.
            seqidx: Index of sequences to use.
        """

        if seqs is not None:
            self.seqidx = (
                range(seqs.shape[0]) if seqidx is None or len(seqidx) == 0 else seqidx
            )
        else:
            self.seqidx = (
                range(ohe_seqs.shape[0])
                if seqidx is None or len(seqidx) == 0
                else seqidx
            )

        # X
        self.seqs = np.array(seqs[self.seqidx]) if seqs is not None else None
        self.names = np.array(names[self.seqidx]) if names is not None else None
        self.rev_seqs = (
            np.array(rev_seqs[self.seqidx]) if rev_seqs is not None else None
        )
        self.ohe_seqs = (
            np.array(ohe_seqs[self.seqidx]) if ohe_seqs is not None else None
        )
        self.ohe_rev_seqs = (
            np.array(ohe_rev_seqs[self.seqidx]) if ohe_rev_seqs is not None else None
        )

        # n_obs
        if self.seqs is not None:
            self._n_obs = len(self.seqs)
        else:
            self._n_obs = len(self.ohe_seqs)

        # seq_annot (handled by gen dataframe)
        if isinstance(self.seqidx, slice):
            self.seqs_annot = _gen_dataframe(
                seqs_annot, self._n_obs, ["obs_names", "row_names"]
            )[self.seqidx]
        elif type(self.seqidx[0]) in [bool, np.bool_]:
            self.seqs_annot = _gen_dataframe(
                seqs_annot, self._n_obs, ["obs_names", "row_names"]
            ).loc[self.seqidx]
        else:
            self.seqs_annot = _gen_dataframe(
                seqs_annot, self._n_obs, ["obs_names", "row_names"]
            ).iloc[self.seqidx]

        # pos_annot
        if isinstance(pos_annot, dict):
            self.pos_annot = pr.from_dict(pos_annot)
        elif isinstance(pos_annot, str):
            self.pos_annot = pr.read_bed(pos_annot)
        else:
            self.pos_annot = pos_annot

        # uns
        self.uns = uns or OrderedDict()

        # seqsm TODO: Think about consequences of making obsm a group in hdf
        self.seqsm = convert_to_dict(seqsm)

    @property
    def seqs(self) -> np.ndarray:
        """Sequences."""
        return self._seqs

    @seqs.setter
    def seqs(self, seqs: np.ndarray):
        self._seqs = seqs

    @property
    def names(self) -> np.ndarray:
        """Names of sequences."""
        return self._names

    @names.setter
    def names(self, names: np.ndarray):
        self._names = names

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self._n_obs

    @property
    def rev_seqs(self) -> np.ndarray:
        """Reverse complement of sequences."""
        return self._rev_seqs

    @rev_seqs.setter
    def rev_seqs(self, rev_seqs: np.ndarray):
        self._rev_seqs = rev_seqs

    @property
    def ohe_seqs(self) -> np.ndarray:
        """One-hot encoded sequences."""
        return self._ohe_seqs

    @ohe_seqs.setter
    def ohe_seqs(self, ohe_seqs: np.ndarray):
        self._ohe_seqs = ohe_seqs

    @property
    def seqs_annot(self) -> pd.DataFrame:
        """Sequences annotations."""
        return self._seqs_annot

    @seqs_annot.setter
    def seqs_annot(self, seqs_annot: Union[pd.DataFrame, Mapping[str, Iterable[Any]]]):
        self._seqs_annot = _gen_dataframe(
            seqs_annot, self._n_obs, ["obs_names", "row_names"]
        )

    @property
    def pos_annot(self) -> pr.PyRanges:
        """Positional annotations."""
        return self._pos_annot

    @pos_annot.setter
    def pos_annot(self, pos_annot: pr.PyRanges):
        self._pos_annot = pos_annot

    @property
    def ohe_rev_seqs(self) -> np.ndarray:
        """One-hot encoded reverse complement sequences."""
        return self._ohe_rev_seqs

    @ohe_rev_seqs.setter
    def ohe_rev_seqs(self, ohe_rev_seqs: np.ndarray):
        self._ohe_rev_seqs = ohe_rev_seqs

    @property
    def seqsm(self) -> Mapping[str, Sequence[Any]]:
        """Sequences metadata."""
        return self._seqsm

    @seqsm.setter
    def seqsm(self, seqsm: Mapping[str, Sequence[Any]]):
        self._seqsm = seqsm

    @property
    def uns(self) -> Mapping[str, Any]:
        """Unstructured data."""
        return self._uns

    @uns.setter
    def uns(self, uns: Mapping[str, Any]):
        self._uns = uns

    def __getitem__(self, index):
        """Get item from data."""
        if isinstance(index, str):
            return self.seqs_annot[index]
        elif isinstance(index, slice):
            index = np.arange(self.n_obs)[index]
            return SeqData(
                seqs=self.seqs,
                names=self.names,
                rev_seqs=self.rev_seqs,
                seqs_annot=self.seqs_annot,
                pos_annot=self.pos_annot,
                ohe_seqs=self.ohe_seqs,
                ohe_rev_seqs=self.ohe_rev_seqs,
                seqsm=self.seqsm,
                uns=self.uns,
                seqidx=index,
            )
        # TODO: this is where we could fix the way bools are handled
        else:
            return SeqData(
                seqs=self.seqs,
                names=self.names,
                rev_seqs=self.rev_seqs,
                seqs_annot=self.seqs_annot,
                pos_annot=self.pos_annot,
                ohe_seqs=self.ohe_seqs,
                ohe_rev_seqs=self.ohe_rev_seqs,
                seqsm=self.seqsm,
                uns=self.uns,
                seqidx=index,
            )

    def __setitem__(self, index, value):
        """Set item in data."""
        if isinstance(index, str):
            self.seqs_annot[index] = value
        else:
            raise ValueError(
                "SeqData only supports setting seq_annot columns with indexing."
            )

    def __repr__(self):
        descr = f"SeqData object with = {self._n_obs} seqs"
        for attr in [
            "seqs",
            "names",
            "rev_seqs",
            "ohe_seqs",
            "ohe_rev_seqs",
            "seqs_annot",
            "pos_annot",
            "seqsm",
            "uns",
        ]:
            if attr in ["seqs", "names", "rev_seqs", "ohe_seqs", "ohe_rev_seqs"]:
                if getattr(self, attr) is not None:
                    descr += f"\n{attr} = {getattr(self, attr).shape}"
                else:
                    descr += f"\n{attr} = None"
            elif attr in ["seqs_annot"]:
                keys = getattr(self, attr).keys()
                if len(keys) > 0:
                    descr += f"\n{attr}: {str(list(keys))[1:-1]}"
            elif attr in ["pos_annot"]:
                if getattr(self, attr) is not None:
                    descr += f"\n{attr}: PyRanges object with {len(getattr(self, attr))} features"
                else:
                    descr += f"\n{attr}: None"
            elif attr in ["seqsm", "uns"]:
                if len(getattr(self, attr)) > 0:
                    descr += f"\n{attr}: {str(list(getattr(self, attr).keys()))[1:-1]}"
                else:
                    descr += f"\n{attr}: None"
        return descr


    def copy(self):
        """Return a copy of the object."""
        return deepcopy(self)


    def write_h5sd(self, path: PathLike, mode: str = "w"):
        """Write SeqData object to h5sd file.

        Args:
            path: Path to h5sd file.
            mode: Mode to open h5sd file.
        """
        from .._io import write_h5sd

        write_h5sd(self, path, mode)

    def to_dataset(
        self,
        target_keys: Union[str, List[str]] = None,
        seq_transforms: List[str] = None,
        transform_kwargs: dict = {}
    ) -> SeqDataset:
        """Convert SeqData object to SeqDataset.

        Parameters
        ----------
        target: str or list of str, optional
            Target to use for training. If None, -1 will be used for all sequences.
        seq_transforms: list of str, optional
            List of sequence transforms to apply.
        transform_kwargs: dict, optional
            Keyword arguments to pass to sequence transforms.

        Returns
        -------
        SeqDataset
        """
        from .._transforms import Augment, ReverseComplement, OneHotEncode, ToTensor
        from torchvision import transforms as torch_transforms

        transforms = []
        if target_keys is None:
            targs = None
        else:
            targs = self.seqs_annot[target_keys].values  

        if seq_transforms is None:
            print("No transforms given, assuming just need to tensorize.")
            transforms = [ToTensor(**transform_kwargs)]
            return SeqDataset(
                self.ohe_seqs,
                names=self.names,
                targets=targs if target_keys is not None else None,
                rev_seqs=self.ohe_rev_seqs,
                transform=torch_transforms.Compose(transforms),
            )

        if "augment" in seq_transforms:
            if self.seqs is not None:
                transforms.append(Augment(**transform_kwargs))
            else:
                Raise(ValueError("Cannot augment sequences if seqs is None"))

        if "reverse_complement" in seq_transforms:
            if self.seqs is not None:
                transforms.append(ReverseComplement(**transform_kwargs))
            else:
                Raise(ValueError("Cannot reverse complement sequences if seqs is None"))

        ohe_flag = True
        if "one_hot_encode" in seq_transforms:
            if self.seqs is not None:
                transforms.append(OneHotEncode(**transform_kwargs))
            else:
                Raise(ValueError("Cannot one hot encode sequences if seqs is None"))
            ohe_flag = False

        transforms.append(ToTensor(**transform_kwargs))

        if ohe_flag:
            return SeqDataset(
                self.ohe_seqs,
                names=self.names,
                targets=targs if target_keys is not None else None,
                rev_seqs=self.ohe_rev_seqs,
                transform=torch_transforms.Compose(transforms),
            )
        else:
            return SeqDataset(
                self.seqs,
                names=self.names,
                targets=targs if target_keys is not None else None,
                rev_seqs=self.rev_seqs,
                transform=torch_transforms.Compose(transforms),
            )

    def make_names_unique(self):
        n_digits = len(str(self.n_obs))
        new_index = np.array(
            [
                "seq{num:0{width}}".format(num=i, width=n_digits)
                for i in range(self.n_obs)
            ]
        )
        self.names = new_index
        self.seqs_annot.index = new_index


# TODO: move this to utils or something
@singledispatch
def _gen_dataframe(anno, length, index_names):
    if anno is None or len(anno) == 0:
        return pd.DataFrame(index=pd.RangeIndex(0, length, name=None).astype(str))
    for index_name in index_names:
        if index_name in anno:
            return pd.DataFrame(
                anno,
                index=anno[index_name],
                columns=[k for k in anno.keys() if k != index_name],
            )
    return pd.DataFrame(anno, index=pd.RangeIndex(0, length, name=None).astype(str))


@_gen_dataframe.register(pd.DataFrame)
def _(anno, length, index_names):
    anno = anno.copy(deep=False)
    if not is_string_dtype(anno.index):
        # warnings.warn("Transforming to str index.", ImplicitModificationWarning)
        anno.index = anno.index.astype(str)
    return anno


@_gen_dataframe.register(pd.Series)
@_gen_dataframe.register(pd.Index)
def _(anno, length, index_names):
    raise ValueError(f"Cannot convert {type(anno)} to DataFrame")


@singledispatch
def convert_to_dict(obj) -> dict:
    return dict(obj)


@convert_to_dict.register(dict)
def convert_to_dict_dict(obj: dict):
    return obj


@convert_to_dict.register(np.ndarray)
def convert_to_dict_ndarray(obj: np.ndarray):
    if obj.dtype.fields is None:
        raise TypeError(
            "Can only convert np.ndarray with compound dtypes to dict, "
            f"passed array had “{obj.dtype}”."
        )
    return {k: obj[k] for k in obj.dtype.fields.keys()}


@convert_to_dict.register(type(None))
def convert_to_dict_nonetype(obj: None):
    return dict()
