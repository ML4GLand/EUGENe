import h5py
import numpy as np
import pandas as pd
from typing import List, Union, Optional, Iterable
from os import PathLike
import pyranges as pr
from .dataloaders import SeqData
from ._utils import _read_and_concat_dataframes
from ..preprocess import reverse_complement_seqs, decode_seqs


def read_csv(
    filename: Union[PathLike, List[PathLike]],
    seq_col: Optional[str] = "SEQ",
    name_col: Optional[str] = None,
    target_col: Union[str, Iterable[str]] = None,
    rev_comp: bool = False,
    sep: str = "\t",
    low_memory: bool = False,
    return_numpy: bool = False,
    return_dataframe: bool = False,
    col_names: Iterable = None,
    auto_name: bool = True,
    compression: str = "infer",
    **kwargs,
):
    """Read sequences into SeqData object from csv/tsv files.

    Also allows for returning np.ndarray and pd.DataFrame objects if specified.

    Parameters
    ----------
    file : PathLike
        File path to read the data from.
    seq_col : str, optional
        Column name containing sequences. Defaults to "SEQ".
    name_col : str, optional
        Column name containing identifiers. Defaults to None.
    target_col : str, optional
        Column name containing targets. Defaults to None.
    rev_comp : bool, optional
        Whether to generate reverse complements for sequences. Defaults to False.
    sep : str, optional
        Delimiter to use. Defaults to "\\t".
    low_memory : bool, optional
        Whether to use low memory mode. Defaults to False.
    return_numpy : bool, optional
        Whether to return numpy arrays. Defaults to False.
    return_dataframe : bool, optional
        Whether to return pandas dataframe. Defaults to False.
    col_names : Iterable, optional
        Column names to use. Defaults to None. If not provided, uses first line of file.
    auto_name : bool, optional
        Whether to automatically generate identifiers. Defaults to True.
    compression : str, optional
        Compression type to use. Defaults to "infer".
    **kwargs : kwargs, dict
        Keyword arguments to pass to pandas.read_csv. Defaults to {}.

    Returns
    -------
    sdata : SeqData
        Returns SeqData object containing sequences and identifiers by default
    tuple :
        Returns numpy arrays of identifiers, sequences, reverse complement sequences and targets.
        If return_numpy is True. If any are not provided they are set to none.
    dataframe : pandas.DataFrame
        Returns pandas dataframe containing sequences and identifiers if return_dataframe is True.
    """
    dataframe = _read_and_concat_dataframes(
        file_names=filename,
        col_names=col_names,
        sep=sep,
        low_memory=low_memory,
        compression=compression,
        **kwargs,
    )
    seqs = dataframe[seq_col].to_numpy(dtype=str)
    targets = dataframe[target_col].to_numpy(float) if target_col is not None else None
    if name_col is not None:
        ids = dataframe[name_col].to_numpy(dtype=str)
    else:
        if auto_name:
            n_digits = len(str(len(dataframe) - 1))
            dataframe.index = np.array(
                [
                    "seq{num:0{width}}".format(num=i, width=n_digits)
                    for i in range(len(dataframe))
                ]
            )
            ids = dataframe.index.to_numpy()
        else:
            ids = None
    if rev_comp:
        from ..preprocess import reverse_complement_seqs

        rev_seqs = reverse_complement_seqs(seqs)
    else:
        rev_seqs = None
    if return_numpy:
        return ids, seqs, rev_seqs, targets
    elif return_dataframe:
        return dataframe
    else:
        return SeqData(
            names=ids,
            seqs=seqs,
            rev_seqs=rev_seqs,
            seqs_annot=pd.DataFrame(data=targets, index=ids, columns=["target"]),
        )


def read_fasta(
    seq_file, target_file=None, rev_comp=False, is_target_text=False, return_numpy=False
):
    """Read sequences into SeqData object from fasta files.

    Parameters
    ----------
    seq_file : str
        Fasta file path to read
    target_file : str
        .npy or .txt file path containing targets. Defaults to None.
    rev_comp : bool, optional
        Whether to generate reverse complements for sequences. Defaults to False.
    is_target_text : bool, optional
        Whether the file is compressed or plaintext. Defaults to False.
    return_numpy : bool, optional
        Whether to return numpy arrays. Defaults to False.

    Returns
    -------
    sdata : SeqData
        Returns SeqData object containing sequences and identifiers by default
    tuple :
        Returns numpy arrays of identifiers, sequences, reverse complement sequences and targets.
        If return_numpy is True. If any are not provided they are set to none.
    """
    seqs = np.array([x.rstrip() for (i, x) in enumerate(open(seq_file)) if i % 2 == 1])
    ids = np.array(
        [
            x.rstrip().replace(">", "")
            for (i, x) in enumerate(open(seq_file))
            if i % 2 == 0
        ]
    )
    if target_file is not None:
        if is_target_text:
            targets = np.loadtxt(target_file, dtype=float)
        else:
            targets = np.load(target_file)
    else:
        targets = None
    if rev_comp:
        from ..preprocess import reverse_complement_seqs

        rev_seqs = reverse_complement_seqs(seqs)
    else:
        rev_seqs = None
    if return_numpy:
        return ids, seqs, rev_seqs, targets
    elif targets is not None:
        return SeqData(
            names=ids,
            seqs=seqs,
            rev_seqs=rev_seqs,
            seqs_annot=pd.DataFrame(data=targets, columns=["target"]),
        )
    else:
        return SeqData(names=ids, seqs=seqs, rev_seqs=rev_seqs)


def read_numpy(
    seq_file,
    names_file=None,
    target_file=None,
    rev_seq_file=None,
    is_names_text=False,
    is_seq_text=False,
    is_target_text=False,
    delim="\n",
    ohe=False,
    return_numpy=False,
):
    """Read sequences into numpy objects from numpy compressed files.

    Note if you pass only one hot encoded sequences in, you must pass in reverse complements if you want them to be included.

    Parameters
    ----------
    seq_file : str
        .npy file path containing sequences.
    names_file : str
        .npy or .txt file path containing identifiers. Defaults to None.
    target_file : str
        .npy or .txt file path containing targets. Defaults to None.
    rev_seq_file : str, optional
        .npy or .txt file path containing reverse complement sequences. Defaults to None.
    is_names_text : bool, optional
        Whether the file is compressed (.npy) or plaintext (.txt). Defaults to False.
    is_seq_text : bool, optional
         Whether the file is (.npy) or plaintext (.txt). Defaults to False.
    is_target_text : bool, optional
        Whether the file is (.npy) or plaintext (.txt). Defaults to False.
    delim : str, optional
        Defaults to "\\n".
    ohe : bool, optional
        Whether the sequences are one hot encoded. Defaults to False.
    return_numpy : bool, optional
        Whether to return numpy arrays. Defaults to False.

    Returns
    --------
    sdata : SeqData
        Returns SeqData object containing sequences and identifiers by default.
    tuple :
        Numpy arrays of identifiers, sequences, reverse complement sequences and targets.
        If return_numpy is True. If any are not provided they are set to none.
    """
    if is_seq_text:
        seqs = np.loadtxt(seq_file, dtype=str, delim=delim)
        if rev_seq_file is not None:
            rev_seqs = np.loadtxt(rev_seq_file, dtype=str)
        else:
            rev_seqs = None
    else:
        seqs = np.load(seq_file, allow_pickle=True)
        if rev_seq_file is not None:
            rev_seqs = np.load(rev_seq_file, allow_pickle=True)
        else:
            rev_seqs = None
    if names_file is not None:
        if is_names_text:
            ids = np.loadtxt(names_file, dtype=str)
        else:
            ids = np.load(names_file)
    else:
        ids = None
    if target_file is not None:
        if is_target_text:
            targets = np.loadtxt(target_file, dtype=float)
        else:
            targets = np.load(target_file)
    else:
        targets = None
    if return_numpy:
        return ids, seqs, rev_seqs, targets
    elif ohe:
        return SeqData(
            names=ids,
            ohe_seqs=seqs,
            rev_seqs=rev_seqs,
            seqs_annot=pd.DataFrame(data=targets, columns=["targets"]),
        )
    else:
        return SeqData(
            names=ids,
            seqs=seqs,
            rev_seqs=rev_seqs,
            seqs_annot=pd.DataFrame(data=targets, columns=["targets"]),
        )


def read_h5sd(filename: Optional[PathLike], sdata=None, mode: str = "r"):
    """
    Read sequences into SeqData objects from h5sd files.

    Parameters
    ----------
    filename (str):
        .h5sd file path to read.
    sdata (SeqData, optional):
        SeqData object to load data into. Defaults to None.
    mode (str, optional):
        Mode to open file. Defaults to "r".

    Returns
    -------
        sdata: SeqData object containing sequences and identifiers.
    """
    with h5py.File(filename, "r") as f:
        d = {}
        if "seqs" in f:
            d["seqs"] = np.array([n.decode("ascii", "ignore") for n in f["seqs"][:]])
        if "names" in f:
            d["names"] = np.array([n.decode("ascii", "ignore") for n in f["names"][:]])
        if "ohe_seqs" in f:
            d["ohe_seqs"] = f["ohe_seqs"][:]
        if "rev_seqs" in f:
            d["rev_seqs"] = np.array(
                [n.decode("ascii", "ignore") for n in f["rev_seqs"][:]]
            )
        if "ohe_rev_seqs" in f:
            d["ohe_rev_seqs"] = f["ohe_rev_seqs"][:]
        if "seqs_annot" in f:
            out_dict = {}
            for key in f["seqs_annot"].keys():
                out = f["seqs_annot"][key][()]
                if isinstance(out[0], bytes):
                    out_dict[key] = np.array([n.decode("ascii", "ignore") for n in out])
                else:
                    out_dict[key] = out
            if "names" in f:
                d["seqs_annot"] = pd.DataFrame(index=d["names"], data=out_dict).replace(
                    "NA", np.nan
                )
            else:
                n_digits = len(str(len(d["seqs"])))
                idx = np.array(
                    [
                        "seq{num:0{width}}".format(num=i, width=n_digits)
                        for i in range(len(d["seqs"]))
                    ]
                )
                d["seqs_annot"] = pd.DataFrame(index=idx, data=out_dict).replace(
                    "NA", np.nan
                )
        if "pos_annot" in f:
            out_dict = {}
            for key in f["pos_annot"].keys():
                out = f["pos_annot"][key][()]
                if isinstance(out[0], bytes):
                    out_dict[key] = np.array([n.decode("ascii", "ignore") for n in out])
                else:
                    out_dict[key] = out
            d["pos_annot"] = pr.from_dict(out_dict)
        if "seqsm" in f:
            out_dict = {}
            for key in f["seqsm"].keys():
                out = f["seqsm"][key][()]
                if isinstance(out[0], bytes):
                    out_dict[key] = np.array([n.decode("ascii", "ignore") for n in out])
                else:
                    out_dict[key] = out
            d["seqsm"] = out_dict
        if "uns" in f:
            out_dict = {}
            for key in f["uns"].keys():
                if key == "pfms":
                    pfm_dfs = {}
                    for i, pfm in enumerate(f["uns"][key][()]):
                        pfm_dfs[i] = pd.DataFrame(pfm, columns=["A", "C", "G", "T"])
                    out_dict[key] = pfm_dfs
                else:
                    out = f["uns"][key][()]
                    if isinstance(out[0], bytes):
                        out_dict[key] = np.array(
                            [n.decode("ascii", "ignore") for n in out]
                        )
                    else:
                        out_dict[key] = out
            d["uns"] = out_dict
    return SeqData(**d)


def read_bed(
    bed_file: str,
    roi_file: str,
    ref_file: str,
    dnaflank=0,
    resolution=None,
    collapser="max",
    add_seqs=False,
    return_janggu=False,
    **kwargs,
):
    """
    Read sequences from a BED file.

    Parameters
    ----------
    bed_file : str
        Path to the BED file where peaks are stored.
    roi_file : str
        Path to the file containing the regions of interest under consideration.
    ref_file : str
        Path to the genome reference file.
    dnaflank : int, optional
        Number of nucleotides to flank the sequence. Defaults to None.
    resolution : int, optional
        Resolution of the sequence. Defaults to None.
    collapser : str, optional
        Collapser to use. Defaults to "max".
    add_seqs : bool, optional
        Add sequences to the DataFrame. Defaults to False.
    return_janggu : bool, optional
        Return a Janggu object. Defaults to False.
    **kwargs : dict
        Additional arguments to pass to as Janggu's parameters for loading.

    Returns
    -------
    sdata : SeqData
        SeqData object containing the peaks.
    """
    try:
        from ..external.janggu.data import Bioseq, Cover
    except ImportError:
        raise ImportError(
            "Please install janggu dependencies `pip install eugene[janggu]`"
        )
    dna = Bioseq.create_from_refgenome(
        name="dna", refgenome=ref_file, roi=roi_file, flank=dnaflank, **kwargs
    )
    cover = Cover.create_from_bed(
        "cover",
        bedfiles=bed_file,
        roi=roi_file,
        resolution=resolution,
        collapser=collapser,
        **kwargs,
    )
    if return_janggu:
        return dna, cover
    ids = np.array(list(dna.garray.region2index.keys()))
    ohe_seqs = dna[:][:, :, 0, :]
    targets = cover[:].squeeze()
    seqs = np.array(decode_seqs(ohe_seqs)) if add_seqs else None
    rev_seqs = np.array(reverse_complement_seqs(seqs)) if add_seqs else None
    return SeqData(
        names=ids,
        seqs=seqs,
        ohe_seqs=ohe_seqs,
        rev_seqs=rev_seqs,
        seqs_annot=pd.DataFrame(data=targets, index=ids, columns=["target"]),
    )


def read_bam(
    bam_file: str,
    roi_file: str,
    ref_file: str,
    dnaflank=0,
    resolution=None,
    normalizer=None,
    add_seqs=False,
    return_janggu=False,
    **kwargs,
):
    """
    Read sequences from a BAM file.

    Parameters
    ----------
    bam_file : str
        Path to the BED file where peaks are stored.
    roi_file : str
        Path to the file containing the regions of interest under consideration.
    ref_file : str
        Path to the genome reference file.
    dnaflank : int, optional
        Number of nucleotides to flank the sequence. Defaults to None.
    resolution : int, optional
        Resolution of the sequence. Defaults to None.
    collapser : str, optional
        Collapser to use. Defaults to "max".
    add_seqs : bool, optional
        Add sequences to the DataFrame. Defaults to False.
    return_janggu : bool, optional
        Return a Janggu object. Defaults to False.
    **kwargs : dict
        Additional arguments to pass to as Janggu's parameters for loading.

    Returns
    -------
    sdata : SeqData
        SeqData object containing the peaks.
    """
    try:
        from ..external.janggu.data import Bioseq, Cover
    except ImportError:
        raise ImportError(
            "Please install janggu dependencies `pip install eugene[janggu]`"
        )
    dna = Bioseq.create_from_refgenome(
        name="dna", refgenome=ref_file, roi=roi_file, flank=dnaflank, **kwargs
    )
    cover = Cover.create_from_bam(
        "cover",
        bamfiles=bam_file,
        roi=roi_file,
        resolution=resolution,
        normalizer=normalizer,
        stranded=False,
        **kwargs,
    )
    if return_janggu:
        return dna, cover
    ids = np.array(list(dna.garray.region2index.keys()))
    ohe_seqs = dna[:][:, :, 0, :]
    targets = cover[:].squeeze(axis=(2, 3))
    seqs = np.array(decode_seqs(ohe_seqs)) if add_seqs else None
    rev_seqs = np.array(reverse_complement_seqs(seqs)) if add_seqs else None
    return SeqData(
        names=ids,
        seqs=seqs,
        ohe_seqs=ohe_seqs,
        rev_seqs=rev_seqs,
        seqs_annot=pd.DataFrame(
            data=targets,
            index=ids,
            columns=[f"target_{i}" for i in range(targets.shape[1])],
        ),
    )


def read_bigwig(
    bigwig_file: str,
    roi_file: str,
    ref_file: str,
    dnaflank=0,
    resolution=None,
    collapser="max",
    add_seqs=False,
    return_janggu=False,
    **kwargs,
):
    """
    Read sequences from a BigWig file.

    Parameters
    ----------
    bigwig_file : str
        Path to the bigwig file where peaks are stored.
    roi_file : str
        Path to the file containing the regions of interest under consideration.
    ref_file : str
        Path to the genome reference file.
    dnaflank : int, optional
        Number of nucleotides to flank the sequence. Defaults to None.
    resolution : int, optional
        Resolution of the sequence. Defaults to None.
    collapser : str, optional
        Collapser to use. Defaults to "max".
    add_seqs : bool, optional
        Add sequences to the DataFrame. Defaults to False.
    return_janggu : bool, optional
        Return a Janggu object. Defaults to False.
    **kwargs : dict
        Additional arguments to pass to as Janggu's parameters for loading.

    Returns
    -------
    sdata : SeqData
        SeqData object containing the peaks.
    """
    try:
        from ..external.janggu.data import Bioseq, Cover
    except ImportError:
        raise ImportError(
            "Please install janggu dependencies `pip install eugene[janggu]`"
        )

    dna = Bioseq.create_from_refgenome(
        name="dna", refgenome=ref_file, roi=roi_file, flank=dnaflank, **kwargs
    )
    cover = Cover.create_from_bigwig(
        "cover",
        bigwigfiles=bigwig_file,
        roi=roi_file,
        resolution=resolution,
        collapser=collapser,
        **kwargs,
    )
    if return_janggu:
        return dna, cover
    ids = np.array(list(dna.garray.region2index.keys()))
    ohe_seqs = dna[:][:, :, 0, :]
    targets = cover[:].squeeze(axis=(2, 3))
    seqs = np.array(decode_seqs(ohe_seqs)) if add_seqs else None
    rev_seqs = np.array(reverse_complement_seqs(seqs)) if add_seqs else None
    return SeqData(
        names=ids,
        seqs=seqs,
        ohe_seqs=ohe_seqs,
        rev_seqs=rev_seqs,
        seqs_annot=pd.DataFrame(
            data=targets,
            index=ids,
            columns=[f"target_{i}" for i in range(targets.shape[1])],
        ),
    )


def read(seq_file, *args, **kwargs):
    """Wrapper function to read sequences based on file extension.

    Parameters
    ----------
    seq_file : str
        File path containing sequences.
    *args : dict
        Positional arguments from read_csv, read_fasta, read_numpy, etc.
    **kwargs : dict
        Keyword arguments from read_csv, read_fasta, read_numpy, etc.

    Returns
    -------
    sdata : SeqData
        SeqData object containing sequences and identifiers
    tuple :
        Numpy arrays of identifiers, sequences, reverse complement sequences and targets.
        If any are not provided they are set to none.
    """
    seq_file_extension = seq_file.split(".")[-1]
    if seq_file_extension in ["csv", "tsv"]:
        return read_csv(seq_file, *args, **kwargs)
    elif seq_file_extension in ["npy"]:
        return read_numpy(seq_file, *args, **kwargs)
    elif seq_file_extension in ["fasta", "fa"]:
        return read_fasta(seq_file, *args, **kwargs)
    elif seq_file_extension in ["h5sd", "h5"]:
        return read_h5sd(seq_file, *args, **kwargs)
    elif seq_file_extension in ["bed"]:
        return read_bed(seq_file, *args, **kwargs)
    elif seq_file_extension in ["bam"]:
        return read_bam(seq_file, *args, **kwargs)
    elif seq_file_extension in ["bw"]:
        return read_bigwig(seq_file, *args, **kwargs)
    else:
        print("Sequence file type not currently supported. Seethe.")
        return


def write_csv(sdata, filename, target_key=None, delim="\t"):
    r"""Write sequences from SeqData to csv files.

    Parameters
    ----------
    sdata : SeqData
        SeqData object containing sequences and identifiers.
    filename : str
        File path to write to.
    target_key : str
        Key in sdata.seqs_annot to use as target.
    delim : str, optional
        Delimiter to use. Defaults to "\\t".
    """
    dataframe = pd.DataFrame(data={"name": sdata.names, "seq": sdata.seqs})
    dataframe = dataframe.merge(sdata.seqs_annot, left_on="name", right_index=True)
    dataframe.to_csv(filename, sep=delim, index=False)


def write_fasta(sdata, filename):
    """Write sequences from SeqData to fasta files.

    Parameters
    ----------
    sdata : SeqData
        SeqData object containing sequences and identifiers.
    filename : str
        File path to write to.
    """
    with open(filename, "w") as f:
        for i in range(len(sdata.seqs)):
            f.write(">" + sdata.names[i] + "\n")
            f.write(sdata.seqs[i] + "\n")


def write_numpy(sdata, filename, ohe=False, target_key=None):
    """Write sequences from SeqData to numpy files.

    Parameters
    ----------
    sdata : SeqData
        SeqData object containing sequences and identifiers.
    filename : str
        File path to write to.
    ohe : bool
        Whether to include ohe sequences in a separate file.
    target_key : str, optional
        Optionally save targets from a SeqData object using a key.
    """
    if ohe:
        np.save(filename + "_ohe_seqs.npy", sdata.ohe_seqs)

    np.save(filename + "_seqs.npy", sdata.seqs)

    if target_key is not None:
        np.save(filename + "_targets.npy", sdata.seqs_annot[target_key])


def write_h5sd(sdata, filename: Optional[PathLike] = None, mode: str = "w"):
    """Write SeqData object to h5sd file.

    Parameters
    ----------
    sdata : SeqData
        SeqData object containing sequences and identifiers.
    filename : str, optional
        File path to write to. Defaults to None.
    mode : str, optional
        Mode to open file. Defaults to "w".
    """
    with h5py.File(filename, mode) as f:
        f = f["/"]
        f.attrs.setdefault("encoding-type", "SeqData")
        f.attrs.setdefault("encoding-version", "0.0.0")
        if sdata.seqs is not None:
            f.create_dataset(
                "seqs", data=np.array([n.encode("ascii", "ignore") for n in sdata.seqs])
            )
        if sdata.names is not None:
            f.create_dataset(
                "names",
                data=np.array([n.encode("ascii", "ignore") for n in sdata.names]),
            )
        if sdata.ohe_seqs is not None:
            f.create_dataset("ohe_seqs", data=sdata.ohe_seqs)
        if sdata.rev_seqs is not None:
            f.create_dataset(
                "rev_seqs",
                data=np.array([n.encode("ascii", "ignore") for n in sdata.rev_seqs]),
            )
        if sdata.ohe_rev_seqs is not None:
            f.create_dataset("ohe_rev_seqs", data=sdata.ohe_rev_seqs)
        if sdata.seqs_annot is not None:
            for key, item in dict(sdata.seqs_annot).items():
                # note that not all variable types are supported but string and int are
                if item.dtype == "object":
                    f["seqs_annot/" + str(key)] = np.array(
                        [
                            n.encode("ascii", "ignore")
                            for n in item.replace(np.nan, "NA")
                        ]
                    )
                else:
                    f["seqs_annot/" + str(key)] = item
        if sdata.pos_annot is not None:
            for key, item in dict(sdata.pos_annot.df).items():
                if item.dtype in ["object", "category"]:
                    f["pos_annot/" + str(key)] = np.array(
                        [
                            n.encode("ascii", "ignore")
                            for n in item.replace(np.nan, "NA")
                        ]
                    )
                else:
                    f["pos_annot/" + str(key)] = item
        if sdata.seqsm is not None:
            for key, item in dict(sdata.seqsm).items():
                f["seqsm/" + str(key)] = item
        if sdata.uns is not None:
            for key, item in dict(sdata.uns).items():
                if "pfms" in key:
                    pfms = np.zeros((len(item), *item[list(item.keys())[0]].shape))
                    for i, in_key in enumerate(item.keys()):
                        pfms[i, :, :] = item[in_key]
                    item = pfms
                try:
                    f["uns/" + str(key)] = item
                except TypeError:
                    print(f"Unsupported type for {key}")
                    continue


def write(sdata, filename, *args, **kwargs):
    """Wrapper function to write SeqData objects to various file types.

    Parameters
    ----------
    sdata : SeqData
        SeqData object containing sequences and identifiers.
    filename : str
        File path to write to.
    *args : args, dict
        Positional arguments from write_csv, write_fasta, write_numpy.
    **kwargs : kwargs, dict
        Keyword arguments from write_csv, write_fasta, write_numpy.
    """
    seq_file_extension = filename.split(".")[-1]
    if seq_file_extension in ["csv", "tsv"]:
        write_csv(sdata, filename, *args, **kwargs)
    elif seq_file_extension in ["npy"]:
        write_numpy(sdata, filename, *args, **kwargs)
    elif seq_file_extension in ["fasta", "fa"]:
        write_fasta(sdata, filename, *args, **kwargs)
    elif seq_file_extension in ["h5sd", "h5"]:
        write_h5sd(sdata, filename, *args, **kwargs)
    else:
        print("Sequence file type not currently supported.")
        return
