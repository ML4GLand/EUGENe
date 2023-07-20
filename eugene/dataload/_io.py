import h5py
import pyfaidx
import pyBigWig
import torch
import pyranges as pr
import numpy as np
import pandas as pd
from os import PathLike
from tqdm.auto import tqdm
from typing import List, Union, Optional, Iterable
from .datastructures import SeqData
from ._utils import _read_and_concat_dataframes
from ..preprocess._seq_preprocess import reverse_complement_seqs, decode_seqs, ohe_seq


def read_csv(
    filename: Union[PathLike, List[PathLike]],
    seq_col: Optional[str] = "seq",
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
        Column name containing sequences. Defaults to "seq".
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
    seq_file, 
    target_file=None, 
    rev_comp=False, 
    is_target_text=False, 
    return_numpy=False
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
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)
        return SeqData(
            names=ids,
            seqs=seqs,
            rev_seqs=rev_seqs,
            seqs_annot=pd.DataFrame(data=targets, columns=[f"target_{i}" for i in range(targets.shape[1])]),
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
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)
        seqs_annot = pd.DataFrame(data=targets, columns=[f"target_{i}" for i in range(targets.shape[1])])
    else:
        targets = None
        seqs_annot = None   
    if return_numpy:
        return ids, seqs, rev_seqs, targets
    elif ohe:
        
        return SeqData(
            names=ids,
            ohe_seqs=seqs,
            rev_seqs=rev_seqs,
            seqs_annot=seqs_annot,
        )
    else: 
        return SeqData(
            names=ids,
            seqs=seqs,
            rev_seqs=rev_seqs,
            seqs_annot=seqs,
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
    binsize: int = None,
    flank=0,
    resolution=None,
    collapser="max",
    order=1,
    add_seqs=False,
    return_janggu=False,
    **kwargs,
):
    """Read sequences from a BED file using Janggu data function

    Parameters
    ----------
    bed_file : str
        Path to the BED file where peaks are stored.
    roi_file : str
        Path to the file containing the regions of interest under consideration.
    ref_file : str
        Path to the genome reference file.
    flank : int, optional
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
        name="dna", 
        refgenome=ref_file, 
        roi=roi_file, 
        binsize=binsize,
        flank=flank, 
        order=order,
        **kwargs
    )
    cover = Cover.create_from_bed(
        "cover",
        bedfiles=bed_file,
        roi=roi_file,
        binsize=binsize,
        resolution=resolution,
        collapser=collapser,
        **kwargs,
    )
    if return_janggu:
        return dna, cover
    ids = np.array(list(dna.garray.region2index.keys()))
    ohe_seqs = dna[:][:, :, 0, :].transpose(0,2,1)
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
    flank=0,
    order=1,
    resolution=None,
    normalizer=None,
    add_seqs=False,
    return_janggu=False,
    **kwargs,
):
    """Read sequences from a BAM file using Janggu data function.

    Parameters
    ----------
    bam_file : str
        Path to the BED file where peaks are stored.
    roi_file : str
        Path to the file containing the regions of interest under consideration.
    ref_file : str
        Path to the genome reference file.
    flank : int, optional
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
        name="dna", 
        refgenome=ref_file, 
        roi=roi_file, 
        flank=flank, 
        order=order,
        **kwargs
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
    ohe_seqs = dna[:][:, :, 0, :].transpose(0,2,1)
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
    order=1,
    flank=0,
    resolution=None,
    collapser="max",
    add_seqs=False,
    return_janggu=False,
    **kwargs,
):
    """Read sequences from a BigWig file using Janggu data function.

    Parameters
    ----------
    bigwig_file : str
        Path to the bigwig file where peaks are stored.
    roi_file : str
        Path to the file containing the regions of interest under consideration.
    ref_file : str
        Path to the genome reference file.
    flank : int, optional
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
        name="dna", 
        refgenome=ref_file, 
        roi=roi_file, 
        flank=flank, 
        order=order,
        **kwargs
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
    ohe_seqs = dna[:][:, :, 0, :].transpose(0,2,1)
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
    """Write sequences from SeqData to csv files.

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


def read_profile(
	loci, 
    sequences, 
    signals=None, 
    controls=None, 
    chroms=None, 
	in_window=2114,
    out_window=1000, 
    max_jitter=128, 
    min_counts=None,
	max_counts=None, 
    verbose=False
):
	"""Extract sequences and signals at coordinates from a locus file.
	This function will take in genome-wide sequences, signals, and optionally
	controls, and extract the values of each at the coordinates specified in
	the locus file/s and return them as tensors.
	Signals and controls are both lists with the length of the list, n_s
	and n_c respectively, being the middle dimension of the returned
	tensors. Specifically, the returned tensors of size 
	(len(loci), n_s/n_c, (out_window/in_wndow)+max_jitter*2).
	The values for sequences, signals, and controls, can either be filepaths
	or dictionaries of np arrays or a mix of the two. When a filepath is 
	passed in it is loaded using pyfaidx or pyBigWig respectively.   
	Parameters
	----------
	loci: str or pd.DataFrame or list/tuple of such
		Either the path to a bed file or a pd DataFrame object containing
		three columns: the chromosome, the start, and the end, of each locus
		to train on. Alternatively, a list or tuple of strings/DataFrames where
		the intention is to train on the interleaved concatenation, i.e., when
		you want to train on peaks and negatives.
	sequences: str or dictionary
		Either the path to a fasta file to read from or a dictionary where the
		keys are the unique set of chromosoms and the values are one-hot
		encoded sequences as np arrays or memory maps.
	signals: list of strs or list of dictionaries or None, optional
		A list of filepaths to bigwig files, where each filepath will be read
		using pyBigWig, or a list of dictionaries where the keys are the same
		set of unique chromosomes and the values are np arrays or memory
		maps. If None, no signal tensor is returned. Default is None.
	controls: list of strs or list of dictionaries or None, optional
		A list of filepaths to bigwig files, where each filepath will be read
		using pyBigWig, or a list of dictionaries where the keys are the same
		set of unique chromosomes and the values are np arrays or memory
		maps. If None, no control tensor is returned. Default is None. 
	chroms: list or None, optional
		A set of chromosomes to extact loci from. Loci in other chromosomes
		in the locus file are ignored. If None, all loci are used. Default is
		None.
	in_window: int, optional
		The input window size. Default is 2114.
	out_window: int, optional
		The output window size. Default is 1000.
	max_jitter: int, optional
		The maximum amount of jitter to add, in either direction, to the
		midpoints that are passed in. Default is 128.
	min_counts: float or None, optional
		The minimum number of counts, summed across the length of each example
		and across all tasks, needed to be kept. If None, no minimum. Default 
		is None.
	max_counts: float or None, optional
		The maximum number of counts, summed across the length of each example
		and across all tasks, needed to be kept. If None, no maximum. Default 
		is None.  
	verbose: bool, optional
		Whether to display a progress bar while loading. Default is False.
	Returns
	-------
	seqs: torch.tensor, shape=(n, 4, in_window+2*max_jitter)
		The extracted sequences in the same order as the loci in the locus
		file after optional filtering by chromosome.
	signals: torch.tensor, shape=(n, len(signals), out_window+2*max_jitter)
		The extracted signals where the first dimension is in the same order
		as loci in the locus file after optional filtering by chromosome and
		the second dimension is in the same order as the list of signal files.
		If no signal files are given, this is not returned.
	controls: torch.tensor, shape=(n, len(controls), out_window+2*max_jitter)
		The extracted controls where the first dimension is in the same order
		as loci in the locus file after optional filtering by chromosome and
		the second dimension is in the same order as the list of control files.
		If no control files are given, this is not returned.
	"""

	seqs, signals_, controls_ = [], [], []
	in_width, out_width = in_window // 2, out_window // 2

	# Load the sequences
	if isinstance(sequences, str):
		sequences = pyfaidx.Fasta(sequences)

	names = ['chrom', 'start', 'end']
	if not isinstance(loci, (tuple, list)):
		loci = [loci]

	loci_dfs = []
	for i, df in enumerate(loci):
		if isinstance(df, str):
			df = pd.read_csv(df, sep='\t', usecols=[0, 1, 2], header=None, index_col=False, names=names)
			df['idx'] = np.arange(len(df)) * len(loci) + i
		loci_dfs.append(df)

	loci = pd.concat(loci_dfs).set_index("idx").sort_index().reset_index(drop=True)
	if chroms is not None:
		loci = loci[np.isin(loci['chrom'], chroms)]

	# Load the signal and optional control tracks if filenames are given
	if signals is not None:
		for i, signal in enumerate(signals):
			if isinstance(signal, str):
				signals[i] = pyBigWig.open(signal, "r")

	if controls is not None:
		for i, control in enumerate(controls):
			if isinstance(control, str):
				controls[i] = pyBigWig.open(control, "r")

	desc = "Loading Loci"
	d = not verbose

	max_width = max(in_width, out_width)

	for chrom, start, end in tqdm(loci.values, disable=d, desc=desc):
		mid = start + (end - start) // 2

		if start - max_width - max_jitter < 0:
			continue

		if end + max_width + max_jitter >= len(sequences[chrom]):
			continue
		
		start = mid - out_width - max_jitter
		end = mid + out_width + max_jitter
		
		# Extract the signal from each of the signal files
		if signals is not None:
			signals_.append([])
			for signal in signals:
				if isinstance(signal, dict):
					signal_ = signal[chrom][start:end]
				else:
					signal_ = signal.values(chrom, start, end, numpy=True)
					signal_ = np.nan_to_num(signal_)

				signals_[-1].append(signal_)

		# For the sequences and controls extract a window the size of the input
		start = mid - in_width - max_jitter
		end = mid + in_width + max_jitter

		# Extract the controls from each of the control files
		if controls is not None:
			controls_.append([])
			for control in controls:
				if isinstance(control, dict):
					control_ = control[chrom][start:end]
				else:
					control_ = control.values(chrom, start, end, numpy=True)
					control_ = np.nan_to_num(control_)

				controls_[-1].append(control_)

		# Extract the sequence
		if isinstance(sequences, dict):
			seq = sequences[chrom][start:end].T
		else:
			seq = ohe_seq(sequences[chrom][start:end].seq.upper())
		
		seqs.append(seq)

	seqs = torch.tensor(np.array(seqs), dtype=torch.float32)

	if signals is not None:
		signals_ = torch.tensor(np.array(signals_), dtype=torch.float32)

		idxs = torch.ones(signals_.shape[0], dtype=torch.bool)
		if max_counts is not None:
			idxs = (idxs) & (signals_.sum(dim=(1, 2)) < max_counts)
		if min_counts is not None:
			idxs = (idxs) & (signals_.sum(dim=(1, 2)) > min_counts)

		if controls is not None:
			controls_ = torch.tensor(np.array(controls_), dtype=torch.float32)
			return seqs[idxs], signals_[idxs], controls_[idxs]

		return seqs[idxs], signals_[idxs]
	else:
		if controls is not None:
			controls_ = torch.tensor(np.array(controls_), dtype=torch.float32)
			return seqs, controls_

		return seqs