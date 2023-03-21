import h5py
from os import PathLike
import numpy as np
from ._Motif import Motif, MotifSet
from ._convert import (
    from_biopython
)
from ._utils import (
    _parse_version, 
    _parse_alphabet, 
    _parse_strands, 
    _parse_background, 
    _parse_motif, 
)
from ...preprocess import decode_seq
from ...preprocess._utils import _token2one_hot

def _read_meme_pymemesuite(
    filename: PathLike
):
    """Read motifs from a MEME file into pymemesuite.common.Motif objects.

    Parameters
    ----------
    filename : str
        MEME filename

    Returns
    -------
    list of pymemesuite.common.Motif
        List of motifs
    pymemesuite.common.Background
        A pymemesuite.common.Array object representing the background frequencies
    """
    memesuite_motifs = []
    try:
        from pymemesuite.common import MotifFile
    except ImportError:
        raise ImportError("Please install pymemesuite dependency with `pip install pymemesuite`")
    with MotifFile(filename) as motif_file:
        for motif in motif_file:
            memesuite_motifs.append(motif)
        bg = motif_file.background
    return memesuite_motifs, bg

def _read_meme_MotifSet(
    filename: PathLike
):
    """Read motifs from a MEME file into a MotifSet object.

    Parameters
    ----------
    filename : str
        MEME filename
    
    Returns
    -------
    MotifSet
        MotifSet object
    """
    motifs = {}
    version = None
    alphabet = None
    strands = None
    background = None
    background_source = None
    with open(filename) as meme_file:
        line = meme_file.readline()
        version = _parse_version(line)
        line = meme_file.readline()
        while line:
            if line.startswith("ALPHABET"):
                if alphabet is None:
                    alphabet = _parse_alphabet(line)
                    line = meme_file.readline()
                else:
                    raise RuntimeError("Multiple alphabet definitions encountered in MEME file")
            elif line.startswith("strands: "):
                if strands is None:
                    strands = _parse_strands(line)
                    line = meme_file.readline()
                else:
                    raise RuntimeError("Multiple strand definitions encountered in MEME file")
            elif line.startswith("Background letter frequencies"):
                if background is None:
                    line, background = _parse_background(line, meme_file)
                else:
                    raise RuntimeError("Multiple background frequency definitions encountered in MEME file")
            elif line.startswith("MOTIF"):
                motif = _parse_motif(line, meme_file)
                if motif.identifier in motifs:
                    raise RuntimeError("Motif identifiers not unique within file")
                motifs[motif.identifier] = motif
                line = meme_file.readline()
            else:
                line = meme_file.readline()
    return MotifSet(
        motifs=motifs,
        version=version,
        alphabet=alphabet,
        strands=strands,
        background=background,
        background_source=background_source,
    )

MEME_READER_REGISTRY = {
    "pymemesuite": _read_meme_pymemesuite,
    "MotifSet": _read_meme_MotifSet,
}

def read_meme(
    filename: PathLike,
    return_type: str = "MotifSet"
):
    """Read motifs from a MEME file into a MotifSet object.
    
    Parameters
    ----------
    filename : str
        MEME filename
    
    Returns
    -------
    MotifSet
        MotifSet object
    """
    return MEME_READER_REGISTRY[return_type](filename)

def read_homer(
    filename: PathLike,
    transpose: bool = False,
    counts: bool = False,
    alphabet: str = "ACGT",
    background: dict = None,
):
    """Read motifs from a .motif file into a MotifSet object.

    Parameters
    ----------
    filename : str
        .motif filename
    transpose : bool, optional
        whether to transpose the matrices, by default True
    counts : bool, optional
        whether the input matrices are counts and should be converted to pfm, by default True

    Returns
    -------
    MotifSet
        MotifSet object
    """
    with open(filename) as motif_file:
        data = motif_file.readlines()
    pfm_rows, pfms, ids, names = [], [], [], []
    for line in data:
        line = line.rstrip()
        if line.startswith(">"):
            ids.append(line.split()[0].replace(">", ""))
            names.append(line.split()[1])
            if len(pfm_rows) > 0:
                pfms.append(np.vstack(pfm_rows))
                pfm_rows = []
        else:
            pfm_row = np.array(list(map(float, line.split())))
            pfm_rows.append(pfm_row)
    pfms.append(np.vstack(pfm_rows))
    motifs = {}
    for i in range(len(pfms)):
        if transpose:
            pfms[i] = pfms[i].T
        if counts:
            pfms[i] = np.divide(pfms[i], pfms[i].sum(axis=1)[:,None])
        consensus = decode_seq(_token2one_hot(pfms[i].argmax(axis=1)))
        motif = Motif(
            pfm=pfms[i],
            identifier=ids[i],
            name=names[i],
            consensus=consensus,
            length=len(consensus)
        )
        motifs[ids[i]] = motif
    if background is None:
        uniform = 1/len(alphabet) if background is None else background
        background = {a: uniform for a in alphabet}
    return MotifSet(
        motifs=motifs,
        alphabet=alphabet,
        background=background,
    )

def _load_jaspar(
    motif_accs=None, 
    motif_names=None, 
    collection=None, 
    release="JASPAR2022",
    **kwargs,
):
    """Load motifs from JASPAR database into Bio.motifs.jaspar.Motif objects.

    Utility function for load_jaspar.

    Parameters
    ----------
    motif_accs : list of str, optional
        List of motif accessions, by default None
    motif_names : list of str, optional
        List of motif names, by default None
    collection : str, optional
        Collection name, by default None
    release : str, optional
        JASPAR release, by default "JASPAR2020"

    Returns
    -------
    list of bio.motifs.jaspar.Motif
    """
    assert (motif_accs or motif_names or collection), "Must provide either motif_accs, motif_names, or collection"
    try:
        from pyjaspar import jaspardb
    except ImportError:
        raise ImportError("Please install pyjaspar dependency with `pip install pyjaspar`")
    jdb_obj = jaspardb(release=release)
    if motif_accs:
        if isinstance(motif_accs, str):
            motif_accs = [motif_accs]
        motifs = [jdb_obj.fetch_motif_by_id(acc) for acc in motif_accs]
    elif motif_names:
        if isinstance(motif_names, str):
            motif_names = [motif_names]
        motifs = [jdb_obj.fetch_motifs_by_name(name) for name in motif_names]
    elif collection:
        motifs = jdb_obj.fetch_motifs(collection=collection, **kwargs)
    return motifs

def load_jaspar(
    motif_accs=None,
    motif_names=None,
    collection=None,
    release="JASPAR2022",
    identifier_number=0,
    verbose=False,
    **kwargs,   
):
    """Load motifs from JASPAR database into a MotifSet object

    Parameters
    ----------
    motif_accs : list of str, optional
        List of motif accessions, by default None
    motif_names : list of str, optional
        List of motif names, by default None
    collection : str, optional
        Collection name, by default None
    release : str, optional
        JASPAR release, by default "JASPAR2020"
    identifier_number : int, optional
        Identifier number, by default 0
    verbose : bool, optional
        Verbose, by default False

    Returns
    -------
    MotifSet
    """
    motifs = _load_jaspar(
        motif_accs=motif_accs,
        motif_names=motif_names,
        collection=collection,
        release=release,
        **kwargs,
    )
    motif_set = from_biopython(
        motifs, 
        identifier_number=identifier_number, 
        verbose=verbose
    )
    return motif_set

def read_h5(filename):
    motif_set = MotifSet()
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            pfm = f[key][:]
            consensus = decode_seq(_token2one_hot(pfm.argmax(axis=1)))
            motif_set.add_motif(
                Motif(
                    identifier=key,
                    name=key,
                    pfm=pfm,
                    consensus=consensus,
                    length=len(consensus)
                )
            )
    return motif_set

def write_meme(
    motif_set: MotifSet,
    filename: str
):
    """Write MotifSet object to MEME file

    Parameters
    ----------
    motif_set : MotifSet
        MotifSet object
    filename : str
        filename to write to
    """
    alphabet = motif_set.alphabet
    version = motif_set.version
    background = motif_set.background
    strands = motif_set.strands
    meme_file = open(filename, "w")
    meme_file.write(f"MEME version {version}\n\n")
    meme_file.write(f"ALPHABET= {alphabet}\n\n")
    meme_file.write(f"strands: {strands}\n\n")
    meme_file.write("Background letter frequencies\n")
    meme_file.write(
        f"{alphabet[0]} {background[alphabet[0]]} " \
        f"{alphabet[1]} {background[alphabet[1]]} " \
        f"{alphabet[2]} {background[alphabet[2]]} " \
        f"{alphabet[3]} {background[alphabet[3]]}" \
        f"\n"
    )
    for motif in motif_set:
        ident = motif.identifier
        name = motif.name
        pfm = motif.pfm
        if np.sum(pfm) > 0:
            meme_file.write("\n")
            meme_file.write(f"MOTIF {ident} {name} \n")
            meme_file.write(f"letter-probability matrix: alength= 4 w= {len(pfm)} \n")
        for j in range(0, len(pfm)):
            if np.sum(pfm[j]) > 0:
                meme_file.write("\t".join(pfm[j].astype(str)) + "\n")
    meme_file.close()
    print("Saved pfm in MEME format as: {}".format(filename))

def write_homer(
    motif_set: MotifSet,
    filename: PathLike,
    log_odds_threshold: np.ndarray = None
):
    """Write MotifSet object to HOMER file format

    Parameters
    ----------
    motif_set : MotifSet
        MotifSet object
    filename : str
        filename to write to
    """
    motif_file = open(filename, "w")
    log_odds_threshold = np.zeros(len(motif_set)) if log_odds_threshold is None else log_odds_threshold 
    for i, motif in enumerate(motif_set):
        ident = motif.identifier
        name = motif.name
        pfm = motif.pfm
        motif_file.write(f">{ident}\t{name}\t{log_odds_threshold[i]}\n")
        for j in range(0, len(pfm)):
            if np.sum(pfm) > 0:
                motif_file.write("\t".join(pfm[j].astype(str)) + "\n")
    motif_file.close()
    print("Saved pfms in .motifs format as: {}".format(filename))

def write_h5(
    motif_set: MotifSet,
    filename: PathLike
):
    """Write MotifSet object to h5 file format

    Parameters
    ----------
    motif_set : MotifSet
        MotifSet object
    filename : str
        filename to write to
    """
    with h5py.File(filename, "w") as f:
        for motif in motif_set.motifs.values():
            f.create_dataset(motif.identifier, data=motif.pfm)
    print("Saved pfm in h5 format as: {}".format(filename))