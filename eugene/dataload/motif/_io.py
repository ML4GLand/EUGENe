import numpy as np
from ._Motif import Motif, MotifSet
from ._convert import (
    _from_biopython
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

def _read_meme(
    filename
):
    """Read motifs from a MEME file into a list of pymemesuite.common.Motif objects.

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

def read_meme(
    filename
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
                    line = _parse_background(line, meme_file)
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

def read_motifs(
    filename,
    transpose=True,
    counts=True
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
    """

    with open(filename) as motif_file:
        data = motif_file.readlines()
    pfm_rows = []
    pfms = []
    ids = []
    names = []
    for line in data:
        line = line.rstrip()
        if line.startswith(">"):
            ids.append(line.split()[0])
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
    return MotifSet(motifs=motifs)

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
    motif_set = _from_biopython(
        motifs, 
        identifier_number=identifier_number, 
        verbose=verbose
    )
    return motif_set

def read_array(
    filename,
    transpose=True,
):
"""Read from a NumPy array file into a MotifSet object.
TODO: test this with a real case
"""
    pass

def write_meme( 
    motif_set,
    filename,
    background=[0.25, 0.25, 0.25, 0.25],
    **kwargs,
):
    """Write a MotifSet object to a MEME file.

    Parameters
    ----------
    motif_set : MotifSet
        MotifSet object
    filename : str
        Output filename
    background : list of float, optional
        Background frequencies, by default [0.25, 0.25, 0.25, 0.25]
    """
    pass

def write_motifs(
    motif_set,
    filename,
    format="homer",
    **kwargs,
):
    """Write a MotifSet object to a file.

    Parameters
    ----------
    motif_set : MotifSet
        MotifSet object
    filename : str
        Output filename
    format : str, optional
        Output format, by default "homer"
    """
    pass

def write_meme_from_array(
    array,
    outfile,
    vocab="DNA",
    background=[0.25, 0.25, 0.25, 0.25],
):
    """
    Function to convert pwm as ndarray to meme file
    Adapted from:: nnexplain GitHub:
    TODO: Depracate in favor of first converting to MotifSet and then writing to file

    Parameters
    ----------
    array:
        numpy.array, often pwm matrices, shape (U, 4, filter_size), where U - number of units
    outfile:
        string, the name of the output meme file
    """
    from ...preprocess._utils import _get_vocab

    vocab = "".join(_get_vocab(vocab))
    n_filters = array.shape[0]
    filter_size = array.shape[2]
    meme_file = open(outfile, "w")
    meme_file.write("MEME version 4\n\n")
    meme_file.write(f"ALPHABET= {vocab}\n\n")
    meme_file.write("strands: + -\n\n")
    meme_file.write("Background letter frequencies\n")
    meme_file.write(f"{vocab[0]} {background[0]} {vocab[1]} {background[1]} {vocab[2]} {background[2]} {vocab[3]} {background[3]}\n")

    for i in range(0, n_filters):
        if np.sum(array[i, :, :]) > 0:
            meme_file.write("\n")
            meme_file.write("MOTIF filter%s\n" % i)
            meme_file.write(
                "letter-probability matrix: alength= 4 w= %d \n"
                % np.count_nonzero(np.sum(array[i, :, :], axis=0))
            )
        for j in range(0, filter_size):
            if np.sum(array[i, :, j]) > 0:
                meme_file.write(
                    str(array[i, 0, j])
                    + "\t"
                    + str(array[i, 1, j])
                    + "\t"
                    + str(array[i, 2, j])
                    + "\t"
                    + str(array[i, 3, j])
                    + "\n"
                )
    meme_file.close()
    print("Saved array in as : {}".format(outfile))
