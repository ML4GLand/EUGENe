import re
import os
import numpy as np
from io import TextIOBase
from ._Motif import Motif, MotifSet
from ...preprocess import decode_seq
from ...preprocess._utils import _token2one_hot

__version_regex = re.compile("^MEME version ([0-9]+)")
__background_regex = re.compile( "^Background letter frequencies(?: \(from (.+)\))?")
__background_sum_error = 0.00001
__pfm_header_regex = re.compile("^letter-probability matrix:(?: alength= ([0-9]+))?(?: w= ([0-9]+))") 

def _parse_version(line: str) -> str:
        match = re.match(__version_regex, line)
        if match:
            return match.group(1)
        else:
            raise RuntimeError("Minimal MEME file missing version string on first line")

def _parse_alphabet(line: str) -> str:
    if line.startswith("ALPHABET "):
        raise NotImplementedError("Alphabet definitions not supported")
    elif line.startswith("ALPHABET= "):
        return line.rstrip()[10:]
    else:
        raise RuntimeError("Unable to parse alphabet line")

def _parse_strands(line: str) -> str:
    strands = line.rstrip()[9:]
    if not ((strands == "+") or (strands == "+ -")):
        raise RuntimeError("Invalid strand specification")
    else:
        return strands

def _parse_background(line: str, handle: TextIOBase) -> str:
    match = re.match(__background_regex, line)
    if match:
        if match.group(1) is not None:
            background_source = match.group(1)
    else:
        raise RuntimeError("Unable to parse background frequency line")

    background = {}
    line = handle.readline()
    while line:
        if (not line.rstrip()) or line.startswith("MOTIF"):
            if (
                abs(1 - sum(background.values()))
                <= __background_sum_error
            ):
                return line
            else:
                raise RuntimeError("Background frequencies do not sum to 1")
        else:
            line_freqs = line.rstrip().split(" ")
            if len(line_freqs) % 2 != 0:
                raise RuntimeError("Invalid background frequency definition")
            for residue, freq in zip(line_freqs[0::2], line_freqs[1::2]):
                background[residue] = float(freq)
        line = handle.readline()

def _parse_motif(line: str, handle: TextIOBase) -> str:
    
    # parse motif identifier
    line_split = line.rstrip().split(" ")
    if (len(line_split) < 2) or (len(line_split) > 3):
        raise RuntimeError("Invalid motif name line")
    motif_identifier = line_split[1]
    motif_name = line_split[2] if len(line_split) == 3 else None
    
    # parse letter probability matrix header
    line = handle.readline()
    if not line.startswith("letter-probability matrix:"):
        raise RuntimeError(
            "No letter-probability matrix header line in motif entry"
        )
    match = re.match(__pfm_header_regex, line)
    if match:
        motif_alphabet_length = (
            int(match.group(1)) if match.group(1) is not None else None
        )
        motif_length = int(match.group(2)) if match.group(2) is not None else None
    else:
        raise RuntimeError("Unable to parse letter-probability matrix header")
    
    # parse letter probability matrix
    line = handle.readline()
    pfm_rows = []
    while line:
        line_split = line.rstrip().split()
        if motif_alphabet_length is None:
            motif_alphabet_length = len(line_split)
        elif motif_alphabet_length != len(line_split):
            raise RuntimeError(
                "Letter-probability matrix row length doesn't equal alphabet length"
            )
        pfm_row = np.array([float(s) for s in line_split])
        pfm_rows.append(pfm_row)
        line = handle.readline()

        if (line.strip() == "") or line.startswith("MOTIF"):
            pfm = np.stack(pfm_rows)
            if motif_length is None:
                motif_length = pfm.shape[0]
            elif motif_length != pfm.shape[0]:
                raise RuntimeError(
                    "Provided motif length is not consistent with the letter-probability matrix shape"
                )
            consensus = decode_seq(_token2one_hot(pfm.argmax(axis=1)))
            motif = Motif(
                identifier=motif_identifier,
                pfm=pfm,
                consensus=consensus,
                alphabet_length=motif_alphabet_length,
                length=motif_length,
                name=motif_name
            )
            return motif

def _load_pyjaspar(
    motif_accs=None, 
    motif_names=None, 
    collection=None, 
    release="JASPAR2022",
    **kwargs,
):
    """Get and return motifs from JASPAR database

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
    assert (
        motif_accs or motif_names or collection
    ), "Must provide either motif_accs, motif_names, or collection"
    try:
        from pyjaspar import jaspardb
    except ImportError:
        raise ImportError(
            "Please install pyjaspar dependency with `pip install pyjaspar`"
        )
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


def _from_biopython_motifs(
    biopyhon_motifs,
    identifier_number=0,
    verbose=False
):
    """Convert BioPython motif to Motif object

    Parameters
    ----------
    biopyhon_motif : Bio.motifs.Motif
        BioPython motif

    Returns
    -------
    Motif
        Motif object
    """
    # flatten list of lists if exists
    if isinstance(biopyhon_motifs[0], list):
        biopyhon_motifs = [item for sublist in biopyhon_motifs for item in sublist]
    motifs = {}
    for biopyhon_motif in biopyhon_motifs:
        if len(biopyhon_motif.acc) != 0:
           acc = biopyhon_motif.acc[identifier_number] 
        else:
            acc = ""
        norm_cnts = biopyhon_motif.counts.normalize()
        pfm = np.array([list(val) for val in norm_cnts.values()]).T
        curr_motif = Motif(
            identifier=acc,
            pfm=pfm,
            consensus=str(biopyhon_motif.consensus),
            alphabet_length=len(biopyhon_motif.alphabet),
            length=biopyhon_motif.length,
            name=biopyhon_motif.name,
        )
        if curr_motif.identifier in motifs:
            if verbose:
                print(f"Duplicate motif identifier: {curr_motif.name}, skipping")
        else:
            motifs[curr_motif.identifier] = curr_motif
    return MotifSet(motifs=motifs)


def _load_pymemesuite_motifs(
    filename
):
    """Load MEME file

    Parameters
    ----------
    filename : str
        MEME filename

    Returns
    -------
    list of pymemesuite.motif.Motif
        List of motifs
    pymemesuite.background.Background
        Background
    """
    memesuite_motifs = []
    try:
        from pymemesuite.common import MotifFile
    except ImportError:
        raise ImportError(
            "Please install pymemesuite dependency with `pip install pymemesuite`"
        )
    with MotifFile(filename) as motif_file:
        for motif in motif_file:
            memesuite_motifs.append(motif)
        bg = motif_file.background
    return memesuite_motifs, bg

def _from_pymemesuite_motif(
    pymemesuite_motifs,
    verbose=False
):
    """
    Function to convert pymemesuite motif to eugene motif
    Parameters
    ----------
    pymemesuite_motifs:
        list, list of pymemesuite motifs
    verbose:
        bool, verbose
    Returns
    -------
    MotifSet
    """
    motif_set = MotifSet()
    for motif in pymemesuite_motifs:
        motif_set.add_motif(
            Motif(
                 identifier=motif.accession.decode("utf-8"),
                 pfm=np.array(motif.frequencies),
                 consensus=motif.consensus,
                 name=motif.name.decode("utf-8"),
                 length=motif.width
            )
        )
    return motif_set