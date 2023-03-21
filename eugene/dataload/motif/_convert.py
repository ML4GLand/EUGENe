import torch
import numpy as np
from typing import List, Dict, Union, Optional, Iterator
import Bio
from ._Motif import Motif, MotifSet
from ...preprocess import decode_seq
from ...preprocess._utils import _token2one_hot

def to_biopython(
    motif_set: MotifSet,
    **kwargs
):
    """Convert MotifSet object to list of Bio.motif.Motif objects
    
    Parameters
    ----------
    motif_set : MotifSet
        MotifSet object
    **kwargs
        Additional arguments to pass to Bio.motif.jaspar.Motif constructor
    """
    biopython_motifs = []
    alphabet = motif_set.alphabet
    for motif in motif_set:
        pfm = motif.pfm
        motif_counts = {a: pfm[:, i] for i, a in enumerate(alphabet)}
        biomotif = Bio.motifs.jaspar.Motif(
            motif.identifier,
            motif.name,
            counts=motif_counts,
            alphabet=alphabet,
            **kwargs
        )
        biopython_motifs.append(biomotif)
    return biopython_motifs

def from_biopython(
    biopyhon_motifs: List[Bio.motifs.jaspar.Motif],
    verbose=False
):
    """Convert Bio.motif.Motif objects to MotifSet object

    Parameters
    ----------
    biopyhon_motif : List[Bio.motifs.jaspar.Motif]
        BioPython motif objects in list, can be nested
    verbose : bool, optional
        whether to print out duplicate identifiers that are skipped, by default False
    Returns
    -------
    MotifSet
        MotifSet object
    """
    # flatten list of lists if exists
    if isinstance(biopyhon_motifs[0], list):
        biopyhon_motifs = [item for sublist in biopyhon_motifs for item in sublist]
    motifs = {}
    for biopyhon_motif in biopyhon_motifs:
        norm_cnts = biopyhon_motif.counts.normalize()
        pfm = np.array([list(val) for val in norm_cnts.values()]).T
        curr_motif = Motif(
            identifier=biopyhon_motif.matrix_id,
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

def to_pymemesute(
    motif_set: MotifSet,
):
    """TODO: Convert MotifSet object to list of pymemesuite.common.Motif objects
    
    Parameters
    ----------
    motif_set : MotifSet
        MotifSet object
    
    Returns
    -------
    list
        list of pymemesuite.common.Motif objects
    """

    pass

def from_pymemesuite(
    pymemesuite_motifs,
):
    """
    Convert list of pymemesuite.common.Motif objects to MotifSet object

    Parameters
    ----------
    pymemesuite_motifs:
        list, list of pymemesuite.common.Motif objects
    
    Returns
    -------
    MotifSet
    """
    motifs = {}
    for motif in pymemesuite_motifs:
        motif = Motif(
            identifier=motif.accession.decode("utf-8"),
            pfm=np.array(motif.frequencies),
            consensus=motif.consensus,
            name=motif.name.decode("utf-8"),
            length=motif.width
        )
        motifs[motif.identifier] = motif
    return MotifSet(motifs=motifs) 
    return motif_set

def from_kernel(
    kernel: np.ndarray,
    identifiers: List[str] = None,
    names: List[str] = None,
    alphabet: str = "ACGT",
    bg: Optional[Dict[str, float]] = None,
    strands: str = "+ -",
):
    """Convert array of motif weights to MotifSet object
    
    Parameters
    ----------
    kernel : np.ndarray
        array of motif weights, shape (n_filters, n_channels, len_filter)
    identifiers : List[str], optional
        list of motif identifiers, by default None
    names : List[str], optional
        list of motif names, by default None
    alphabet : str, optional
        alphabet of motifs, by default "ACGT"
    bg : Optional[Dict[str, float]], optional
        background distribution of motifs, by default None
    strands : str, optional
        strands of motifs, by default "+ -"
    
    Returns
    -------
    MotifSet
        MotifSet object
    """
    n_filters, n_chennels, len_filter = kernel.shape
    identifiers = identifiers or [f"filter_{i}" for i in range(n_filters)]
    names = names or [f"filter_{i}" for i in range(n_filters)]
    motifs = {}
    for i in range(n_filters):
        name = names[i]
        pfm = kernel[i].T
        consensus = decode_seq(_token2one_hot(pfm.argmax(axis=1)))
        motifs[name] = Motif(
            identifier=identifiers[i],
            name=names[i],
            pfm=pfm,
            consensus=consensus,
            length=len_filter,
            alphabet_length=len(alphabet)
        )
    return MotifSet(
        motifs=motifs,
        alphabet=alphabet,
        version="5",
        background=bg,
        strands=strands
    )

def to_kernel(
    motif_set: MotifSet, 
    tensor: torch.Tensor = None,
    size: tuple = None,
    convert_to_pwm=True
) -> np.ndarray:
    """Convert MotifSet object to array of motif weights"""
    if tensor is None:
        assert size is not None
        kernel = torch.zeros(size)
        torch.nn.init.xavier_uniform_(kernel)
    else:
        kernel = tensor
        size = kernel.shape
    if len(size) != 3:
        raise RuntimeError("Kernel matrix size must be a tuple of length 3")
    motifs = motif_set.motifs
    for i, motif_id in enumerate(motifs):
        motif = motifs[motif_id]
        if convert_to_pwm:
            new_weight = torch.tensor(motif.pfm[: min(len(motif), kernel.shape[2]), :] / 0.25).transpose(0, 1)
        else:
            new_weight = torch.tensor(motif.pfm[: min(len(motif), kernel.shape[2]), :]).transpose(0, 1)
        kernel[i, :, : min(len(motif), kernel.shape[2])] = new_weight
    return kernel
