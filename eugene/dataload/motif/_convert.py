import torch
import numpy as np
from typing import Dict
from ._Motif import Motif, MotifSet

def _to_biopython(motif_set):
    """Convert MotifSet object to list of Bio.motif.Motif objects"""
    pass    

def _from_biopython(
    biopyhon_motifs,
    identifier_number=0,
    verbose=False
):
    """Convert Bio.motif.Motif objects to MotifSet object

    Parameters
    ----------
    biopyhon_motif : Bio.motifs.Motif
        BioPython motif
    identifier_number : int, optional
        if there are multiple identifiers, which one to use by index, by default 0
    verbose : bool, optional
        whether to print outany duplicate identifiers that are skipped, by default False
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

def _to_pymemesute(motif_set):
    """Convert MotifSet object to list of pymemesuite.common.Motif objects
    
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

def _from_pymemesuite(
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

def _from_array():
    """Convert array of motif weights to MotifSet object"""
    pass

def _to_array(
    motifs: MotifSet, 
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
    motifs = motifs.motifs
    for i, motif_id in enumerate(motifs):
        motif = motifs[motif_id]
        if convert_to_pwm:
            new_weight = torch.tensor(motif.pfm[: min(len(motif), kernel.shape[2]), :] / 0.25).transpose(0, 1)
        else:
            new_weight = torch.tensor(motif.pfm[: min(len(motif), kernel.shape[2]), :]).transpose(0, 1)
        kernel[i, :, : min(len(motif), kernel.shape[2])] = new_weight
    return kernel
