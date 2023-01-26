import numpy as np
import pandas as pd
import pyranges as pr
from ...utils import track


def fimo_motifs(sdata, pymeme_motifs, background):
    """Run FIMO on a list of motifs

    Parameters
    ----------
    sdata : eugene.data.SeqData
        SeqData object
    pymeme_motifs : list of pymemesuite.motif.Motif
        List of motifs
    background : pymemesuite.background.Background
        Background

    Returns
    -------
    list of list
        List of FIMO scores
    """
    try:
        from pymemesuite.common import MotifFile, Sequence
        from pymemesuite.fimo import FIMO
    except ImportError:
        raise ImportError(
            "Please install pymemesuite dependency with `pip install pymemesuite`"
        )
    pymeme_seqs = [
        Sequence(str(seq), name.encode()) for seq, name in zip(sdata.seqs, sdata.names)
    ]
    fimo = FIMO(both_strands=True)
    motif_scores = []
    for motif in pymeme_motifs:
        pattern = fimo.score_motif(motif, pymeme_seqs, background)
        for m in pattern.matched_elements:
            motif_scores.append(
                [
                    m.source.accession.decode(),
                    m.start,
                    m.stop,
                    m.strand,
                    m.score,
                    m.pvalue,
                    m.qvalue,
                    motif.accession.decode(),
                    motif.name.decode(),
                ]
            )
    return motif_scores


def score_seqs(
    sdata,
    motif_accs=None,
    motif_names=None,
    collection=None,
    release="JASPAR2020",
    filename="motifs.meme",
):
    """Score sequences with JASPAR motifs

    Parameters
    ----------
    sdata : eugene.data.SeqData
        SeqData object
    motif_accs : list of str, optional
        List of motif accessions, by default None
    motif_names : list of str, optional
        List of motif names, by default None
    collection : str, optional
        Collection name, by default None
    release : str, optional
        JASPAR release, by default "JASPAR2020"
    filename : str, optional
        MEME filename, by default "motifs.meme"
    """
    assert (
        motif_accs or motif_names or collection
    ), "Must provide either motif_accs, motif_names, or collection"
    motifs = get_jaspar_motifs(
        motif_accs=motif_accs,
        motif_names=motif_names,
        collection=collection,
        release=release,
    )
    save_motifs_as_meme(motifs, filename)
    memesuite_motifs, bg = load_meme(filename)
    scores = fimo_motifs(sdata, memesuite_motifs, bg)
    dataframe = pr.PyRanges(
        pd.DataFrame(
            scores,
            columns=[
                "Chromosome",
                "Start",
                "End",
                "Strand",
                "Score",
                "Pvalue",
                "Qvalue",
                "Accession",
                "Name",
            ],
        )
    )
    return dataframe


@track
def jaspar_annots_sdata(
    sdata,
    motif_accs=None,
    motif_names=None,
    collection=None,
    release="JASPAR2020",
    filename="motifs.meme",
    copy=False,
):
    """Annotate SeqData with JASPAR motifs

    Parameters
    ----------
    sdata : eugene.data.SeqData
        SeqData object
    motif_accs : list of str, optional
        List of motif accessions, by default None
    motif_names : list of str, optional
        List of motif names, by default None
    collection : str, optional
        Collection name, by default None
    release : str, optional
        JASPAR release, by default "JASPAR2020"
    filename : str, optional
        MEME filename, by default "motifs.meme"
    copy : bool, optional
        Copy SeqData, by default False

    Returns
    -------
    eugene.data.SeqData
        SeqData object
    """
    sdata = sdata.copy() if copy else sdata
    sdata.pos_annot = score_seqs(
        sdata,
        motif_accs=motif_accs,
        motif_names=motif_names,
        collection=collection,
        release=release,
        filename=filename,
    )
    return sdata if copy else None
