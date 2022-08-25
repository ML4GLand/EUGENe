import numpy as np
import pandas as pd
from pyjaspar import jaspardb
import pyranges as pr
from pymemesuite.common import MotifFile, Sequence
from pymemesuite.fimo import FIMO
#from ...utils import track


def get_jaspar_motifs(
    motif_accs=None,
    motif_names=None,
    collection=None,
    release='JASPAR2022'
):
    assert motif_accs or motif_names or collection, "Must provide either motif_accs, motif_names, or collection"
    jdb_obj = jaspardb(release=release)
    if motif_accs:
        motifs = [jdb_obj.fetch_motif_by_id(acc) for acc in motif_accs]
    elif motif_names:
        motifs = [motif for name in motif_names for motif in jdb_obj.fetch_motifs_by_name(name)]
    elif collection:
        motifs = jdb_obj.fetch_motifs(
            collection = collection,
            tax_group = ['vertebrates']
        )
    return motifs


def save_motifs_as_meme(
    jaspar_motifs,
    filename
):
    meme_file = open(filename, 'w')
    meme_file.write("MEME version 4 \n")
    print(f'Saved PWM File as : {filename}')
    for motif in jaspar_motifs:
        acc = motif.base_id
        name = motif.name
        pwm = np.array(list(motif.pwm.values()))
        filter_size = pwm.shape[1]
        meme_file.write("\n")
        meme_file.write(f"MOTIF {acc} {name}\n")
        meme_file.write(
            "letter-probability matrix: alength= 4 w= %d \n" % np.count_nonzero(np.sum(pwm[:, :], axis=0)))
        for j in range(0, filter_size):
            if np.sum(pwm[:, j]) > 0:
                meme_file.write(
                    str(pwm[0, j]) + "\t" + 
                    str(pwm[1, j]) + "\t" + 
                    str(pwm[2, j]) + "\t" + 
                    str(pwm[3, j]) + "\n"
                )
    meme_file.close()


def load_meme(filename):
    memesuite_motifs = []
    with MotifFile(filename) as motif_file:
        for motif in motif_file:
            memesuite_motifs.append(motif)
        bg = motif_file.background
    return memesuite_motifs, bg


def fimo_motifs(
    sdata,
    pymeme_motifs,
    background
):
    pymeme_seqs = [Sequence(str(seq), name.encode()) for seq, name in zip(sdata.seqs, sdata.names)]
    fimo = FIMO(both_strands=True)
    motif_scores = []
    for motif in pymeme_motifs:
        pattern = fimo.score_motif(motif, pymeme_seqs, background)
        for m in pattern.matched_elements:
            motif_scores.append([
                m.source.accession.decode(),
                m.start,
                m.stop,
                m.strand,
                m.score,
                m.pvalue,
                m.qvalue,
                motif.accession.decode(),
                motif.name.decode()
            ])
    return motif_scores


def score_seqs(
    sdata,
    motif_accs=None,
    motif_names=None,
    collection=None,
    release='JASPAR2020',
    filename="motifs.meme"
):
    assert motif_accs or motif_names or collection, "Must provide either motif_accs, motif_names, or collection"
    motifs = get_jaspar_motifs(
        motif_accs=motif_accs,
        motif_names=motif_names,
        collection=collection,
        release=release
    )
    save_motifs_as_meme(motifs, filename)
    memesuite_motifs, bg = load_meme(filename)
    scores = fimo_motifs(sdata, memesuite_motifs, bg)
    dataframe = pr.PyRanges(pd.DataFrame(scores, columns=['Chromosome', 'Start', 'End', 'Strand', 'Score', 'Pvalue', 'Qvalue', 'Accession', 'Name']))
    return dataframe

#@track
def jaspar_annots_sdata(
    sdata,
    motif_accs=None,
    motif_names=None,
    collection=None,
    release='JASPAR2020',
    filename="motifs.meme",
    copy=False
):
    sdata = sdata.copy() if copy else sdata
    sdata.pos_annot = score_seqs(
        sdata,
        motif_accs=motif_accs,
        motif_names=motif_names,
        collection=collection,
        release=release,
        filename=filename
    )
    return sdata if copy else None