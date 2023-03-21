import subprocess
import numpy as np

def check_homer_installation(path='findMotifsGenome.pl'):
    """ check if Homer is installed """
    try:
        subprocess.check_output(['findMotifsGenome.pl', '-version'])
    except:
        raise Exception('Homer not installed')

def find_motifs(
    seqs, 
    output_path, 
    motif_length=8, 
    motif_number=5, 
    homer_path='findMotifsGenome.pl'
):
    "run homer's findMotifs.pl script on motifs and seqs native to EUGENe"
    cmd = [homer_path, seqs, 'hg19', output_path, '-len', str(motif_length), '-nMotifs', str(motif_number)]

def annotate_peaks(
    peaks, 
    output_path, 
    homer_path='annotatePeaks.pl'
):
    "run homer's annotatePeaks.pl script on peaks native to EUGENe"
    cmd = [homer_path, peaks, 'hg19', '-go', output_path]


    


