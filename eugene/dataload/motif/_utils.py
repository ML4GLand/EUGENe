import re
import numpy as np
from io import TextIOBase
from ._Motif import Motif
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

def _info_content(pwm, transpose=False, bg_gc=0.415):
    ''' Compute PWM information content.
    In the original analysis, I used a bg_gc=0.5. For any
    future analysis, I ought to switch to the true hg19
    value of 0.415.
    '''
    pseudoc = 1e-9

    if transpose:
        pwm = np.transpose(pwm)

    bg_pwm = [1-bg_gc, bg_gc, bg_gc, 1-bg_gc]

    ic = 0
    for i in range(pwm.shape[0]):
        for j in range(4):
            # ic += 0.5 + pwm[i][j]*np.log2(pseudoc+pwm[i][j])
            ic += -bg_pwm[j]*np.log2(bg_pwm[j]) + pwm[i][j]*np.log2(pseudoc+pwm[i][j])

    return ic