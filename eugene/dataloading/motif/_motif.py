import numpy as np
import pandas as pd
import torch
import re
from dataclasses import dataclass
from typing import Optional, Dict
from io import TextIOBase
from ...preprocessing import decode_seq
from ...preprocessing._utils import _token2one_hot

# Taken from https://github.com/tobjores/Synthetic-Promoter-Designs-Enabled-by-a-Comprehensive-Analysis-of-Plant-Core-Promoters/blob/main/CNN/CNN_train%2Bevaluate.ipynb
@dataclass
class Motif:
    identifier: str
    pfm: np.ndarray
    consensus: str
    alphabet_length: int
    length: int
    name: Optional[str] = None
    source_sites: Optional[int] = None
    source_evalue: Optional[float] = None

    def __len__(self) -> int:
        return self.length


class MinimalMEME:
    """http://meme-suite.org/doc/meme-format.html"""

    __version_regex = re.compile("^MEME version ([0-9]+)$")
    __background_regex = re.compile(
        "^Background letter frequencies(?: \(from (.+)\))?$"
    )
    __background_sum_error = 0.00001
    __pfm_header_regex = re.compile(
        "^letter-probability matrix:(?: alength= ([0-9]+))?(?: w= ([0-9]+))?(?: nsites= ([0-9]+))?(?: E= ([0-9.e-]+))?$"
    )
    version = None
    alphabet = None
    strands = None
    background = None
    background_source = None
    motifs = None
    consensus = None

    def __init__(self, path):
        self.motifs = {}

        # parse the minimal MEME file
        with open(path) as minimal_meme_file:
            line = minimal_meme_file.readline()
            # first line must be version
            self.version = self._parse_version(line)

            line = minimal_meme_file.readline()
            while line:
                if line.startswith("ALPHABET"):
                    if self.alphabet is None:
                        self.alphabet = self._parse_alphabet(line)
                        line = minimal_meme_file.readline()
                    else:
                        raise RuntimeError(
                            "Multiple alphabet definitions encountered in MEME file"
                        )
                elif line.startswith("strands: "):
                    if self.strands is None:
                        self.strands = self._parse_strands(line)
                        line = minimal_meme_file.readline()
                    else:
                        raise RuntimeError(
                            "Multiple strand definitions encountered in MEME file"
                        )
                elif line.startswith("Background letter frequencies"):
                    if self.background is None:
                        line = self._parse_background(line, minimal_meme_file)
                    else:
                        raise RuntimeError(
                            "Multiple background frequency definitions encountered in MEME file"
                        )
                elif line.startswith("MOTIF"):
                    line = self._parse_motif(line, minimal_meme_file)
                else:
                    line = minimal_meme_file.readline()

    def _parse_version(self, line: str) -> str:
        match = re.match(self.__version_regex, line)
        if match:
            return match.group(1)
        else:
            raise RuntimeError("Minimal MEME file missing version string on first line")

    def _parse_alphabet(self, line: str) -> str:
        if line.startswith("ALPHABET "):
            raise NotImplementedError("Alphabet definitions not supported")
        elif line.startswith("ALPHABET= "):
            return line.rstrip()[10:]
        else:
            raise RuntimeError("Unable to parse alphabet line")

    def _parse_strands(self, line: str) -> str:
        strands = line.rstrip()[9:]
        if not ((strands == "+") or (strands == "+ -")):
            raise RuntimeError("Invalid strand specification")
        else:
            return strands

    def _parse_background(self, line: str, handle: TextIOBase) -> str:
        match = re.match(self.__background_regex, line)
        if match:
            if match.group(1) is not None:
                self.background_source = match.group(1)
        else:
            raise RuntimeError("Unable to parse background frequency line")

        self.background = {}
        # start parsing possibly multiple lines of background frequencies
        line = handle.readline()
        while line:
            if (not line.rstrip()) or line.startswith("MOTIF"):
                if (
                    abs(1 - sum(self.background.values()))
                    <= self.__background_sum_error
                ):
                    return line
                else:
                    raise RuntimeError("Background frequencies do not sum to 1")
            else:
                line_freqs = line.rstrip().split(" ")
                if len(line_freqs) % 2 != 0:
                    raise RuntimeError("Invalid background frequency definition")
                for residue, freq in zip(line_freqs[0::2], line_freqs[1::2]):
                    self.background[residue] = float(freq)
            line = handle.readline()

    def _parse_motif(self, line: str, handle: TextIOBase) -> str:
        # parse motif identifier
        line_split = line.rstrip().split(" ")
        if (len(line_split) < 2) or (len(line_split) > 3):
            raise RuntimeError("Invalid motif name line")
        motif_identifier = line_split[1]
        motif_name = line_split[2] if len(line_split) == 3 else None

        line = handle.readline()
        # parse letter probability matrix header
        if not line.startswith("letter-probability matrix:"):
            raise RuntimeError(
                "No letter-probability matrix header line in motif entry"
            )
        match = re.match(self.__pfm_header_regex, line)
        if match:
            motif_alphabet_length = (
                int(match.group(1)) if match.group(1) is not None else None
            )
            motif_length = int(match.group(2)) if match.group(2) is not None else None
            motif_source_sites = (
                int(match.group(3)) if match.group(3) is not None else None
            )
            motif_source_evalue = (
                float(match.group(4)) if match.group(4) is not None else None
            )
        else:
            raise RuntimeError("Unable to parse letter-probability matrix header")

        # parse letter probability matrix
        line = handle.readline()
        pfm_rows = []
        while line:
            if (not line.rstrip()) or line.startswith("MOTIF"):
                if motif_identifier in self.motifs:
                    raise RuntimeError("Motif identifiers not unique within file")
                pfm = np.stack(pfm_rows)
                if motif_length is None:
                    motif_length = pfm.shape[0]
                elif motif_length != pfm.shape[0]:
                    raise RuntimeError(
                        "Provided motif length is not consistent with the letter-probability matrix shape"
                    )
                consensus = decode_seq(_token2one_hot(pfm.argmax(axis=1)))
                self.motifs[motif_identifier] = Motif(
                    identifier=motif_identifier,
                    pfm=pfm,
                    consensus=consensus,
                    alphabet_length=motif_alphabet_length,
                    length=motif_length,
                    name=motif_name,
                    source_sites=motif_source_sites,
                    source_evalue=motif_source_evalue,
                )
                return line
            else:
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


def _create_kernel_matrix(
    size: tuple, motifs: Dict[str, Motif], convert_to_pwm=True
) -> np.ndarray:
    if len(size) != 3:
        raise RuntimeError("Kernel matrix size must be a tuple of length 3")
    kernel = torch.zeros(size)
    torch.nn.init.xavier_uniform_(kernel)
    for i, motif_id in enumerate(motifs):
        motif = motifs[motif_id]
        if convert_to_pwm:
            new_weight = torch.tensor(
                motif.pfm[: min(len(motif), kernel.shape[2]), :] / 0.25
            ).transpose(0, 1)
        else:
            new_weight = torch.tensor(
                motif.pfm[: min(len(motif), kernel.shape[2]), :]
            ).transpose(0, 1)
        kernel[i, :, : min(len(motif), kernel.shape[2])] = new_weight
    return kernel
