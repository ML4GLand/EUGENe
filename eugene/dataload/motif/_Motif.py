import numpy as np
from typing import Optional, Dict, Iterator


class Motif:
    """
    Motif class for storing motif information.

    Adapted from https://github.com/tobjores/Synthetic-Promoter-Designs-Enabled-by-a-Comprehensive-Analysis-of-Plant-Core-Promoters/blob/main/CNN/CNN_train%2Bevaluate.ipynb
    """

    def __init__(
        self,
        identifier: str,
        pfm: np.ndarray,
        consensus: str,
        length: int,
        alphabet_length: int = None,
        name: Optional[str] = None,
    ):
        self.identifier = identifier
        self.pfm = pfm
        self.consensus = consensus
        self.alphabet_length = alphabet_length
        self.length = length
        self.name = name

    def __len__(self) -> int:
        return self.length

    def __str__(self) -> str:
        output = "Motif %s" % self.identifier
        if self.name is not None:
            output += " (%s)" % self.name
        output += " with %d positions" % (
            self.length,
        )
        return output

    def __repr__(self) -> str:
        return self.__str__()


class MotifSet:
    """
    Stores a set of Motifs.

    Adapted from https://github.com/tobjores/Synthetic-Promoter-Designs-Enabled-by-a-Comprehensive-Analysis-of-Plant-Core-Promoters/blob/main/CNN/CNN_train%2Bevaluate.ipynb
    MEME format: http://meme-suite.org/doc/meme-format.html
    """
    def __init__(
        self,
        motifs: Dict[str, Motif] = {},
        alphabet: Optional[str] = None,
        version: Optional[str] = None,
        strands: Optional[str] = None,
        background: Optional[str] = None,
        background_source: Optional[str] = None,
    ) -> None:

        self.motifs = motifs
        self.alphabet = alphabet
        self.version = version
        self.strands = strands
        self.background = background
        self.background_source = background_source

    def add_motif(self, motif: Motif) -> None:
        self.motifs[motif.identifier] = motif

    def __str__(self) -> str:
        return "MotifSet with %d motifs" % len(self.motifs)
    
    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.motifs)

    def __getitem__(self, key: str) -> Motif:
        return self.motifs[key]

    def __iter__(self) -> Iterator[Motif]:
        return iter(self.motifs.values())

    def __contains__(self, key: str) -> bool:
        return key in self.motifs
