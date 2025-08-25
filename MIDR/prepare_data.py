"""
    Compose corpus
"""

import numpy as np
from pathlib import Path
import argparse

from MIDR.corpus.conv_midi2events import midi2note
from MIDR.corpus.conv_evnt2mirepr import note2helix, note2torus
from corpus.conv_repr2label import midr2label
from corpus.conv_wav2feat import wav2feat


def build_corpus():
    return 0


if __name__=="__main__":
    wav = Path("./MIDR/test_files/test1.wav")