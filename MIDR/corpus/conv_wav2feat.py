"""
    Convertion from MIDI to note sequences

    Based on: https://github.com/sony/hFT-Transformer
"""

from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from MIDR.representations.spec_repr import Spec_LogMel


def wav2feat(wav_path):
    # (1) Read WAV file

    # (2) Convert to feature
    return 0


def feat2wav():
    return 0



if __name__=="__main__":
    wav = Path("./MIDR/test_files/test1.wav")
    log_mel = Spec_LogMel()
    log_mel.visualize(wav, 40, 45)

