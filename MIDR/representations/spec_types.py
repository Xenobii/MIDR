""" 
    Spectrogram classes 
"""

from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torchaudio


class SpecTransforms():
    def __init__(self, config):
        self.config = config
    
    def MelSpectrogram(self, wav):
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.config['feature']['sr'],
                                                    n_fft=self.config['feature']['fft_bins'],
                                                    win_length=self.config['feature']['window_length'],
                                                    hop_length=self.config['feature']['hop_sample'],
                                                    pad_mode=self.config['feature']['pad_mode'],
                                                    n_mels=self.config['feature']['mel_bins'],
                                                    norm='slaney')
        return mel_transform(wav)
    
    def CQT():
        return 0
    
    def HCQT():
        return 0


if __name__=="__main__":
    wav = Path("./MIDR/test_files/test1.wav")
    log_mel = Spec_LogMel()
    log_mel.visualize(wav, 40, 45)