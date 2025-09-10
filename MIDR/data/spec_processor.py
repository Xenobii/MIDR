"""
    Spec Processor

    Preprocessing algorithm is based on this: https://github.com/sony/hFT-Transformer
"""

import torchaudio
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pretty_midi
import json
import warnings
import math

from pathlib import Path
from collections import defaultdict
import torch.nn.functional as F

from MIDR.data.spec_types import SpecTransforms
from MIDR.data.note_types import MidiTransforms, MidiNote


class SpecProcessor():
    def __init__(self):
        # Add config here for now
        self.config = {
            "feature": {
                "sr"            : 16000,
                "fft_bins"      : 2048,
                "window_length" : 2048,
                "hop_sample"    : 256,
                "mel_bins"      : 256,
                "pad_mode"      : "constant"
            },
            "input" : {
                "margin_left"   : 32,
                "margin_right"  : 32,
                "num_frame"     : 128
            }
        }
    
    
    def wav2spec(self, wav_path):
        """
            Input   : WAV (2, L)
            Output  : Spec (C, F, N)

            T: Audio samples
            C: Channels
            F: Frequency bins
            L: Total time frames
        """
        # (1) Import wav, match to config
        wav, sr = torchaudio.load(wav_path)
        wav = torch.mean(wav, dim=0)
        wav = torchaudio.transforms.Resample(sr, self.config["feature"]["sr"])(wav) # 16000
        
        # (2) Convert to spec and dB
        transform = SpecTransforms(self.config)
        spec = transform.MelSpectrogram(wav)
        spec = torchaudio.transforms.AmplitudeToDB()(spec)

        return spec
    
    def augment_spec(self, spec):
        return 0
    
    def plot_spec(self, spec):
        """
            Helper function for plotting a spectrogram (F x N)
        """
        sr = self.config["feature"]["sr"]
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            spec.numpy(),
            sr=sr,
            cmap="magma"
        )
        # plt.colorbar(format="%+2.0f dB")
        plt.title("Spec")
        plt.tight_layout()
        plt.show()
    
    def spec2chunks(self, spec):
        """
            Input   : Spec (C, F, L)
            Output  : Array of padded chunks (K x (C, F, (N + 2M)))

            C: Channels
            F: Frequency bins
            L: Total time frames
            K: Number of chunks
            N: Number of frames per chunk
            M: Pad margin
        """
        # (1) Pad end to allow smooth splitting into chunks
        pad_value = -80.0   # It's a dB spectrogram

        num_frame_spec  = spec.shape[1]
        num_frame_chunk = self.config["input"]["num_frame"] # 128
        num_chunks      = math.ceil(num_frame_spec / num_frame_chunk)
        num_frame_pad   = num_chunks * num_frame_chunk - num_frame_spec
        spec_padded     = F.pad(spec, (0, num_frame_pad, 0, 0), mode="constant", value=pad_value)

        # (2) Pad edges
        margin_left     = self.config["input"]["margin_left"]   # 32
        margin_right    = self.config["input"]["margin_right"]  # 32
        mode            = self.config["feature"]["pad_mode"]    # constant
        spec_padded     = F.pad(spec_padded, (margin_left, margin_right, 0, 0), mode=mode, value=pad_value)
        # (3) Split into chunks (prepad chunks)
        chunks = []
        for n in range(0, num_chunks):
            # Trust me this accounts for padding
            chunk = spec_padded[:, 
                                n*num_frame_chunk:
                                margin_left + n*num_frame_chunk + num_frame_chunk + margin_right]
            chunks.append(chunk)

        return chunks
    
    def chunks2frames(self, chunk):
        """
            Input   : Padded chunk (C, F, (N + 2M))
            Output  : Array of frame features (N x (C, F, (M + 1 + M)))

            C: Channels
            F: Frequency bins
            N: Number of frames per chunk
            M: Pad margin
        """
        # (1) Split into frames
        margin_left = self.config["input"]["margin_left"]   # 32
        margin_right = self.config["input"]["margin_right"] # 32
        frames = []
        for n in range(margin_left, chunk.shape[1] - margin_right):
            # (1.1) Get the M + 1 + M frames around n
            frame = chunk[:, n-margin_left:n+margin_right+1]
            frames.append(frame)
        
        return frames



if __name__=="__main__":
    wav = Path("./MIDR/test_files/test1.wav")
    spec_processor = SpecProcessor()
    spec = spec_processor.wav2spec(wav)
    chunks = spec_processor.spec2chunks(spec)
    frames = spec_processor.chunks2frames(chunks[0])
    print(len(chunks))