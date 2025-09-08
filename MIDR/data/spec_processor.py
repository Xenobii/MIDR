"""
    Spec preprocessor

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



def get_centroid(notes):
    if not notes:
        return 0
    
    notes   = np.array(notes)
    coords  = notes[:, :3]
    weights = notes[:, 3]

    if (all(weights) == 0):
        return 0
    
    centroid    = np.average(coords, axis=0, weights=weights)
    weight      = np.average(weights, axis=0)
    return np.hstack([centroid, weight])


def get_diameter(notes):
    if not notes:
        return -1
    
    notes = np.array(notes)
    coords = notes[:, :3]

    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)

    diameter = np.max(dists)
    return diameter


def plot_spec_and_label(spec, label, sr=16000, hop_length=256):
    """
    Plot spectrogram and label aligned in time on the x-axis
    as two separate subplots.
    """
    if not isinstance(label, torch.Tensor):
        label = torch.from_numpy(label)

    # Reduce label to (frames, notes) shape if needed
    if label.ndim == 4:  
        label = label[:, 0, 0, :].T

    # Get dimensions
    n_mels, n_frames = spec.shape
    _, label_frames = label.shape
    assert n_frames == label_frames, f"Mismatch: spec has {n_frames} frames, label has {label_frames}"

    # Time axis (in seconds) for alignment
    times = np.arange(n_frames) * hop_length / sr
    duration = times[-1]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # --- Spectrogram ---
    img = librosa.display.specshow(
        spec.numpy(),
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax1
    )
    ax1.set_title("Spectrogram")
    # fig.colorbar(img, ax=ax1, format="%+2.0f dB")

    # --- Label ---
    ax2.imshow(
        label.numpy(),
        aspect="auto",
        origin="lower",
        cmap="Greys",
        extent=[0, duration, 0, label.shape[1]]  # stretch to same time axis
    )
    ax2.set_title("Label")
    ax2.set_ylabel("Notes")
    ax2.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()



if __name__=="__main__":
    wav = Path("./MIDR/test_files/test1.wav")
    spec_processor = SpecProcessor()
    spec = spec_processor.wav2spec(wav)
    chunks = spec_processor.spec2chunks(spec)
    frames = spec_processor.chunks2frames(chunks[0])
    print(len(chunks))