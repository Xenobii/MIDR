"""
    Model file
"""
import torchaudio
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa

from pathlib import Path
import json
import torch.nn.functional as F



class AMT():
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
        # (1) Import wav, match to config
        wav, sr = torchaudio.load(wav_path)
        wav = torch.mean(wav, dim=0)
        wav = torchaudio.transforms.Resample(sr, self.config["feature"]["sr"])(wav) # 16000
        
        # (2) Convert to spec and dB
        transform = Transforms(self.config)
        spec = transform.MelSpectrogram(wav)
        spec = torchaudio.transforms.AmplitudeToDB()(spec)

        return spec
    
    def plot_spec(self, spec):
        sr = self.config["feature"]["sr"]
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            spec.numpy(),
            sr=sr,
            cmap="magma"
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spec")
        plt.tight_layout()
        plt.show()
    
    def spec2chunks(self, spec):
        # (1) Pad end to allow smooth splitting into chunks
        num_frame_spec  = spec.shape[1]
        num_frame_chunk = self.config["input"]["num_frame"] # 128
        num_frame_pad   = num_frame_spec % num_frame_chunk
        spec_padded     = F.pad(spec, (0, num_frame_pad, 0, 0), mode="constant", value=0)

        # (2) Pad edges
        margin_left     = self.config["input"]["margin_left"]   # 32
        margin_right    = self.config["input"]["margin_right"]  # 32
        mode            = self.config["feature"]["pad_mode"]    # constant
        spec_padded     = F.pad(spec_padded, (margin_left, margin_right, 0, 0), mode=mode, value=0)
        
        # (3) Split into chunks (prepad chunks)
        chunks = []
        num_chunks = num_frame_spec // num_frame_chunk
        for n in range(0, num_chunks):
            # Trust me this accounts for padding
            chunk = spec_padded[:, 
                                n*num_frame_chunk:
                                margin_left + n*num_frame_chunk + num_frame_chunk + margin_right]
            chunks.append(chunk)

        # plot specs in between for debugging

        return chunks
    
    def chunks2frames(self, chunk):
        # (1) Split into frames
        margin_left = self.config["input"]["margin_left"]   # 32
        margin_right = self.config["input"]["margin_right"] # 32
        frames = []
        for n in range(margin_left, chunk.shape[1] - margin_right):
            # (1) Get the M + 1 + M frames around n
            frame = chunk[:, n-margin_left:n+margin_right+1]
            frames.append(frame)
            print(n)
        
        return frames



class Transforms():
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
    amt = AMT()
    spec = amt.wav2spec(wav)
    chunks = amt.spec2chunks(spec)
    frames = amt.chunks2frames(chunks[0])
    print(spec.shape)
    print(chunks[0].shape, len(chunks))
    print(frames[0].shape, len(frames))
