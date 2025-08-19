""" 
    Spectrogram classes 
"""

from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


class Spec_LogMel:
    def __init__(self,
                 n_fft = 1024,
                 hop_length = 256,
                 win_length = 1024,
                 sr = 22050,
                 window = "hann",
                 n_mels = 128,
                 fmin = 20,
                 fmax = 11025):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sr = sr
        self.window = window
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def wav_to_spec(self, wav_path):
        y, sr = librosa.load(wav_path, sr=self.sr)
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
    
    def visualize(self, wav_path, start=0, end=5):
        log_mel_spec = self.wav_to_spec(wav_path)

        time = librosa.frames_to_time(
            np.arange(log_mel_spec.shape[1]),
            sr=self.sr,
            hop_length=self.hop_length)

        if end is None:
            end = time[-1]
        mask = (time >= start) & (time <= end)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            log_mel_spec[:, mask],
            sr=self.sr,
            hop_length=self.hop_length,
            x_axis="time",
            y_axis="mel",
            fmin=self.fmin,
            fmax=self.fmax
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Log Mel Spectrogram")
        plt.tight_layout()
        plt.show()


class Spec_STFT:
    def __init__(self):
        return 0


class Spec_CQT:
    def __init__(self):
        return 0
    

class Spec_HCQT:
    def __init__(self):
        return 0


class Spec_Chroma:
    def __init__(self):
        return 0
    

class Spec_Torus:
    def _init__(self):
        return 0
    

class Spec_Helix:
    def __init__(self):
        return 0
    

if __name__=="__main__":
    wav = Path("./MIDR/test_files/test1.wav")
    log_mel = Spec_LogMel()
    log_mel.visualize(wav, 40, 45)