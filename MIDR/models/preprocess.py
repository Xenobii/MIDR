"""
    Model file

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

from MIDR.representations.spec_types import SpecTransforms
from MIDR.representations.note_types import MidiTransforms, MidiNote


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


class MidiProcessor():
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
    
    def midi2events(self, midi_path, supportedCC = [64, 66, 67], extendSustainPedal=True):
        """
            Input   : Midi file
            Output  : Event list (list of note events)
        """
        # (1) Open midi file
        try:
            midi_file = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as e:
            print(f"Error opening file: {e}")

        # (2) Assert single instrument
        assert(len(midi_file.instruments) == 1)
        if len(midi_file.instruments) > 1:
            raise Exception("Provided midi contains multiple instruments")
        
        i = midi_file.instruments[0]

        # (3) Get note events
        note_events = [MidiNote(n.start, n.end, n.pitch, n.velocity) for n in i.notes]
        note_events.sort(key = lambda x: (x.onset, x.offset, x.pitch))
        endT = max([n.offset for n in note_events]) # final offset in sequence

        # (3.1) If extendSustainPedal, extend notes to its duration
        if extendSustainPedal:
            ccSeq = pedal2list(i.control_changes, ccNum = 64, onThreshold = 64, endT = endT)
            ccSeq.sort(key = lambda x: (x.onset, x.offset, x.pitch))
            note_events.sort(key = lambda x: (x.onset, x.offset, x.pitch))
            note_events = extend_pedal(note_events, ccSeq)
        
        # (4) Resolve overlapping and validate
        note_events = resolve_overlapping(note_events)
        validate_notes(note_events)

        eventSeqs = [note_events]
        
        # (5) Get pedal events
        for ccNum in supportedCC:
            ccSeq = pedal2list(i.control_changes, ccNum, onThreshold = 64, endT = endT)
            ccSeq.sort(key = lambda x: (x.onset, x.offset, x.pitch))
            eventSeqs.append(ccSeq)

        # (6) Name sequences
        events = {"notes"       : eventSeqs[0],
                  "sustain"     : eventSeqs[1],
                  "sostenuto"   : eventSeqs[2],
                  "soft"        : eventSeqs[3]}

        return events
    
    def events2midr(self, events, repr_type="standard", delta=0):
        """
            Input   : Note event list (list of note events)
            Output  : MIDR list (list of midr events)
        """
        # (1) Retrieve events separately
        midi_notes  = events["notes"]
        sustain     = events["sustain"]

        # (2) Iterate through notes and convert to midr
        midr_notes = []
        for midi_note in midi_notes:
            midr_note = MidiTransforms(midi_note, repr_type=repr_type, delta=delta)
            midr_notes.append(midr_note.to_dict())
        
        # (3) Parse events
        midrep = {"midrep_notes": midr_notes,
                  "sustain"     : sustain,
                  "type"        : repr_type}
        
        return midrep
    
    def midr2label(self, midr, use_offset_duration_tolerance=False):
        """
            Input   : MIDR list (list of midr events)
            Ouptut  : Label matrices (4 x ((X, Y, Z), L))

            L: Total time bins
            X, Y, Z: 3D coordinates in space
        """
        # (1) Calculate settings

        sr          = self.config["feature"]["sr"]          # 16000
        hop_sample  = self.config["feature"]["hop_sample"]  # 256

        frames_per_sec  = sr / hop_sample           # 62.5
        hop_ms          = 1000 * hop_sample / sr    # 16

        onset_tolerance     = int(50.0 / hop_ms + 0.5)  # 50 ms
        offset_tolerance    = int(50.0 / hop_ms + 0.5)  # 50 ms

        # (2) Get length of labels matrices (we'll pad later)
        max_offset = 0
        midr_notes = midr["midrep_notes"]
        for note in midr_notes:
            if max_offset < note["offset"]:
                max_offset = note["offset"]
        nframe = int(max_offset * frames_per_sec + 0.5) + 1

        # (2) Create label matrices
        # matrix dimensions are temporary, must convert to 3d
        mpe_label       = np.zeros((nframe, 10, 10, 100), dtype=np.bool)
        onset_label     = np.zeros((nframe, 10, 10, 100), dtype=np.float32)
        offset_label    = np.zeros((nframe, 10, 10, 100), dtype=np.float32)
        velocity_label  = np.zeros((nframe, 10, 10, 100), dtype=np.int8)

        # (3) Fill label matrices 
        for note in midr_notes:
            x           = int(note["x"] + 0.5)
            y           = int(note["y"] + 0.5)
            z           = int(note["z"] + 0.5)
            onset       = note["onset"]
            offset      = note["offset"]
            velocity    = note["velocity"]

            # (3.1) Calculate frame for onset
            onset_frame     = int(onset * frames_per_sec + 0.5)
            onset_ms        = onset * 1000.0
            onset_sharpness = onset_tolerance   # 3 frames

            # (3.2) Calculate frame for onset
            offset_frame     = int(offset * frames_per_sec + 0.5)
            offset_ms        = offset * 1000.0
            offset_sharpness = offset_tolerance # 3 frames

            # (3.3) If offset_tolerance_flag, the offset is calculated based
            #       on the duration of the note itself (20% of the note's duration)
            if use_offset_duration_tolerance:
                offset_duration_tolerance = int((offset_ms - onset_ms) * 0.2 / hop_ms + 0.5)
                offset_sharpness = max(offset_tolerance, offset_duration_tolerance)

            # (3.4) Spread onset label along a window around the onset frame
            #       If onset value is over 0.5, add velocity to that position

            # (3.4.1) Look ahead
            for j in range(0, onset_sharpness+1):
                onset_ms_q      = (onset_frame + j) * hop_ms
                onset_ms_diff   = onset_ms_q - onset_ms                                             # 16 * j    (I did the math)
                onset_val       = max(0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms))) # 1 - j/3   (I did it again)
                
                if onset_frame + j < nframe:
                    onset_label[onset_frame+j][x][y][z] = max(onset_label[onset_frame+j][x][y][z], onset_val)

                    if (onset_label[onset_frame+j][x][y][z] >= 0.5):
                        velocity_label[onset_frame+j][x][y][z] = velocity
            # (3.4.2) Look behind
            for j in range(1, onset_sharpness+1):
                onset_ms_q      = (onset_frame - j) * hop_ms
                onset_ms_diff   = onset_ms_q - onset_ms
                onset_val       = max(0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms)))
                
                if onset_frame - j >= 0:
                    onset_label[onset_frame-j][x][y][z] = max(onset_label[onset_frame-j][x][y][z], onset_val)
                    
                    if (onset_label[onset_frame-j][x][y][z] >= 0.5):
                        velocity_label[onset_frame-j][x][y][z] = 1

            # (3.6) Fill out mpe
            for j in range(onset_frame, offset_frame + 1):
                mpe_label[j][x][y][z] = 1

            # (3.7) Fill out offset
            #       Ignore if a note at the same position has an onset 
            offset_flag = True
            for _note in midr_notes:
                if _note["pitch"] != note["pitch"]:
                    continue
                if _note["offset"] == note["onset"]:
                    offset_flag = False
                    break
                         
            if offset_flag:
                for j in range(0, offset_sharpness+1):
                    offset_ms_q      = (offset_frame + j) * hop_ms
                    offset_ms_diff   = offset_ms_q - offset_ms                                             # 16 * j    (I did the math)
                    offset_val       = max(0.0, 1.0 - (abs(offset_ms_diff) / (offset_sharpness * hop_ms))) # 1 - j/3   (I did it again)
                    offset_val = 1
                    
                    if offset_frame + j < nframe:
                        offset_label[offset_frame+j][x][y][z] = max(offset_label[offset_frame+j][x][y][z], offset_val)

                for j in range(1, offset_sharpness+1):
                    offset_ms_q      = (offset_frame - j) * hop_ms
                    offset_ms_diff   = offset_ms_q - offset_ms
                    offset_val       = max(0.0, 1.0 - (abs(offset_ms_diff) / (offset_sharpness * hop_ms)))
                    
                    if offset_frame - j >= 0:
                        offset_label[offset_frame-j][x][y][z] = max(offset_label[offset_frame-j][x][y][z], offset_val)
        
        # (4) Return label files
        # mpe        : 0 or 1
        # onset      : 0.0-1.0
        # offset     : 0.0-1.0
        # velocity   : 0 - 127
        return mpe_label, onset_label, offset_label, velocity_label
    
    def label2chunks(self, label):
        """
            Input   : Label matrix ((X, Y, Z), L)
            Output  : Array of padded chunks (K x ((X, Y, Z), (N + 2M)))

            X, Y, Z: Coordinates in space
            L: Total time frames
            K: Number of chunks
            N: Number of frames per chunk
            M: Pad margin
        """
        # (1) Pad end to allow smooth splitting into chunks
        num_frame_label = label.shape[0]
        num_frame_chunk = self.config["input"]["num_frame"] # 128
        num_chunks      = math.ceil(num_frame_label / num_frame_chunk)
        num_frame_pad   = num_chunks * num_frame_chunk - num_frame_label

        label_2d        = label[:, 0, 0, :].T               # Temp logic
        label_2d        = torch.from_numpy(label_2d)        # Do this more cleanly elsewhere
        label_padded    = F.pad(label_2d, (0, num_frame_pad, 0, 0), mode="constant",)

        # (2) Pad edges
        margin_left     = self.config["input"]["margin_left"]   # 32
        margin_right    = self.config["input"]["margin_right"]  # 32
        mode            = self.config["feature"]["pad_mode"]    # constant
        label_padded    = F.pad(label_padded, (margin_left, margin_right, 0, 0), mode=mode, value=0)

        # (3) Split into chunks (prepaded)
        chunks = []
        for n in range(0, num_chunks):
            chunk = label_padded[:,
                                 n*num_frame_chunk:
                                 margin_left + n*num_frame_chunk + num_frame_chunk + margin_right]
            chunks.append(chunk)

        return chunks

    
    def plot_label(self, label):
        """
            Helper function for plotting a label after collapsing it to 2D (P x N)
        """
        if not isinstance(label, torch.Tensor):
            label = label[:, 0, 0, :].T
        
        plt.imshow(label, aspect="auto", origin="lower", cmap="magma")
        plt.title("Label")
        plt.show()



def pedal2list(ccSeq, ccNum, onThreshold=64, endT = None):
    """
    Get pedal note events

    Args:
        ccSeq (prettyMidi.control_changes)
        ccNum (int): Pedal number
        onThreshold (int, optional). Defaults to 64.
        endT (float32, optional): Final offset in the sequence. Defaults to None.

    Returns:
        List(StdNote objects): The list of pedal events as negative note events
    """
    currentStatus = False
    runningStatus = False

    currentEvent = None
    seqEvent = []

    time = 0

    for c in ccSeq:
        # Get status (on/off)
        if c.number == ccNum:
            time = c.time
            if c.value>=onThreshold:
                currentStatus = True
            else:
                currentStatus = False
                
        # Find onset and offset position
        if runningStatus != currentStatus:
            if currentStatus == True:
                # Use negative number as pitch for control change event
                currentEvent = MidiNote(time, None, -ccNum, 127)
            else:
                currentEvent.offset = time
                seqEvent.append(currentEvent)

        runningStatus = currentStatus

    # Process the case where state is not closed off at the offset
    if runningStatus and endT is not None:
        currentEvent.offset = max(endT, time)
        if currentEvent.offset > currentEvent.onset:
            seqEvent.append(currentEvent)

    return seqEvent


def resolve_overlapping(note_list):
    """
    Resolve overlapping note segments by slicing off the offset of overlapping notes

    Args:
        note_list (StdNote objects)

    Returns:
        List(StdNote objects): The corrected list of notes 
    """
    
    buffer_dict  = {}
    ex_notes = []
    idx = 0

    # For all overlapping notes of the same pitch, slice the offset of the first note
    for note in note_list:
        pitch = note.pitch

        if pitch in buffer_dict.keys():
            _idx = buffer_dict[pitch]
            if ex_notes[_idx].offset > note.onset:
                ex_notes[_idx].offset = note.onset

        buffer_dict[pitch] = idx
        idx += 1

        ex_notes.append(note)

    ex_notes.sort(key = lambda x: (x.onset, x.offset, x.pitch))

    # Detect errors
    error_notes = [n for n in ex_notes if not n.onset<n.offset]
    if len(error_notes) > 0:
        warnings.warn("There are error notes in given midi")

    return ex_notes


def extend_pedal(note_events, pedal_events):
    """
    Extend notes if sustain pedal is on 

    Args:
        note_events  (List(StdNote Object))
        pedal_events (List(StdNote Object))

    Returns:
        List(StdNote Object): List of adjusted notes 
    """
    ex_notes = []

    idx = 0

    buffer_dict = {}
    nIn = len(note_events)

    for note in note_events:
        pitch = note.pitch
        if pitch in buffer_dict.keys():
            _idx = buffer_dict[pitch]
            if ex_notes[_idx].offset > note.onset:
                ex_notes[_idx].offset = note.onset

        for pedal in pedal_events:
            if note.offset < pedal.offset and note.offset>pedal.onset:
                note.offset = pedal.offset
        
        buffer_dict[note] = idx
        idx += 1
        ex_notes.append(note)

    ex_notes.sort(key = lambda x: (x.onset, x.offset, x.pitch))

    nOut = len(ex_notes)
    assert(nOut == nIn)

    return ex_notes


def validate_notes(notes):
    """
    Validate that note events don't overlap and no notes offset before they onset

    Args:
        notes (List(StdNote object)): A list of StdNote objects
    """
    pitches = defaultdict(list)
    for n in notes:
        if len(pitches[n.pitch])>0:
            # Make sure no notes overlap
            nPrev = pitches[n.pitch][-1]
            assert n.onset >= nPrev.offset, str(n) + str(nPrev)
        # Make sure no notes offset before they onset
        assert n.onset < n.offset, n

        pitches[n.pitch].append(n)


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
    wav = Path("./MIDR/test_files/test_midi_maestro.wav")
    spec_processor = SpecProcessor()
    spec = spec_processor.wav2spec(wav)
    chunks = spec_processor.spec2chunks(spec)
    frames = spec_processor.chunks2frames(chunks[0])
    print(len(chunks))

    midi_processor = MidiProcessor()
    midi_path = Path("./MIDR/test_files/test_midi_maestro.midi")
    events = midi_processor.midi2events(midi_path)
    midr_notes = midi_processor.events2midr(events)
    mpe_label, onset_label, offset_label, velocity_label = midi_processor.midr2label(midr_notes)
    offset_chunks = midi_processor.label2chunks(offset_label)
    [print(len(offset_chunks))]

    # spec_processor.plot_spec(chunks[102])
    # midi_processor.plot_label(mpe_chunks[0])

    plot_spec_and_label(chunks[101], offset_chunks[101])

    
