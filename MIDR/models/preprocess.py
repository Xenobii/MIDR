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
        plt.colorbar(format="%+2.0f dB")
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
    
    def midr2label(self):
        """
            Input   : MIDR list (list of midr events)
            Ouptut  : Label matrices (4 x ((X, Y, Z), L))

            L: Total time bins
            X, Y, Z: 3D coordinates in space
        """
        return 0
    
    def label2chunks(self):
        """
            Input   : Label matrix ((X, Y, Z), L)
            Output  : Array of padded chunks (K x ((X, Y, Z), (N + 2M)))

            X, Y, Z: Coordinates in space
            L: Total time frames
            K: Number of chunks
            N: Number of frames per chunk
            M: Pad margin
        """
        return 0



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



if __name__=="__main__":
    # wav = Path("./MIDR/test_files/test1.wav")
    # processor = SpecPreprocessor()
    # spec = processor.wav2spec(wav)
    # chunks = processor.spec2chunks(spec)
    # frames = processor.chunks2frames(chunks[0])
    # print(spec.shape)
    # print(chunks[0].shape, len(chunks))
    # print(frames[0].shape, len(frames))

    midi_processor = MidiProcessor()
    midi_path = Path("./MIDR/test_files/test_midi.MID")
    events = midi_processor.midi2events(midi_path)
    midr_notes = midi_processor.events2midr(events)
    print(midr_notes)
