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

from MIDR.data.spec_types import SpecTransforms
from MIDR.data.note_types import MidiTransforms, MidiNote



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
    
    def midr2frame(self, midr, use_offset_duration_tolerance=True):
        """
            Input   : MIDR list (list of midr events)
            Ouptut  : Note Arrays (4 x (Array x L))

            L: Total time bins
            Array: Array of notes [x, y, z, velocity]
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

        # (3) Create label matrices
        mpe_array       = [[] for _ in range(nframe)]
        onset_array     = [[] for _ in range(nframe)]
        offset_array    = [[] for _ in range(nframe)]
        velocity_array  = [[] for _ in range(nframe)]

        # (4) Fill label matrices 
        for note in midr_notes:
            x           = note["x"]
            y           = note["y"]
            z           = note["z"]
            onset       = note["onset"]
            offset      = note["offset"]
            velocity    = note["velocity"]

            # (4.1) Calculate onset frame
            onset_frame     = int(onset * frames_per_sec + 0.5)
            onset_ms        = onset * 1000.0
            onset_sharpness = onset_tolerance   # 3 frames

            # (4.2) Calculate offset frame
            offset_frame     = int(offset * frames_per_sec + 0.5)
            offset_ms        = offset * 1000.0
            offset_sharpness = offset_tolerance # 3 frames

            # (4.3) If offset_tolerance_flag, the offset is calculated based
            #       on the duration of the note itself (20% of the note's duration)
            if use_offset_duration_tolerance:
                offset_duration_tolerance = int((offset_ms - onset_ms) * 0.2 / hop_ms + 0.5)
                offset_sharpness = max(offset_tolerance, offset_duration_tolerance)

            # (4.4) Spread onset label along a window around the onset frame
            #       If onset value is over 0.5, add velocity to that position

            # (4.4.1) Look ahead and update weighted average
            for j in range(0, onset_sharpness+1):
                onset_ms_q      = (onset_frame + j) * hop_ms
                onset_ms_diff   = onset_ms_q - onset_ms                                             # 16 * j    (I did the math)
                onset_val       = max(0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms))) # 1 - j/3   (I did it again)
                
                if onset_frame + j < nframe:
                    # if onset_label[onset_frame+j, 3] < onset_val:
                    onset_array[onset_frame+j].append([x, y, z, onset_val])

                    if onset_val >= 0.5:
                        velocity_array[onset_frame+j].append([x, y, z, velocity])

            # (4.4.2) Look behind
            for j in range(1, onset_sharpness+1):
                onset_ms_q      = (onset_frame - j) * hop_ms
                onset_ms_diff   = onset_ms_q - onset_ms
                onset_val       = max(0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms)))
                
                if onset_frame - j >= 0:
                    # if onset_label[onset_frame-j, 3] < onset_val:
                    onset_array[onset_frame-j].append([x, y, z, onset_val])
                    if onset_val >= 0.5:
                        velocity_array[onset_frame-j].append([x, y, z, velocity])

            # (4.6) Fill out mpe
            for j in range(onset_frame, offset_frame + 1):
                mpe_array[j].append([x, y, z, velocity])

            # (4.7) Fill out offset
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
                    
                    if offset_frame + j < nframe:
                        # if offset_label[offset_frame+j, 3] < offset_val:
                        offset_array[offset_frame+j].append([x, y, z, offset_val])

                for j in range(1, offset_sharpness+1):
                    offset_ms_q      = (offset_frame - j) * hop_ms
                    offset_ms_diff   = offset_ms_q - offset_ms
                    offset_val       = max(0.0, 1.0 - (abs(offset_ms_diff) / (offset_sharpness * hop_ms)))
                    
                    if offset_frame - j >= 0:
                        # if offset_label[offset_frame-j, 3] < offset_val:
                        offset_array[offset_frame-j].append([x, y, z, offset_val])

        # (5) Return arrays
        return mpe_array, onset_array, offset_array, velocity_array
    

    def frame2notes(self, array):
        """
            Input   : Note Array(L, Array([x, y, z, vel]))
            Output  : Note label matrix (L, (X, Y, Z))
        """
        nframe  = len(array)
        label   = np.zeros((nframe, 1, 1, 88), dtype=np.int8) # Temp logic

        for i in range(0, nframe):
            for note in array[i]:
                x = int(note[0] + 0.5)
                y = int(note[1] + 0.5)
                z = int(note[2] + 0.5)
                velocity = note[3]
                label[i, x, y, z] = velocity

        return label
    

    def frame2center(self, array):
        """
            Input   : Note Array(L, Array([x, y, z, vel]))
            Output  : Cloud center matrix (L, [x, y, z, vel, diam])
        """
        nframe = len(array)
        center = np.zeros((nframe, 5), dtype=np.float32)

        for i in range(0, nframe):
            center[i, :4]    = get_centroid(array[i])
            center[i, 4]     = get_diameter(array[i])
        return center

    def label2chunks(self, label):
        """
            Input   : Label matrix (L, 5)
            Output  : Array of padded chunks (K x ((N + 2M), [x, y, z, vel, cm]))

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

        label           = torch.from_numpy(label)        # Do this more cleanly elsewhere
        label_padded    = F.pad(label, (0, 0, 0, num_frame_pad), mode="constant", value = 0)

        # (2) Pad edges
        margin_left     = self.config["input"]["margin_left"]   # 32
        margin_right    = self.config["input"]["margin_right"]  # 32
        mode            = self.config["feature"]["pad_mode"]    # constant
        label_padded    = F.pad(label_padded, (0, 0, margin_left, margin_right), mode=mode, value=0)

        # (3) Split into chunks (prepaded)
        chunks = []
        for n in range(0, num_chunks):
            chunk = label_padded[n*num_frame_chunk:
                                 margin_left + n*num_frame_chunk + num_frame_chunk + margin_right,
                                 :]
            chunks.append(chunk)

        return chunks
    
    
    def plot_label(self, label):
        """
            Helper function for plotting a label after collapsing it to 2D (P x N)
        """
        nframe = label.shape[0]
        grid = np.zeros((nframe, 88))

        for i in range(nframe):
            z           = int(label[i, 2] + 0.5)
            velocity    = label[i, 3]
            diameter    = label[i, 4]
            grid[i, z]  = diameter

        plt.imshow(grid.T, aspect='auto', origin='lower', cmap='magma')
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


if __name__=="__main__":
    midi_processor = MidiProcessor()
    midi_path = Path("./MIDR/test_files/test_midi.mid")
    events = midi_processor.midi2events(midi_path)
    midr_notes = midi_processor.events2midr(events, "sna")
    mpe_label, onset_label, offset_label, velocity_label = midi_processor.midr2frame(midr_notes)
    mpe_label = midi_processor.frame2center(mpe_label)
    mpe_chunks = midi_processor.label2chunks(mpe_label)
    midi_processor.plot_label(mpe_label)