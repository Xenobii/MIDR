"""
    Midi Processor

    Preprocessing algorithm is based on this: https://github.com/sony/hFT-Transformer
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import json
import warnings
import math

from pathlib import Path
from collections import defaultdict
import torch.nn.functional as F

from MIDR.data.note_types import MidiTransforms, MidiNote


class MidiProcessor():
    def __init__(self, config, repr_type="standard"):
        self.config     = config
        self.repr_type  = repr_type
    
    # PREPROCESSING
    
    def midi2evnt(self, midi_path, supportedCC = [64, 66, 67], extendSustainPedal=False):
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
            ccSeq = self.pedal2list(i.control_changes, ccNum = 64, onThreshold = 64, endT = endT)
            ccSeq.sort(key = lambda x: (x.onset, x.offset, x.pitch))
            note_events.sort(key = lambda x: (x.onset, x.offset, x.pitch))
            note_events = self.extend_pedal(note_events, ccSeq)
        
        # (4) Resolve overlapping and validate
        note_events = self.resolve_overlapping(note_events)
        self.validate_notes(note_events)

        eventSeqs = [note_events]
        
        # (5) Get pedal events
        for ccNum in supportedCC:
            ccSeq = self.pedal2list(i.control_changes, ccNum, onThreshold = 64, endT = endT)
            ccSeq.sort(key = lambda x: (x.onset, x.offset, x.pitch))
            eventSeqs.append(ccSeq)

        # (6) Name sequences
        events = {"notes"       : eventSeqs[0],
                  "sustain"     : eventSeqs[1],
                  "sostenuto"   : eventSeqs[2],
                  "soft"        : eventSeqs[3]}

        return events
    
    def evnt2midr(self, events, delta=0):
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
            midr_note = MidiTransforms(midi_note, repr_type=self.repr_type, delta=delta)
            midr_notes.append(midr_note.to_dict())
        
        # (3) Parse events
        midrep = {"midrep_notes": midr_notes,
                  "sustain"     : sustain,
                  "type"        : self.repr_type}
        
        return midrep
    
    def midr2seq(self, midr, use_offset_duration_tolerance=True):
        """
            Input   : MIDR list (list of midr events)
            Ouptut  : Note Arrays (4 x (Array x L))

            L: Total time bins
            Array: Array of notes at a given fram [[x, y, z, velocity],[...],...]
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
        mpe_seq       = [[] for _ in range(nframe)]
        onset_seq     = [[] for _ in range(nframe)]
        offset_seq    = [[] for _ in range(nframe)]
        velocity_seq  = [[] for _ in range(nframe)]

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
                    onset_seq[onset_frame+j].append([x, y, z, onset_val])

                    if onset_val >= 0.5:
                        velocity_seq[onset_frame+j].append([x, y, z, velocity])

            # (4.4.2) Look behind
            for j in range(1, onset_sharpness+1):
                onset_ms_q      = (onset_frame - j) * hop_ms
                onset_ms_diff   = onset_ms_q - onset_ms
                onset_val       = max(0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms)))
                
                if onset_frame - j >= 0:
                    # if onset_label[onset_frame-j, 3] < onset_val:
                    onset_seq[onset_frame-j].append([x, y, z, onset_val])
                    if onset_val >= 0.5:
                        velocity_seq[onset_frame-j].append([x, y, z, velocity])

            # (4.6) Fill out mpe
            for j in range(onset_frame, offset_frame + 1):
                mpe_seq[j].append([x, y, z, velocity])

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
                        offset_seq[offset_frame+j].append([x, y, z, offset_val])

                for j in range(1, offset_sharpness+1):
                    offset_ms_q      = (offset_frame - j) * hop_ms
                    offset_ms_diff   = offset_ms_q - offset_ms
                    offset_val       = max(0.0, 1.0 - (abs(offset_ms_diff) / (offset_sharpness * hop_ms)))
                    
                    if offset_frame - j >= 0:
                        # if offset_label[offset_frame-j, 3] < offset_val:
                        offset_seq[offset_frame-j].append([x, y, z, offset_val])

        # (5) Return arrays
        return mpe_seq, onset_seq, offset_seq, velocity_seq

    def seq2note(self, seq, int_velocity=False):
        """
            Input   : Note Array(L, Array([x, y, z, vel]))
            Output  : Note label matrix (L, X, Y, Z)
        """
        nframe  = len(seq)
        X, Y, Z = MidiTransforms.get_repr_space(self.repr_type)
        
        label   = np.zeros((nframe, X, Y, Z), dtype=np.int8) # Temp logic

        for i in range(0, nframe):
            for note in seq[i]:
                x = int(note[0] + 0.5)
                y = int(note[1] + 0.5)
                z = int(note[2] + 0.5)
                velocity = note[3]
                if not int_velocity:
                    velocity = int(velocity * 127)
                label[i, x, y, z] = velocity
                
        return label
    
    def seq2centr(self, seq):
        """
            Input   : Note Array(L, Array([x, y, z, vel]))
            Output  : Cloud center matrix (L, [x, y, z, vel, diam])
        """
        nframe = len(seq)
        center = np.zeros((nframe, 5), dtype=np.float32)

        for i in range(0, nframe):
            center[i, :4]    = self.get_centroid(seq[i])
            center[i, 4]     = self.get_diameter(seq[i])
        return center

    def lbl2chunks(self, label):
        """
            Input   : Label matrix (L, ?)
            Output  : Array of padded chunks (K x ((N + 2M), ?))

            L:  Total time frames
            K:  Number of chunks
            N:  Number of frames per chunk
            M:  Pad margin
            ?:  Input can be [x, y, z, vel, diam] or [x, y, z, vel]
        """
        # (1) Pad end to allow smooth splitting into chunks
        num_frame_label = label.shape[0]
        num_frame_chunk = self.config["input"]["num_frame"] # 128
        num_chunks      = math.ceil(num_frame_label / num_frame_chunk)
        num_frame_pad   = num_chunks * num_frame_chunk - num_frame_label

        label           = torch.from_numpy(label)        # Do this more cleanly elsewhere
        label_padded    = F.pad(label, (0, 0, 0, 0, 0, 0, 0, num_frame_pad), mode="constant", value = 0)

        # (2) Pad edges
        margin_left     = self.config["input"]["margin_b"]  # 32
        margin_right    = self.config["input"]["margin_f"]  # 32
        mode            = self.config["feature"]["pad_mode"]    # constant
        label_padded    = F.pad(label_padded, (0, 0, 0, 0, 0, 0, margin_left, margin_right), mode=mode, value=0)

        # (3) Split into chunks (prepaded)
        chunks = []
        for n in range(0, num_chunks):
            chunk = label_padded[n*num_frame_chunk:
                                 margin_left + n*num_frame_chunk + num_frame_chunk + margin_right,
                                 :]
            chunks.append(chunk)

        return chunks
    
    def ctr2chunks(self, center):
        """
            Input   : Label matrix (L, ?)
            Output  : Array of padded chunks (K x ((N + 2M), ?))

            L:  Total time frames
            K:  Number of chunks
            N:  Number of frames per chunk
            M:  Pad margin
            ?:  Input can be [x, y, z, vel, diam] or [x, y, z, vel]
        """
        # (1) Pad end to allow smooth splitting into chunks
        num_frame_label = center.shape[0]
        num_frame_chunk = self.config["input"]["num_frame"] # 128
        num_chunks      = math.ceil(num_frame_label / num_frame_chunk)
        num_frame_pad   = num_chunks * num_frame_chunk - num_frame_label

        center          = torch.from_numpy(center)        # Do this more cleanly elsewhere
        center_padded    = F.pad(center, (0, 0, 0, num_frame_pad), mode="constant", value = 0)
        # (2) Pad edges
        margin_left     = self.config["input"]["margin_b"]  # 32
        margin_right    = self.config["input"]["margin_f"]  # 32
        mode            = self.config["feature"]["pad_mode"]    # constant
        center_padded    = F.pad(center_padded, (0, 0, margin_left, margin_right), mode=mode, value=0)

        # (3) Split into chunks (prepaded)
        chunks = []
        for n in range(0, num_chunks):
            chunk = center_padded[n*num_frame_chunk:
                                 margin_left + n*num_frame_chunk + num_frame_chunk + margin_right,
                                 :]
            chunks.append(chunk)

        return chunks
    
    def midi_to_labels(self, midi_path):
        """
            Input   : Midi file path
            Output  : Note label matrices 4 x (L, X, Y, Z)
        """
        events  = self.midi2evnt(midi_path)
        midr    = self.evnt2midr(events)

        mpe_seq, onset_seq, offset_seq, velocity_seq = self.midr2seq(midr)

        onset_label     = self.seq2note(onset_seq)
        offset_label    = self.seq2note(offset_seq)
        mpe_label       = self.seq2note(mpe_seq, int_velocity=True)
        velocity_label  = self.seq2note(velocity_seq, int_velocity=True)

        return onset_label, offset_label, mpe_label, velocity_label
    
    def midi_to_centr(self, midi_path):
        """
            Input   : Midi file path
            Output  : Note center matrices 4 x (L, [x, y, z, vel, diam])
        """
        events  = self.midi2evnt(midi_path)
        midr    = self.evnt2midr(events)

        mpe_seq, onset_seq, offset_seq, velocity_seq = self.midr2seq(midr)

        onset_center    = self.seq2centr(onset_seq)
        offset_center   = self.seq2centr(offset_seq)
        mpe_center      = self.seq2centr(mpe_seq)
        velocity_center = self.seq2centr(velocity_seq)

        return onset_center, offset_center, mpe_center, velocity_center

    # POSTPROCESSING

    def label2midr(self,
                  onset_label, offset_label, mpe_label, velocity_label,
                  onset_thresh=0.1, offset_thresh=0.1, mpe_thresh=0.1,
                  mode_velocity="ignore_zero", mode_offset="shorter"):
        """ 
            Input   : Note label matrices 4 x (L, X, Y, Z)
            Output  : MIDR list (list of midr events)
        """
        # (1) Calculate settings
        sr          = self.config["feature"]["sr"]
        hop_sample  = self.config["feature"]["hop_sample"]
        hop_sec     = float(hop_sample / sr)

        L, X, Y, Z = onset_label.shape

        midr_notes = []

        onset_label_mask = onset_label >= onset_thresh
        # THIS IS PRETTY SCUFFED AND UNOPTIMIZED, MAYBE REWORK LATER
        # (2) Iterate through every coordinate and check for local maxima
        for z in range(Z):
            # Move to next iteration if no element of an axis is above threshold
            # through the whole piece
            if not onset_label_mask[:, :, :, z].any():
                continue

            for y in range(Y):
                if not onset_label_mask[:, :, y, z].any():
                    continue

                for x in range(X):
                    if not onset_label_mask[:, x, y, z].any():
                        continue

                    # (2.1) Onset detection
                    onset_seq   = []
                    for i in range(L):
                        if onset_label[i, x, y, z] >= onset_thresh:
                            # (2.1.1) Check if i is local maxima or not
                            left_flag=True
                            for ii in range(i-1, -1, -1):
                                if onset_label[i, x, y, z] > onset_label[ii, x, y, z]:
                                    left_flag = True
                                    break
                                elif onset_label[i, x, y, z] < onset_label[ii, x, y, z]:
                                    left_flag = False
                                    break
                            right_flag = True
                            for ii in range(i+1, L):
                                if onset_label[i, x, y, z] > onset_label[ii, x, y, z]:
                                    right_flag = True
                                    break
                                elif onset_label[i, x, y, z] < onset_label[ii, x, y, z]:
                                    right_flag = False
                                    break

                            # (2.1.2) If i is local maximum, save its onset time
                            if left_flag and right_flag:
                                if i == 0 or i == L-1:
                                    onset_time = i * hop_sec
                                else:
                                    onset_curr  = onset_label[i, x, y, z]
                                    onset_prev  = onset_label[i-1, x, y, z]
                                    onset_next  = onset_label[i+1, x, y, z]

                                # (2.1.3) Approximate the onset time based on the surrounding values
                                    if onset_prev == onset_next:
                                        onset_time = i * hop_sec    
                                    elif onset_prev > onset_next:
                                        onset_time = i * hop_sec - (hop_sec * 0.5 * (onset_prev - onset_next) / (onset_curr - onset_next))
                                    else:
                                        onset_time = i * hop_sec + (hop_sec * 0.5 * (onset_next - onset_prev) / (onset_curr - onset_prev))
                                
                                onset_seq.append({'loc': i,
                                                  'onset_time': onset_time})
                    # (2.2) Offset detection
                    offset_seq  = []
                    for i in range(L):
                        if offset_label[i, x, y, z] >= offset_thresh:
                            # (2.2.1) Check if i is local maxima or not
                            left_flag=True
                            for ii in range(i-1, -1, -1):
                                if offset_label[i, x, y, z] > offset_label[ii, x, y, z]:
                                    left_flag = True
                                    break
                                elif offset_label[i, x, y, z] < offset_label[ii, x, y, z]:
                                    left_flag = False
                                    break
                            right_flag = True
                            for ii in range(i+1, L):
                                if offset_label[i, x, y, z] > offset_label[ii, x, y, z]:
                                    right_flag = True
                                    break
                                elif offset_label[i, x, y, z] < offset_label[ii, x, y, z]:
                                    right_flag = False
                                    break

                            # (2.2.2) If i is local maximum, save its onset time
                            if left_flag and right_flag:
                                if i == 0 or i == L-1:
                                    offset_time = i * hop_sec
                                else:
                                    offset_curr  = offset_label[i, x, y, z]
                                    offset_prev  = offset_label[i-1, x, y, z]
                                    offset_next  = offset_label[i+1, x, y, z]

                                # (2.2.3) Approximate the onset time based on the surrounding values
                                    if offset_prev == offset_next:
                                        offset_time = i * hop_sec    
                                    elif offset_prev > offset_next:
                                        offset_time = i * hop_sec - (hop_sec * 0.5 * (offset_prev - offset_next) / (offset_curr - offset_next))
                                    else:
                                        offset_time = i * hop_sec + (hop_sec * 0.5 * (offset_next - offset_prev) / (offset_curr - offset_prev))
                                
                                offset_seq.append({'loc': i,
                                                   'offset_time': offset_time})

                    time_next   = 0.0
                    time_offset = 0.0
                    time_mpe    = 0.0
                    
                    # (3) Return a note for each onset
                    for onset_idx in range(len(onset_seq)):
                        # (3.1) Define onset times
                        loc_onset   = onset_seq[onset_idx]["loc"]
                        time_onset  = onset_seq[onset_idx]["onset_time"]

                        if onset_idx + 1 < len(onset_seq):
                            loc_next    = onset_seq[onset_idx+1]["loc"]
                            time_next   = onset_seq[onset_idx+1]["onset_time"]
                        else:
                            loc_next    = len(mpe_label)
                            time_next  = (loc_next-1) * hop_sec

                        # (3.2) Define offset times from offset_seq
                        loc_offset  = loc_onset+1
                        flag_offset = False

                        for offset_idx in range(len(offset_seq)):
                            if loc_onset < offset_seq[offset_idx]["loc"]:
                                loc_offset  = offset_seq[offset_idx]["loc"]
                                time_offset = offset_seq[offset_idx]["offset_time"]
                                flag_offset = True
                                break
                        if loc_offset > loc_next:
                            loc_offset  = loc_next
                            time_offset = time_next

                        # (3.3) Define offset times from mpe (1 frame longer)
                        loc_mpe     = loc_onset+1
                        flag_mpe    = False

                        for ii_mpe in range(loc_onset+1, loc_next):
                            if mpe_label[ii_mpe, x, y, z] < mpe_thresh:
                                loc_mpe     = ii_mpe
                                flag_mpe    = True
                                time_mpe    = loc_mpe * hop_sec
                                break
                        
                        velocity = int(velocity_label[loc_onset, x, y, z])

                        # (3.4) Get the appropriate offset time
                        if (flag_offset is False) and (flag_mpe is False):
                            offset = float(time_next)
                        elif (flag_offset is True) and (flag_mpe is False):
                            offset = float(time_offset)
                        elif (flag_offset is False) and (flag_mpe is True):
                            offset = float(time_mpe)
                        else:
                            if mode_offset == "offset":
                                offset = float(time_offset)
                            elif mode_offset == "longer":
                                if loc_offset >= loc_mpe:
                                    offset = float(time_offset)
                                else:
                                    offset = float(time_mpe)

                            else:
                                if loc_offset <= loc_mpe:
                                    offset = float(time_offset)
                                else:
                                    offset = float(time_mpe)

                        # (3.5) Handle velocity
                        if mode_velocity != "ignore_zero":
                            # NOTE Check
                            midr_note = MidiTransforms.from_coords(time_onset,
                                                                   offset,
                                                                   x, y, z,
                                                                   velocity,
                                                                   repr_type=self.repr_type)
                            midr_notes.append(midr_note)
                        else:
                            if velocity > 0:
                                midr_note = MidiTransforms.from_coords(time_onset,
                                                                       offset,
                                                                       x, y, z,
                                                                       velocity,
                                                                       repr_type=self.repr_type)
                                midr_notes.append(midr_note)

                        # (3.6) Trim overlapping notes
                        if  (len(midr_notes) > 1) and \
                            (midr_notes[len(midr_notes)-1].pitch == midr_notes[len(midr_notes)-2].pitch) and \
                            (midr_notes[len(midr_notes)-1].onset < midr_notes[len(midr_notes)-2].offset):
                                midr_notes[len(midr_notes)-2].offset = midr_notes[len(midr_notes)-1].onset
                            
        midr_notes = sorted(sorted(midr_notes, key=lambda x: x.pitch), key=lambda x: x.onset)
        return midr_notes

    def midr2notes(self, midr_notes):
        notes = []
        for midr_note in midr_notes:
            note = midr_note.to_midi_note()
            notes.append(note)
        return notes
    
    def notes2midi(self, notes, midi_path):
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        for note in notes:
            instrument.notes.append(pretty_midi.Note(velocity=note.velocity,
                                                     pitch=note.pitch,
                                                     start=note.onset,
                                                     end=note.offset))
        midi.instruments.append(instrument)
        midi.write(str(midi_path))
        return

    def labels_to_midi(self, osnet_label, offset_label, mpe_label, velocity_label, midi_path):
        """
            Input   : Note label matrices 4 x (L, (X, Y, Z))
            Output  : Midi file path
        """
        midr_notes = self.label2midr(osnet_label, offset_label, mpe_label, velocity_label)
        notes = self.midr2notes(midr_notes)
        self.notes2midi(notes, midi_path)
        return

    # UTILS

    def get_centroid(self, notes):
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

    def get_diameter(self, notes):
        if not notes:
            return -1
        
        notes = np.array(notes)
        coords = notes[:, :3]

        diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)

        diameter = np.max(dists)
        return diameter

    def plot_cntr(self, label):
        """
            Helper function for plotting a center after projecting the z axis to 2D (L x z)
        """
        nframe = label.shape[0]
        grid = np.zeros((nframe, 88))

        for i in range(nframe):
            z           = int(label[i, 2] + 0.5)
            velocity    = label[i, 3]
            diameter    = label[i, 4]
            grid[i, z]  = diameter * velocity

        plt.imshow(grid.T, aspect='auto', origin='lower', cmap='magma')
        plt.show()

    def plot_label(self, label):
        """
            Helper function for plotting a label after projecting the z axis to 2D (L x z)
        """
        nframe      = label.shape[0]
        num_notes   = label.shape[3]
        grid = np.zeros((nframe, num_notes))

        for i in range(nframe):
            for z in range(num_notes):
                velocity    = label[i, 0, 0, z]
                if velocity > 0:
                    grid[i, z]  = velocity

        plt.imshow(grid.T, aspect='auto', origin='lower', cmap='magma')
        plt.show()

    def pedal2list(self, ccSeq, ccNum, onThreshold=64, endT = None):
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

    def resolve_overlapping(self, note_list):
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

    def extend_pedal(self, note_events, pedal_events):
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

    def validate_notes(self, notes):
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
    # Load config midi
    midi_config_path = str(Path("MIDR/data/midi_config.json"))
    with open(midi_config_path, 'r', encoding='utf-8') as f:
        midi_config = json.load(f)

    # Load processor
    mp              = MidiProcessor(midi_config)
    midi_path       = Path("./MIDR/test_files/test_midi.mid")
    midi_path_out   = Path("./MIDR/test_files/test_midi_out.mid")

    onset_label, offset_label, mpe_label, velocity_label = mp.midi_to_labels(midi_path)
    onset_cntr, offset_cntr, mpe_cntr, velocity_centr = mp.midi_to_centr(midi_path)
    
    onset_chunks, offset_chunks, mpe_chunks, velocity_chunks = mp.lbl2chunks(onset_label), mp.lbl2chunks(offset_label), mp.lbl2chunks(mpe_label), mp.lbl2chunks(velocity_label)
    mp.labels_to_midi(onset_label, offset_label, mpe_label, velocity_label, midi_path_out)
