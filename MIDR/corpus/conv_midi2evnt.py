"""
    This module imports a specified midi file and returns a list of intervals describing each event
    Based on https://github.com/yujia-yan/transkun

    NOTE works only for single instrument midis
"""

from pathlib import Path
import argparse
from collections import defaultdict
import warnings
import json

import pretty_midi

from MIDR.representations.note_types import StdNote


def midi2events(midi_path, supportedCC = [64, 66, 67], extendSustainPedal = True):
    """
    Convert midi to a series of events

    Args:
        midi_path (string): Path to the midi file
        supportedCC (list, optional): Pedal event keys. Defaults to [64, 66, 67].
        extendSustainPedal (bool, optional): Defaults to True.

    Returns:
        List(List(StdNote Objects)): A list of note/pedal sequences
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
    note_events = notes2list(i.notes)
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

    # Name sequences
    events = {"notes"       : eventSeqs[0],
              "sustain"     : eventSeqs[1],
              "sostenuto"   : eventSeqs[2],
              "soft"        : eventSeqs[3]}

    return events


def notes2list(notes):
    """
    Converts notes from a prettyMidi object to a list

    Args:
        notes (prettyMidi.instruments.notes)

    Returns:
        List: A list of notes containing:
            {
                onset:    np.float32,
                offset:   np.float32,
                pitch:    int,
                velocity: int
            }
    """
    # Create list of note events
    notes_list = [StdNote(n.start, n.end, n.pitch, n.velocity) for n in notes]
    notes_list.sort(key = lambda x: (x.onset, x.offset, x.pitch))
    
    return notes_list


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
                currentEvent = StdNote(time, None, -ccNum, 127)
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




def midi2events_json(midi_path, json_path):
    """
    Converts a midi to a list of events and stores them in a json file. 
    """

    # (1) Read files
    midi_path = Path(midi_path)
    json_path = Path(json_path)

    if not midi_path.exists():
        raise ValueError(f"Path does not exst: {midi_path}")

    # (2) Convert to events
    midi_events = midi2events(midi_path)

    # (3) Convert to json structure
    midi_events_json = {
        key: [event.to_dict() for event in events]
        for key, events in midi_events.items()
    }
    
    # (4) Dump to json
    with open(str(json_path), "w") as f:
        json.dump(midi_events_json, f, indent=4)



if __name__ == "__main__":
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("midi_path", help="path to the midi file")
    argumentParser.add_argument("json_path", help="path to the output json")

    args = argumentParser.parse_args()

    midi_path = args.midi_path
    json_path = args.json_path

    midi2events_json(midi_path, json_path)
    