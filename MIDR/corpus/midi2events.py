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

from classes import Note



def midi2events(midi_path, supportedCC = [64, 66, 67], extendSustainPedal = False):
    """
    Convert midi to a series of events

    Args:
        midi_path (string): Path to the midi file
        supportedCC (list, optional): Pedal event keys. Defaults to [64, 66, 67].
        extendSustainPedal (bool, optional): Defaults to True.

    Returns:
        List(List(Note Objects)): A list of note/pedal sequences
    """

    # Open midi file
    try:
        midi_file = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"Error opening file: {e}")

    # Assert single instrument
    assert(len(midi_file.instruments) == 1)
    if len(midi_file.instruments) > 1:
        raise Exception("Provided midi contains multiple instruments")
    
    i = midi_file.instruments[0]

    # Get note events
    note_events  = notes2list(i.notes)
    endT = max([n.end for n in note_events]) # final offset in sequence

    # If extendSustainPedal, extend notes to its duration
    if extendSustainPedal:
        ccSeq = pedal2list(i.control_changes, ccNum = 64, onThreshold = 64, endT = endT)
        ccSeq      .sort(key = lambda x: (x.start, x.end, x.pitch))
        note_events.sort(key = lambda x: (x.start, x.end, x.pitch))
        note_events = extendPedal(note_events, ccSeq)
    
    # Resolve overlapping and validate
    note_events = resolve_overlapping(note_events)
    validate_notes(note_events)

    eventSeqs = [note_events]
    
    # Get pedal events
    for ccNum in supportedCC:
        ccSeq = pedal2list(i.control_changes, ccNum, onThreshold = 64, endT = endT)
        ccSeq.sort(key = lambda x: (x.start, x.end, x.pitch))
        eventSeqs.append(ccSeq)

    # Name sequences
    events = {"notes":     eventSeqs[0],
              "sustain":   eventSeqs[1],
              "sostenuto": eventSeqs[2],
              "soft":      eventSeqs[3]}

    return events


def notes2list(notes):
    """
    Converts notes from a prettyMidi object to a list

    Args:
        notes (prettyMidi.instruments.notes)

    Returns:
        List: A list of notes containing:
            {
                start:    np.float32,
                end:      np.float32,
                pitch:    int,
                velocity: int
            }
    """
    # Create list of note events
    notes_list = [Note(**n.__dict__) for n in notes]
    notes_list.sort(key = lambda x: (x.start, x.end, x.pitch))

    # Validate notes
    validate_notes(notes_list)
    
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
        List(Note objects): The list of pedal events as negative note events
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
                
        # Find start and end position
        if runningStatus != currentStatus:
            if currentStatus == True:
                # Use negative number as pitch for control change event
                currentEvent = Note(time, None, -ccNum, 127)
            else:
                currentEvent.end = time
                seqEvent.append(currentEvent)

        runningStatus = currentStatus

    # Process the case where state is not closed off at the end
    if runningStatus and endT is not None:
        currentEvent.end = max(endT, time)
        if currentEvent.end > currentEvent.start:
            seqEvent.append(currentEvent)

    return seqEvent


def resolve_overlapping(note_list):
    """
    Resolve overlapping note segments by slicing off the end of overlapping notes

    Args:
        note_list (Note objects)

    Returns:
        List(Note objects): The corrected list of notes 
    """
    
    buffer_dict  = {}
    ex_notes = []
    idx = 0

    # For all overlapping notes of the same pitch, slice the end of the first note
    for note in note_list:
        pitch = note.pitch

        if pitch in buffer_dict.keys():
            _idx = buffer_dict[pitch]
            if ex_notes[_idx].end > note.start:
                ex_notes[_idx].end = note.start

        buffer_dict[pitch] = idx
        idx += 1

        ex_notes.append(note)

    ex_notes.sort(key = lambda x: (x.start, x.end, x.pitch))

    # Detect errors
    error_notes = [n for n in ex_notes if not n.start<n.end]
    if len(error_notes) > 0:
        warnings.warn("There are error notes in given midi")

    return ex_notes


def extendPedal(note_events, pedal_events):
    """
    Extend notes if sustain pedal is on 

    Args:
        note_events  (List(Note Object))
        pedal_events (List(Note Object))

    Returns:
        List(Note Object): List of adjusted notes 
    """
    ex_notes = []

    idx = 0

    buffer_dict = {}
    nIn = len(note_events)

    for note in note_events:
        pitch = note.pitch
        if pitch in buffer_dict.keys():
            _idx = buffer_dict[pitch]
            if ex_notes[_idx].end > note.start:
                ex_notes[_idx].end = note.start

        for pedal in pedal_events:
            if note.end < pedal.end and note.end>pedal.start:
                note.end = pedal.end
        
        buffer_dict[note] = idx
        idx += 1
        ex_notes.append(note)

    ex_notes.sort(key = lambda x: (x.start, x.end, x.pitch))

    nOut = len(ex_notes)
    assert(nOut == nIn)

    return ex_notes


def validate_notes(notes):
    """
    Validate that note events don't overlap and no notes end before they start

    Args:
        notes (List(Note object)): A list of Note objects
    """
    pitches = defaultdict(list)
    for n in notes:
        if len(pitches[n.pitch])>0:
            # Make sure no notes overlap
            nPrev = pitches[n.pitch][-1]
            assert n.start >= nPrev.end, str(n) + str(nPrev)
        # Make sure no notes end before they start
        assert n.start < n.end, n

        pitches[n.pitch].append(n)


def main(midi_path, json_path):
    """
    Converts a midi to a list of events and stores them in a json file. 
    """

    midi_path = Path(midi_path)
    json_path = Path(json_path)

    if not midi_path.exists():
        raise ValueError(f"Path does not exst: {midi_path}")
    
    if not json_path.exists():
        raise ValueError(f"Path does not exst: {json_path}")

    midi_events = midi2events(midi_path)

    midi_events_json = {
        key: [event.to_dict() for event in events]
        for key, events in midi_events.items()
    }
    print(midi_events_json)
    
    with open(str(json_path), "w") as f:
        json.dump(midi_events_json, f, indent=4)




if __name__ == "__main__":
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("midi_path", help="path to the midi file")
    argumentParser.add_argument("json_path", help="path to the output json")

    args = argumentParser.parse_args()

    midi_path = args.midi_path
    json_path = args.json_path

    main(midi_path, json_path)
    