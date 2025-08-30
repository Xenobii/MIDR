"""
    Convertion from MIDI notes to other represenations
"""

import argparse
import json
from pathlib import Path
from MIDR.representations.note_types import StdNote, SnaNote



def notes2std(notes, delta):
    # (1) Convert midi notes to standard notes
    std_notes = []
    for note in notes:
        std_note = StdNote.from_midi_note(note, delta)
        std_notes.append(std_note.to_dict())
    
    return std_notes


def notes2sna(notes, delta):
    # (1) Convert midi notes to sna_notes
    sna_notes = []
    for note in notes:
        sna_note = SnaNote.from_midi_note(note, delta)
        sna_notes.append(sna_note.to_dict())

    return sna_notes



def evnt2midr(events, repr_type="standard", delta=0):
    """
    Converts a list of notes to MIDR representations
    """
    # (1) Retrieve events separately
    notes = events["notes"]
    sustain = events["sustain"]

    # (2) Convert to defined midr 
    if repr_type == "standard":
        midrep_notes = notes2std(notes, delta)
    
    elif repr_type == "sna":
        midrep_notes = notes2sna(notes, delta)
    
    else:
        raise ValueError("Invalid midrep type")
    
    # (3) Parse events
    midrep = {"midrep_notes": midrep_notes,
              "sustain"     : sustain,
              "type"        : repr_type}
    
    return midrep



def evnt2midr_json(evnt_path, json_path, repr_type):
    """
    Converts a list of notes to MIDR representations and exports to json
    """
    # (1) Read files
    evnt_path = Path(evnt_path)
    json_path = Path(json_path)

    if not evnt_path.exists():
        raise ValueError(f"Path does not exist: {evnt_path}")
    
    with open(str(evnt_path), "r") as f:
        events = json.load(f)
    
    # (2) Convert to midrep
    midrep = evnt2midr(events, repr_type=repr_type)

    # (3) Dump to json
    with open(str(json_path), "w") as f:
        json.dump(midrep, f, indent=4)
    


if __name__ == '__main__':
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("evnt_path", help="path to the note list json")
    argumentParser.add_argument("json_path", help="path to the output json")
    argumentParser.add_argument("repr_type", help="The type of representation to be used")

    args = argumentParser.parse_args()

    evnt_path = args.evnt_path
    json_path = args.json_path
    repr_type = args.repr_type

    evnt2midr_json(evnt_path, json_path, repr_type)


# TODO:
# - Implement all representation functions