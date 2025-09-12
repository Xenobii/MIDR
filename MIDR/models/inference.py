import torch
import numpy as np
import json

from pathlib import Path

from MIDR.models.midi_model import MIDINet_Std_S, MIDINet_Std_M
from MIDR.data.midi_processor import MidiProcessor
from MIDR.dataset.dataset import MIDI_DataLoader, MIDI_Dataset


def inference(midi_path, midi_path_out):
    # (1) Settings
    repr_type   = "standard"
    midi_config_path = str(Path("MIDR/data/midi_config.json"))
    with open(midi_config_path, 'r', encoding='utf-8') as f:
        midi_config = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # (2) Preprocessing
    print("Preprocessing file...")
    mp = MidiProcessor(midi_config, repr_type)
    
    onset_cntr, offset_cntr, mpe_cntr, velocity_cntr = mp.midi_to_centr(midi_path)

    # Temp
    # onset_cntr      = mp.ctr2chunks(onset_cntr)[5].numpy()
    # offset_cntr     = mp.ctr2chunks(offset_cntr)[5].numpy()
    # mpe_cntr        = mp.ctr2chunks(mpe_cntr)[5].numpy()
    # velocity_cntr   = mp.ctr2chunks(velocity_cntr)[5].numpy()
    
    onset_cntr      = torch.from_numpy(onset_cntr).to(device)
    offset_cntr     = torch.from_numpy(offset_cntr).to(device)
    mpe_cntr        = torch.from_numpy(mpe_cntr).to(device)
    velocity_cntr   = torch.from_numpy(velocity_cntr).to(device)

    # (3) Load model
    model_path  = str(Path("./MIDR/test_files/test_model.pth"))

    model = MIDINet_Std_M()
    model.load_state_dict(torch.load(model_path))

    model.eval()
    model = model.to(device)

    # (4) Run inference
    # Since we do it frame-by-frame for now it's a mess
    print("Runnning inference...")
    with torch.no_grad():
        onset       = model(onset_cntr).squeeze(1).cpu().numpy()
        offset      = model(offset_cntr).squeeze(1).cpu().numpy()
        mpe         = model(mpe_cntr).squeeze(1).cpu().numpy()
        velocity    = model(velocity_cntr).squeeze(1).cpu().numpy()

    # (5) Postprocessing
    print("Post-processing...")
    mp.labels_to_midi(onset, offset, mpe, velocity, midi_path_out)

    print(f"Saved midi to {midi_path_out}.")


if __name__=="__main__":
    midi_path       = str(Path("./MIDR/test_files/test_midi.mid"))
    midi_path_out   = str(Path("./MIDR/test_files/test_midi_out.mid"))

    inference(midi_path, midi_path_out)

