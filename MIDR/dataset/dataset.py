import torch
import pickle
import numpy as np
import os
import json

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from MIDR.data.midi_processor import MidiProcessor


class MIDI_Dataset(Dataset):
    def __init__(self, pkl_path):
        super().__init__()

        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        centers = torch.tensor(item["centers"], dtype=torch.float32)
        labels  = torch.tensor(item["labels"], dtype=torch.float32)
        file    = item["file"]
        return {"centers"   : centers,
                "labels"    : labels,
                "file"      : file}

    @classmethod
    def create_dataset(cls, midi_folder, output_path):
        # (1) Load Midi Processor
        midi_config_path = str(Path("MIDR/data/midi_config.json"))
        with open(midi_config_path, 'r', encoding='utf-8') as f:
            midi_config = json.load(f)
        
        mp = MidiProcessor(midi_config)

        dataset = []
        for midi_file in os.listdir(midi_folder):
            if not midi_file.endswith(".mid"):
                continue
            
            # (2) Load midi
            midi_path = os.path.join(midi_folder, midi_file)

            # (3) Extract labels and centers
            onset_label, offset_label, mpe_label, velocity_label = mp.midi_to_labels(midi_path)
            onset_cntr, offset_cntr, mpe_cntr, velocity_centr = mp.midi_to_centr(midi_path)

            # (3.1) Normalize
            mpe_label   = mpe_label.astype(np.float32) / 127.0
            mpe_cntr    = mpe_cntr.astype(np.float32)

            # # (3.2) Temp - Convert to chunk
            # mpe_label   = mp.lbl2chunks(mpe_label)[5].numpy()
            # mpe_cntr    = mp.ctr2chunks(mpe_cntr)[5].numpy()

            # (4) Stack labels across axis 0
            # FOR NOW DO ONLY MPE SO NO STACKING

            # (5) Add to dataset
            dataset.append({
                "file"      : midi_file,
                "centers"   : mpe_cntr,
                "labels"    : mpe_label
            })

        # (6) Save dataset as pkl
        with open(output_path, "wb") as f:
            pickle.dump(dataset, f)

        print(f"Saved dataset with {len(dataset)} MIDI files to {output_path}")


class MIDI_DataLoader(DataLoader):
    def __init__(self, dataset, batch_size=10, shuffle=True, num_workers=1):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    @classmethod
    def create_dataloader(cls, pkl_path, batch_size=10):
        dataset = MIDI_Dataset(pkl_path)
        return cls(dataset, batch_size)


if __name__=="__main__":
    midi_folder = str(Path("./MIDR/test_files/"))
    output_path = str(Path("./MIDR/test_files/test.pkl"))
    
    MIDI_Dataset.create_dataset(midi_folder, output_path)