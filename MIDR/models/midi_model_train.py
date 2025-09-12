import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from MIDR.models.midi_model import MIDINet_Std_S, MIDINet_Std_M
from MIDR.dataset.dataset import MIDI_DataLoader, MIDI_Dataset


def midi_model_train(repr_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print(f"Model is running on {device}!")
        breakpoint()

    # (2) Config model
    print("-- Model Settings -- ")
    print("-- repr_type    : " + str(repr_type))
    
    if repr_type == "standard":
        model = MIDINet_Std_M()
    else: 
        raise ValueError(f"Invalid representation type {repr_type}")
    
    model.to(device)
    model_path = str(Path("./MIDR/test_files/test_model.pth"))

    # (2.1) Print parameters
    params = model.count_params()
    print(f"--Trainable parameters: {params}.")

    # (3) Config training
    print("-- Training settings -- ")
    print("-- ")

    batch_size  = 10
    epochs      = 50
    lr          = 0.001
    
    optimizer   = optim.Adam(model.parameters(), lr=lr)
    pos_weight  = torch.tensor([10.0]).to(device)
    criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # (4) Load dataset
    print("-- Loading dataset --")

    pkl_path = str(Path("./MIDR/test_files/test.pkl"))
    dataloader_train = MIDI_DataLoader.create_dataloader(pkl_path, batch_size)

    # (5) Train
    print("-- Training Start --")
    for epoch in range(epochs):
        model.train()
        
        total_loss      = 0.0
        total_frames    = 0

        for idx, batch in enumerate(dataloader_train):
            centers = batch["centers"].to(device)
            labels  = batch["labels"].to(device)

            B, L, *_ = centers.shape

            optimizer.zero_grad()
            
            batch_loss  = 0.0
            frame_count = 0

            # Go frame by frame
            with tqdm(total=B*L, desc=f"Epoch {epoch+1}/{epochs} [Batch {idx+1}]", leave=False) as prog_bar:
                for b in range(B):
                    for t in range(L):
                        center  = centers[b, t]
                        label   = labels[b, t]

                        # Forward pass
                        pred = model(center.unsqueeze(0))
                        pred = pred.squeeze((0, 1))
                        loss = criterion(pred, label)

                        # Accumulate
                        batch_loss  += loss
                        frame_count += 1

                    # Update progress bar
                    prog_bar.update(1)
                
            
            # Normalize
            batch_loss = batch_loss / frame_count

            # Backpropagation
            batch_loss.backward()
            optimizer.step()

            total_loss      += batch_loss.item() * frame_count
            total_frames    += frame_count

        avg_loss = total_loss / total_frames
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss per frame {avg_loss: .3f}")

    # (6) Save model
    torch.save(model.state_dict(), model_path)
    
    # (7) Done!
    print(f"-- Finished training! --")


if __name__=="__main__":
    midi_model_train("standard")