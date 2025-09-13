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
        model = MIDINet_Std_S()
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
    epochs      = 400
    lr          = 0.001
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion_onset     = nn.BCELoss()
    criterion_offset    = nn.BCELoss()
    criterion_mpe       = nn.BCELoss()
    criterion_velocity  = nn.CrossEntropyLoss()
    
    pos_weight          = torch.tensor([100.0]).to(device)
    pos_weight_onset    = torch.tensor([400.0]).to(device)
    criterion_onset     = nn.BCEWithLogitsLoss(pos_weight=pos_weight_onset)
    criterion_offset    = nn.BCEWithLogitsLoss(pos_weight=pos_weight_onset)
    criterion_mpe       = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_velocity  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

        pbar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for idx, batch in enumerate(pbar):
            centers     = batch["centers"].to(device)
            onset_label       = batch["onset_label"].to(device)
            offset_label      = batch["offset_label"].to(device)
            mpe_label         = batch["mpe_label"].to(device)
            velocity_label    = batch["velocity_label"].to(device)

            B, L = centers.shape[:2]

            optimizer.zero_grad()
            
            # Flatten batch and frames
            centers         = centers.view(B*L, *centers.shape[2:])
            onset_label     = onset_label.view(B*L, *onset_label.shape[2:])
            offset_label    = offset_label.view(B*L, *offset_label.shape[2:])
            mpe_label       = mpe_label.view(B*L, *mpe_label.shape[2:])
            velocity_label  = velocity_label.view(B*L, *velocity_label.shape[2:])

            # Temp test - bool mpe, cce velocity
            mpe_label       = (mpe_label != 0).float()

            # Forward
            onset_pred, offset_pred, mpe_pred, velocity_pred = model(centers)

            # Loss
            loss_onset      = criterion_onset(onset_pred, onset_label)
            loss_offset     = criterion_offset(offset_pred, offset_label)
            loss_mpe        = criterion_mpe(mpe_pred, mpe_label)
            loss_velocity   = criterion_velocity(velocity_pred, velocity_label)
            loss = loss_onset + loss_offset + loss_mpe + loss_velocity
            
            # Backprop
            loss.backward()
            optimizer.step()

            # Accumulate
            batch_frames    = B * L
            total_loss      += loss.item() * batch_frames
            total_frames    += batch_frames

            # pbar.set_postfix({"batch_loss": loss.item()})

        avg_loss = total_loss / total_frames
        if (epoch+1) % 10  == 0:
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss per frame {avg_loss: .3f}")
            if (epoch+1) % 50 == 0:
                print(round(onset_pred.max().item(), 3),
                      round(offset_pred.max().item(), 3),
                      round(mpe_pred.max().item(), 3),
                      round(velocity_pred.max().item(), 3))
                print(round(onset_pred.min().item(), 3),
                      round(offset_pred.min().item(), 3),
                      round(mpe_pred.min().item(), 3),
                      round(velocity_pred.min().item(), 3))

    # (6) Save model
    torch.save(model.state_dict(), model_path)
    
    # (7) Done!
    print(f"-- Finished training! --")


if __name__=="__main__":
    midi_model_train("standard")