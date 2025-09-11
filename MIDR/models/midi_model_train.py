import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from MIDR.models.midi_model import MIDINet_Std


def midi_model_train(repr_type):
    device = 'cpu' # Temp

    # (2) Config model
    print("-- Model Settings -- ")
    print("-- repr_type    : " + str(repr_type))
    print("\n")
    
    if repr_type == "standard":
        model = MIDINet_Std()
    else: 
        raise ValueError(f"Invalid representation type {repr_type}")
    
    model.to(device)

    # (2.1) Print parameters
    params = model.count_params()
    print(f"--Trainable parameters: {params}.")
    print("\n")

    # (3) Config training
    print("-- Training settings -- ")
    print("-- ")
    print("\n")

    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    performance = {
        'loss': [],
        'current_epoch': []
    }