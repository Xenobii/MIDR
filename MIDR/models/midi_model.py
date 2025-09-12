import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MIDINet_Std_S(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size = 3, padding=1),
            nn.ReLU()
        )

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        # FC
        x = self.fc(x)

        # Transpose to 3d 
        x = x.view(-1, 8, 1, 1, 22)

        # 3D deconv
        x = self.deconv(x)

        return x
    
class MIDINet_Std_M(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size = 3, padding=1),
            nn.ReLU()
        )

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        # FC
        x = self.fc(x)

        # Transpose to 3d 
        x = x.view(-1, 16, 1, 1, 22)

        # 3D deconv
        x = self.deconv(x)

        return x

if __name__ == "__main__":
    device = "cpu"
    model = MIDINet_Std_S()
    model.to(device)
    params = model.count_params()
    print(params)
    
    model = MIDINet_Std_M()
    model.to(device)
    params = model.count_params()
    print(params)