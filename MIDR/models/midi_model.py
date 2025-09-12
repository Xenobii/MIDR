import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MIDINet_Std_S(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(5, 14),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.ReLU(),
            nn.Linear(12, 88),
            nn.ReLU(),
            nn.Linear(88, 88 * 3),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(3, 8, kernel_size=(3, 3, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(8, 1, kernel_size=(3, 3, 1), padding=(0, 0, 0)),
            nn.Sigmoid(),
        )

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        # FC
        x = self.fc(x)

        # Transpose to 3d 
        x = x.view(-1, 3, 1, 1, 88)

        # 3D deconv
        x = self.deconv(x)

        return x
    
class MIDINet_Std_M(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(5, 14),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.ReLU(),
            nn.Linear(12, 88),
            nn.ReLU(),
            nn.Linear(88, 88 * 3),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(3, 8, kernel_size=(3, 3, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 16, kernel_size=(3, 3, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(16, 8, kernel_size=(3, 3, 1), padding=(0, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(8, 1, kernel_size=(3, 3, 1), padding=(0, 0, 0)),
            nn.Sigmoid(),
        )

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        # FC
        x = self.fc(x)

        # Transpose to 3d 
        x = x.view(-1, 3, 1, 1, 88)

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