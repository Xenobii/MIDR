import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MIDINet_Std(nn.Module):
    def __init__(self):
        super().__init__()

        # CONFIG
        self.in_dim         = 5
        self.hidden_dim     = 10
        self.out_channels   = 1

        # LAYERS
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim * 1 * 1 * 22)
        )
    
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(self.hidden_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv3d(16, self.out_channels, kernel_size = 3, padding=1),
            nn.AdaptiveAvgPool3d((1,1,88)),
            nn.Sigmoid()
        )

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    
    def forward(self, x):
        # FC
        x = self.fc(x)

        # Transpose to 3d 
        x = x.view(-1, self.hidden_dim, 1, 1, 22)

        # 3D deconv
        x = self.deconv(x)

        return x

if __name__ == "__main__":
    device = "cpu"
    model = MIDINet_Std(device)
    params = model.count_params()
    print(params)