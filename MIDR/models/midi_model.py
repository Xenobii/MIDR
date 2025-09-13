import torch
import torch.nn as nn


class MIDINet_Std_S(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(5, 88),
            nn.ReLU(),
            nn.Linear(88, 88),
            nn.ReLU(),
        )

        self.decoder_onset = nn.Sequential(
            nn.Linear(88 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.decoder_offset = nn.Sequential(
            nn.Linear(88 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.decoder_mpe = nn.Sequential(
            nn.Linear(88 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.decoder_velocity = nn.Sequential(
            nn.Linear(88 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        coords_x = torch.linspace(-1, 1, 1)
        coords_y = torch.linspace(-1, 1, 1)
        coords_z = torch.linspace(-1, 1, 88)
        X, Y, Z = torch.meshgrid(coords_x, coords_y, coords_z, indexing="ij")
        grid_coords = torch.stack([X, Y, Z], dim=-1)
        self.register_buffer("grid_coords", grid_coords, persistent=False)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        B = x.size(0)

        # Encoder
        z = self.encoder(x) # [B, 88]

        # Expand to all voxels
        z = z[:, None, None, None, :]   # [B, 1, 1, 1, 88]
        z = z.expand(B, 1, 1, 88, -1)   # [B, 1, 1, 88, 88]

        # Repeat across batch
        coords = self.grid_coords[None, ...]    # [1, 1, 88, 3]
        coords = coords.expand(B, -1, -1, -1, -1)   # [B, 1, 1, 88, 3]

        # Concat with voxel coords
        features = torch.cat([z, coords], dim=-1)   # [B, 1, 1, 88, 88 + 3]

        # Flatten
        features = features.view(B * 1 * 1 * 88, -1)    # [B * 1 * 1 * 88, 88 + 3]

        # Decode
        onset       = self.decoder_onset(features)      # [B * 1 * 1 * 88, 1]
        offset      = self.decoder_offset(features)     # [B * 1 * 1 * 88, 1]
        mpe         = self.decoder_mpe(features)        # [B * 1 * 1 * 88, 1]
        velocity    = self.decoder_velocity(features)   # [B * 1 * 1 * 88, 1]

        onset       = onset.view(B, 1, 1, 88)     # [B, 1, 1, 88]
        offset      = offset.view(B, 1, 1, 88)     # [B, 1, 1, 88]
        mpe         = mpe.view(B, 1, 1, 88)     # [B, 1, 1, 88]
        velocity    = velocity.view(B, 1, 1, 88)     # [B, 1, 1, 88]

        return onset, offset, mpe, velocity



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