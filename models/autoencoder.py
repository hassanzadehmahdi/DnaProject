import torch
import torch.nn as nn


class ResidueAutoencoder(nn.Module):
    """
    Simple MLP autoencoder for per-residue embeddings (dim=1024).
    Encoder → bottleneck → decoder.
    """
    def __init__(self, input_dim=1024, bottleneck_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, bottleneck_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
