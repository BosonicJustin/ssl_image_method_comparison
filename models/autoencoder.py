"""Convolutional autoencoder for STL-10 (96x96 RGB)."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class UpConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(self.upsample(x))))


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.block1 = ConvBlock(3, 32)     # 96 -> 48
        self.block2 = ConvBlock(32, 64)    # 48 -> 24
        self.block3 = ConvBlock(64, 128)   # 24 -> 12
        self.block4 = ConvBlock(128, 256)  # 12 -> 6
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Identity() if latent_dim == 256 else nn.Linear(256, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.project(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 6 * 6 * 256)
        self.block1 = UpConvBlock(256, 128)  # 6  -> 12
        self.block2 = UpConvBlock(128, 64)   # 12 -> 24
        self.block3 = UpConvBlock(64, 32)    # 24 -> 48
        self.block4 = UpConvBlock(32, 16)    # 48 -> 96
        self.to_rgb = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 256, 6, 6)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.activation(self.to_rgb(x))


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


if __name__ == "__main__":
    model = Autoencoder(latent_dim=256)
    x = torch.randn(4, 3, 96, 96)
    x_hat, z = model(x)

    print(f"Input:         {tuple(x.shape)}")
    print(f"Latent z:      {tuple(z.shape)}")
    print(f"Reconstruction:{tuple(x_hat.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"\nTotal params: {n_params:>10,}")
    print(f"  Encoder:    {n_enc:>10,}")
    print(f"  Decoder:    {n_dec:>10,}")

    assert x_hat.shape == x.shape
    assert z.shape == (4, 256)
    print("\nShape checks passed.")
