"""Convolutional autoencoder for STL-10 (96x96 RGB)."""

import torch
import torch.nn as nn
import timm


class UpConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(self.upsample(x))))


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 512):
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
    def __init__(self, backbone: str = "resnet18"):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.latent_dim = self.encoder.num_features  # 512 for resnet18
        self.decoder = Decoder(latent_dim=self.latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


if __name__ == "__main__":
    model = Autoencoder(backbone="resnet18")
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
    assert z.shape == (4, 512)
    print("\nShape checks passed.")
