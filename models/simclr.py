"""SimCLR contrastive learning for STL-10 (96x96 RGB)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms

STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent (normalized temperature-scaled cross-entropy) loss."""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                      # (2B, D)
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature                          # (2B, 2B)
    sim.fill_diagonal_(float("-inf"))
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)], dim=0).to(z.device)
    return F.cross_entropy(sim, labels)


class SimCLRAugmentation:
    """Dual-view augmentation for SimCLR pretraining."""

    def __init__(self, image_size: int = 96):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
        ])

    def __call__(self, image):
        return self.transform(image), self.transform(image)


class SimCLR(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        proj_hidden_dim: int = 512,
        proj_output_dim: int = 128,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.latent_dim = self.encoder.num_features
        self.projection_head = ProjectionHead(self.latent_dim, proj_hidden_dim, proj_output_dim)
        self.temperature = temperature

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Features BEFORE projection head (for downstream eval)."""
        return self.encoder(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward. Returns (z1, z2) projections."""
        h1, h2 = self.encoder(x1), self.encoder(x2)
        return self.projection_head(h1), self.projection_head(h2)


if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    model = SimCLR(backbone="resnet18", proj_output_dim=128)

    # Forward pass
    x1 = torch.randn(4, 3, 96, 96)
    x2 = torch.randn(4, 3, 96, 96)
    z1, z2 = model(x1, x2)
    print(f"x1:            {tuple(x1.shape)}")
    print(f"z1 (proj):     {tuple(z1.shape)}")
    print(f"z2 (proj):     {tuple(z2.shape)}")
    assert z1.shape == (4, 128), f"Expected (4, 128), got {z1.shape}"
    assert z2.shape == (4, 128), f"Expected (4, 128), got {z2.shape}"

    # Encode (before projection head)
    h = model.encode(x1)
    print(f"h  (encode):   {tuple(h.shape)}")
    assert h.shape == (4, model.latent_dim), f"Expected (4, {model.latent_dim}), got {h.shape}"

    # Encoder consistency
    h_direct = model.encoder(x1)
    assert torch.equal(h, h_direct), "encode(x) must equal encoder(x)"

    # Loss
    loss = nt_xent_loss(z1, z2)
    print(f"NT-Xent loss:  {loss.item():.4f}")
    assert loss.dim() == 0, "Loss must be a scalar"
    assert loss.item() > 0, "Loss must be positive"

    # Augmentation
    pil_img = Image.fromarray(np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8))
    aug = SimCLRAugmentation(image_size=96)
    view1, view2 = aug(pil_img)
    print(f"view1:         {tuple(view1.shape)}")
    print(f"view2:         {tuple(view2.shape)}")
    assert view1.shape == (3, 96, 96), f"Expected (3, 96, 96), got {view1.shape}"
    assert view2.shape == (3, 96, 96), f"Expected (3, 96, 96), got {view2.shape}"
    assert not torch.equal(view1, view2), "Two views must differ"

    # Parameter counts
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_proj = sum(p.numel() for p in model.projection_head.parameters())
    print(f"\nBackbone:      {model.encoder.default_cfg['architecture']}")
    print(f"Feature dim:   {model.latent_dim}")
    print(f"Total params:  {n_params:>10,}")
    print(f"  Encoder:     {n_enc:>10,}")
    print(f"  Proj. head:  {n_proj:>10,}")

    print("\nAll checks passed.")
