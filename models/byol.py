"""BYOL (Bootstrap Your Own Latent) for STL-10 (96x96 RGB)."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms

STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)


class MLP(nn.Module):
    """Two-layer MLP used for both projector and predictor."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BYOLAugmentation:
    """Dual-view augmentation for BYOL pretraining."""

    def __init__(self, image_size: int = 96):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
        ])

    def __call__(self, image):
        return self.transform(image), self.transform(image)


def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """MSE between L2-normalized online prediction and target projection."""
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=1).mean()


class BYOL(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        proj_hidden_dim: int = 1024,
        proj_output_dim: int = 256,
        pred_hidden_dim: int = 1024,
        momentum: float = 0.996,
    ):
        super().__init__()

        # Online network: encoder + projector + predictor
        self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.latent_dim = self.encoder.num_features
        self.projector = MLP(self.latent_dim, proj_hidden_dim, proj_output_dim)
        self.predictor = MLP(proj_output_dim, pred_hidden_dim, proj_output_dim)

        # Target network: encoder + projector (no predictor)
        # Starts as exact copy of online, then updated via EMA each step
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.momentum = momentum

    @torch.no_grad()
    def update_target(self):
        """EMA update: target = m * target + (1 - m) * online."""
        m = self.momentum
        for online, target in [
            (self.encoder.parameters(), self.target_encoder.parameters()),
            (self.projector.parameters(), self.target_projector.parameters()),
        ]:
            for p_o, p_t in zip(online, target):
                p_t.data.mul_(m).add_(p_o.data, alpha=1 - m)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder features for downstream eval (before projector)."""
        return self.encoder(x)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (pred1, target2, pred2, target1).

        Loss = byol_loss(pred1, target2) + byol_loss(pred2, target1)
        """
        p1 = self.predictor(self.projector(self.encoder(x1)))
        p2 = self.predictor(self.projector(self.encoder(x2)))

        with torch.no_grad():
            z1 = self.target_projector(self.target_encoder(x1))
            z2 = self.target_projector(self.target_encoder(x2))

        return p1, z2, p2, z1


if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    model = BYOL(backbone="resnet18")

    # Forward pass
    x1 = torch.randn(4, 3, 96, 96)
    x2 = torch.randn(4, 3, 96, 96)
    p1, z2, p2, z1 = model(x1, x2)
    print(f"x1:            {tuple(x1.shape)}")
    print(f"p1 (pred):     {tuple(p1.shape)}")
    print(f"z2 (target):   {tuple(z2.shape)}")
    assert p1.shape == (4, 256)
    assert z2.shape == (4, 256)

    # Encode (before projector)
    h = model.encode(x1)
    print(f"h  (encode):   {tuple(h.shape)}")
    assert h.shape == (4, model.latent_dim)

    # Loss
    loss = byol_loss(p1, z2) + byol_loss(p2, z1)
    print(f"BYOL loss:     {loss.item():.4f}")
    assert loss.dim() == 0
    assert loss.item() >= 0

    # EMA update
    old_param = next(model.target_encoder.parameters()).clone()
    model.update_target()
    new_param = next(model.target_encoder.parameters())
    assert not torch.equal(old_param, new_param)
    print(f"EMA update:    OK (momentum={model.momentum})")

    # Augmentation
    pil_img = Image.fromarray(np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8))
    aug = BYOLAugmentation(image_size=96)
    view1, view2 = aug(pil_img)
    print(f"view1:         {tuple(view1.shape)}")
    assert view1.shape == (3, 96, 96)
    assert not torch.equal(view1, view2)

    # Parameter counts
    online_params = (
        sum(p.numel() for p in model.encoder.parameters())
        + sum(p.numel() for p in model.projector.parameters())
        + sum(p.numel() for p in model.predictor.parameters())
    )
    target_params = (
        sum(p.numel() for p in model.target_encoder.parameters())
        + sum(p.numel() for p in model.target_projector.parameters())
    )
    print(f"\nFeature dim:   {model.latent_dim}")
    print(f"Online params: {online_params:>10,}")
    print(f"  Encoder:     {sum(p.numel() for p in model.encoder.parameters()):>10,}")
    print(f"  Projector:   {sum(p.numel() for p in model.projector.parameters()):>10,}")
    print(f"  Predictor:   {sum(p.numel() for p in model.predictor.parameters()):>10,}")
    print(f"Target params: {target_params:>10,} (EMA copy, no grad)")

    print("\nAll checks passed.")
