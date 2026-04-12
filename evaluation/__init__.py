"""Evaluation utilities for frozen encoder features."""

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_features(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run encoder on all images in loader. Returns (features, labels) on CPU."""
    was_training = encoder.training
    encoder.to(device)
    encoder.eval()
    feats, labels = [], []
    for imgs, y in loader:
        feats.append(encoder(imgs.to(device)).cpu())
        labels.append(y)
    encoder.train(was_training)
    return torch.cat(feats), torch.cat(labels)
