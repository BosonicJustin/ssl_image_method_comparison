"""k-NN evaluation on frozen encoder features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import extract_features


def knn_classify(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    k: int = 20,
) -> float:
    """k-NN accuracy from pre-extracted, L2-normalised features.

    Useful when you want to extract features once and sweep over k values.
    """
    num_classes = int(train_labels.max().item()) + 1
    correct = 0
    total = test_features.size(0)

    chunk_size = 256
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        sim = test_features[start:end] @ train_features.T
        _, topk_idx = sim.topk(k, dim=1)
        topk_labels = train_labels[topk_idx]
        votes = torch.zeros(end - start, num_classes, dtype=torch.long)
        votes.scatter_add_(1, topk_labels, torch.ones_like(topk_labels))
        correct += (votes.argmax(1) == test_labels[start:end]).sum().item()

    return correct / total


@torch.no_grad()
def knn_accuracy(
    encoder: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    k: int = 20,
    device: str | torch.device = "cuda",
) -> float:
    """Compute k-NN accuracy (cosine similarity, majority vote).

    Extracts features from the full train set as the reference bank,
    classifies every test image by majority vote of its k nearest
    train-set neighbors. Runs entirely in no-grad mode so it's safe
    to call mid-training.

    Args:
        encoder: Module whose forward pass maps (B, C, H, W) -> (B, D).
        train_loader: Labeled train split (reference).
        test_loader: Labeled test split.
        k: Number of neighbors.
        device: Device for encoder forward passes.

    Returns:
        Top-1 accuracy in [0, 1].
    """
    train_feats, train_labels = extract_features(encoder, train_loader, device)
    test_feats, test_labels = extract_features(encoder, test_loader, device)

    # L2-normalize for cosine similarity
    train_feats = F.normalize(train_feats, dim=1)
    test_feats = F.normalize(test_feats, dim=1)

    # (N_test, N_train) similarity matrix — small enough for CPU
    sim = test_feats @ train_feats.T

    # Top-k neighbors -> majority vote
    _, indices = sim.topk(k, dim=1)
    neighbor_labels = train_labels[indices]  # (N_test, k)
    predictions = neighbor_labels.mode(dim=1).values

    return (predictions == test_labels).float().mean().item()
