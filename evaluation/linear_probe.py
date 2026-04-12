"""Linear probe evaluation on frozen encoder features."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import extract_features


class LinearProbe:
    """Train a linear classifier on frozen encoder features.

    Usage:
        probe = LinearProbe(feature_dim=256)
        probe.fit(encoder, train_loader)
        acc = probe.evaluate(encoder, test_loader)
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 10,
        lr: float = 0.1,
        epochs: int = 100,
        batch_size: int = 256,
        device: str | torch.device = "cuda",
    ):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.head = nn.Linear(feature_dim, num_classes).to(device)
        self.optimizer = torch.optim.SGD(
            self.head.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs,
        )
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, encoder: nn.Module, train_loader: DataLoader) -> list[float]:
        """Train linear head on frozen features from the full train set.

        Returns:
            Per-epoch training loss.
        """
        train_feats, train_labels = extract_features(encoder, train_loader, self.device)
        feat_loader = DataLoader(
            TensorDataset(train_feats, train_labels),
            batch_size=self.batch_size,
            shuffle=True,
        )

        losses = []
        self.head.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for feats, labels in feat_loader:
                feats, labels = feats.to(self.device), labels.to(self.device)
                loss = self.criterion(self.head(feats), labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * feats.size(0)
            self.scheduler.step()
            losses.append(epoch_loss / len(train_feats))
        return losses

    @torch.no_grad()
    def evaluate(self, encoder: nn.Module, test_loader: DataLoader) -> float:
        """Evaluate on test set. Returns top-1 accuracy in [0, 1]."""
        test_feats, test_labels = extract_features(encoder, test_loader, self.device)
        feat_loader = DataLoader(
            TensorDataset(test_feats, test_labels),
            batch_size=self.batch_size,
        )

        self.head.eval()
        correct = 0
        for feats, labels in feat_loader:
            feats, labels = feats.to(self.device), labels.to(self.device)
            correct += (self.head(feats).argmax(1) == labels).sum().item()
        return correct / len(test_feats)
