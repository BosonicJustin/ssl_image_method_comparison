"""STL-10 data loading."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# STL-10 channel statistics
STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def get_base_transform(image_size: int = 96) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
    ])


def get_pretrain_loader(
    batch_size: int = 256,
    num_workers: int = 4,
    image_size: int = 96,
    data_dir: str | Path = DATA_DIR,
) -> DataLoader:
    """100k unlabeled split for pretraining."""
    dataset = datasets.STL10(
        root=str(data_dir), split="unlabeled", download=False,
        transform=get_base_transform(image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_eval_loaders(
    batch_size: int = 256,
    num_workers: int = 4,
    image_size: int = 96,
    data_dir: str | Path = DATA_DIR,
    low_data_fraction: float = 0.01,
    low_data_seed: int = 42,
) -> dict[str, DataLoader]:
    """Labeled splits for evaluation.

    Returns:
        "train": 5k labeled (for k-NN / linear probe)
        "test": 8k labeled
        "train_lowdata": ~50 images (1% subset for low-data probe)
    """
    transform = get_base_transform(image_size)

    train_ds = datasets.STL10(
        root=str(data_dir), split="train", download=False, transform=transform,
    )
    test_ds = datasets.STL10(
        root=str(data_dir), split="test", download=False, transform=transform,
    )

    n_low = max(1, int(len(train_ds) * low_data_fraction))
    gen = torch.Generator().manual_seed(low_data_seed)
    indices = torch.randperm(len(train_ds), generator=gen)[:n_low].tolist()
    low_ds = Subset(train_ds, indices)

    loader_kwargs = dict(
        batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    )

    return {
        "train": DataLoader(train_ds, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_ds, shuffle=False, **loader_kwargs),
        "train_lowdata": DataLoader(low_ds, shuffle=True, drop_last=False, **loader_kwargs),
    }
