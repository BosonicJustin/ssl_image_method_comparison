"""Unified evaluation script for all SSL methods on STL-10.

Usage:
    python scripts/evaluate.py --method simclr --checkpoint results/simclr/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

# Allow running as `python scripts/evaluate.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import extract_features
from evaluation.knn import knn_classify
from evaluation.linear_probe import LinearProbe
from utils.config import load_shared_config
from utils.data import get_eval_loaders, DATA_DIR

METHOD_CHOICES = ["autoencoder", "simclr", "byol", "mae"]


def parse_args():
    cfg = load_shared_config()

    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained SSL encoder on STL-10",
    )
    parser.add_argument(
        "--method", type=str, required=True, choices=METHOD_CHOICES,
        help="SSL method to evaluate",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for eval_results.json (default: results/{method})",
    )
    # Data
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--num-workers", type=int, default=cfg["num_workers"])
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    # Evaluation hyper-parameters
    parser.add_argument(
        "--knn-k-values", type=int, nargs="+",
        default=cfg["knn_k_values"],
    )
    parser.add_argument("--probe-epochs", type=int, default=cfg["probe_epochs"])
    parser.add_argument("--probe-lr", type=float, default=cfg["probe_lr"])
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"results/{args.method}"

    return args


def load_model(method: str, checkpoint_path: str, device: torch.device):
    """Instantiate the correct model and load checkpoint weights.

    Returns:
        (model, latent_dim, checkpoint_dict)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    if method == "autoencoder":
        from models.autoencoder import Autoencoder
        model = Autoencoder(backbone=ckpt_args.get("backbone", "resnet18"))
        latent_dim = 512

    elif method == "simclr":
        from models.simclr import SimCLR
        model = SimCLR(
            backbone=ckpt_args.get("backbone", "resnet18"),
            proj_output_dim=ckpt_args.get("proj_dim", 128),
        )
        latent_dim = 512

    elif method == "byol":
        from models.byol import BYOL
        model = BYOL(
            backbone=ckpt_args.get("backbone", "resnet18"),
            proj_output_dim=ckpt_args.get("proj_dim", 256),
        )
        latent_dim = 512

    elif method == "mae":
        from models.mae import MAE
        model = MAE(
            encoder_dim=ckpt_args.get("encoder_dim", 384),
            encoder_depth=ckpt_args.get("encoder_depth", 6),
            encoder_heads=ckpt_args.get("encoder_heads", 6),
            decoder_dim=ckpt_args.get("decoder_dim", 192),
            decoder_depth=ckpt_args.get("decoder_depth", 4),
            decoder_heads=ckpt_args.get("decoder_heads", 3),
            patch_size=ckpt_args.get("patch_size", 4),
            mask_ratio=ckpt_args.get("mask_ratio", 0.75),
        )
        latent_dim = 384

    else:
        raise ValueError(f"Unknown method: {method}")

    model.load_state_dict(ckpt["model_state_dict"])
    return model, latent_dim, ckpt


def get_autoencoder_eval_transform(image_size: int = 96) -> transforms.Compose:
    """Autoencoder eval transform: [0, 1] range, no normalization.

    The autoencoder decoder uses Sigmoid, so it was trained on unnormalized
    images. Evaluation must match.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])


def main():
    args = parse_args()

    # -- Reproducibility --
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Load model --
    print(f"Loading {args.method} checkpoint from {args.checkpoint}")
    model, latent_dim, ckpt = load_model(args.method, args.checkpoint, device)
    model.to(device)
    model.eval()

    # -- Data loaders --
    # Autoencoder needs unnormalized [0,1] images; all others use default normalized.
    eval_transform = None
    if args.method == "autoencoder":
        eval_transform = get_autoencoder_eval_transform()

    loaders = get_eval_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        transform=eval_transform,
    )

    # -- Extract features once for k-NN --
    print("Extracting features ...")
    train_feats, train_labels = extract_features(model.encoder, loaders["train"], device)
    test_feats, test_labels = extract_features(model.encoder, loaders["test"], device)

    # L2-normalize for k-NN (cosine similarity)
    train_norm = F.normalize(train_feats, dim=1)
    test_norm = F.normalize(test_feats, dim=1)

    # -- k-NN sweep --
    print("Running k-NN sweep ...")
    knn_results = {}
    for k in args.knn_k_values:
        acc = knn_classify(train_norm, train_labels, test_norm, test_labels, k=k)
        knn_results[str(k)] = acc
        print(f"  k={k:<4d}  accuracy={acc:.4f}")

    # -- Linear probe (full data) --
    print("Training linear probe (full data) ...")
    probe = LinearProbe(
        feature_dim=latent_dim,
        num_classes=10,
        lr=args.probe_lr,
        epochs=args.probe_epochs,
        device=device,
    )
    probe.fit(model.encoder, loaders["train"])
    probe_acc = probe.evaluate(model.encoder, loaders["test"])
    print(f"  Linear probe accuracy: {probe_acc:.4f}")

    # -- Linear probe (low data) --
    print("Training linear probe (low data) ...")
    probe_low = LinearProbe(
        feature_dim=latent_dim,
        num_classes=10,
        lr=args.probe_lr,
        epochs=args.probe_epochs,
        device=device,
    )
    probe_low.fit(model.encoder, loaders["train_lowdata"])
    probe_low_acc = probe_low.evaluate(model.encoder, loaders["test"])
    print(f"  Linear probe (low data) accuracy: {probe_low_acc:.4f}")

    # -- Assemble results --
    results = {
        "method": args.method,
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch", -1),
        "training_loss": ckpt.get("loss", float("nan")),
        "knn": knn_results,
        "linear_probe": {
            "accuracy": probe_acc,
            "epochs": args.probe_epochs,
        },
        "linear_probe_lowdata": {
            "accuracy": probe_low_acc,
            "epochs": args.probe_epochs,
        },
    }

    # -- Save --
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # -- Summary --
    best_knn_k = max(knn_results, key=knn_results.get)
    print(f"\n{'=' * 50}")
    print(f"  Method:             {args.method}")
    print(f"  Checkpoint epoch:   {results['epoch']}")
    print(f"  Training loss:      {results['training_loss']:.6f}")
    print(f"  Best k-NN:          k={best_knn_k}  acc={knn_results[best_knn_k]:.4f}")
    print(f"  Linear probe:       {probe_acc:.4f}")
    print(f"  Linear probe (low): {probe_low_acc:.4f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
