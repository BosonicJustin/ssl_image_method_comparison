"""Evaluate ALL checkpoints across all methods. Saves combined results."""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import extract_features
from evaluation.knn import knn_classify
from evaluation.linear_probe import LinearProbe
from utils.data import get_eval_loaders, DATA_DIR

RESULTS_DIR = Path("results")
KNN_K_VALUES = [1, 5, 10, 20, 50, 100, 200]
PROBE_EPOCHS = 200
PROBE_LR = 0.1


def load_model(method, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    if method == "autoencoder":
        from models.autoencoder import Autoencoder
        model = Autoencoder(backbone=ckpt_args.get("backbone", "resnet18"))
    elif method == "simclr":
        from models.simclr import SimCLR
        model = SimCLR(backbone=ckpt_args.get("backbone", "resnet18"),
                       proj_output_dim=ckpt_args.get("proj_dim", 128))
    elif method == "byol":
        from models.byol import BYOL
        model = BYOL(backbone=ckpt_args.get("backbone", "resnet18"),
                     proj_output_dim=ckpt_args.get("proj_dim", 256))
    elif method == "mae":
        from models.mae import MAE
        model = MAE(encoder_dim=ckpt_args.get("encoder_dim", 384),
                    encoder_depth=ckpt_args.get("encoder_depth", 6),
                    encoder_heads=ckpt_args.get("encoder_heads", 6),
                    decoder_dim=ckpt_args.get("decoder_dim", 192),
                    decoder_depth=ckpt_args.get("decoder_depth", 4),
                    decoder_heads=ckpt_args.get("decoder_heads", 3),
                    patch_size=ckpt_args.get("patch_size", 4),
                    mask_ratio=ckpt_args.get("mask_ratio", 0.75))
    else:
        raise ValueError(f"Unknown method: {method}")

    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data ONCE — normalized (for simclr/byol/mae)
    print("Loading eval data (normalized) ...")
    loaders_norm = get_eval_loaders(batch_size=256, data_dir=DATA_DIR)

    # Autoencoder needs unnormalized
    print("Loading eval data (unnormalized for autoencoder) ...")
    ae_transform = transforms.Compose([
        transforms.Resize(96), transforms.CenterCrop(96), transforms.ToTensor(),
    ])
    loaders_ae = get_eval_loaders(batch_size=256, data_dir=DATA_DIR, transform=ae_transform)

    # Find all methods and checkpoints
    all_results = []
    for method_dir in sorted(RESULTS_DIR.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        if method not in ("autoencoder", "simclr", "byol", "mae"):
            continue

        checkpoints = sorted(method_dir.glob("*.pt"))
        if not checkpoints:
            continue

        loaders = loaders_ae if method == "autoencoder" else loaders_norm
        print(f"\n{'='*60}")
        print(f"Method: {method}  ({len(checkpoints)} checkpoints)")
        print(f"{'='*60}")

        for ckpt_path in checkpoints:
            print(f"\n  {ckpt_path.name} ...")

            model, ckpt = load_model(method, ckpt_path, device)
            model.to(device).eval()
            latent_dim = model.latent_dim

            # Extract features
            train_feats, train_labels = extract_features(model.encoder, loaders["train"], device)
            test_feats, test_labels = extract_features(model.encoder, loaders["test"], device)
            train_norm = F.normalize(train_feats, dim=1)
            test_norm = F.normalize(test_feats, dim=1)

            # k-NN sweep
            knn = {}
            for k in KNN_K_VALUES:
                knn[str(k)] = knn_classify(train_norm, train_labels, test_norm, test_labels, k=k)
            best_k = max(knn, key=knn.get)

            # Linear probe (full)
            probe = LinearProbe(feature_dim=latent_dim, num_classes=10,
                                lr=PROBE_LR, epochs=PROBE_EPOCHS, device=device)
            probe.fit(model.encoder, loaders["train"])
            probe_acc = probe.evaluate(model.encoder, loaders["test"])

            # Linear probe (low data)
            probe_low = LinearProbe(feature_dim=latent_dim, num_classes=10,
                                    lr=PROBE_LR, epochs=PROBE_EPOCHS, device=device)
            probe_low.fit(model.encoder, loaders["train_lowdata"])
            probe_low_acc = probe_low.evaluate(model.encoder, loaders["test"])

            result = {
                "method": method,
                "checkpoint": str(ckpt_path),
                "checkpoint_name": ckpt_path.name,
                "epoch": ckpt.get("epoch", -1),
                "training_loss": ckpt.get("loss", None),
                "knn": knn,
                "linear_probe": probe_acc,
                "linear_probe_lowdata": probe_low_acc,
            }
            all_results.append(result)

            print(f"    epoch={result['epoch']:>3d}  "
                  f"knn(k={best_k})={knn[best_k]:.4f}  "
                  f"probe={probe_acc:.4f}  "
                  f"probe_low={probe_low_acc:.4f}")

            # Save incrementally
            with open(RESULTS_DIR / "all_eval_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

            del model
            torch.cuda.empty_cache()

    print(f"\n\nDone. {len(all_results)} checkpoints evaluated.")
    print(f"Results saved to {RESULTS_DIR / 'all_eval_results.json'}")


if __name__ == "__main__":
    main()
