"""Training script for convolutional autoencoder on STL-10."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

# Allow running as `python scripts/train_autoencoder.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.knn import knn_accuracy
from models.autoencoder import Autoencoder
from utils.config import load_shared_config
from utils.data import get_eval_loaders, get_pretrain_loader, DATA_DIR


def parse_args():
    # Shared config provides defaults; CLI flags override.
    cfg = load_shared_config()

    parser = argparse.ArgumentParser(description="Train autoencoder on STL-10")
    # Model
    parser.add_argument("--backbone", type=str, default="resnet18")
    # Training (defaults from configs/shared.yaml)
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--lr", type=float, default=cfg["lr"])
    parser.add_argument("--weight-decay", type=float, default=cfg["weight_decay"])
    # Data
    parser.add_argument("--num-workers", type=int, default=cfg["num_workers"])
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    # Logging & checkpoints
    parser.add_argument("--log-dir", type=str, default="results/autoencoder")
    parser.add_argument("--log-interval", type=int, default=cfg["log_interval"])
    parser.add_argument("--eval-interval", type=int, default=cfg["eval_interval"],
                        help="Log epoch loss + run k-NN eval every N epochs")
    parser.add_argument("--vis-interval", type=int, default=cfg["vis_interval"])
    parser.add_argument("--save-interval", type=int, default=cfg["save_interval"])
    # Reproducibility
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    # Wandb (optional)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="ssl-comparison")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def get_ae_transform(image_size: int = 96) -> transforms.Compose:
    """Autoencoder transform: maps to [0, 1] without normalization.

    The decoder uses Sigmoid, so reconstruction targets must be in [0, 1].
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


@torch.no_grad()
def log_reconstruction_grid(
    model: nn.Module,
    fixed_batch: torch.Tensor,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    was_training = model.training
    model.eval()
    x = fixed_batch.to(device)
    x_hat, _ = model(x)
    grid = make_grid(torch.cat([x, x_hat], dim=0), nrow=len(x), padding=2)
    writer.add_image("reconstructions", grid, epoch)
    model.train(was_training)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    log_interval: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:>3d}", leave=False)
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)

        x_hat, _ = model(images)
        loss = criterion(x_hat, images)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        n_batches += 1

        if batch_idx % log_interval == 0:
            step = epoch * len(loader) + batch_idx
            writer.add_scalar("train/loss_step", loss_val, step)
            pbar.set_postfix(loss=f"{loss_val:.4f}")

    return total_loss / n_batches


def main():
    args = parse_args()

    # -- Reproducibility --
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Output directory --
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # -- Data --
    transform = get_ae_transform()
    loader = get_pretrain_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        transform=transform,
    )
    fixed_batch = next(iter(loader))[0][:16].clone()

    eval_loaders = get_eval_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        transform=transform,
    )

    # -- Model --
    model = Autoencoder(backbone=args.backbone).to(device)

    # -- Optimizer, scheduler & loss --
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # -- Resume --
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", ckpt["loss"])
        print(f"Resumed from {args.resume}  (epoch {start_epoch}, loss {best_loss:.4f})")

    # -- Logging --
    writer = SummaryWriter(log_dir=str(log_dir / "tb"))
    history = []
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or "autoencoder",
            config=vars(args),
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training autoencoder on {device}")
    print(f"  Backbone:    {args.backbone} (feat_dim={model.latent_dim})")
    print(f"  Epochs:      {start_epoch} -> {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr} (cosine -> 0)")
    print(f"  Parameters:  {n_params:,}")
    print(f"  Log dir:     {log_dir}")
    print()

    # -- Training loop --
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(
            model, loader, optimizer, criterion, device, epoch, writer, args.log_interval,
        )
        scheduler.step()

        is_eval_epoch = (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1

        if is_eval_epoch:
            lr = optimizer.param_groups[0]["lr"]
            knn_acc = knn_accuracy(
                model.encoder, eval_loaders["train"], eval_loaders["test"],
                k=20, device=device,
            )

            writer.add_scalar("train/loss_epoch", avg_loss, epoch)
            writer.add_scalar("train/lr", lr, epoch)
            writer.add_scalar("eval/knn_top1", knn_acc, epoch)

            history.append({"epoch": epoch, "loss": avg_loss, "knn_top1": knn_acc, "lr": lr})
            with open(log_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            if args.wandb:
                wandb.log({"train/loss": avg_loss, "train/lr": lr,
                           "eval/knn_top1": knn_acc, "epoch": epoch})
            print(f"Epoch {epoch:>3d}/{args.epochs}  loss={avg_loss:.6f}  "
                  f"knn={knn_acc:.4f}  lr={lr:.2e}")

        if (epoch + 1) % args.vis_interval == 0:
            log_reconstruction_grid(model, fixed_batch, writer, epoch, device)

        # -- Checkpointing --
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "best_loss": best_loss,
            "args": vars(args),
        }

        if is_best:
            torch.save(checkpoint, log_dir / "best.pt")
        if (epoch + 1) % args.save_interval == 0:
            torch.save(checkpoint, log_dir / f"checkpoint_{epoch:04d}.pt")
        torch.save(checkpoint, log_dir / "last.pt")

    writer.close()
    if args.wandb:
        wandb.finish()

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to {log_dir}")


if __name__ == "__main__":
    main()
