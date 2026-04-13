"""Training script for I-JEPA on STL-10."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Allow running as `python scripts/train_ijepa.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.knn import knn_accuracy
from models.ijepa import IJEPA
from utils.config import load_shared_config
from utils.data import get_eval_loaders, get_pretrain_loader, DATA_DIR


def parse_args():
    cfg = load_shared_config()

    parser = argparse.ArgumentParser(description="Train I-JEPA on STL-10")
    # Model
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--encoder-dim", type=int, default=384)
    parser.add_argument("--encoder-depth", type=int, default=6)
    parser.add_argument("--encoder-heads", type=int, default=6)
    parser.add_argument("--pred-depth", type=int, default=4)
    parser.add_argument("--pred-heads", type=int, default=6)
    parser.add_argument("--momentum", type=float, default=0.996)
    parser.add_argument("--num-target-blocks", type=int, default=4)
    parser.add_argument("--target-block-min", type=int, default=4)
    parser.add_argument("--target-block-max", type=int, default=8)
    # Training (defaults from configs/shared.yaml)
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--lr", type=float, default=cfg["lr"])
    parser.add_argument("--weight-decay", type=float, default=cfg["weight_decay"])
    # Data
    parser.add_argument("--num-workers", type=int, default=cfg["num_workers"])
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    # Logging & checkpoints
    parser.add_argument("--log-dir", type=str, default="results/ijepa")
    parser.add_argument("--log-interval", type=int, default=cfg["log_interval"])
    parser.add_argument("--eval-interval", type=int, default=cfg["eval_interval"],
                        help="Log epoch loss + run k-NN eval every N epochs")
    parser.add_argument("--save-interval", type=int, default=cfg["save_interval"])
    # Reproducibility
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def train_one_epoch(
    model: IJEPA,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
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

        loss = model(images)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        model.update_target()

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
    loader = get_pretrain_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )

    eval_loaders = get_eval_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )

    # -- Model --
    model = IJEPA(
        patch_size=args.patch_size,
        embed_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        pred_depth=args.pred_depth,
        pred_heads=args.pred_heads,
        momentum=args.momentum,
        num_target_blocks=args.num_target_blocks,
        target_block_min=args.target_block_min,
        target_block_max=args.target_block_max,
    ).to(device)

    # Only optimize context encoder + predictor (target is EMA)
    online_params = (
        list(model.encoder.parameters())
        + list(model.predictor.parameters())
    )
    optimizer = torch.optim.AdamW(
        online_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

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

    n_online = sum(p.numel() for p in online_params)
    print(f"Training I-JEPA on {device}")
    print(f"  Encoder:       dim={args.encoder_dim}, depth={args.encoder_depth}, heads={args.encoder_heads}")
    print(f"  Predictor:     depth={args.pred_depth}, heads={args.pred_heads}")
    print(f"  EMA momentum:  {args.momentum}")
    print(f"  Target blocks: {args.num_target_blocks} x ({args.target_block_min}-{args.target_block_max})")
    print(f"  Epochs:        {start_epoch} -> {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  LR:            {args.lr} (cosine -> 0)")
    print(f"  Online params: {n_online:,}")
    print(f"  Log dir:       {log_dir}")
    print()

    # -- Training loop --
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(
            model, loader, optimizer, device, epoch, writer, args.log_interval,
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

            print(f"Epoch {epoch:>3d}/{args.epochs}  loss={avg_loss:.6f}  "
                  f"knn={knn_acc:.4f}  lr={lr:.2e}")

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

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to {log_dir}")


if __name__ == "__main__":
    main()
