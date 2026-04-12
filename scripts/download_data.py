"""Download STL-10 dataset to data/ directory."""

import argparse
import sys
from pathlib import Path

import torchvision.datasets as datasets


def main():
    parser = argparse.ArgumentParser(description="Download STL-10 dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Directory to download data into (default: <project>/data/)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading STL-10 to {data_dir} ...")

    # Download all splits — torchvision handles caching, so re-running is safe
    for split in ("train", "test", "unlabeled"):
        print(f"  [{split}] ", end="", flush=True)
        ds = datasets.STL10(root=str(data_dir), split=split, download=True)
        print(f"{len(ds)} images")

    print("\nDone. Contents:")
    for p in sorted(data_dir.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.relative_to(data_dir)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
