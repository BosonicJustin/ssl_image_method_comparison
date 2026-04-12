"""Load shared training config and merge with CLI overrides."""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "shared.yaml"


def load_shared_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load shared.yaml and return a flat dict suitable for argparse defaults.

    Nested keys are flattened with underscores:
        training.batch_size -> batch_size
        data.num_workers    -> num_workers
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    flat: dict = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat
