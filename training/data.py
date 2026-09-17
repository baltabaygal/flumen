"""Data loading routines for flumen training and evaluation."""

from pathlib import Path
import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
USPACE_DIR = DATA_ROOT / "u_space"


def load_uspace_split(split="train", data_dir=USPACE_DIR):
    """Load pre-processed u-space arrays: u, idx, ctx, meta."""
    path = Path(data_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"u-space dataset for split '{split}' not found at {path}")
    data = np.load(path)
    return {
        "u": data["u"],
        "idx": data["idx"],
        "ctx": data["ctx"],
        "meta": data["meta"],
    }


def load_training_uspace(data_dir=USPACE_DIR, delta=0.05):
    """Load train and validation splits for production v3 flow training."""
    tr = load_uspace_split("train", data_dir)
    va = load_uspace_split("validation", data_dir)
    return tr, va
