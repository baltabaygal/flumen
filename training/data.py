"""Load the bundled HDF5 simulation splits for model training."""

from pathlib import Path

import h5py
import numpy as np


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DATASETS = (DATA_ROOT / "production_data",
            DATA_ROOT / "production_data_lostruct_aug",
            DATA_ROOT / "production_data_lostruct_aug_r2")
AUGMENTATION_DATASET = DATA_ROOT / "production_data_highstruct_highz_aug"


def load_split(path):
    with h5py.File(path, "r") as handle:
        result = {"lnmu": handle["samples/lnmu"][:],
                  "valid_counts": handle["samples/valid_counts"][:],
                  "z": handle["samples/z"][:]}
        for key in ("h", "OmegaM", "sigma8", "OmegaB", "ns", "zeq"):
            result[key] = handle[f"samples/{key}"][:]
        return result


def contexts(split):
    return np.stack([split["z"], split["h"], split["OmegaM"], split["sigma8"],
                     split["OmegaB"], split["ns"], split["zeq"] / 1000.0], axis=-1)


def available_splits(dataset_dir):
    result = {}
    for name in ("train", "validation", "test", "heldout"):
        path = dataset_dir / name / f"dataset_{name}.h5"
        if path.exists():
            result[name] = load_split(path)
    return result


def _flatten(split):
    context = contexts(split)
    rows_context, rows_lnmu = [], []
    for i, count in enumerate(split["valid_counts"].astype(int)):
        rows_context.append(np.tile(context[i], (count, 1)))
        rows_lnmu.append(split["lnmu"][i, :count])
    return np.concatenate(rows_context), np.concatenate(rows_lnmu)


def load_training_rows(n_train=2_700_000, n_validation=594_000, seed=0):
    train_context, train_lnmu, val_context, val_lnmu = [], [], [], []
    for directory in DATASETS:
        dataset = available_splits(directory)
        context, lnmu = _flatten(dataset["train"])
        train_context.append(context); train_lnmu.append(lnmu)
        if "validation" in dataset:
            context, lnmu = _flatten(dataset["validation"])
            val_context.append(context); val_lnmu.append(lnmu)
    x_train, y_train = np.concatenate(train_context), np.concatenate(train_lnmu)
    if val_context:
        x_val, y_val = np.concatenate(val_context), np.concatenate(val_lnmu)
    else:
        x_val, y_val = x_train, y_train
    rng = np.random.default_rng(seed)
    if len(x_train) > n_train:
        take = rng.choice(len(x_train), n_train, replace=False)
        x_train, y_train = x_train[take], y_train[take]
    if len(x_val) > n_validation:
        take = rng.choice(len(x_val), n_validation, replace=False)
        x_val, y_val = x_val[take], y_val[take]
    return x_train, y_train, x_val, y_val


def _subsample(context, lnmu, target, rng):
    if len(context) > target:
        take = rng.choice(len(context), target, replace=False)
        return context[take], lnmu[take]
    return context, lnmu


def load_finetune_v1_rows(seed=0, rows_scale=1.0):
    """Recreate the additive row allocation used by the shipped v1 fine-tune.

    The three reference datasets retain their historical 2.7M/594k budget.
    The high-structure/high-z dataset contributes an additional 400k/90k rows.
    It has no named validation split, so its training pool was independently
    resampled for the validation objective, matching the 2026-09-02 run.
    """
    rng = np.random.default_rng(seed)
    old_train_context, old_train_lnmu = [], []
    old_val_context, old_val_lnmu = [], []
    for directory in DATASETS:
        dataset = available_splits(directory)
        context, lnmu = _flatten(dataset["train"])
        old_train_context.append(context); old_train_lnmu.append(lnmu)
        if "validation" in dataset:
            context, lnmu = _flatten(dataset["validation"])
            old_val_context.append(context); old_val_lnmu.append(lnmu)

    x_train = np.concatenate(old_train_context)
    y_train = np.concatenate(old_train_lnmu)
    x_val = np.concatenate(old_val_context)
    y_val = np.concatenate(old_val_lnmu)
    x_train, y_train = _subsample(
        x_train, y_train, int(2_700_000 * rows_scale), rng)
    x_val, y_val = _subsample(
        x_val, y_val, int(594_000 * rows_scale), rng)

    augmented = available_splits(AUGMENTATION_DATASET)
    x_extra, y_extra = _flatten(augmented["train"])
    # Historical v1 behavior: independently draw both allocations from the
    # train pool. The heldout file remains available for evaluation/audit only.
    x_extra_train, y_extra_train = _subsample(
        x_extra, y_extra, int(400_000 * rows_scale), rng)
    x_extra_val, y_extra_val = _subsample(
        x_extra, y_extra, int(90_000 * rows_scale), rng)

    return (np.concatenate([x_train, x_extra_train]),
            np.concatenate([y_train, y_extra_train]),
            np.concatenate([x_val, x_extra_val]),
            np.concatenate([y_val, y_extra_val]))
