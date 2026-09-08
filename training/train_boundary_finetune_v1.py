"""Reproduce the selected epoch-11 boundary-body fine-tuning recipe.

This is a warm-started research training program. It writes a new checkpoint
under ``training/`` and never overwrites the frozen production checkpoint.
"""

import os
from pathlib import Path
import time

import numpy as np
import torch

from flumen.model.architecture import build_flow
from flumen.model.features import edge_features
from flumen.model.scale import loc_scale_ctx
from flumen.training.data import load_finetune_v1_rows
from flumen.training.transform import y_bound_from_edge, y_to_u


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "training" / "starting_checkpoints" / "boundary_body_epoch14.pt"
OUTPUT = Path(os.environ.get(
    "OUTPUT", ROOT / "training" / "boundary_finetune_v1_retrained.pt"))
ROWS_SCALE = float(os.environ.get("ROWS_SCALE", 1.0))
EPOCHS = int(os.environ.get("EPOCHS", 40))
BATCH = 8192
LR = 3e-4
PATIENCE = 8


def prepare(context, lnmu, stats):
    m, s = loc_scale_ctx(context, stats["scale_fit"])
    y = (lnmu - m) / s
    edge = edge_features(
        context[:, 0], tuple(context[:, j] for j in range(1, 7))) \
        @ np.asarray(stats["edge_coef"])
    z_nodes = np.asarray(stats["sigma_edge_z"])
    sigma_nodes = np.asarray(stats["sigma_edge_v"])
    sigma = np.interp(
        np.log(np.clip(context[:, 0], z_nodes.min(), z_nodes.max())),
        np.log(z_nodes), sigma_nodes)
    bound = y_bound_from_edge(edge, sigma, stats["k_margin"])
    keep = y > bound
    return (context[keep].astype(np.float32),
            y_to_u(y[keep], bound[keep]).astype(np.float32))


def main():
    original = torch.load(START, map_location="cpu", weights_only=False)
    stats = original["stats"]
    x_train, y_train, x_val, y_val = load_finetune_v1_rows(
        seed=0, rows_scale=ROWS_SCALE)
    x_train, u_train = prepare(x_train, y_train, stats)
    x_val, u_val = prepare(x_val, y_val, stats)

    mean = np.asarray(stats["context_mean"])
    std = np.asarray(stats["context_std"])
    std_safe = std.copy(); std_safe[std_safe == 0] = 1.0
    x_train = ((x_train - mean) / std_safe).astype(np.float32)
    x_val = ((x_val - mean) / std_safe).astype(np.float32)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    flow = build_flow(original["config"]).to(device)
    flow.load_state_dict(original["state_dict"])
    optimizer = torch.optim.Adam(flow.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=3, min_lr=1e-6)

    xt = torch.from_numpy(x_train)
    ut = torch.from_numpy(u_train).unsqueeze(-1)
    xv = torch.from_numpy(x_val).to(device)
    uv = torch.from_numpy(u_val).unsqueeze(-1).to(device)

    def validation_nll():
        flow.eval(); total = 0.0
        with torch.no_grad():
            for start in range(0, len(xv), BATCH):
                loss = -flow(xv[start:start+BATCH]).log_prob(
                    uv[start:start+BATCH]).mean()
                total += float(loss) * len(xv[start:start+BATCH])
        return total / len(xv)

    best = validation_nll()
    bad = 0
    started = time.time()
    print(f"baseline val_nll(u)={best:.6f}; device={device}")
    for epoch in range(EPOCHS):
        flow.train(); permutation = torch.randperm(len(xt))
        for start in range(0, len(xt), BATCH):
            ids = permutation[start:start+BATCH]
            loss = -flow(xt[ids].to(device)).log_prob(ut[ids].to(device)).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            optimizer.step()
        val = validation_nll(); scheduler.step(val)
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch} val_nll_u={val:.6f} lr={lr:.2e} "
              f"elapsed={time.time()-started:.0f}s")
        if val < best - 1e-5:
            best, bad = val, 0
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": {k: v.cpu().clone() for k, v in flow.state_dict().items()},
                "config": original["config"],
                "stats": stats,
                "best_val_nll": best,
                "epoch": epoch,
                "finetuned_from": str(START),
            }, OUTPUT)
        else:
            bad += 1
            if bad >= PATIENCE and lr <= 1.5e-6:
                break
    print(f"saved {OUTPUT}; best val NLL(u)={best:.6f}")


if __name__ == "__main__":
    main()
