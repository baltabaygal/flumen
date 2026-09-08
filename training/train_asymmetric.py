"""Retrain the split-normal SOSPF component on bundled data."""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Independent
from zuko.lazy import UnconditionalDistribution

from flumen.model.architecture import build_flow
from flumen.model.hybrid_body import SplitNormal
from flumen.model.scale import loc_scale_ctx
from flumen.training.data import load_training_rows


ROOT = Path(__file__).resolve().parents[1]
EDGE_FIT = ROOT / "model" / "calibration" / "edge_fit.json"
OUTPUT = Path(os.environ.get("OUTPUT", ROOT / "training" / "asymmetric_retrained.pt"))
ROWS_SCALE = float(os.environ.get("ROWS_SCALE", 1.0))
EPOCHS = int(os.environ.get("EPOCHS", 200))
BATCH, LR, PATIENCE = 8192, 3e-3, 8
SIGMA_LEFT = float(os.environ.get("SIGMA_LEFT", 0.6))
SIGMA_RIGHT = float(os.environ.get("SIGMA_RIGHT", 1.3))
CONFIG = dict(family="sospf", context=7, transforms=3, hidden=128, depth=2,
              degree=8, polynomials=5, slope=1e-3, base="normal", seed=0)


def main():
    scale_fit = json.loads(EDGE_FIT.read_text())["scale_fit"]
    n_train, n_val = int(2_700_000*ROWS_SCALE), int(594_000*ROWS_SCALE)
    x_train, lnmu_train, x_val, lnmu_val = load_training_rows(n_train, n_val, seed=0)
    mt, st = loc_scale_ctx(x_train, scale_fit); mv, sv = loc_scale_ctx(x_val, scale_fit)
    y_train = ((lnmu_train-mt)/st).astype(np.float32)
    y_val = ((lnmu_val-mv)/sv).astype(np.float32)
    mean = x_train.mean(0, dtype=np.float64); std = x_train.std(0, dtype=np.float64)
    std[std == 0] = 1
    x_train = ((x_train-mean)/std).astype(np.float32)
    x_val = ((x_val-mean)/std).astype(np.float32)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    flow = build_flow(CONFIG)
    flow.base = UnconditionalDistribution(
        lambda left, right: Independent(SplitNormal(left, right), 1),
        torch.tensor([SIGMA_LEFT]), torch.tensor([SIGMA_RIGHT]), buffer=True)
    flow = flow.to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=3, min_lr=1e-5)
    xt, yt = torch.from_numpy(x_train), torch.from_numpy(y_train).unsqueeze(-1)
    xv = torch.from_numpy(x_val).to(device)
    yv = torch.from_numpy(y_val).unsqueeze(-1).to(device)
    best, bad = float("inf"), 0
    for epoch in range(EPOCHS):
        flow.train(); permutation = torch.randperm(len(xt))
        for i in range(0, len(xt), BATCH):
            ids = permutation[i:i+BATCH]
            loss = -flow(xt[ids].to(device)).log_prob(yt[ids].to(device)).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); optimizer.step()
        flow.eval()
        with torch.no_grad():
            losses = [float(-flow(xv[i:i+BATCH]).log_prob(yv[i:i+BATCH]).mean())
                      * len(xv[i:i+BATCH]) for i in range(0, len(xv), BATCH)]
        val = sum(losses)/len(xv); scheduler.step(val)
        print(f"epoch={epoch} val_nll_y={val:.6f} lr={optimizer.param_groups[0]['lr']:.2e}")
        if val < best-1e-4:
            best, bad = val, 0
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=flow.state_dict(), config=CONFIG,
                            stats=dict(context_mean=mean.tolist(), context_std=std.tolist(),
                                       scale_fit=scale_fit, base="splitnormal",
                                       sigma_left=SIGMA_LEFT, sigma_right=SIGMA_RIGHT),
                            best_val_nll=best, epoch=epoch), OUTPUT)
        else:
            bad += 1
            if bad >= PATIENCE and optimizer.param_groups[0]["lr"] <= 1.5e-5:
                break
    print(f"saved {OUTPUT}; best val NLL(y)={best:.6f}")


if __name__ == "__main__":
    main()
