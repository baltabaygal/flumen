"""Retrain the exact-lower-boundary SOSPF component on bundled data."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from flumen.model.architecture import build_flow
from flumen.model.features import edge_features
from flumen.model.scale import loc_scale_ctx
from flumen.training.data import load_training_rows
from flumen.training.transform import y_bound_from_edge, y_to_u


ROOT = Path(__file__).resolve().parents[1]
EDGE_FIT = ROOT / "model" / "calibration" / "edge_fit.json"
OUTPUT = Path(os.environ.get("OUTPUT", ROOT / "training" / "boundary_retrained.pt"))
ROWS_SCALE = float(os.environ.get("ROWS_SCALE", 1.0))
EPOCHS = int(os.environ.get("EPOCHS", 200))
K_MARGIN, BATCH, LR, PATIENCE = 4.0, 8192, 3e-3, 8
SIGMA_Z = np.array([0.2, 0.5, 1.0, 2.5, 6.0])
SIGMA_V = np.array([0.0742, 0.0794, 0.0936, 0.1226, 0.1815])
CONFIG = dict(family="sospf", context=7, transforms=3, hidden=128, depth=2,
              degree=8, polynomials=5, slope=1e-3, base="normal", seed=0)


def sigma_edge(z):
    return np.interp(np.log(np.clip(z, SIGMA_Z.min(), SIGMA_Z.max())),
                     np.log(SIGMA_Z), SIGMA_V)


def prepare(context, lnmu, fit):
    m, s = loc_scale_ctx(context, fit["scale_fit"])
    y = (lnmu-m)/s
    edge = edge_features(context[:, 0], tuple(context[:, j] for j in range(1, 7))) \
           @ np.asarray(fit["edge_coef"])
    bound = y_bound_from_edge(edge, sigma_edge(context[:, 0]), K_MARGIN)
    keep = y > bound
    return context[keep].astype(np.float32), y_to_u(y[keep], bound[keep]).astype(np.float32)


def main():
    fit = json.loads(EDGE_FIT.read_text())
    n_train = int(2_700_000*ROWS_SCALE)
    n_val = int(594_000*ROWS_SCALE)
    x_train, y_train, x_val, y_val = load_training_rows(n_train, n_val, seed=0)
    x_train, u_train = prepare(x_train, y_train, fit)
    x_val, u_val = prepare(x_val, y_val, fit)
    mean = x_train.mean(0, dtype=np.float64); std = x_train.std(0, dtype=np.float64)
    std[std == 0] = 1
    x_train = ((x_train-mean)/std).astype(np.float32)
    x_val = ((x_val-mean)/std).astype(np.float32)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    flow = build_flow(CONFIG).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=3, min_lr=1e-5)
    xt, yt = torch.from_numpy(x_train), torch.from_numpy(u_train).unsqueeze(-1)
    xv = torch.from_numpy(x_val).to(device)
    yv = torch.from_numpy(u_val).unsqueeze(-1).to(device)
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
        print(f"epoch={epoch} val_nll_u={val:.6f} lr={optimizer.param_groups[0]['lr']:.2e}")
        if val < best-1e-4:
            best, bad = val, 0
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=flow.state_dict(), config=CONFIG,
                            stats=dict(context_mean=mean.tolist(), context_std=std.tolist(),
                                       scale_fit=fit["scale_fit"], edge_coef=fit["edge_coef"],
                                       edge_target="relative", k_margin=K_MARGIN,
                                       sigma_edge_z=SIGMA_Z.tolist(), sigma_edge_v=SIGMA_V.tolist()),
                            best_val_nll=best, epoch=epoch), OUTPUT)
        else:
            bad += 1
            if bad >= PATIENCE and optimizer.param_groups[0]["lr"] <= 1.5e-5:
                break
    print(f"saved {OUTPUT}; best val NLL(u)={best:.6f}")


if __name__ == "__main__":
    main()
