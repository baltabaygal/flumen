"""Loader for the exact-lower-boundary SOSPF body."""

from pathlib import Path

import numpy as np
import torch

from .architecture import build_flow
from .features import edge_features
from .scale import loc_scale


DEFAULT_MODEL = Path(__file__).resolve().parent / "checkpoints" / "boundary_body.pt"


def load_boundary_body(path=DEFAULT_MODEL, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    flow = build_flow(checkpoint["config"])
    flow.load_state_dict(checkpoint["state_dict"])
    flow = flow.to(torch.device(device)).eval()
    stats = checkpoint["stats"]
    cmean = np.asarray(stats["context_mean"], float)
    cstd = np.asarray(stats["context_std"], float)
    edge_coef = np.asarray(stats["edge_coef"], float)
    sig_z = np.asarray(stats["sigma_edge_z"], float)
    sig_v = np.asarray(stats["sigma_edge_v"], float)
    k_margin = float(stats["k_margin"])
    scale_fit = stats["scale_fit"]

    def controls(z, theta):
        theta = tuple(float(v) for v in theta)
        m, s = (float(v) for v in loc_scale(float(z), theta, scale_fit))
        feats = edge_features(np.array([float(z)]),
                              tuple(np.array([v]) for v in theta))
        y_edge = float((feats @ edge_coef).item())
        lz = np.log(np.clip(float(z), sig_z.min(), sig_z.max()))
        sigma_edge = float(np.interp(lz, np.log(sig_z), sig_v))
        return m, s, y_edge, y_edge - k_margin * sigma_edge

    def log_prob(lnmu, z, theta):
        x = np.atleast_1d(np.asarray(lnmu, dtype=np.float64))
        theta = tuple(float(v) for v in theta)
        m, s, _, y_bound = controls(z, theta)
        y = (x - m) / s
        out = np.full(x.size, -np.inf, dtype=np.float64)
        keep = y > y_bound
        if np.any(keep):
            u = np.log(y[keep] - y_bound)
            raw_context = np.array([float(z), *theta], float)
            context = (raw_context - cmean) / cstd
            context_t = torch.tensor(np.tile(context, (keep.sum(), 1)),
                                     dtype=torch.float32, device=device)
            u_t = torch.tensor(u[:, None], dtype=torch.float32, device=device)
            with torch.no_grad():
                lp_u = flow(context_t).log_prob(u_t).cpu().numpy().ravel()
            out[keep] = lp_u - u - np.log(s)
        return out

    log_prob.controls = controls
    log_prob.checkpoint = checkpoint
    return log_prob
