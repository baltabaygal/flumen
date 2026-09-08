"""Smooth high-redshift/left-flank hybrid of the two frozen SOSPF bodies."""

import math
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Distribution, Independent, constraints
from zuko.lazy import UnconditionalDistribution

from .architecture import build_flow
from .boundary_body import load_boundary_body
from .scale import loc_scale


HERE = Path(__file__).resolve().parent
ASYMMETRIC_MODEL = HERE / "checkpoints" / "asymmetric_body_sl0p6.pt"
Z_BLEND_LO, Z_BLEND_HI = 2.0, 3.5
Y_BLEND_LO, Y_BLEND_HI = -1.0, 0.0
ISLAND_S8_LO, ISLAND_S8_HI = 0.9, 1.5
ISLAND_Y_FAR, ISLAND_Y_NEAR, ISLAND_WIDTH = -4.0, -2.55, 0.25
FLANK_SHIFT_MAX, FLANK_SHIFT_Z_POWER = 0.18, 0.7


class SplitNormal(Distribution):
    arg_constraints = {}
    support = constraints.real
    has_rsample = True

    def __init__(self, sigma_left, sigma_right, validate_args=None):
        self.sigma_left = torch.as_tensor(sigma_left)
        self.sigma_right = torch.as_tensor(sigma_right)
        batch = torch.broadcast_shapes(self.sigma_left.shape, self.sigma_right.shape)
        super().__init__(batch_shape=batch, validate_args=validate_args)

    def log_prob(self, y):
        y = torch.as_tensor(y)
        sigma = torch.where(y < 0, self.sigma_left, self.sigma_right)
        log_c = 0.5 * math.log(2.0 / math.pi) - torch.log(
            self.sigma_left + self.sigma_right)
        return log_c - 0.5 * (y / sigma) ** 2

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SplitNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.sigma_left = self.sigma_left.expand(batch_shape)
        new.sigma_right = self.sigma_right.expand(batch_shape)
        super(SplitNormal, new).__init__(batch_shape, validate_args=False)
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, device=self.sigma_left.device)
        p_left = self.sigma_left / (self.sigma_left + self.sigma_right)
        left = -torch.abs(torch.randn(shape, device=self.sigma_left.device)) * self.sigma_left
        right = torch.abs(torch.randn(shape, device=self.sigma_left.device)) * self.sigma_right
        return torch.where(u < p_left, left, right)


def compact_switch(x, lo, hi):
    x = np.asarray(x, dtype=np.float64)
    t = (x - lo) / (hi - lo)
    out = np.zeros_like(t)
    out[t >= 1.0] = 1.0
    mid = (t > 0.0) & (t < 1.0)
    tm = t[mid]
    logit = -1.0 / tm + 1.0 / (1.0 - tm)
    out[mid] = 1.0 / (1.0 + np.exp(-np.clip(logit, -700.0, 700.0)))
    return out


def _load_asymmetric_body(path=ASYMMETRIC_MODEL, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    flow = build_flow(checkpoint["config"])
    stats = checkpoint["stats"]
    sl, sr = float(stats["sigma_left"]), float(stats["sigma_right"])
    flow.base = UnconditionalDistribution(
        lambda a, b: Independent(SplitNormal(a, b), 1),
        torch.tensor([sl]), torch.tensor([sr]), buffer=True)
    flow.load_state_dict(checkpoint["state_dict"])
    flow = flow.to(torch.device(device)).eval()
    cmean = np.asarray(stats["context_mean"], float)
    cstd = np.asarray(stats["context_std"], float)
    scale_fit = stats["scale_fit"]

    def log_prob(lnmu, z, theta):
        x = np.atleast_1d(np.asarray(lnmu, dtype=np.float64))
        theta = tuple(float(v) for v in theta)
        m, s = (float(v) for v in loc_scale(float(z), theta, scale_fit))
        y = (x - m) / s
        context = (np.array([float(z), *theta]) - cmean) / cstd
        context_t = torch.tensor(np.tile(context, (x.size, 1)),
                                 dtype=torch.float32, device=device)
        y_t = torch.tensor(y[:, None], dtype=torch.float32, device=device)
        with torch.no_grad():
            lp_y = flow(context_t).log_prob(y_t).cpu().numpy().ravel()
        return lp_y - np.log(s)

    log_prob.checkpoint = checkpoint
    return log_prob


def load_hybrid_body(device="cpu"):
    v1 = load_boundary_body(device=device)
    v2 = _load_asymmetric_body(device=device)

    def log_prob(lnmu, z, theta):
        x = np.atleast_1d(np.asarray(lnmu, dtype=np.float64))
        theta = tuple(float(v) for v in theta)
        m, s, _, _ = v1.controls(float(z), theta)
        y = (x - m) / s
        wz = float(compact_switch([np.log(float(z))],
                                  np.log(Z_BLEND_LO), np.log(Z_BLEND_HI))[0])
        w = wz * (1.0 - compact_switch(y, Y_BLEND_LO, Y_BLEND_HI))
        l1 = np.asarray(v1(x, z, theta), float)
        structure = theta[2] * np.sqrt(theta[1] / 0.3)
        high_structure = float(compact_switch(
            [structure], ISLAND_S8_LO, ISLAND_S8_HI)[0])
        flank_shift = (high_structure * FLANK_SHIFT_MAX
                       * (3.5 / max(float(z), 3.5)) ** FLANK_SHIFT_Z_POWER)
        l2 = np.asarray(v2(x - s * flank_shift, z, theta), float)
        p1 = np.zeros_like(l1)
        finite = np.isfinite(l1)
        p1[finite] = np.exp(np.clip(l1[finite], -745.0, 700.0))
        p2 = np.exp(np.clip(l2, -745.0, 700.0))
        floor = ISLAND_Y_FAR + high_structure * (ISLAND_Y_NEAR - ISLAND_Y_FAR)
        p2 *= compact_switch(y, floor, floor + ISLAND_WIDTH)
        p = (1.0 - w) * p1 + w * p2
        out = np.full(x.size, -np.inf)
        positive = p > 0.0
        out[positive] = np.log(p[positive])
        return out

    log_prob.controls = v1.controls
    log_prob.v1 = v1
    log_prob.v2 = v2
    return log_prob
