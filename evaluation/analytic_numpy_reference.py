"""Independent pure-NumPy float64 evaluation of the analytic emulator's NETWORK stages (context MLP ensemble and the NSF body flows), used to verify the PyTorch-evaluated port.

``NumpyAnalyticPDF`` is :class:`flumen.model.AnalyticPDF` with ``_context`` and ``_logpU`` re-implemented without torch: float64 MLPs (SiLU conditioner), the
rational-quadratic-spline transforms of zuko 1.6 (``MonotonicRQSTransform``: softmax widths/heights, min slope 1e-3, bound 5, identity outside) and the seed mixture
(log-mean-exp).  Everything downstream (left guard, tail, blend, normalisation, planes) is the SAME NumPy code, so any difference is a network-evaluation difference.
Weights are read from the loaded torch modules (exact float32 -> float64 conversion)."""
from __future__ import annotations

import math

import numpy as np
import torch.nn as nn

from flumen.model.analytic import ANCH, DELTA, AnalyticPDF, _law_through, Y_C_REL, LNMU_C_ABS, features

_SLOPE = 1e-3


def _act(kind, x):
    assert kind == "silu"
    return x / (1.0 + np.exp(-x))


def _mlp(layers, kind, x):
    """layers: list of (W, b); activation after every layer but the last."""
    h = x
    for i, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if i < len(layers) - 1:
            h = _act(kind, h)
    return h


def _rqs(phi, z, bins):
    """zuko MonotonicRQSTransform (bound 5) applied to z (n,), parameters phi (3*bins-1,): returns (z', ladj)."""
    L = math.log(_SLOPE)
    w, h, d = phi[:bins], phi[bins:2 * bins], phi[2 * bins:]
    w = w / (1 + np.abs(2 * w / L)); h = h / (1 + np.abs(2 * h / L)); d = d / (1 + np.abs(d / L))
    sm = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
    w, h = np.concatenate([[0.0], sm(w)]), np.concatenate([[0.0], sm(h)]); d = np.exp(np.concatenate([[0.0], d, [0.0]]))
    hor, ver = 5.0 * (2 * np.cumsum(w) - 1), 5.0 * (2 * np.cumsum(h) - 1)
    k = np.searchsorted(hor, z, side="left") - 1
    mask = (k >= 0) & (k < bins); kk = np.where(mask, k, 0)
    x0, x1, y0, y1, d0, d1 = hor[kk], hor[kk + 1], ver[kk], ver[kk + 1], d[kk], d[kk + 1]
    s = (y1 - y0) / (x1 - x0)
    t = np.where(mask, (z - x0) / (x1 - x0), 0.0)
    den = s + (d0 + d1 - 2 * s) * t * (1 - t)
    y = y0 + (y1 - y0) * (s * t ** 2 + d0 * t * (1 - t)) / den
    jac = s ** 2 * (2 * s * t * (1 - t) + d0 * (1 - t) ** 2 + d1 * t ** 2) / den ** 2
    return np.where(mask, y, z), np.where(mask, np.log(jac), 0.0)


class NumpyAnalyticPDF(AnalyticPDF):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        lin = lambda seq: [(m.weight.detach().cpu().numpy().astype(np.float64), m.bias.detach().cpu().numpy().astype(np.float64)) for m in seq if hasattr(m, "weight") and m.weight.ndim == 2]
        self._np_ctx = [(lin(net.net), mu, sd) for net, mu, sd in self.ctx_nets]
        self._np_flows = []
        for f in self.flows:
            ts = []
            for t in f.transform.transforms:
                mods = list(t.hyper)
                assert any(isinstance(m, nn.SiLU) for m in mods), "only the SiLU conditioner is supported"
                ts.append((lin(mods), "silu"))
            self._np_flows.append(ts)
        self._bins = self.cfg["bins"]

    def _context(self, p):
        X = features(p)
        P = np.mean([_mlp(L, "silu", X) * sd + mu for L, mu, sd in self._np_ctx], axis=0)[0]
        g = dict(zip(self.names, P))
        s = float(np.exp(g["ln_s"])); m = float(g["m_over_s"] * s); yb = float(g["y_b"])
        anch = np.array([g[f"tail_{int(a)}"] for a in ANCH])
        return dict(m=m, s=s, yb=yb, law=_law_through(anch), lnmu_c=max(LNMU_C_ABS, m + Y_C_REL * s), cut=m + s * (yb - DELTA))

    def _logpU(self, p, u, c):
        Cnp = features(p) * 2 - 1
        if self.cfg.get("ctx_extra"):
            st = self.cfg["extra_stats"]
            e = (np.array([math.log(c["s"]), c["m"] / c["s"], c["yb"]]) - np.array(st["mu"])) / np.array(st["sd"])
            Cnp = np.column_stack([Cnp, e[None, :]])
        C = Cnp[0]
        z0 = (np.asarray(u, dtype=np.float64) - self.U0) / self.US
        lp = []
        for ts in self._np_flows:
            z, ladj = z0.copy(), np.zeros_like(z0)
            for layers, kind in ts:
                z, la = _rqs(_mlp(layers, kind, C), z, self._bins); ladj += la
            lp.append(-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) + ladj)
        lp = np.array(lp)
        mx = lp.max(0)
        out = mx + np.log(np.mean(np.exp(lp - mx), axis=0))
        return out - math.log(self.US)
