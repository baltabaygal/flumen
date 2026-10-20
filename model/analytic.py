"""Analytic-chain backend: an emulator of the semi-analytic compound-Poisson (Levy-Khintchine) magnification PDF.

Trained on 3,903 deterministic PDFs of the analytic chain (stochastic_wl_analytic, draft model: r200 halo edges,
xi_lin Gaussian clustering, exact moment subhalos, filaments, pseudo-elliptical halos; training resolution "aggr2p"),
cleaned of Fourier-inversion artifacts before training. Self-contained inference code; the training pipeline lives in
stochastic_wl_analytic/learning (README sections 6-11).

Native quantity: the SOURCE-plane PDF P_S(ln mu) (random source positions; mu^-2 tail in ln mu). The IMAGE-plane PDF
(random lines of sight, the quantity of the MC backend) is P_I(ln mu) = mu P_S(ln mu) / <mu>_S.

Pieces (as validated in stochastic_wl_analytic/learning/emulator.py):
  context   MLP ensemble (5 x 3 x 128) -> m, s (median / robust width of ln mu), y_b (cutoff in widths), ln P at ln mu 4..9
  body      zuko NSF flow (2 seeds, mixture) for p_U(u), u = ln(y - y_b + DELTA), y = (ln mu - m)/s; context = the 10
            parameters + the predicted (ln s, m/s, y_b);  P_S(ln mu) = p_U(u) e^-u / s, exactly 0 below the cutoff
  left      below where the body first drops under EPS_LEFT of its peak: the log-quadratic edge law in t = ln mu - cut
  tail      beyond ln mu_c = min(max(ln 20, m + 10 s), where the body < P_HANDOVER of its peak): ln P = a - 2 ln mu
            + b/mu + c/mu^2, a predicted, (b, c) matching the body's value and slope; smoothstep blend of +-BLEND
  norm      quadrature of the joined density (analytic mu^-2 remainder beyond ln mu = 12)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import zuko

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / "analytic"

AXES = ("z_s", "h", "Om", "sigma8", "Ob", "ns", "zeq", "Ok", "w0", "wa")
BOX = dict(z_s=(0.5, 12.0), h=(0.55, 0.85), Om=(0.15, 0.50), sigma8=(0.40, 1.50), Ob=(0.03, 0.07),
           ns=(0.94, 0.99), zeq=(3300.0, 3500.0), Ok=(-0.15, 0.05), w0=(-1.5, 0.5), wa=(-2.0, 1.0))
LOG_AXES = ("z_s", "sigma8", "Om")
ANCH = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
DELTA = 0.2
LNMU_C_ABS, Y_C_REL = math.log(20.0), 10.0
LNMU_TAIL_MAX = 12.0
EPS_LEFT = 1e-8
P_HANDOVER = 1e-5
BLEND = 0.15


def features(params):
    """(n, 10) parameters in AXES order -> the unit box (log for z_s, sigma8, Omega_m)."""
    params = np.atleast_2d(np.asarray(params, dtype=np.float64))
    cols = []
    for i, k in enumerate(AXES):
        v, (lo, hi) = params[:, i], BOX[k]
        if k in LOG_AXES:
            v, lo, hi = np.log(v), np.log(lo), np.log(hi)
        cols.append((v - lo) / (hi - lo))
    return np.column_stack(cols)


class _CtxMLP(nn.Module):
    def __init__(self, n_in, n_out, hidden, depth):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _build_flow(cfg):
    return zuko.flows.NSF(features=1, context=cfg.get("n_ctx", 10), transforms=cfg["transforms"],
                          hidden_features=[cfg["hidden"]] * cfg["depth"], bins=cfg["bins"])


def _law_through(lnP_anch):
    X = np.column_stack([np.ones_like(ANCH), np.exp(-ANCH), np.exp(-2 * ANCH)])
    c, *_ = np.linalg.lstsq(X, lnP_anch + 2 * ANCH, rcond=None)
    return c


def _law(c, x):
    return c[0] - 2 * x + c[1] * np.exp(-x) + c[2] * np.exp(-2 * x)


def _hybrid_bc(xR, yR, sR, a):
    e = np.exp(-xR)
    M = np.array([[e, e * e], [-e, -2 * e * e]])
    b, c = np.linalg.solve(M, [yR - a + 2 * xR, sR + 2])
    return float(b), float(c)


class AnalyticPDF:
    """Source/image-plane magnification PDF of the analytic chain, conditional on (z_s, 9 cosmological parameters).

    Parameters are given as ``params = (z_s, h, Om, sigma8, Ob, ns, zeq, Ok, w0, wa)`` (zeq in absolute units).
    ``plane="source"`` (default, as for ``flumen.generate_pdf`` since 0.4.0; the chain's native quantity: random source
    positions, the one for standard sirens) or ``plane="image"`` (random lines of sight, the MC rays' measure).
    Densities are with respect to d ln(mu) unless stated otherwise.
    """

    def __init__(self, device="cpu", ckpt_dir=CKPT_DIR, seeds=(0, 1)):
        self.device = torch.device(device)
        ck = torch.load(Path(ckpt_dir) / "ctx.pt", map_location=self.device, weights_only=True)
        self.names = list(ck["names"])
        self.ctx_nets = []
        for mem in ck["members"]:
            net = _CtxMLP(10, len(self.names), ck["hidden"], ck["depth"]).to(self.device)
            net.load_state_dict(mem["state"]); net.eval()
            self.ctx_nets.append((net, mem["mu"].numpy(), mem["sd"].numpy()))
        self.flows = []
        for s in seeds:
            b = torch.load(Path(ckpt_dir) / f"body_s{s}.pt", map_location=self.device, weights_only=True)
            f = _build_flow(b["cfg"]).to(self.device); f.load_state_dict(b["state"]); f.eval()
            self.flows.append(f)
            self.cfg, self.U0, self.US = b["cfg"], float(b["U0"]), float(b["US"])
        self._cache = {}

    # ------------------------------------------------------------ context + body
    def _context(self, p):
        X = torch.tensor(features(p), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            P = np.mean([net(X).cpu().numpy() * sd + mu for net, mu, sd in self.ctx_nets], axis=0)[0]
        g = dict(zip(self.names, P))
        s = float(np.exp(g["ln_s"])); m = float(g["m_over_s"] * s); yb = float(g["y_b"])
        anch = np.array([g[f"tail_{int(a)}"] for a in ANCH])
        return dict(m=m, s=s, yb=yb, law=_law_through(anch), lnmu_c=max(LNMU_C_ABS, m + Y_C_REL * s),
                    cut=m + s * (yb - DELTA))

    def _logpU(self, p, u, c):
        Cnp = features(p) * 2 - 1
        if self.cfg.get("ctx_extra"):
            st = self.cfg["extra_stats"]
            e = (np.array([math.log(c["s"]), c["m"] / c["s"], c["yb"]]) - np.array(st["mu"])) / np.array(st["sd"])
            Cnp = np.column_stack([Cnp, e[None, :]])
        C = torch.tensor(Cnp, dtype=torch.float32, device=self.device)
        us = torch.tensor(((np.asarray(u) - self.U0) / self.US)[:, None, None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            lp = np.array([f(C).log_prob(us)[:, 0].cpu().numpy() for f in self.flows], dtype=np.float64)
        mx = lp.max(0)
        with np.errstate(invalid="ignore"):
            out = mx + np.log(np.mean(np.exp(lp - mx), axis=0))                    # mixture of the seeds
        return np.where(np.isfinite(mx), out, -np.inf) - math.log(self.US)

    def _flow_lnP(self, p, c, x):
        x = np.asarray(x, float)
        out = np.full(x.shape, -np.inf)
        ok = x > c["cut"]
        if ok.any():
            y = (x[ok] - c["m"]) / c["s"]
            u = np.log(y - c["yb"] + DELTA)
            out[ok] = self._logpU(p, u, c) - u - math.log(c["s"])
        return out

    def _left_guard(self, p, c):
        s, cut = c["s"], c["cut"]
        x = cut + np.geomspace(1e-4, 1.0, 1200) * (c["m"] + 3 * s - cut)
        lp = self._flow_lnP(p, c, x)
        ipk = int(np.argmax(lp)); lpk = lp[ipk]
        left = lp[:ipk + 1][::-1]
        rise = np.flatnonzero(np.diff(left) > 0)
        low = np.flatnonzero(left < lpk + math.log(EPS_LEFT))
        k_rise = rise[0] if rise.size else len(left)
        k_low = low[0] if low.size else len(left)
        k = k_low if k_low <= k_rise else max(1, int(0.7 * k_rise))
        if k >= len(left):
            return None
        xE = x[ipk - k]
        h = 0.01 * s
        a0, a1, a2 = self._flow_lnP(p, c, np.array([xE - h, xE, xE + h]))
        d1, d2 = (a2 - a0) / (2 * h), (a2 - 2 * a1 + a0) / h ** 2
        tE = xE - cut
        g1, g2 = tE * d1, tE * d1 + tE ** 2 * d2
        beta = max(-0.5 * g2, 0.0)
        n = g1 + 2 * beta * math.log(tE)
        return (xE, a1, n, beta) if g1 > 0 else None

    def _body_lnP(self, p, c, x):
        x = np.asarray(x, float)
        out = self._flow_lnP(p, c, x)
        g = c.get("left")
        if g is not None:
            xE, lE, n, beta = g
            m = (x > c["cut"]) & (x < xE)
            if m.any():
                L, LE = np.log(x[m] - c["cut"]), math.log(xE - c["cut"])
                out[m] = lE + n * (L - LE) - beta * (L ** 2 - LE ** 2)
        return out

    # ------------------------------------------------------------ tail + join
    @staticmethod
    def _tail_lnP(tail, x):
        a, b, cc = tail[1]
        return a - 2 * x + b * np.exp(-x) + cc * np.exp(-2 * x)

    def _tail(self, p, c):
        xc, h = c["lnmu_c"], 1e-3
        yv = self._body_lnP(p, c, np.array([xc - h, xc, xc + h]))
        y0, s0 = yv[1], (yv[2] - yv[0]) / (2 * h)
        a = c["law"][0]
        b, cc = _hybrid_bc(xc, y0, s0, a)
        xt = np.linspace(xc, LNMU_TAIL_MAX, 2000)
        if np.all(np.diff(self._tail_lnP(("hybrid", (a, b, cc)), xt)) < 0):
            return ("hybrid", (a, b, cc))
        law = c["law"].copy()
        law[0] += y0 - _law(c["law"], xc)
        return ("law", tuple(law))

    def _blend_window(self, c):
        return max(min(BLEND, 0.5 * (c["lnmu_c"] - (c["m"] - 0.5 * c["s"]))), 1e-3)

    def _joined(self, p, c, x):
        x = np.asarray(x, float)
        D, xc = self._blend_window(c), c["lnmu_c"]
        out = np.empty(x.shape)
        lo, hi = x <= xc - D, x >= xc + D
        mid = ~(lo | hi)
        if lo.any():
            out[lo] = self._body_lnP(p, c, x[lo])
        if hi.any():
            out[hi] = self._tail_lnP(c["tail"], x[hi])
        if mid.any():
            s = (x[mid] - (xc - D)) / (2 * D)
            w = s * s * s * (10 - 15 * s + 6 * s * s)
            out[mid] = (1 - w) * self._body_lnP(p, c, x[mid]) + w * self._tail_lnP(c["tail"], x[mid])
        return out

    def _moments(self, p, c):
        """(mass, <mu>-integral) of the joined, unnormalised density."""
        D = self._blend_window(c)
        xa = c["lnmu_c"] - D
        u_a = math.log((xa - c["m"]) / c["s"] - c["yb"] + DELTA)
        u = np.linspace(math.log(1e-9), u_a, 8001)
        x = c["m"] + c["s"] * (np.exp(u) + c["yb"] - DELTA)
        pb = np.exp(self._body_lnP(p, c, x)) * c["s"] * np.exp(u)
        xm = np.linspace(xa, c["lnmu_c"] + D, 801)
        pm = np.exp(self._joined(p, c, xm))
        xt = np.linspace(c["lnmu_c"] + D, LNMU_TAIL_MAX, 4001)
        pt = np.exp(self._tail_lnP(c["tail"], xt))
        mass = np.trapezoid(pb, u) + np.trapezoid(pm, xm) + np.trapezoid(pt, xt) + 0.5 * pt[-1]
        mu1 = (np.trapezoid(pb * np.exp(x), u) + np.trapezoid(pm * np.exp(xm), xm)
               + np.trapezoid(pt * np.exp(xt), xt) + pt[-1] * np.exp(xt[-1]))   # mu^-2 remainder: P(12) e^12
        return mass, mu1

    # ------------------------------------------------------------ public
    def prepare(self, params):
        key = tuple(float(v) for v in params)
        if key in self._cache:
            return self._cache[key]
        p = np.asarray(params, dtype=np.float64)
        c = self._context(p)
        c["left"] = self._left_guard(p, c)
        xs = np.linspace(c["m"], c["lnmu_c"], 800)
        lp = self._flow_lnP(p, c, xs)
        ipk = int(np.argmax(lp))
        below = np.flatnonzero((np.arange(len(xs)) > ipk) & (lp < lp[ipk] + math.log(P_HANDOVER)))
        if below.size:
            c["lnmu_c"] = max(float(xs[below[0]]), c["m"] + 5 * c["s"])
        c["tail"] = self._tail(p, c)
        mass, mu1 = self._moments(p, c)
        c["norm"], c["mean_mu"] = mass, mu1 / mass
        if len(self._cache) > 256:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = c
        return c

    def log_prob_lnmu(self, lnmu, params, plane="source"):
        """ln of the density with respect to d ln(mu); source plane by default, ``plane="image"`` for lines of sight."""
        if plane not in ("source", "image"):
            raise ValueError("plane must be 'source' or 'image'")
        c = self.prepare(params)
        x = np.asarray(lnmu, dtype=np.float64)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.full(x.shape, -np.inf)
        ok = x > c["cut"]
        out[ok] = self._joined(np.asarray(params, float), c, x[ok]) - math.log(c["norm"])
        if plane == "image":
            out = out + x - math.log(c["mean_mu"])
        return float(out[0]) if scalar else out

    def pdf_lnmu(self, lnmu, params, plane="source"):
        return np.exp(self.log_prob_lnmu(lnmu, params, plane))

    def pdf_mu(self, mu, params, plane="source"):
        """Density with respect to d mu."""
        mu = np.asarray(mu, dtype=np.float64)
        if np.any(mu <= 0):
            raise ValueError("Magnification mu must be strictly positive.")
        return self.pdf_lnmu(np.log(mu), params, plane) / mu

    def mean_mu_source(self, params):
        """<mu> of the source-plane PDF (= 1/<1/mu> of the image-plane one); the chain's second-order flux deficit
        makes it differ from 1 (median -0.6% on the training set, down to -9% at high z_s / sigma8)."""
        return float(self.prepare(params)["mean_mu"])


def load_analytic_model(device="cpu"):
    """Load the analytic-chain emulator (2-seed ensemble, trained on train + validation)."""
    return AnalyticPDF(device=device)
