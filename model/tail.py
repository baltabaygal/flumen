"""Frozen calibrated peaks-over-threshold tail in log-magnification space."""

import json
from pathlib import Path

import numpy as np

from .features import tail_features
from .hybrid_body import compact_switch


FIT_PATH = Path(__file__).resolve().parent / "calibration" / "tail_fit.json"
POT = dict(u2=2.0, u3=3.0, u8=8.0, width=0.30, wsp=0.25,
           k_min=1.0, k_max=8.0, anchor="u3")
_FIT = None


def _fit():
    global _FIT
    if _FIT is None:
        _FIT = json.loads(FIT_PATH.read_text())
    return _FIT


def _softplus(t):
    return np.where(t > 30, t, np.log1p(np.exp(np.minimum(t, 30.0))))


# Precomputed constants for exact S(8) anchor preservation
X2, X3, X8 = float(np.log(POT["u2"])), float(np.log(POT["u3"])), float(np.log(POT["u8"]))
DX = X8 - X3
W_SP = POT["wsp"]

_SP_DX_W = float(_softplus(DX / W_SP))
_SP_0 = float(_softplus(0.0))
_SP_NEG_DX_W = float(_softplus(-DX / W_SP))
_A = W_SP * (_SP_DX_W - _SP_0)
_B = W_SP * (_SP_0 - _SP_NEG_DX_W)
_A_MINUS_B = _A - _B


def pot_tail_logp(lnmu, z, theta):
    """Calibrated peaks-over-threshold tail with exact S(8) anchor preservation.

    The intermediate slope k2 is derived in closed form to ensure the smooth
    survival curve passes through both S(3) and S(8) exactly, eliminating the
    softplus corner-smoothing erosion at mu=8 without post-hoc switch ramps.
    """
    fit = _fit()
    features = tail_features(z, theta)
    ln_s2 = float(features @ np.asarray(fit["coef"]["2.0"]))
    ln_s3 = float(features @ np.asarray(fit["coef"]["3.0"]))
    ln_s8 = float(features @ np.asarray(fit["coef"]["8.0"]))

    k1 = np.clip((ln_s2 - ln_s3) / (X3 - X2), POT["k_min"], POT["k_max"])
    # Closed-form k2 that satisfies ln_survival(X8) == ln_s8 exactly
    k2 = np.clip((ln_s3 - ln_s8 - k1 * (DX - _A) - _B) / _A_MINUS_B, 1.05, POT["k_max"])

    x = np.asarray(lnmu, dtype=np.float64)
    t3, t8 = (x - X3) / W_SP, (x - X8) / W_SP
    ln_survival = (ln_s3 - k1 * (x - X3)
                   + (k1 - k2) * W_SP * (_softplus(t3) - _SP_0)
                   + (k2 - 1.0) * W_SP * (_softplus(t8) - _SP_NEG_DX_W))
    slope = (k1 - (k1 - k2) / (1.0 + np.exp(-t3))
             - (k2 - 1.0) / (1.0 + np.exp(-t8)))
    return np.log(np.maximum(slope, 1e-10)) + ln_survival


def pot_tail_logp_corrected(lnmu, z, theta, ramp_lo=None):
    """Compatibility alias: pot_tail_logp now preserves the S(8) anchor by construction."""
    return pot_tail_logp(lnmu, z, theta)

