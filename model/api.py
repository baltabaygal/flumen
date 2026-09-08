"""Stable public API for the production candidate."""

import numpy as np

from .composite import make_composite
from .hybrid_body import load_hybrid_body


class MagnificationPDF:
    """Conditional image-plane magnification PDF.

    ``theta`` is ``(h, OmegaM, sigma8, OmegaB, ns, zeq/1000)``.
    Default output enforces the positive-PDF unit inverse-mean convention.
    ``flux_mode="legacy"`` retains the previous capped calibration. Unit mode
    shifts the magnification scale; it does not preserve fixed-mu tail anchors.
    """

    def __init__(self, device="cpu", *, flux_mode="unit"):
        self._log_prob = make_composite(load_hybrid_body(device=device), flux_mode=flux_mode)
        self.flux_mode = flux_mode

    def log_prob_lnmu(self, lnmu, z_s, theta):
        """Log density with respect to ``d ln(mu)``."""
        return self._log_prob(lnmu, z_s, theta)

    def pdf_lnmu(self, lnmu, z_s, theta):
        return np.exp(self.log_prob_lnmu(lnmu, z_s, theta))

    def pdf_mu(self, mu, z_s, theta):
        """Density with respect to ``d mu``."""
        mu = np.asarray(mu, dtype=np.float64)
        if np.any(mu <= 0):
            raise ValueError("magnification mu must be positive")
        return self.pdf_lnmu(np.log(mu), z_s, theta) / mu


def load_model(device="cpu", *, flux_mode="unit"):
    """Load with unit inverse mean; use flux_mode='legacy' for old calibration."""
    return MagnificationPDF(device=device, flux_mode=flux_mode)
