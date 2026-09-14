"""Public API for the production single-body magnification PDF emulator (v3)."""

import numpy as np
from .composite import SingleBodyComposite


class MagnificationPDF:
    """Conditional image-plane magnification PDF emulator.

    Parameters
    ----------
    device : str, default="cpu"
        Computation device ("cpu", "cuda", or "mps").
    flux_mode : str, default="unit"
        "unit" enforces unit inverse-magnification moment <1/mu> = 1.0.
        "standard" enforces unit normalization integral p(mu) dmu = 1.0.
    """

    def __init__(self, device="cpu", *, flux_mode="unit"):
        self.composite = SingleBodyComposite(device=device, flux_mode=flux_mode)
        self.flux_mode = flux_mode

    def log_prob_lnmu(self, lnmu, z_s, theta):
        """Log probability density with respect to d ln(mu).

        Parameters
        ----------
        lnmu : float or array_like
            Natural log of magnification ln(mu).
        z_s : float
            Source redshift.
        theta : tuple of 6 floats
            Cosmological parameters (h, OmegaM, sigma8, OmegaB, ns, zeq).
        """
        return self.composite.log_prob_lnmu(lnmu, z_s, theta)

    def pdf_lnmu(self, lnmu, z_s, theta):
        """Probability density with respect to d ln(mu)."""
        return np.exp(self.log_prob_lnmu(lnmu, z_s, theta))

    def pdf_mu(self, mu, z_s, theta):
        """Probability density with respect to d mu.

        Parameters
        ----------
        mu : float or array_like
            Magnification factor mu (must be positive).
        z_s : float
            Source redshift.
        theta : tuple of 6 floats
            Cosmological parameters (h, OmegaM, sigma8, OmegaB, ns, zeq).
        """
        mu_arr = np.asarray(mu, dtype=np.float64)
        if np.any(mu_arr <= 0):
            raise ValueError("Magnification mu must be strictly positive.")
        return self.pdf_lnmu(np.log(mu_arr), z_s, theta) / mu_arr


def load_model(device="cpu", *, flux_mode="unit"):
    """Load the single-body production emulator."""
    return MagnificationPDF(device=device, flux_mode=flux_mode)
