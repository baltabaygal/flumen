"""Quadrature routines for PDF normalization and unit flux calibration."""

from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss


@lru_cache(maxsize=4)
def _gauss(order):
    return leggauss(order)


def calibration_quadrature(m, s, y_b, delta=0.05, y_c=10.0, order=16, subdivisions=1):
    """Gauss-Legendre panels resolving the sharp boundary cutoff and broad tail."""
    lo = m + s * (y_b - delta + 1e-7)
    hi = max(40.0, m + 25.0 * s)
    cutoff_edge = lo

    edges = np.unique(
        np.concatenate(
            (
                # Near the hard boundary: geometric spacing to capture the steep slope
                cutoff_edge + s * np.geomspace(1e-8, 1.0, 41 * subdivisions),
                # Core body panels
                np.linspace(cutoff_edge + s, m + s * min(y_c, 15.0), 301 * subdivisions),
                # Handover to tail
                np.linspace(m + s * min(y_c, 15.0), np.log(400.0), 101 * subdivisions),
                # Far tail
                np.linspace(np.log(400.0), hi, 51 * subdivisions),
            )
        )
    )
    nodes, weights = _gauss(order)
    widths = np.diff(edges)[:, None]
    x = ((edges[:-1, None] + edges[1:, None]) / 2.0 + widths * nodes / 2.0).ravel()
    w = (widths * weights / 2.0).ravel()
    return x, w


def compute_normalization_and_flux(density_fn, m, s, y_b, delta=0.05, y_c=10.0):
    """Compute total probability mass M and inverse moment J."""
    x, w = calibration_quadrature(m, s, y_b, delta=delta, y_c=y_c, order=16)
    values = density_fn(x)
    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)

    mass = float(w @ values)
    inverse = float(w @ (values * np.exp(-x)))
    return mass, inverse
