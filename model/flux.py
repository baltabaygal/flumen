"""Full-range quadrature for the explicit unit-inverse-magnification convention."""
import numpy as np
from functools import lru_cache
from numpy.polynomial.legendre import leggauss


class FluxCalibrationError(RuntimeError):
    """A context could not be calibrated to the requested numerical accuracy."""


def calibration_grid(body, z, theta, resolution=1):
    m, s, _, boundary = body.controls(z, theta)
    if not np.isfinite(m+s+boundary) or s <= 0:
        raise FluxCalibrationError('invalid body location, scale or support')
    # Both body branches have exact support gates. The hybrid branch begins
    # no earlier than standardized coordinate -4; the v1 bound is explicit.
    lo = min(-8., m+s*min(boundary, -4.)-1.)
    hi = max(40., m+20*s)
    return np.unique(np.concatenate((
        np.linspace(lo, np.log(400.), 3001*resolution),
        np.linspace(max(lo, m-14*s), min(hi, m+20*s), 6001*resolution),
        np.linspace(np.log(400.), hi, 1601*resolution))))


@lru_cache(maxsize=4)
def _gauss(order):
    return leggauss(order)


def calibration_quadrature(body, z, theta, order=16, subdivisions=1):
    """Gauss-Legendre panels resolve the narrow body without trapezoid aliasing."""
    m, s, _, boundary = body.controls(z, theta)
    if not np.isfinite(m+s+boundary) or s <= 0:
        raise FluxCalibrationError('invalid body location, scale or support')
    lo = min(-8., m+s*min(boundary, -4.)-1.)
    hi = max(40., m+20*s)
    edges = np.unique(np.concatenate((
        np.linspace(lo, np.log(400.), 201*subdivisions),
        m+s*np.linspace(min(boundary, -4.), 20., 401*subdivisions),
        # The logarithmic boundary transform can concentrate small amounts of
        # mass extremely close to its exact lower bound. Linear panels alias it.
        m+s*boundary+s*np.geomspace(1e-14, 1., 81*subdivisions),
        np.linspace(np.log(400.), hi, 101*subdivisions))))
    nodes, weights = _gauss(order)
    widths = np.diff(edges)[:, None]
    x = ((edges[:-1, None]+edges[1:, None])/2+widths*nodes/2).ravel()
    w = (widths*weights/2).ravel()
    return x, w


def unit_flux_context(density, body, z, theta):
    """Return normalization and shift, checked at successively finer grids.

    If r is the raw density, M=integral r and J=integral exp(-x)r,
    q(x)=r(x-log(J/M))/M has integral 1 and inverse moment 1.
    No shift clipping is applied HERE. Accuracy is numerical, not symbolic --
    this function honestly reports whatever shift exact unit flux requires,
    including large ones at extreme (z, theta). Callers that want a bounded
    correction (e.g. `composite.make_composite`'s default policy, which caps
    at `FLUX_DELTA_MAX` -- see its 2026-09-07 comment for why) apply that cap
    themselves on this function's return value; it is a deliberate policy
    choice made by the caller, not a property of this calibration itself.
    """
    previous = None
    for order in (8, 16, 32, 64):
        grid, weights = calibration_quadrature(body, z, theta, order)
        values = np.asarray(density(grid, z, theta), dtype=np.float64)
        if values.shape != grid.shape or np.any(~np.isfinite(values)) or np.any(values < 0):
            raise FluxCalibrationError('nonfinite or negative composite density')
        mass = float(weights @ values)
        inverse = float(weights @ (values*np.exp(-grid)))
        if not (np.isfinite(mass+inverse) and mass > 0 and inverse > 0):
            raise FluxCalibrationError('invalid full-range mass or inverse moment')
        current = np.array([mass, inverse])
        if previous is not None and np.max(np.abs(current/previous-1)) < 2e-6:
            return mass, float(np.log(inverse/mass))
        previous = current
    raise FluxCalibrationError(f'quadrature did not converge for z={z}, theta={theta}')
