"""
flumen — Fast Lensing Unified Magnification Emulator with Normalizing flows.

A normalizing-flow emulator for the gravitational-wave weak-lensing
magnification PDF, trained on Monte-Carlo simulations of the Vaskonen (2026)
lensing model. Wraps the frozen ``model/`` bundle behind one convenience
function so the PDF for a given cosmology and source redshift is a single
call away.

Quick start
-----------
>>> import flumen
>>> mu, pdf = flumen.generate_pdf(z_s=1.0, Om=0.315, h=0.674, sigma8=0.811)
>>> mu.shape, pdf.shape
((2000,), (2000,))

See ``generate_pdf`` for the full parameter list and ``examples/basic_usage.py``
for a runnable script. Full architecture, training and validation details are
in ``docs/MODEL_CARD.md``.
"""

import numpy as np

from .model import load_model

__version__ = "0.3.0"
__all__ = ["generate_pdf", "generate_pdf_lnmu", "get_model", "TRAINING_RANGE", "__version__"]

# Confirmed training/validity box (see docs/MODEL_CARD.md). Calls outside this
# box are extrapolation -- generate_pdf warns rather than raises, since a
# small overshoot (e.g. z_s=12.5) is usually still fine.
TRAINING_RANGE = {
    "z_s": (0.2, 12.0),
    "h": (0.55, 0.80),
    "Om": (0.15, 0.50),
    "sigma8": (0.40, 1.50),
    "ns": (0.94, 0.99),
    "Ob": (0.03, 0.07),
    "zeq": (3300.0, 3500.0),
}

# Planck-like fiducial cosmology, used as the default for every parameter
# except z_s (which has no sensible default and must be given explicitly).
_FIDUCIAL = dict(h=0.674, Om=0.315, sigma8=0.811, Ob=0.0493, ns=0.965, zeq=3402.0)

_MODEL_CACHE = {}


def get_model(device="cpu", flux_mode="unit"):
    """Return the (cached) loaded :class:`model.MagnificationPDF`.

    The checkpoint and calibration files are only read from disk once per
    ``(device, flux_mode)`` combination; repeated calls reuse the same
    in-memory model. Most users do not need this directly -- use
    ``generate_pdf`` instead.
    """
    key = (device, flux_mode)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(device=device, flux_mode=flux_mode)
    return _MODEL_CACHE[key]


def _check_range(name, value, warn):
    lo, hi = TRAINING_RANGE[name]
    if not (lo <= value <= hi) and warn:
        import warnings

        warnings.warn(
            f"{name}={value} is outside the confirmed training range "
            f"[{lo}, {hi}]; treat the result as extrapolation.",
            stacklevel=3,
        )


def generate_pdf(
    z_s,
    h=_FIDUCIAL["h"],
    Om=_FIDUCIAL["Om"],
    sigma8=_FIDUCIAL["sigma8"],
    Ob=_FIDUCIAL["Ob"],
    ns=_FIDUCIAL["ns"],
    zeq=_FIDUCIAL["zeq"],
    mu=None,
    mu_min=0.3,
    mu_max=30.0,
    n_mu=2000,
    device="cpu",
    flux_mode="unit",
    warn_out_of_range=True,
):
    """Predict the GW weak-lensing magnification PDF p(mu) for one cosmology.

    Parameters
    ----------
    z_s : float
        Source redshift. Required -- confirmed training range 0.2-12.
    h : float, optional
        Dimensionless Hubble parameter. Default 0.674 (Planck-like).
    Om : float, optional
        Matter density parameter Omega_M. Default 0.315.
    sigma8 : float, optional
        Amplitude of matter fluctuations (top-hat, 8 Mpc/h). Default 0.811.
    Ob : float, optional
        Baryon density parameter Omega_B. Default 0.0493.
    ns : float, optional
        Scalar spectral index. Default 0.965.
    zeq : float, optional
        Matter-radiation equality redshift. Default 3402.0.
    mu : array_like, optional
        Magnification values to evaluate the PDF on. If not given, a
        log-spaced grid from ``mu_min`` to ``mu_max`` with ``n_mu`` points
        is used.
    mu_min, mu_max, n_mu : float, float, int, optional
        Bounds and resolution of the default log-spaced mu grid. Ignored if
        ``mu`` is given explicitly.
    device : str, optional
        "cpu" (default) or "cuda".
    flux_mode : str, optional
        "unit" (default) enforces exact <1/mu> = 1. "legacy" uses the
        previous capped calibration -- see docs/MODEL_CARD.md for the
        tradeoff between the two.
    warn_out_of_range : bool, optional
        If True (default), warn when a parameter falls outside
        ``TRAINING_RANGE``. Does not raise -- the model still evaluates.

    Returns
    -------
    mu : numpy.ndarray
        The magnification grid the PDF was evaluated on, shape ``(n_mu,)``
        (or matching the input ``mu`` if one was given).
    pdf : numpy.ndarray
        p(mu), the magnification PDF (density with respect to d mu), same
        shape as ``mu``. Integrates to 1 over ``mu``.

    Examples
    --------
    >>> import flumen
    >>> mu, pdf = flumen.generate_pdf(z_s=1.0)
    >>> mu, pdf = flumen.generate_pdf(z_s=2.0, Om=0.28, sigma8=0.9, h=0.7)
    """
    if warn_out_of_range:
        _check_range("z_s", z_s, warn_out_of_range)
        _check_range("h", h, warn_out_of_range)
        _check_range("Om", Om, warn_out_of_range)
        _check_range("sigma8", sigma8, warn_out_of_range)
        _check_range("ns", ns, warn_out_of_range)
        _check_range("Ob", Ob, warn_out_of_range)
        _check_range("zeq", zeq, warn_out_of_range)

    model = get_model(device=device, flux_mode=flux_mode)

    if mu is None:
        mu = np.geomspace(mu_min, mu_max, n_mu)
    else:
        mu = np.asarray(mu, dtype=np.float64)

    theta = (h, Om, sigma8, Ob, ns, zeq / 1000.0)
    pdf = model.pdf_mu(mu, z_s=z_s, theta=theta)
    return mu, pdf


def generate_pdf_lnmu(
    z_s,
    h=_FIDUCIAL["h"],
    Om=_FIDUCIAL["Om"],
    sigma8=_FIDUCIAL["sigma8"],
    Ob=_FIDUCIAL["Ob"],
    ns=_FIDUCIAL["ns"],
    zeq=_FIDUCIAL["zeq"],
    lnmu=None,
    lnmu_min=-1.5,
    lnmu_max=3.0,
    n_lnmu=2000,
    device="cpu",
    flux_mode="unit",
    warn_out_of_range=True,
):
    """Same as :func:`generate_pdf`, but on the ln(mu) grid/density.

    Returns ``(lnmu, pdf_lnmu)`` where ``pdf_lnmu`` is the density with
    respect to ``d ln(mu)`` (integrates to 1 over ``lnmu``).
    """
    if warn_out_of_range:
        _check_range("z_s", z_s, warn_out_of_range)
        _check_range("h", h, warn_out_of_range)
        _check_range("Om", Om, warn_out_of_range)
        _check_range("sigma8", sigma8, warn_out_of_range)
        _check_range("ns", ns, warn_out_of_range)
        _check_range("Ob", Ob, warn_out_of_range)
        _check_range("zeq", zeq, warn_out_of_range)

    model = get_model(device=device, flux_mode=flux_mode)

    if lnmu is None:
        lnmu = np.linspace(lnmu_min, lnmu_max, n_lnmu)
    else:
        lnmu = np.asarray(lnmu, dtype=np.float64)

    theta = (h, Om, sigma8, Ob, ns, zeq / 1000.0)
    pdf = model.pdf_lnmu(lnmu, z_s=z_s, theta=theta)
    return lnmu, pdf
