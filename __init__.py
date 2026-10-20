"""
flumen — Fast Lensing Universal Magnification Emulator with Normalizing flows.

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

Backends
--------
``backend="mc"`` (default): the normalizing-flow emulator of the Vaskonen (2026) Monte-Carlo engine, flat LCDM.
``backend="analytic"``: an emulator of the semi-analytic compound-Poisson chain (stochastic_wl_analytic), which also
takes ``Ok``, ``w0``, ``wa``. ``plane="image"`` (default; random lines of sight, the MC engine's measure) or
``plane="source"`` (random source positions). See ``docs/MODEL_CARD.md`` section "Analytic backend".
"""

import numpy as np

from .model import load_analytic_model, load_model

__version__ = "0.3.0"
__all__ = ["generate_pdf", "generate_pdf_lnmu", "get_model", "TRAINING_RANGE", "ANALYTIC_TRAINING_RANGE",
           "__version__"]

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

# Training box of the analytic backend (stochastic_wl_analytic training set; z_s >= 0.5, h to 0.85, curvature and
# CPL dark energy). Its sgl.Cosmology domain also requires w0 + wa < 0 and rho_DE/rho_m(z = 1100) <= 1e-2.
ANALYTIC_TRAINING_RANGE = {
    "z_s": (0.5, 12.0),
    "h": (0.55, 0.85),
    "Om": (0.15, 0.50),
    "sigma8": (0.40, 1.50),
    "ns": (0.94, 0.99),
    "Ob": (0.03, 0.07),
    "zeq": (3300.0, 3500.0),
    "Ok": (-0.15, 0.05),
    "w0": (-1.5, 0.5),
    "wa": (-2.0, 1.0),
}

# Planck-like fiducial cosmology, used as the default for every parameter
# except z_s (which has no sensible default and must be given explicitly).
_FIDUCIAL = dict(h=0.674, Om=0.315, sigma8=0.811, Ob=0.0493, ns=0.965, zeq=3402.0)

_MODEL_CACHE = {}


def get_model(device="cpu", flux_mode="unit", tail_mode="asymptotic", backend="mc"):
    """Return the (cached) loaded model: :class:`model.MagnificationPDF` (``backend="mc"``) or
    :class:`model.AnalyticPDF` (``backend="analytic"``; ``flux_mode``/``tail_mode`` do not apply to it).

    The checkpoint and calibration files are only read from disk once per
    ``(device, flux_mode, tail_mode)`` combination; repeated calls reuse the same
    in-memory model. Most users do not need this directly -- use
    ``generate_pdf`` instead.
    """
    if backend == "analytic":
        key = ("analytic", device)
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = load_analytic_model(device=device)
        return _MODEL_CACHE[key]
    if backend != "mc":
        raise ValueError(f"backend must be 'mc' or 'analytic', not {backend!r}")
    key = (device, flux_mode, tail_mode)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(device=device, flux_mode=flux_mode, tail_mode=tail_mode)
    return _MODEL_CACHE[key]


def _check_range(name, value, warn, box=None):
    lo, hi = (TRAINING_RANGE if box is None else box)[name]
    if not (lo <= value <= hi) and warn:
        import warnings

        warnings.warn(
            f"{name}={value} is outside the confirmed training range "
            f"[{lo}, {hi}]; treat the result as extrapolation.",
            stacklevel=3,
        )


def _evaluate(z_s, cosmo, x, density, device, flux_mode, tail_mode, warn_out_of_range, backend, plane):
    """Shared core of generate_pdf / generate_pdf_lnmu. x = mu (density 'mu') or ln mu (density 'lnmu')."""
    if plane not in ("image", "source"):
        raise ValueError(f"plane must be 'image' or 'source', not {plane!r}")
    h, Om, sigma8, Ob, ns, zeq, Ok, w0, wa = cosmo
    if backend == "analytic":
        if warn_out_of_range:
            for k, v in zip(("z_s", "h", "Om", "sigma8", "ns", "Ob", "zeq", "Ok", "w0", "wa"),
                            (z_s, h, Om, sigma8, ns, Ob, zeq, Ok, w0, wa)):
                _check_range(k, v, warn_out_of_range, ANALYTIC_TRAINING_RANGE)
            if not (w0 + wa < 0):
                import warnings
                warnings.warn("w0 + wa >= 0 is outside the analytic backend's dark-energy domain; "
                              "treat the result as extrapolation.", stacklevel=3)
        model = get_model(device=device, backend="analytic")
        params = (z_s, h, Om, sigma8, Ob, ns, zeq, Ok, w0, wa)
        return model.pdf_mu(x, params, plane) if density == "mu" else model.pdf_lnmu(x, params, plane)
    if (Ok, w0, wa) != (0.0, -1.0, 0.0):
        raise ValueError("the MC backend is flat LCDM (Ok = 0, w0 = -1, wa = 0); use backend='analytic' "
                         "for curvature or dark energy")
    if warn_out_of_range:
        for k, v in zip(("z_s", "h", "Om", "sigma8", "ns", "Ob", "zeq"), (z_s, h, Om, sigma8, ns, Ob, zeq)):
            _check_range(k, v, warn_out_of_range)
    model = get_model(device=device, flux_mode=flux_mode, tail_mode=tail_mode)
    theta = (h, Om, sigma8, Ob, ns, zeq / 1000.0)
    pdf = model.pdf_mu(x, z_s=z_s, theta=theta) if density == "mu" else model.pdf_lnmu(x, z_s=z_s, theta=theta)
    if plane == "source":
        # P_S = P_I / (mu <1/mu>_I); exact under the unit-flux calibration (<1/mu>_I = 1)
        if flux_mode != "unit":
            raise ValueError("plane='source' with the MC backend needs flux_mode='unit'")
        pdf = pdf / (x if density == "mu" else np.exp(x))
    return pdf


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
    tail_mode="asymptotic",
    warn_out_of_range=True,
    backend="mc",
    plane="image",
    Ok=0.0,
    w0=-1.0,
    wa=0.0,
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
        "unit" (default) enforces exact <1/mu> = 1. "standard" enforces unit normalization.
    tail_mode : str, optional
        "asymptotic" (default) uses the smooth C^inf relaxation approaching mu^-2 at rate O(1/mu).
        "hermite" uses the finite C^1 cubic Hermite bridge.
    warn_out_of_range : bool, optional
        If True (default), warn when a parameter falls outside
        ``TRAINING_RANGE`` (``ANALYTIC_TRAINING_RANGE`` for the analytic backend).
        Does not raise -- the model still evaluates.
    backend : str, optional
        "mc" (default): the emulator of the Monte-Carlo engine (flat LCDM).
        "analytic": the emulator of the semi-analytic chain; accepts ``Ok``, ``w0``,
        ``wa``; ``flux_mode`` and ``tail_mode`` do not apply to it.
    plane : str, optional
        "image" (default, kept for backward compatibility): random lines of sight, the measure of the MC engine
        (<1/mu> = 1). "source": random source positions, P_S(mu) = P_I(mu) / (mu <1/mu>_I) -- use this for
        standard sirens (sources at random positions, before detection selection). The image-plane PDF is shifted
        toward magnification by about Var(ln mu) (<ln mu> +0.010 vs -0.009 at z_s = 2, Planck-like).
    Ok, w0, wa : float, optional
        Curvature and CPL dark-energy parameters (analytic backend only).
        Defaults 0, -1, 0.

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
    if mu is None:
        mu = np.geomspace(mu_min, mu_max, n_mu)
    else:
        mu = np.asarray(mu, dtype=np.float64)
    pdf = _evaluate(z_s, (h, Om, sigma8, Ob, ns, zeq, Ok, w0, wa), mu, "mu", device, flux_mode, tail_mode,
                    warn_out_of_range, backend, plane)
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
    tail_mode="asymptotic",
    warn_out_of_range=True,
    backend="mc",
    plane="image",
    Ok=0.0,
    w0=-1.0,
    wa=0.0,
):
    """Same as :func:`generate_pdf`, but on the ln(mu) grid/density. Image plane by default; pass
    ``plane="source"`` for sources at random positions (standard sirens).

    Returns ``(lnmu, pdf_lnmu)`` where ``pdf_lnmu`` is the density with
    respect to ``d ln(mu)`` (integrates to 1 over ``lnmu``).
    """
    if lnmu is None:
        lnmu = np.linspace(lnmu_min, lnmu_max, n_lnmu)
    else:
        lnmu = np.asarray(lnmu, dtype=np.float64)
    pdf = _evaluate(z_s, (h, Om, sigma8, Ob, ns, zeq, Ok, w0, wa), lnmu, "lnmu", device, flux_mode, tail_mode,
                    warn_out_of_range, backend, plane)
    return lnmu, pdf
