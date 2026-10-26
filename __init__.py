"""FLUMEN: weak-lensing magnification PDFs.

The analytic model is the default:
    mu, pdf = flumen.generate_pdf(z_s=1.0)

PDFs use the source plane by default. Pass plane="image" for the image plane,
or backend="mc" to explicitly select the older Monte-Carlo model.
"""

import numpy as np

from .model import load_analytic_model, load_model

__version__ = "0.5.0"
__all__ = ["generate_pdf", "generate_pdf_lnmu", "get_model", "TRAINING_RANGE", "ANALYTIC_TRAINING_RANGE",
           "__version__"]

# Confirmed training/validity box (see docs/MODEL_CARD.md). Calls outside this
# box are extrapolation -- generate_pdf warns rather than raises, since a
# small overshoot (e.g. z_s=12.5) is usually still fine.
MC_TRAINING_RANGE = {
    "z_s": (0.2, 12.0),
    "h": (0.55, 0.80),
    "Om": (0.15, 0.50),
    "sigma8": (0.40, 1.50),
    "ns": (0.94, 0.99),
    "Ob": (0.03, 0.07),
    "zeq": (3300.0, 3500.0),
}

# Training box of the default mix3 analytic backend (z_s >= 0.2, h to 0.85, curvature and
# CPL dark energy). Its sgl.Cosmology domain also requires w0 + wa < 0 and rho_DE/rho_m(z = 1100) <= 1e-2.
ANALYTIC_TRAINING_RANGE = {
    "z_s": (0.2, 12.0),
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

TRAINING_RANGE = ANALYTIC_TRAINING_RANGE

# Planck-like fiducial cosmology, used as the default for every parameter
# except z_s (which has no sensible default and must be given explicitly).
_FIDUCIAL = dict(h=0.674, Om=0.315, sigma8=0.811, Ob=0.0493, ns=0.965, zeq=3402.0)

_MODEL_CACHE = {}


def get_model(device="cpu", flux_mode="unit", tail_mode="asymptotic", backend="analytic"):
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
            _check_range(k, v, warn_out_of_range, MC_TRAINING_RANGE)
    model = get_model(device=device, flux_mode=flux_mode, tail_mode=tail_mode, backend="mc")
    theta = (h, Om, sigma8, Ob, ns, zeq / 1000.0)
    pdf = model.pdf_mu(x, z_s=z_s, theta=theta) if density == "mu" else model.pdf_lnmu(x, z_s=z_s, theta=theta)
    if plane == "source":
        # P_S = P_I / (mu <1/mu>_I); <1/mu>_I = 1 exactly under the unit-flux calibration, by quadrature otherwise
        inv_mu = 1.0 if flux_mode == "unit" else _mc_inverse_mu(model, z_s, theta)
        pdf = pdf / ((x if density == "mu" else np.exp(x)) * inv_mu)
    return pdf


def _mc_inverse_mu(model, z_s, theta):
    """<1/mu> of the MC backend's image-plane PDF (needed for the source plane when flux_mode != "unit")."""
    x = np.linspace(-3.0, 12.0, 120001)
    return float(np.trapezoid(model.pdf_lnmu(x, z_s=z_s, theta=theta) * np.exp(-x), x))


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
    backend="analytic",
    plane="source",
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
        MC backend only: "unit" enforces <1/mu> = 1; "standard" enforces unit normalization.
    tail_mode : str, optional
        MC backend only: "asymptotic" uses smooth relaxation approaching mu^-2.
        "hermite" uses the finite C^1 cubic Hermite bridge.
    warn_out_of_range : bool, optional
        If True (default), warn when a parameter falls outside
        ``TRAINING_RANGE`` (``ANALYTIC_TRAINING_RANGE`` for the analytic backend).
        Does not raise -- the model still evaluates.
    backend : str, optional
        "analytic" (default): the analytic model; accepts ``Ok``, ``w0``, ``wa``.
        "mc": the older Monte-Carlo model (flat LCDM).
        ``flux_mode`` and ``tail_mode`` apply only to "mc".
    plane : str, optional
        "source" (default since 0.4.0): random source positions -- the PDF for standard sirens (before
        detection selection), P_S(mu) = P_I(mu) / (mu <1/mu>_I). "image": random lines of sight, the measure of
        the MC rays (<1/mu> = 1 under flux_mode="unit") and the default of flumen <= 0.3. The image-plane PDF is
        shifted toward magnification by about Var(ln mu) (<ln mu> +0.010 vs -0.009 at z_s = 2, Planck-like).
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
    backend="analytic",
    plane="source",
    Ok=0.0,
    w0=-1.0,
    wa=0.0,
):
    """Same as :func:`generate_pdf`, but on the ln(mu) grid/density. Source plane by default (since 0.4.0);
    ``plane="image"`` for random lines of sight.

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
