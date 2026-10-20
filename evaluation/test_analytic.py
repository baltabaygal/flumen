"""Unit tests for the analytic-chain backend (flumen.model.analytic) and the backend/plane options of the API."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import flumen
from flumen.model import load_analytic_model, load_model

CKPT_DIR = Path(__file__).resolve().parents[1] / "model" / "checkpoints" / "analytic"
EXPECTED_SHA256 = {
    "ctx.pt": "21c2d3f3dfb4a96be31bc497264925358c593d235f28c44715a780b8d34a1a11",
    "body_s0.pt": "d6211ce4698c33c58e512afca85bcb928eb276398a519216b801e12326bb7080",
    "body_s1.pt": "f89b3aed1dade768583855677190cf0b09bc7f8a35a133ae6e0c3e0635a57fe3",
}

PLANCK = (1.0, 0.674, 0.315, 0.811, 0.0493, 0.965, 3402.0, 0.0, -1.0, 0.0)   # (z_s, h, Om, s8, Ob, ns, zeq, Ok, w0, wa)
GOLDEN_CASES = [  # SOURCE-plane ln P(ln mu) (the chain's native quantity)
    (PLANCK, [-0.05, 0.0, 0.3, 2.0, 6.0],
     [2.0804599334475644, 1.8414951026834963, -2.9259215369852534, -12.034122436767518, -21.161181311740112]),
    ((5.0, 0.80, 0.45, 1.30, 0.045, 0.95, 3350.0, -0.10, -0.8, -0.5), [-0.3, 0.0, 0.5, 3.0, 8.0],
     [0.34348355938454794, -0.06945925438158436, -1.6841290355685454, -8.39720738647535, -18.528825878947053]),
    ((0.6, 0.60, 0.25, 0.60, 0.040, 0.98, 3450.0, 0.03, -1.2, 0.2), [-0.01, 0.0, 0.05, 1.0, 4.0],
     [3.19049548809207, 2.8790421569956113, 0.4110886541959419, -12.71954262731258, -23.090352403674615]),
]


@pytest.fixture(scope="module")
def model():
    return load_analytic_model(device="cpu")


def _grid(model, p):
    cut = model.prepare(p)["cut"]
    return np.concatenate([np.linspace(cut + 1e-7, cut + 1.0, 60001)[:-1], np.linspace(cut + 1.0, 16.0, 60001)])


def test_analytic_checkpoint_hashes():
    for name, sha in EXPECTED_SHA256.items():
        assert hashlib.sha256((CKPT_DIR / name).read_bytes()).hexdigest() == sha, name


@pytest.mark.parametrize("params, lnmu, expected", GOLDEN_CASES)
def test_analytic_golden_log_density(model, params, lnmu, expected):
    assert np.allclose(model.log_prob_lnmu(np.array(lnmu), params, plane="source"), expected, rtol=0, atol=1e-4)


@pytest.mark.parametrize("z_s", [0.6, 1.0, 3.0, 8.0])
def test_analytic_normalisation_both_planes(model, z_s):
    p = (z_s,) + PLANCK[1:]
    x = _grid(model, p)
    ps, pi = model.pdf_lnmu(x, p, "source"), model.pdf_lnmu(x, p, "image")
    mass_s = np.trapezoid(ps, x) + 0.5 * ps[-1]
    mass_i = np.trapezoid(pi, x) + pi[-1]
    assert mass_s == pytest.approx(1.0, abs=2e-3)
    assert mass_i == pytest.approx(1.0, abs=2e-3)
    inv_mu_image = np.trapezoid(pi * np.exp(-x), x)
    assert inv_mu_image == pytest.approx(1.0 / model.mean_mu_source(p), rel=2e-3)


def test_analytic_positivity_cutoff_and_left_flank(model):
    for z_s in (0.5, 1.0, 4.0, 12.0):
        p = (z_s,) + PLANCK[1:]
        c = model.prepare(p)
        x = np.linspace(c["cut"] - 0.05, 9.0, 20000)
        P = model.pdf_lnmu(x, p)
        assert np.all(P >= 0) and np.all(np.isfinite(P))
        assert np.all(P[x <= c["cut"]] == 0.0)
        left = (x > c["cut"]) & (x < x[np.argmax(P)]) & (P < 0.5 * P.max())
        assert np.all(np.diff(P[left]) >= 0), "left flank must rise monotonically to the peak"


def test_analytic_tail_slopes(model):
    for z_s in (1.0, 5.0):
        p = (z_s,) + PLANCK[1:]
        for plane, slope in (("source", -2.0), ("image", -1.0)):
            a, b = model.log_prob_lnmu(np.array([9.0, 11.0]), p, plane)
            assert (b - a) / 2.0 == pytest.approx(slope, abs=0.01)


def test_generate_pdf_analytic_backend_and_de_parameters():
    mu, pdf = flumen.generate_pdf(z_s=2.0, backend="analytic")
    assert mu.shape == pdf.shape and np.all(pdf >= 0)
    _, pdf_de = flumen.generate_pdf(z_s=2.0, backend="analytic", Ok=-0.05, w0=-0.9, wa=-0.3)
    assert not np.allclose(pdf, pdf_de)
    lnmu, p_src = flumen.generate_pdf_lnmu(z_s=2.0, backend="analytic", plane="source")
    assert np.all(p_src >= 0)


def test_mc_backend_rejects_curvature_and_dark_energy():
    with pytest.raises(ValueError):
        flumen.generate_pdf(z_s=1.0, w0=-0.9)
    with pytest.raises(ValueError):
        flumen.generate_pdf(z_s=1.0, Ok=0.01)


def test_mc_backend_planes():
    """Since 0.4.0 generate_pdf returns the source plane by default; plane="image" is the low-level MC class."""
    model_mc = load_model(device="cpu", flux_mode="unit")
    theta = (0.674, 0.315, 0.811, 0.0493, 0.965, 3.402)
    mu = np.geomspace(0.5, 20.0, 500)
    _, p_img = flumen.generate_pdf(z_s=1.0, mu=mu, plane="image")
    assert np.allclose(p_img, model_mc.pdf_mu(mu, z_s=1.0, theta=theta))
    _, p_def = flumen.generate_pdf(z_s=1.0, mu=mu)
    _, p_src = flumen.generate_pdf(z_s=1.0, mu=mu, plane="source")
    assert np.allclose(p_def, p_src)
    assert np.allclose(p_src, p_img / mu)                     # unit flux: <1/mu>_I = 1
    grid = np.unique(np.concatenate([np.linspace(0.85, 1.15, 3000), np.geomspace(0.4, 80.0, 3000)]))
    for flux_mode in ("unit", "standard"):
        _, ps = flumen.generate_pdf(z_s=2.0, mu=grid, flux_mode=flux_mode)
        assert np.trapezoid(ps, grid) == pytest.approx(1.0, abs=5e-3), flux_mode


def test_analytic_default_plane_matches_toplevel(model):
    """Source plane by default everywhere for the analytic backend (class and generate_pdf, since 0.4.0)."""
    p = PLANCK
    lnmu = np.linspace(-0.1, 3.0, 400)
    _, top = flumen.generate_pdf_lnmu(z_s=p[0], lnmu=lnmu, backend="analytic")
    assert np.allclose(model.pdf_lnmu(lnmu, p), top)
    assert np.allclose(model.pdf_lnmu(lnmu, p), model.pdf_lnmu(lnmu, p, plane="source"))
    assert not np.allclose(model.pdf_lnmu(lnmu, p), model.pdf_lnmu(lnmu, p, plane="image"))
