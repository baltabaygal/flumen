"""Unit test suite for production_model_v3."""

import hashlib
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flumen.model import load_model
import flumen

MODEL_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = MODEL_DIR / "model" / "checkpoints" / "single_body_gauss.pt"
EXPECTED_SHA256 = "a16f54238adae1c0580f6e3b85d9ee879762d521161e6db356c383ebc1e5a6e5"

THETA_FIDUCIAL = (0.6774, 0.3089, 0.8159, 0.0486, 0.9667, 3371.0)

GOLDEN_CASES = [
    (0.2, (0.6774, 0.3089, 0.8159, 0.0486, 0.9667, 3371.0), [-np.inf, 3.98584177062861, -19.28244890543799, -21.059998811962007]),
    (1.0, (0.6774, 0.3089, 0.8159, 0.0486, 0.9667, 3371.0), [-np.inf, 1.8382197026757205, -5.825231303765528, -8.177180397814494]),
    (2.0, (0.6774, 0.3089, 0.8159, 0.0486, 0.9667, 3371.0), [-0.6217705756994054, 1.2246067078419238, -3.943859360297862, -7.405962981911451]),
    (5.0, (0.8, 0.5, 1.5, 0.0493, 0.965, 3402.0), [-0.0023895598015362604, -0.06716875123248396, -1.4569779001659353, -3.1141984728400898]),
]



@pytest.fixture(scope="module")
def model():
    return load_model(device="cpu", flux_mode="unit")


def test_checkpoint_hash():
    """Verify production checkpoint matches expected SHA-256."""
    h = hashlib.sha256()
    with open(CHECKPOINT_PATH, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    assert h.hexdigest() == EXPECTED_SHA256, f"Checkpoint SHA-256 mismatch: {h.hexdigest()}"


def test_scalar_and_array_agreement(model):
    """Scalar evaluation must match 1D array slice evaluation."""
    z_s = 1.5
    theta = THETA_FIDUCIAL
    mu_scalar = 1.2
    mu_array = np.array([0.9, 1.2, 1.5])

    val_scalar = model.pdf_mu(mu_scalar, z_s=z_s, theta=theta)
    val_array = model.pdf_mu(mu_array, z_s=z_s, theta=theta)

    assert isinstance(val_scalar, float)
    assert isinstance(val_array, np.ndarray)
    assert val_scalar == pytest.approx(val_array[1], rel=1e-5)


@pytest.mark.parametrize("z_s, theta, expected", GOLDEN_CASES)
def test_golden_log_density(model, z_s, theta, expected):
    """Verify log densities match certified golden values to 1e-6."""
    lnmu_eval = np.array([-0.2, 0.0, np.log(2.0), np.log(5.0)])
    actual = model.log_prob_lnmu(lnmu_eval, z_s=z_s, theta=theta)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("z_s", [0.2, 0.5, 1.0, 2.0, 5.0, 8.0])
def test_normalization_and_unit_flux(model, z_s):
    """Probability density must normalize to 1 and conserve unit inverse mean."""
    mu_grid = np.unique(np.concatenate([
        np.linspace(0.85, 1.15, 3000),
        np.geomspace(0.4, 80.0, 3000),
    ]))
    pdf = model.pdf_mu(mu_grid, z_s=z_s, theta=THETA_FIDUCIAL)

    mass = np.trapezoid(pdf, mu_grid)
    flux = np.trapezoid(pdf / mu_grid, mu_grid)

    assert mass == pytest.approx(1.0, abs=5e-3)
    assert flux == pytest.approx(1.0, abs=5e-3)


@pytest.mark.parametrize("z_s", [0.5, 1.0, 3.0])
def test_positivity(model, z_s):
    """Densities must be non-negative everywhere."""
    mu_grid = np.geomspace(0.1, 100.0, 500)
    pdf = model.pdf_mu(mu_grid, z_s=z_s, theta=THETA_FIDUCIAL)
    assert np.all(pdf >= 0.0)
    assert np.all(np.isfinite(pdf))


def test_asymptotic_tail_slope(model):
    """The high-magnification tail must follow a power-law slope of exactly -2."""
    z_s = 2.0
    mu1 = 100.0
    mu2 = 200.0
    p1 = model.pdf_mu(mu1, z_s=z_s, theta=THETA_FIDUCIAL)
    p2 = model.pdf_mu(mu2, z_s=z_s, theta=THETA_FIDUCIAL)

    # slope = d ln(p) / d ln(mu)
    slope = np.log(p2 / p1) / np.log(mu2 / mu1)
    assert slope == pytest.approx(-2.0000, abs=1e-3)


def test_c1_hermite_bridge_smoothness(model):
    """Verify C1 continuity (value and first derivative match) across bridge boundaries."""
    comp = model.composite
    z_s = 2.0
    m, s, yb, _ = comp.predictor.predict(z_s, THETA_FIDUCIAL)
    y0 = max(float(comp.y_c_rel), yb - comp.delta + 1.0)
    h = max(comp.h_bridge_min, comp.h_factor / s)
    y1 = y0 + h

    eps = 1e-4
    # Points across y0
    ys_0 = np.array([y0 - eps, y0, y0 + eps])
    lnmus_0 = m + s * ys_0
    lps_0 = comp._raw_log_prob_lnmu(lnmus_0, z_s, THETA_FIDUCIAL, m, s, yb)
    # Continuity of value
    assert lps_0[1] == pytest.approx((lps_0[0] + lps_0[2]) / 2.0, abs=1e-3)

    # Continuity of derivative across y1
    ys_1 = np.array([y1 - eps, y1, y1 + eps])
    lnmus_1 = m + s * ys_1
    lps_1 = comp._raw_log_prob_lnmu(lnmus_1, z_s, THETA_FIDUCIAL, m, s, yb)
    slope_left = (lps_1[1] - lps_1[0]) / (eps * s)
    slope_right = (lps_1[2] - lps_1[1]) / (eps * s)
    assert slope_left == pytest.approx(-1.0, abs=1e-3)
    assert slope_right == pytest.approx(-1.0, abs=1e-3)


def test_empty_beam_cutoff_suppression(model):
    """Densities below the hard cutoff must be completely zeroed."""
    z_s = 2.0
    deep_under_cutoff = 0.2
    pdf_val = model.pdf_mu(deep_under_cutoff, z_s=z_s, theta=THETA_FIDUCIAL)
    assert pdf_val == 0.0


def test_asymptotic_cutoff_monotonicity_and_smoothness(model):
    """At high redshift, density approaching the cutoff must decay strictly monotonically without shelves."""
    z_s = 8.0
    theta = THETA_FIDUCIAL
    m, s, y_b, _ = model.composite.predictor.predict(z_s, theta)
    yb_eff = y_b - model.composite.eps_b
    mu_cut = np.exp(m + s * (yb_eff - model.composite.delta))

    # Sample approaching cutoff from above
    mu_eval = np.linspace(mu_cut * 1.0001, mu_cut * 1.03, 30)
    pdf_vals = model.pdf_mu(mu_eval, z_s=z_s, theta=theta)

    # 1. Non-negative everywhere
    assert np.all(pdf_vals >= 0.0)

    # 2. Strict monotonicity: as mu increases away from cutoff, density increases towards peak
    diffs = np.diff(pdf_vals)
    assert np.all(diffs >= 0.0), f"Non-monotonic cutoff approach detected: {diffs}"

    # 3. Density near cutoff edge is smoothly vanishing (< 1e-5)
    assert pdf_vals[0] < 1e-5

    # 4. Strictly 0 below cutoff
    mu_below = np.array([mu_cut * 0.999, mu_cut * 0.95])
    assert np.all(model.pdf_mu(mu_below, z_s=z_s, theta=theta) == 0.0)


def test_flumen_generate_pdf_convenience():
    """Top-level flumen.generate_pdf and generate_pdf_lnmu must return valid densities."""
    mu, pdf = flumen.generate_pdf(z_s=1.0)
    assert mu.shape == (2000,)
    assert pdf.shape == (2000,)
    assert np.all(pdf >= 0.0)
    integral = np.trapezoid(pdf, mu)
    assert integral == pytest.approx(1.0, abs=1e-3)

    lnmu, pdf_lnmu = flumen.generate_pdf_lnmu(z_s=1.0)
    assert lnmu.shape == (2000,)
    assert pdf_lnmu.shape == (2000,)
    assert np.all(pdf_lnmu >= 0.0)
    integral_lnmu = np.trapezoid(pdf_lnmu, lnmu)
    assert integral_lnmu == pytest.approx(1.0, abs=1e-3)
