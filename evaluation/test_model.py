import hashlib
from pathlib import Path

import numpy as np
import pytest

from flumen.model import load_model
from flumen.model.composite import UnverifiedFallbackError, make_composite


ROOT = Path(__file__).resolve().parents[1]
THETA = (.67, .30, .85, .0493, .965, 3.402)
REDSHIFTS = (.2, .5, 1., 2., 3.5, 5., 8., 10.)
PANEL_CONTEXTS = (
    THETA,
    (.72, .38, 1., .0493, .965, 3.402),
    (.60, .22, .70, .0493, .965, 3.402),
    (.80, .50, 1.50, .0493, .965, 3.402),
    (.55, .15, .40, .0493, .965, 3.402),
)


def test_frozen_checkpoint_hashes():
    expected = {
        "boundary_body.pt": "1a8a07f9deb6937fa47fd98bb9bc3303b1aad4e8265dd35796735dbec2ef4df3",
        "asymmetric_body_sl0p6.pt": "fc88382c50b9a6d04bf28e2b3efa033afad5843872719f4ef0380e6f3581df19",
    }
    for name, digest in expected.items():
        actual = hashlib.sha256((ROOT / "model/checkpoints" / name).read_bytes()).hexdigest()
        assert actual == digest


# Golden values updated 2026-09-07 for the seamless centered C1 Hermite
# bridge transition in composite.py, eliminating the low-structure knee.
GOLDEN = [
    (.2, THETA, [-np.inf, 3.8942003660781515, -9.535280936193987, -11.799959461598691]),
    (3.5, (.8,.5,1.5,.06,1.02,4.5), [-0.4236229711032995, -0.028585307381582954, -1.1194308562033284, -3.0057161045923615]),
    (5., (.8,.5,1.5,.06,1.02,4.5), [-0.9100685608094995, -0.4727119922551272, -0.803840073232868, -2.293684825496909]),
    (10., (.55,.15,.4,.04,.9,2.5), [-23.135928389702734, 2.5588573834062194, -6.248655861696914, -11.841805334541071]),
]


@pytest.mark.parametrize("z,theta,expected", GOLDEN)
def test_golden_log_density(z, theta, expected):
    model = load_model(flux_mode="legacy")
    x = np.array([-.2, 0., np.log(2.), np.log(5.)])
    np.testing.assert_allclose(model.log_prob_lnmu(x, z, theta), expected,
                               rtol=2e-7, atol=2e-7)



@pytest.mark.parametrize("z", [.2, .5, 3.5, 5., 10.])
def test_normalization_and_flux(z):
    model = load_model()
    grid = np.linspace(-4., np.log(400.), 12000)
    density = model.pdf_lnmu(grid, z, THETA)
    assert np.trapezoid(density, grid) == pytest.approx(1., abs=2e-3)
    assert np.trapezoid(np.exp(-grid)*density, grid) == pytest.approx(1., abs=2e-3)


def test_mu_and_lnmu_density_jacobian():
    model = load_model(); mu = np.geomspace(.5, 20., 100)
    np.testing.assert_allclose(model.pdf_mu(mu, 2., THETA)*mu,
                               model.pdf_lnmu(np.log(mu), 2., THETA))


def test_nonpositive_mu_rejected():
    with pytest.raises(ValueError):
        load_model().pdf_mu([0., 1.], 1., THETA)


def test_qualified_composite_mode_and_reference_panel_safety():
    model = load_model()
    assert model._log_prob.plan_mode == "crossing_or_pot_2_3"
    for theta in PANEL_CONTEXTS:
        for z in REDSHIFTS:
            model._log_prob.plan(z, theta)


def test_guard_rejects_a_body_with_no_finite_rescue_window():
    def broken_body(x, z, theta):
        return np.full_like(np.asarray(x, float), np.nan)

    broken_body.controls = lambda z, theta: (0., 1., None, None)
    composite = make_composite(broken_body)
    with pytest.raises(UnverifiedFallbackError):
        composite.plan(.2, THETA)


def test_guard_rescues_an_unsafe_body_when_mu2_to_mu3_is_finite():
    def partly_usable_body(x, z, theta):
        x = np.asarray(x, float)
        return np.where(np.abs(np.exp(x)-2.5) < 1., -1., np.nan)

    partly_usable_body.controls = lambda z, theta: (0., 1., None, None)
    composite = make_composite(partly_usable_body)
    assert composite.plan(.2, THETA) == pytest.approx((np.log(2.), np.log(3.)))
