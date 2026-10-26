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

CKPT_ROOT = Path(__file__).resolve().parents[1] / "model" / "checkpoints"
V1_DIR = CKPT_ROOT / "analytic_silu_v1"                 # "silu-final-v1": seeds 0-3
V2_DIR = CKPT_ROOT / "analytic_silu_v2"                 # default "silu-final-v2": seeds 0-7
EXPECTED_SHA256_V1 = {
    "ctx.pt": "0b20fc36fadcb5bf1db2a4b8745ec3b803fb4f0fd4d569e7735fe3b185235d82",
    "body_s0.pt": "3ddbf7696638862f998f0a1718581bff898ff7ea8efa03b9c5acd13094185f2b",
    "body_s1.pt": "bfadb19c507a76b284c295caa0e50a8ca14a09f562f66b809342a76e1a749dd8",
    "body_s2.pt": "166573be7b3290f06d772f3f627908666a69cf6734a5492cc3192704951e1075",
    "body_s3.pt": "15e296f9d0fe8e2aaac9a3f5708191890a349796e0a62efa87109066676de6a6",
}

EXPECTED_SHA256_V2 = {**EXPECTED_SHA256_V1,
    "body_s4.pt": "72f902dd8fdec441c9284483aefd7bf5874d07aa8197d22faff56b5eacd36c66",
    "body_s5.pt": "5ae62b7c21e1350c424929ab181a2bf7e2735285347e7ac41d57b43660394610",
    "body_s6.pt": "cca35ba6d24a5415a932417f6633dabf3c149905203737f5d44c218d676a4034",
    "body_s7.pt": "ab1a0deaa0b9cec9b11192d1403ba89d31d99e1eebcd41ab99e99c351a94edf3",
}

PLANCK = (1.0, 0.674, 0.315, 0.811, 0.0493, 0.965, 3402.0, 0.0, -1.0, 0.0)   # (z_s, h, Om, s8, Ob, ns, zeq, Ok, w0, wa)
GOLDEN_V1 = [  # "silu-final-v1" (seeds 0-3): SOURCE-plane ln P(ln mu); cross-checked against learning/emulator.py (<= 9e-4) and the pure-NumPy float64 networks (<= 2e-4)
    (PLANCK, [-0.05, 0.0, 0.3, 2.0, 6.0],
     [2.0826083194039984, 1.837586988703233, -2.934962914066601, -12.061702878550713, -21.163620025991314]),
    ((5.0, 0.80, 0.45, 1.30, 0.045, 0.95, 3350.0, -0.10, -0.8, -0.5), [-0.3, 0.0, 0.5, 3.0, 8.0],
     [0.34338980034057087, -0.07321858533421764, -1.6904190927552196, -8.41748190414959, -18.52964807281852]),
    ((0.6, 0.60, 0.25, 0.60, 0.040, 0.98, 3450.0, 0.03, -1.2, 0.2), [-0.01, 0.0, 0.05, 1.0, 4.0],
     [3.1898636802994336, 2.8767621328266433, 0.4151785969281315, -12.734643686194246, -23.241417676753194]),
]
GOLDEN_V2 = [  # "silu-final-v2" (default, seeds 0-7), default guard: SOURCE-plane ln P(ln mu); cross-checked against learning/emulator.py (<= 7e-5) and the pure-NumPy float64 networks (<= 8e-5)
    (PLANCK, [-0.05, 0.0, 0.3, 2.0, 6.0],
     [2.0804309010829116, 1.83994304605144, -2.9321346677924964, -12.047165103206746, -21.163175842394516]),
    ((0.6, 0.60, 0.25, 0.60, 0.040, 0.98, 3450.0, 0.03, -1.2, 0.2), [-0.01, 0.0, 0.05, 1.0, 4.0],
     [3.1890012848184117, 2.8815599101342473, 0.42694498484260496, -12.735224322377276, -23.241930692439432]),
]
# A legitimate configuration on which the ORIGINAL left-edge guard misfires for the 8-seed default (guard anchored at the peak: mass shift +110%, norm 2.095, ln P low by 0.74).
# guard="auto" repairs it (norm 1.0000, values within 0.7% of the 4-seed v1 at the same points).
MISFIRE_V2 = ((5.0, 0.80, 0.45, 1.30, 0.045, 0.95, 3350.0, -0.10, -0.8, -0.5), [-0.3, 0.0, 0.5, 3.0, 8.0],
              [0.3414026924117159, -0.07047423917413158, -1.6836160483963407, -8.409391240921506, -18.52952001236829])
# Known guard-failure configurations of the 4-seed ensemble "silu-final-v1" (mix1 design; the left-edge guard moves 2-24% of the probability mass): id 1580 (test split, z_s 0.26), 1720 (train).
# The 8-seed default does NOT misfire on these two (dM ~ 0, norm 1.0000).
GUARD_CASES = {
    1580: (0.2575686711261008, 0.6847066539243757, 0.16772980301247314, 1.1573735320462715, 0.0329495989414222, 0.9683243391916895, 3491.5502419124086, -0.07993455893411335, -0.3386111764674822, -0.1809706047213595),
    1720: (0.2477461142222782, 0.5651537288749551, 0.20897471152384325, 1.107253474952921, 0.05879484377156601, 0.9565492057274552, 3356.181288082976, -0.11350165441021967, -0.4190204316991035, -0.24727898552600824),
}


@pytest.fixture(scope="module")
def model():
    return load_analytic_model(device="cpu", version="silu-final-v2")  # historical golden/guard regressions


@pytest.fixture(scope="module")
def model_v1():
    return load_analytic_model(device="cpu", version="silu-final-v1")


def _grid(model, p):
    cut = model.prepare(p)["cut"]
    return np.concatenate([np.linspace(cut + 1e-7, cut + 1.0, 60001)[:-1], np.linspace(cut + 1.0, 16.0, 60001)])


def test_analytic_checkpoint_hashes():
    for name, sha in EXPECTED_SHA256_V1.items():               # silu-final-v1
        assert hashlib.sha256((V1_DIR / name).read_bytes()).hexdigest() == sha, name
    for name, sha in EXPECTED_SHA256_V2.items():               # silu-final-v2 (default; seeds 0-3 identical to v1)
        assert hashlib.sha256((V2_DIR / name).read_bytes()).hexdigest() == sha, name


def test_default_version_and_ensemble(model):
    current = load_analytic_model(device="cpu")
    assert current.version == "analytic" and current.seeds == tuple(range(8)) and current.rise_tol == 0.0
    assert model.version == "silu-final-v2" and model.seeds == tuple(range(8)) and model.cfg["act"] == "silu"
    assert model.guard == "current"                            # the original left-edge guard is the default
    with pytest.raises(ValueError):
        load_analytic_model(version="nonexistent")


@pytest.mark.parametrize("params, lnmu, expected", GOLDEN_V2)
def test_analytic_golden_log_density(model, params, lnmu, expected):
    assert np.allclose(model.log_prob_lnmu(np.array(lnmu), params, plane="source"), expected, rtol=0, atol=1e-4)


def test_analytic_guard_misfire_v2_and_auto_repair(model):
    p, lnmu, expected = MISFIRE_V2
    legacy = load_analytic_model(device="cpu", version="silu-final-v2", rise_tol=0.0)                                              # the zero-tolerance rise test of the first release
    d = legacy.diagnose(p)
    assert "guard_mass_shift" in d["flags"] and d["dM"] > 0.5 and abs(d["norm"] - 2.095) < 1e-2          # documented misfire of the original guard
    auto = load_analytic_model(device="cpu", version="silu-final-v2", rise_tol=0.0, guard="auto")
    assert auto.diagnose(p)["fallback"] and abs(auto.diagnose(p)["norm"] - 1.0) < 1e-3
    assert np.allclose(auto.log_prob_lnmu(np.array(lnmu), p, plane="source"), expected, rtol=0, atol=1e-4)
    d2 = model.diagnose(p)                                                                                # default v2: rise tolerance 1e-2 repairs it without a fallback
    assert model.rise_tol == 1e-2 and "guard_mass_shift" not in d2["flags"] and abs(d2["dM"]) < 1e-3 and abs(d2["norm"] - 1.0) < 1e-3
    assert np.allclose(model.log_prob_lnmu(np.array(lnmu), p, plane="source"), expected, rtol=0, atol=1e-3)


def test_analytic_v1_keeps_zero_rise_tolerance(model_v1):
    assert model_v1.rise_tol == 0.0


@pytest.mark.parametrize("params, lnmu, expected", GOLDEN_V1)
def test_analytic_golden_log_density_v1(model_v1, params, lnmu, expected):
    assert model_v1.version == "silu-final-v1" and model_v1.seeds == (0, 1, 2, 3)
    assert np.allclose(model_v1.log_prob_lnmu(np.array(lnmu), params, plane="source"), expected, rtol=0, atol=1e-4)


def test_ensemble_is_the_seed_mixture(model):
    """The 8-seed body density is the arithmetic mean of the single-seed densities (log-mean-exp), to round-off."""
    from flumen.model.analytic import AnalyticPDF, DELTA
    singles = [AnalyticPDF(ckpt_dir=V2_DIR, seeds=(s,)) for s in range(8)]
    p = PLANCK
    c = model._context(np.asarray(p))
    x = np.linspace(c["cut"] + 1e-5, c["m"] + 8 * c["s"], 100)
    u = np.log((x - c["m"]) / c["s"] - c["yb"] + DELTA)
    lps = np.array([m._logpU(np.asarray(p), u, c) for m in singles])
    mix = np.log(np.mean(np.exp(lps), axis=0))
    assert np.max(np.abs(model._logpU(np.asarray(p), u, c) - mix)) < 1e-10


def test_numpy_float64_networks_agree_with_torch(model):
    """Independent pure-NumPy float64 evaluation of the context MLPs and the SiLU NSF flows (evaluation/analytic_numpy_reference.py)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analytic_numpy_reference import NumpyAnalyticPDF
    ref = NumpyAnalyticPDF(ckpt_dir=V2_DIR)
    for p in [PLANCK, (0.6, 0.60, 0.25, 0.60, 0.040, 0.98, 3450.0, 0.03, -1.2, 0.2)]:         # the guard-misfire configuration MISFIRE_V2 is excluded (its guard law is float32-noise sensitive)
        c = model.prepare(p)
        x = np.concatenate([np.linspace(c["cut"] + 1e-5, c["m"] + 8 * c["s"], 200), np.linspace(c["m"] + 8 * c["s"], 11.0, 60)])
        a, b = model.log_prob_lnmu(x, p), ref.log_prob_lnmu(x, p)
        ok = a > a.max() + np.log(1e-10)
        assert np.max(np.abs(a - b)[ok]) < 2e-3
        assert np.max(np.abs(np.exp(a) - np.exp(b))) / np.exp(a).max() < 1e-5
        assert model.prepare(p)["norm"] == pytest.approx(ref.prepare(p)["norm"], rel=1e-6)


def test_guard_diagnostics_and_optional_fallback(model, model_v1):
    """guard="current" is the default; guard="auto" changes nothing where the guard moves < 1e-3 of the mass, and uses the guard-free branch where it does.
    The misfire configurations are those of the 4-seed ensemble (v1); the 8-seed default does not misfire on them."""
    auto = load_analytic_model(device="cpu", version="silu-final-v1", guard="auto")
    x = np.linspace(-0.04, 3.0, 500)
    for m in (model, model_v1):
        assert m.guard == "current"
    assert np.array_equal(model_v1.log_prob_lnmu(x, PLANCK), auto.log_prob_lnmu(x, PLANCK))
    d = model.diagnose(PLANCK)
    assert d["has_guard"] and abs(d["dM"]) < 1e-3 and "guard_mass_shift" not in d["flags"]
    for i, p in GUARD_CASES.items():
        dg = model_v1.diagnose(p)
        assert "guard_mass_shift" in dg["flags"] and abs(dg["dM"]) > 1e-2, i
        da = auto.diagnose(p)
        assert da["fallback"] and abs(da["norm"] - 1.0) < 1e-3, i                    # the fallback restores the unit normalisation
        assert abs(model_v1.diagnose(p)["norm"] - 1.0) > 1e-3, i                     # the default guard does not
        assert "guard_mass_shift" not in model.diagnose(p)["flags"], i               # v2 (8 seeds): no misfire on these two


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
        flumen.generate_pdf(backend="mc", z_s=1.0, w0=-0.9)
    with pytest.raises(ValueError):
        flumen.generate_pdf(backend="mc", z_s=1.0, Ok=0.01)


def test_mc_backend_planes():
    """Since 0.4.0 generate_pdf returns the source plane by default; plane="image" is the low-level MC class."""
    model_mc = load_model(device="cpu", flux_mode="unit")
    theta = (0.674, 0.315, 0.811, 0.0493, 0.965, 3.402)
    mu = np.geomspace(0.5, 20.0, 500)
    _, p_img = flumen.generate_pdf(backend="mc", z_s=1.0, mu=mu, plane="image")
    assert np.allclose(p_img, model_mc.pdf_mu(mu, z_s=1.0, theta=theta))
    _, p_def = flumen.generate_pdf(backend="mc", z_s=1.0, mu=mu)
    _, p_src = flumen.generate_pdf(backend="mc", z_s=1.0, mu=mu, plane="source")
    assert np.allclose(p_def, p_src)
    assert np.allclose(p_src, p_img / mu)                     # unit flux: <1/mu>_I = 1
    grid = np.unique(np.concatenate([np.linspace(0.85, 1.15, 3000), np.geomspace(0.4, 80.0, 3000)]))
    for flux_mode in ("unit", "standard"):
        _, ps = flumen.generate_pdf(backend="mc", z_s=2.0, mu=grid, flux_mode=flux_mode)
        assert np.trapezoid(ps, grid) == pytest.approx(1.0, abs=5e-3), flux_mode


def test_analytic_default_plane_matches_toplevel(model):
    """Source plane by default everywhere for the analytic backend (class and generate_pdf, since 0.4.0)."""
    model = load_analytic_model()  # public API uses the current default, not the historical fixture
    p = PLANCK
    lnmu = np.linspace(-0.1, 3.0, 400)
    _, top = flumen.generate_pdf_lnmu(z_s=p[0], lnmu=lnmu, backend="analytic")
    assert np.allclose(model.pdf_lnmu(lnmu, p), top)
    assert np.allclose(model.pdf_lnmu(lnmu, p), model.pdf_lnmu(lnmu, p, plane="source"))
    assert not np.allclose(model.pdf_lnmu(lnmu, p), model.pdf_lnmu(lnmu, p, plane="image"))
