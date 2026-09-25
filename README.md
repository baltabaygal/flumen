# flumen

**F**ast **L**ensing **U**niversal **M**agnification **E**mulator with
**N**ormalizing flows.

A normalizing-flow emulator for the conditional gravitational-wave
weak-lensing magnification PDF, trained on Monte-Carlo simulations of the
Vaskonen (2026) lensing model. Given a source redshift and a cosmology, it
returns the magnification PDF $p(\mu)$ in milliseconds on CPU, in place of a
Monte-Carlo run.

This is the production model **v3** (released 2026-09-14): a single-body SOS
polynomial flow (20,874 weights with standard Gaussian base) enforcing the
physical Dyer–Roeder (1973) empty-beam cutoff via $u = \ln(y - y_b + \delta)$
(with margin $\delta = 0.05$ and boundary buffer $\epsilon_b = 0.02$), handed off through a smooth
$C^\infty$ asymptotic relaxation bridge ($h = 1.0/s$) to the asymptotic power-law tail
$p(\mu) \propto \mu^{-2.0000}$ at the physical $\mathcal{O}(1/\mu)$ fold-caustic rate, with exact unit-flux ($\langle 1/\mu \rangle = 1.0000$)
calibration. See `docs/MODEL_CARD.md` for architecture, accuracy metrics, and known
limitations, and `PROVENANCE.md` for checkpoint lineage.

## Performance Highlights

- **ACE-Protocol KL**: **0.00476 nats median** (120 width-relative bins in $y$ on $[-6, 12]$ with $N \ge 5$ across all 1,089 held-out test contexts; comparable to the $0.00690$ nats published by ACE-Lensing on its benchmark grid)
- **Quantile Adaptive KL**: **0.00179 nats median** (stabilized against sparse Poisson tail noise)
- **Truncated Wasserstein-1 ($W_1$)**: **0.00591 $\Delta\mu$ median** (mean magnification error on empirical support $\mu \le 1.05\max\mu_{\text{sample}}$)
- **Kolmogorov-Smirnov ($D_{\text{KS}}$)**: **1.73% median** across contexts (context-median of maximum cumulative probability error; 90th percentile $3.66\%$, test-split max $7.80\%$)
- **Total Variation (TVS)**: **3.37% median** across contexts (test-split maximum $8.48\%$ on the 1,089 held-out configurations)
- **Latency**: **~22 ms / eval on CPU** (single flow body, 5.9x faster than multi-body blends)

## Install

```bash
pip install -e .
# or, for the plotting extras used by evaluation/examples:
pip install -e ".[plots]"
```

Requires Python >= 3.12. Pinned dependencies (numpy, torch, zuko, h5py, scikit-learn) are
in `pyproject.toml`.

## Quick usage

The package exposes one convenience function at the root, so getting the
PDF for a cosmology is a single call:

```python
import flumen

# defaults are a Planck-like fiducial cosmology; only z_s is required
mu, pdf = flumen.generate_pdf(z_s=1.0)

# override any cosmology parameter
mu, pdf = flumen.generate_pdf(z_s=2.0, h=0.70, Om=0.28, sigma8=0.90)

# supply your own mu grid instead of the default log-spaced one
import numpy as np
mu, pdf = flumen.generate_pdf(z_s=1.0, mu=np.geomspace(0.5, 15.0, 1000))
```

`mu` and `pdf` are plain `numpy.ndarray`s of the same shape; `pdf` is
`p(mu)` (density w.r.t. `d mu`, integrates to 1 over `mu`). Use
`flumen.generate_pdf_lnmu(...)` for the density w.r.t. `d ln(mu)` instead.

Cosmology parameters accepted: `h`, `Om`, `sigma8`, `Ob`, `ns`, `zeq`. See
`flumen.TRAINING_RANGE` for the confirmed validity box per parameter (calls
outside it emit a warning but still evaluate — treat the result as
extrapolation). The underlying model is loaded once and cached, so repeated
calls are cheap.

A runnable version of the above is in `examples/quickstart.py`.

### Lower-level API

`flumen.generate_pdf` wraps `flumen.model.MagnificationPDF`, which remains
available directly for full control over `theta`/`z_s`/flux calibration:

```python
import numpy as np
from flumen.model import load_model

model = load_model("cpu")
theta = (0.67, 0.30, 0.85, 0.0493, 0.965, 3.402)  # (h, Om, sigma8, Ob, ns, zeq/1000)
mu = np.geomspace(0.7, 20.0, 500)
p_mu = model.pdf_mu(mu, z_s=1.0, theta=theta)
```

`pdf_mu` is a density with respect to `d mu`; `pdf_lnmu` and `log_prob_lnmu`
are densities with respect to `d ln(mu)`. Pass `load_model(flux_mode="standard")`
for the legacy, non-unit-flux calibration — see `docs/MODEL_CARD.md` for
the tradeoff between the two.

## Contents

- `model/`: the inference implementation, context predictor calibration, and the trained
  production checkpoint (`single_body_gauss.pt`). This is all `flumen.generate_pdf` needs at runtime.
- `data/`: $u$-space train/validation/test arrays and simulation datasets with a SHA-256 manifest.
- `training/`: standalone retraining recipes (`train.py`, `fit_predictors.py`, `data.py`) for the flow and context regressions.
- `evaluation/`: validation/test accuracy certificates, reference simulation
  arrays, the KL evaluator (`evaluate_kl.py`), and the regression test suite.
- `qualification/`: machine-readable promotion and qualification records backing the
  claims in `PROVENANCE.md`.
- `docs/MODEL_CARD.md`: architecture, intended use, quantitative results,
  and known limitations.
- `examples/`: minimal runnable usage scripts.

## Verification

Run the test suite:

```bash
python -m pytest -q evaluation/test_model.py
(cd data && shasum -a 256 -c MANIFEST.sha256)
shasum -a 256 -c ARTIFACTS.sha256
```

Recompute the complete untouched test-split KL certificate across all 1,089 held-out contexts:

```bash
python -m flumen.evaluation.evaluate_kl \
  --split test --output evaluation/results/test_summary_rerun.json
```

Recreate the full 5x8 parameter-space atlas, with both simulator-overlay and
model-only variants:

```bash
python -m flumen.evaluation.plot_full_parameter_atlas --ymin 1e-3
```

This writes high-resolution PNG plus vector PDF/SVG files to
`evaluation/figures/`, using only the frozen model and the reference data
packaged in this repository.

## Retraining

Retraining scripts write new checkpoints under `training/`; they never
overwrite the shipped weights. Retrain the production single-body normalizing flow:

```bash
python -m flumen.training.train --epochs 16
```

For a rapid verification of the training pipeline on a small subset:

```bash
python -m flumen.training.train --epochs 1 --max_configs 100 --bs 4096
```

To refit the degree-3 polynomial Ridge regressions for location, scale, anchor $\bar{\kappa}$, and empty-beam bound $y_b$:

```bash
python -m flumen.training.fit_predictors
```

Training reads the $u$-space data bundled under `data/u_space/`. See `PROVENANCE.md` for the exact configuration allocation and checkpoint lineage behind the shipped model.

## Citation

If you use `flumen` in your research, please cite:

```bibtex
@article{Baltabay:2026nmz,
    author = "Baltabay, Galymzhan and Vaskonen, Ville",
    title = "{FLUMEN: Neural Emulator of an Advanced Stochastic Weak Lensing Model}",
    eprint = "2609.21806",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "9",
    year = "2026"
}
```

The underlying Monte-Carlo lensing simulator is described in [vianvask/halos](https://github.com/vianvask/halos).

You can contact me at galymzhan.baltabay@kbfi.ee.
