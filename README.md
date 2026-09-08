# flumen

**F**ast **L**ensing **U**nified **M**agnification **E**mulator with
**N**ormalizing flows.

A normalizing-flow emulator for the conditional gravitational-wave
weak-lensing magnification PDF, trained on Monte-Carlo simulations of the
Vaskonen (2026) lensing model. Given a source redshift and a cosmology, it
returns the magnification PDF p(mu) in milliseconds on CPU, in place of a
Monte-Carlo run.

This is a frozen, qualified production model (last patched 2026-09-07): a
two-body SOS polynomial flow handling the bulk of the distribution, handed
off through a monotonicity-preserving Hermite bridge to an analytic
power-law tail, with an exact unit-flux (`<1/mu> = 1`) calibration. See
`docs/MODEL_CARD.md` for the architecture, accuracy figures, and known
limitations, and `PROVENANCE.md` for the checkpoint lineage and full change
log.

## Install

```bash
pip install -e .
# or, for the plotting extras used by evaluation/examples:
pip install -e ".[plots]"
```

Requires Python >= 3.12. Pinned dependencies (numpy, torch, zuko, h5py) are
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
are densities with respect to `d ln(mu)`. Pass `load_model(flux_mode="legacy")`
for the previous, non-unit-flux calibration — see `docs/MODEL_CARD.md` for
the tradeoff between the two.

## Contents

- `model/`: the inference implementation, calibration files, and two trained
  checkpoints. This is all `flumen.generate_pdf` needs at runtime.
- `data/`: the exact HDF5 train/validation/test and augmentation datasets
  used to produce the shipped checkpoints, with a SHA-256 manifest.
- `training/`: portable retraining scripts and the warm-start checkpoint the
  shipped body was fine-tuned from.
- `evaluation/`: validation/test accuracy certificates, reference simulation
  data, a KL evaluator, and the regression test suite. Figures are not
  checked in — generate them on demand (see "Verification" below).
- `qualification/`: machine-readable promotion/patch records backing the
  claims in `PROVENANCE.md`.
- `docs/MODEL_CARD.md`: architecture, intended use, quantitative results,
  and known limitations.
- `examples/`: minimal runnable usage scripts.

## Verification

```bash
python -m pytest -q evaluation/test_model.py evaluation/test_unit_flux.py
(cd data && shasum -a 256 -c MANIFEST.sha256)
shasum -a 256 -c ARTIFACTS.sha256
```

Recompute the complete untouched test-split KL certificate:

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
overwrite the shipped weights. The shipped body was obtained by warm-start
fine-tuning:

```bash
python -m flumen.training.train_boundary_finetune_v1
```

The from-scratch body and asymmetric-body recipes are also available:

```bash
python -m flumen.training.train_boundary
python -m flumen.training.train_asymmetric
```

For a short pipeline check rather than a full run, set `ROWS_SCALE` and
`EPOCHS`, e.g. `ROWS_SCALE=0.01 EPOCHS=3`. Training reads only the data
bundled under `data/`. See `PROVENANCE.md` for the exact row allocation and
checkpoint lineage behind the shipped model.
