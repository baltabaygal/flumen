# Provenance: Production Model v3

- **Release Date**: 2026-09-14
- **Model Version**: `3.0.0` (Production Shipped)
- **Architecture**: Single-body SOSPF normalizing flow with Gaussian base distribution, margin-buffered empty-beam cutoff ($\delta=0.05$, $\epsilon_b=0.02$), and smooth $C^\infty$ asymptotic relaxation bridge ($h = 1.0/s$) to asymptotic $\mu^{-2.0000}$ tail.
- **Physics Foundation**: Concentration model `cons16` (Ludlow et al. 2016), Subhalo model 5, and real-space top-hat $\sigma_8$ normalization.
- **Repository Source**: Developed across `single_body_new` (dataset generation & physical bound verification) and `single_body_naive` (coordinate discovery & tail fix).

---

## 1. Selected Checkpoint

- **Path**: `model/checkpoints/single_body_gauss.pt`
- **SHA-256**: `a16f54238adae1c0580f6e3b85d9ee879762d521161e6db356c383ebc1e5a6e5`
- **Origin**: `single_body_naive/runs/margin_d0.05_margin_s0.pt` ($\delta=0.05$ margin scan baseline)
- **Conditioner**: MLP depth 1, hidden features 128 (`mlp_d1_h128`, 20,874 parameters)
- **Flow**: SOSPF with 3 transforms, degree 8, 5 polynomials per transform
- **Base Distribution**: Standard Gaussian
- **Coordinate Transformation**: $u = \ln(y - y_b + 0.05)$ with boundary buffer $\epsilon_b = 0.02$

---

## 2. Training Data & Allocation

- **Source Datasets**:
  - `data/base_optimal_lhs`: 1,983 training configs (LHS over 6D space)
  - `data/corner_augmentation`: 1,712 training configs (all 8 corners of $(h, \Omega_M, \sigma_8)$ box $\times$ 8 redshifts)
  - Total Training Rows: 3,695 configurations / 22,170,000 rays (6,000 rays/config)
- **Validation Set**: 1,088 configs (6,528,000 rays)
- **Held-Out Test Set**: 1,089 configs (untouched held-out pool, 6,534,000 rays)

---

## 3. Calibration Assets

- **Context Predictors**: `model/calibration/context_fits.json`
  - **SHA-256**: `f1b4d2bdde8e7bf7ac6a1e2b028fa4e839eacac770c8395f923158dd48a10015`
  - Predicts $s(z_s, \theta)$, $m(z_s, \theta)$, $\bar{\kappa}(z_s, \theta)$, and $y_b(z_s, \theta)$ from 6D cosmological context using degree-3 polynomial Ridge regressions.
  - Scale $s$ median relative error on test split: **0.97%** (95th percentile: 2.55%)
  - Location $m$ median absolute error: **0.00030**
  - Boundary $y_b$ median absolute error: **0.0276** (comfortably absorbed by $\delta = 0.05$)

---

## 4. Verification & Certification Metrics

- **Automated Unit Tests**: 20/20 tests passing (`evaluation/test_model.py`).
- **Held-Out Test Set (Multi-Metric over 1,089 contexts)**:
  - ACE-Protocol $KL$: **0.00476 nats median** (p90: 0.0080, max: 0.0635; comparable to ACE paper published 0.00690 on its separate benchmark grid)
  - Quantile Adaptive $KL$: **0.00179 nats median** (p90: 0.0063)
  - Truncated Wasserstein-1 ($W_1$): **0.00591 $\Delta\mu$ median** (mean physical displacement on empirical support $\mu \le 1.05\max\mu_{\text{sample}}$)
  - Kolmogorov-Smirnov ($D_{\text{KS}}$): **1.73% median** across contexts (90th percentile: 3.66%, test-split max: 7.80%)
  - Total Variation ($TVS$): **3.37% median** across contexts (test-split max: 8.48%)
- **Asymptotic Tail Exponent**: $\frac{d\ln p}{d\ln\mu} = -2.0000 \pm 0.001$ as $\mu \to \infty$.
- **Flux Conservation**: Numerical unit inverse-mean $\langle 1/\mu \rangle = 1.0000 \pm 0.001$.


---

## Analytic backend checkpoints (2026-10-20; REMOVED from the package in 0.5.0, see git history, commit `73e4654`)

- `model/checkpoints/analytic/ctx.pt`: `21c2d3f3dfb4a96be31bc497264925358c593d235f28c44715a780b8d34a1a11`
  (from `stochastic_wl_analytic/learning/models/ctx_final.pt`)
- `model/checkpoints/analytic/body_s0.pt`: `d6211ce4698c33c58e512afca85bcb928eb276398a519216b801e12326bb7080`
  (from `.../learning/models/body_final_s0.pt`)
- `model/checkpoints/analytic/body_s1.pt`: `f89b3aed1dade768583855677190cf0b09bc7f8a35a133ae6e0c3e0635a57fe3`
  (from `.../learning/models/body_final_s1.pt`)
- **Pipeline:** `stochastic_wl_analytic/learning/`; README sections 6–11 there give every step, choice and number.
- **Code:** `model/analytic.py` is a float64 port of `learning/emulator.py`, agreeing to ≤ 8e-4 in ln P at
  P ≥ 1e-12 of the peak (float32 vs float64 seed mixture). Golden values and checks are in
  `evaluation/test_analytic.py`.

---

## Analytic backend `silu-final-v1` (2026-10-23, FLUMEN 0.5.0)

- Checkpoints in `model/checkpoints/analytic_silu_v1/` (`MANIFEST.sha256`, `meta.json`); the 2026-10-20 ReLU checkpoints were removed from FLUMEN in 0.5.0 (the originals remain in `stochastic_wl_analytic/learning/models/`).
  - `ctx.pt`: `0b20fc36fadcb5bf1db2a4b8745ec3b803fb4f0fd4d569e7735fe3b185235d82` (from `stochastic_wl_analytic/learning/models/ctx_mix1_final.pt`)
  - `body_s0.pt`: `3ddbf7696638862f998f0a1718581bff898ff7ea8efa03b9c5acd13094185f2b`, `body_s1.pt`: `bfadb19c507a76b284c295caa0e50a8ca14a09f562f66b809342a76e1a749dd8`
  - `body_s2.pt`: `166573be7b3290f06d772f3f627908666a69cf6734a5492cc3192704951e1075`, `body_s3.pt`: `15e296f9d0fe8e2aaac9a3f5708191890a349796e0a62efa87109066676de6a6`
    (from `.../learning/models/body_silu_final_s{0..3}.pt`, weights copied bit for bit by `learning/export_flumen_analytic.py`)
- **Recipe and reproducibility metadata:** `stochastic_wl_analytic/learning/models/silu_final_metadata.json` (commands, hyper-parameters, seeds, git commit and dirty flag,
  library versions, SHA-256 of all code, data and checkpoints); report `learning/PRODUCTION_READINESS_SILU.md`; one-shot test record `learning/results/test_silu_once.json`.
- **Code:** `model/analytic.py` (port; SiLU conditioner, any number of seeds, optional guard fallback and diagnostics), `evaluation/analytic_numpy_reference.py`
  (independent NumPy float64 networks), `evaluation/test_analytic.py`; freeze record `model/FREEZE_analytic_silu-final-v1.json`.



## Analytic model (2026-10-26; internal artifact `mix3-v1`)

Source: stochastic_wl_analytic/learning/models/ctx_mix3.pt and
body_mix3silu_s0.pt through s7.pt. These are the existing train-only models,
not the continuation or Sobolev trial checkpoints. All tensors are exported
bit-for-bit with weights_only-safe metadata by learning/export_mix3_flumen.py.
Source hashes, split/selection provenance and known limitations are in
model/checkpoints/analytic_mix3_v1/meta.json; checksums in MANIFEST.sha256.
Freeze and verification record: model/FREEZE_analytic_mix3-v1.json.
Original mix1 checkpoints and historical golden regressions are preserved.
This local promotion does not publish a package release or make a git commit.

Public interface: the model is named `analytic` and is the default for generate_pdf, generate_pdf_lnmu and get_model. Artifact identifiers above are internal provenance; no version selection is needed. Weights are unchanged.
