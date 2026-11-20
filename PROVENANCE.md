# Provenance: Production Model v3

## Analytic backend mix3filF-v1: halo-filament overlap correction (FLUMEN 0.7.0, installed 2026-11-20)

The default analytic backend (`backend="analytic"`, artifact `mix3filF-v1`, `model/checkpoints/analytic_mix3filF_v1/`)
replaces `mix3-v1` as the default. `mix3-v1` is kept unchanged in `model/checkpoints/analytic_mix3_v1/` and loads with
`load_analytic_model(version="mix3-v1")`; its tests are pinned to that version (`evaluation/test_analytic_mix3.py`).

- **Physics change**: the same correction as the MC backend v3.1 below, ported to the semi-analytic chain
  (`stochastic_wl_analytic` commit 79b3993, `analytic/filament_overlap.py`, switch `SGL_FILF=1`). F is built per
  cosmology from the same walk table G (SHA-256 `14d40768...3f2c`) and closed per z to the chain's own halo
  fraction. The analytic chain keeps its filament treatment (every filament counted, no convergence threshold).
  In the engine's threshold mode the port reproduces the MC change of sigma_DL at Planck (-10.34 / -7.88 % vs MC
  -10.28 / -8.04 % at z_s = 1 / 3); in the default treatment the change is -13.9 / -12.0 %.
- **Training data**: the 4,372 mix3 configurations (3,826 Gaussian + 546 VP closure; same design, setting aggr2p,
  splits 2,197 / 1,087 / 1,088) regenerated with the correction on the KBFI cluster (2026-11-19/20). Every
  config records the correction; with it off the cluster code reproduces the stored mix3 configs to 1e-12 of the
  peak. Assembled with the mix3 settings (`learning/assemble.py --tag mix3filF`).
- **Training**: the mix3 recipe unchanged (`learning/run_mix3filF.sh`: ctx `--anchors all`, SiLU body, seeds 0-7,
  train-only, validation checkpoint selection).
- **Export check**: `learning/verify_mix3filF_flumen.py --installed`: the installed model equals the training-side
  emulator to 3.8e-16 of the peak (120 validation PDFs) and 6.3e-12 in N10; goldens in
  `evaluation/analytic_mix3filF_goldens.json` come from the training side. Record:
  `model/FREEZE_analytic_mix3filF-v1.json`, `qualification/analytic_mix3filF_export_verification.json`.
- **Validation accuracy** (each model against its own chain targets, 1,087 configs): median max error 0.52 % of
  the peak (mix3-v1: 0.30 %), 95th percentile 2.8 % (1.6 %), median KL 1.3e-5 (4.5e-6). Largest errors at
  z_s < 0.4 (narrow VP peaks, up to 46 % of the peak). No test-split scoring. Low-z N10 not re-checked.
- **Effect at Planck** (source plane): peak of P(ln mu) 8.4 -> 12.4 (z_s = 1), 3.07 -> 3.85 (z_s = 3); width of
  ln mu (|ln mu| <= 1) 0.067 -> 0.058 and 0.161 -> 0.142.
- **Full record**: `stochastic_wl_analytic/tmp/filF/NOTES.md`.

---

## MC backend v3.1: halo-filament overlap correction (FLUMEN 0.6.0, installed 2026-11-19)

The MC backend (`backend="mc"`) now ships the v3.1 checkpoint and context fits. They replace the
v3 files described in sections 1-4 below, which remain in git history (FLUMEN 0.5.0).

- **Physics change**: each filament's density is multiplied by [1 - F(M_f, z)], where F is the mass
  fraction of a filament already inside halos above 1e7 Msun, from excursion-set walks. F is built per
  cosmology from a universal walk table G(S_f/dc^2, S_min/dc^2) and rescaled at each z so that the
  removed filament mass equals the engine's halo fraction f_h. The factor enters both the filament
  threshold count and the sampled filament convergence. This removes the halo-filament double counting
  raised by the A&A referee (aa63347-26, point 4).
- **Training data**: the same v3 designs, seeds and settings as v3 (3,695 / 1,088 / 1,089 configs,
  40k rays per config, kappa_anchor 0, cons16), regenerated with the corrected engine. With the
  correction switched off, the engine reproduces the v3 training rows exactly.
- **Training**: the v3 recipe unchanged (`train_single_body.py --arm margin_d0.05`, seed 0, 8 epochs).
  The same run on the old data reproduces the v3 weights bit for bit.
- **Checkpoint**: `model/checkpoints/single_body_gauss.pt`, SHA-256
  `7644d5fe64b9b8599cb0cb6e767fc01a38d6d04c0c93c23a499421fc7a6074e8` (v3 was `a16f5423...e5a6e5`).
- **Context fits**: `model/calibration/context_fits.json`, SHA-256
  `41f050596ccec282d07abfdd6202ec255807961740b19870e4394ea2d4a4f604`. Test-split errors: s median
  relative 1.01 % (95 %: 2.72 %), m 0.00033, kbar 1.15 %, y_b 0.0170.
- **Held-out test (1,089 configs, new data)**: ACE KL median 0.00482 (p90 0.00735, max 0.0302),
  quantile KL median 0.00139, KS median 1.63 % (max 7.90 %), TV median 3.30 % (max 8.47 %),
  tail slope at mu = 1e3 median -2.00001, max |<1/mu> - 1| = 6.7e-5. The v3 model scored on the
  same new test set gives ACE KL median 0.0234 and KS median 5.0 %.
- **Effect at Planck** (source plane): sigma_DL 0.03440 -> 0.03065 at z_s = 1 (-10.9 %) and
  0.07972 -> 0.07271 at z_s = 3 (-8.8 %). The corrected Monte Carlo itself changes by -10.3 % and -8.0 %.
- **Not updated**: `evaluation/results/`, `qualification/` and sections 1-4 below describe v3.
  The analytic backend (default) does not include the correction yet.
- **Full record**: `flumen_paper/referee_response/production_filF/NOTES.md` (engine patch, G table,
  data generation, training and evaluation logs).

---

## v3 (FLUMEN up to 0.5.0)

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

Superseded as the default in 0.7.0 by `mix3filF-v1` (see the top of this file); still loadable with `load_analytic_model(version="mix3-v1")`. Until 0.6.0: the model is named `analytic` and is the default for generate_pdf, generate_pdf_lnmu and get_model. Artifact identifiers above are internal provenance; no version selection is needed. Weights are unchanged.
