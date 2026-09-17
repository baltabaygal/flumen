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
