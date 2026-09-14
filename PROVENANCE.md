# Provenance: Production Model v3

- **Release Date**: 2026-09-14
- **Model Version**: `3.0.0` (Production Shipped)
- **Architecture**: Single-body SOSPF normalizing flow with Gaussian base distribution, margin-buffered empty-beam cutoff (\(\delta=0.05\), \(\epsilon_b=0.02\)), and width-adaptive \(C^1\) Hermite bridge (\(h = 1.0/s\)) to asymptotic \(\mu^{-2.0000}\) tail.
- **Physics Foundation**: Concentration model `cons16` (Ludlow et al. 2016), Subhalo model 5, and real-space top-hat \(\sigma_8\) normalization.
- **Repository Source**: Local experimental research developed across `single_body_new` (dataset generation & physical bound verification) and `single_body_naive` (coordinate discovery & tail fix).

---

## 1. Selected Checkpoint

- **Path**: `model/checkpoints/single_body_gauss.pt`
- **SHA-256**: `a16f54238adae1c0580f6e3b85d9ee879762d521161e6db356c383ebc1e5a6e5`
- **Origin**: `single_body_naive/runs/gauss_s0.pt`
- **Conditioner**: MLP depth 1, hidden features 128 (`mlp_d1_h128`, 20,874 parameters)
- **Flow**: SOSPF with 3 transforms, degree 8, 5 polynomials per transform
- **Base Distribution**: Standard Gaussian
- **Coordinate Transformation**: \(u = \ln(y - y_b + 0.05)\) with boundary buffer \(\epsilon_b = 0.02\)

---

## 2. Training Data & Allocation

- **Source Datasets**:
  - `single_body_new/data/base_optimal_lhs`: 1,983 training configs (LHS over 6D space)
  - `single_body_new/data/corner_augmentation`: 1,712 training configs (all 8 corners of \((h, \Omega_M, \sigma_8)\) box \(\times\) 8 redshifts)
  - Total Training Rows: 22,170,000 rays (6,000 rays/config)
- **Validation Set**: 1,088 configs (6,528,000 rays)
- **Test Set**: 1,089 configs (untouched held-out pool)

---

## 3. Calibration Assets

- **Context Predictors**: `model/calibration/context_fits.json`
  - **SHA-256**: `c48bf8314ceed1f64add8210c83ef5dc3b13fb990bee915c1e19afc4d0f9f0be`
  - Predicts \(s(z_s, \theta)\), \(m(z_s, \theta)\), \(\bar{\kappa}(z_s, \theta)\), and \(y_b(z_s, \theta)\) from 6D cosmological context.
  - Scale \(s\) median relative error on test split: **2.40%**
  - Location \(m\) median absolute error: **0.0004**
  - Boundary \(y_b\) median absolute error: **0.0369** (comfortably absorbed by \(\delta = 0.05\))

---

## 4. Verification & Certification Metrics

- **Automated Unit Tests**: 19/19 tests passing (`evaluation/test_model.py`).
- **Held-Out Test Set (Multi-Metric)**:
  - ACE-Protocol \(KL\): **0.00477 nats** (beats ACE paper published 0.00690)
  - Quantile Adaptive \(KL\): **0.00153 nats**
  - Wasserstein-1 (\(W_1\)): **0.00495 \(\Delta\mu\)** (mean physical displacement)
  - Kolmogorov-Smirnov (\(D_{\text{KS}}\)): **1.67%** (max CDF percentage error)
  - Total Variation (\(TVS\)): **3.31%** (never exceeds 6.3% anywhere in test set)
- **Asymptotic Tail Exponent**: \(\frac{d\ln p}{d\ln\mu} = -2.0000 \pm 0.001\) for \(\mu > \mu_c\).
- **Flux Conservation**: Numerical unit inverse-mean \(\langle 1/\mu \rangle = 1.0000 \pm 0.001\).
