# Model Card: GW Weak-Lensing Emulator (v3, Single-Body)

## 1. Model Details
- **Architecture**: Single-body Sum-of-Squares Polynomial Flow (SOSPF).
- **Conditioner**: MLP with 1 hidden layer of 128 units (`mlp_d1_h128`).
- **Base Distribution**: Independent standard Gaussian with natural super-exponential descent.
- **Coordinate Transformation**: \(u = \ln(y - y_b + \delta)\) with margin \(\delta = 0.05\) and boundary buffer \(\epsilon_b = 0.02\).
- **Lower Cutoff Behavior**: Hard empty-beam cutoff strictly vanishing for \(\mu < \mu_{\text{cut}} = \exp(m + s(y_b - \epsilon_b - \delta))\). Models the fundamental Dyer–Roeder (1973) empty-beam limit.
- **Tail Handover**: Width-adaptive \(C^1\) cubic Hermite bridge (\(h = 1.0/s\)) connecting the SOSPF core at scale-invariant \(y_0 = 10\sigma\) to the asymptotic \(\mu^{-2.0000}\) power-law tail without derivative discontinuity or artificial shelf artifacts.
- **Context Predictors**: Calibrated Degree-3 polynomial Ridge regression models evaluated in pure NumPy for \((m, s, y_b, \bar{\kappa})\).
- **Total Parameters**: 20,874 flow weights + 120 context regression coefficients.

## 2. Intended Use
- Evaluates the probability density function \(p(\mu \mid z_s, \theta)\) for gravitational-wave source lines-of-sight.
- Intended for likelihood evaluations in cosmological Bayesian inference, Hubble diagram analysis, and mock catalog generation.
- **Valid Parameter Domain**:
  - Source Redshift: \(z_s \in [0.2, 12.0]\)
  - Hubble parameter: \(h \in [0.55, 0.80]\)
  - Matter density: \(\Omega_M \in [0.15, 0.50]\)
  - Amplitude: \(\sigma_8 \in [0.40, 1.50]\) (real-space top-hat normalized)
  - Baryon density: \(\Omega_B \in [0.03, 0.07]\)
  - Spectral index: \(n_s \in [0.94, 0.99]\)
  - Equality redshift: \(z_{\text{eq}} \in [3300, 3500]\)

## 3. Training Data & Physics
- Physics defaults: Concentration model `cons16` (Ludlow et al. 2016), Subhalo model 5 (exact clump retention down to \(10^7\,M_\odot\) with \(\kappa\)-thresholding and mass carve), and correlated clustering void field (`bias_model=1`, `bias_weak=true`).
- Base optimal Latin Hypercube Sampling (1,983 configs) plus systematic up-front 8-corner augmentation (1,712 configs).

## 4. Quantitative Performance & Head-to-Head vs v2
On the held-out test split:

| Metric | Legacy `production_model_v2` (2-Body Hybrid) | `production_model_v3` (Shipped SOTA) | Relative Improvement |
| :--- | :--- | :--- | :--- |
| **Architecture** | 2-body blend (Gumbel + RQS) | Single-body SOSPF (Gaussian base) | Single-flow simplicity |
| **ACE-Protocol \(KL\)** | 0.0086 nats | **0.00477 nats** | **1.8x better** (beats ACE 0.00690) |
| **Quantile Adaptive \(KL\)** | — | **0.00153 nats** | Stabilized against Poisson noise |
| **Wasserstein-1 (\(W_1\))** | — | **0.00495 \(\Delta\mu\)** | Mean physical magnification error \(< 0.005\) |
| **Kolmogorov-Smirnov (\(D_{\text{KS}}\))** | — | **1.67%** | Max CDF error \(\le 1.67\%\) |
| **Worst-Case Max \(KL\)** | 1.0635 nats | **0.01477 nats** | **72x lower outlier error** |
| **Inference Latency** | 130.48 ms / eval | **22.27 ms / eval** | **5.9x faster** |
| **Tail Handover** | Heuristic match | **Width-Adaptive \(C^1\) Bridge** (\(h = 1.0/s\)) | No shelf / smooth \(\mu^{-2}\) |
| **Flux Conservation** | Unit inverse-flux calibrated | **Unit inverse-flux calibrated** (\(\langle 1/\mu \rangle = 1.0000\)) | Exact Gauss-Legendre shift |

## 5. Limitations
- Single-image approximation: Extreme strong-lensing multiple-image micro-interference is outside model scope.
- Sub-cutoff extrapolation: Densities strictly vanish below the physical empty-beam bound (\(\mu < \mu_{\text{cutoff}} = 1/(1+\bar{\kappa})^2\)).
