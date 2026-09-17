# Model Card: GW Weak-Lensing Emulator (v3, Single-Body)

## 1. Model Details
- **Architecture**: Single-body Sum-of-Squares Polynomial Flow (SOSPF).
- **Conditioner**: MLP with 1 hidden layer of 128 units (`mlp_d1_h128`).
- **Base Distribution**: Independent standard Gaussian with natural super-exponential descent.
- **Coordinate Transformation**: $u = \ln(y - y_b + \delta)$ with margin $\delta = 0.05$ and boundary buffer $\epsilon_b = 0.02$.
- **Lower Cutoff Behavior**: Hard empty-beam cutoff strictly vanishing for $\mu < \mu_{\text{cut}} = \exp(m + s(y_b - \epsilon_b - \delta))$. Models the fundamental Dyer–Roeder (1973) empty-beam limit.
- **Tail Handover**: Smooth $C^\infty$ asymptotic relaxation bridge ($q'(y) = -s + (d_0 + s)e^{-(y-y_0)/h}$ with $h = 1.0/s$) connecting the SOSPF core at scale-invariant $y_0 = 10\sigma$ to the asymptotic $\mu^{-2.0000}$ power-law tail at the exact physical $\mathcal{O}(1/\mu)$ fold-caustic rate without derivative discontinuity or artificial shelf artifacts.
- **Context Predictors**: Calibrated Degree-3 polynomial Ridge regression models evaluated in pure NumPy for $(m, s, y_b, \bar{\kappa})$.
- **Total Parameters**: 20,874 flow weights + 120 context regression coefficients.

## 2. Intended Use
- Evaluates the conditional probability density function $p(\mu \mid z_s, \theta)$ for gravitational-wave source lines-of-sight.
- Intended for likelihood evaluations in cosmological Bayesian inference, Hubble diagram analysis, and mock catalog generation.
- **Valid Parameter Domain**:
  - Source Redshift: $z_s \in [0.20, 12.00]$
  - Hubble parameter: $h \in [0.55, 0.80]$
  - Matter density: $\Omega_M \in [0.15, 0.50]$
  - Amplitude: $\sigma_8 \in [0.40, 1.50]$ (real-space top-hat normalized)
  - Baryon density: $\Omega_B \in [0.03, 0.07]$
  - Spectral index: $n_s \in [0.94, 0.99]$
  - Equality redshift: $z_{\text{eq}} \in [3300, 3500]$

## 3. Training Data & Physics
- Physics defaults: Concentration model `cons16` (Ludlow et al. 2016), Subhalo model 5 (exact clump retention down to $10^7\,M_\odot$ with $\kappa$-thresholding and mass carve), and correlated clustering void field (`bias_model=1`, `bias_weak=true`).
- Base optimal Latin Hypercube Sampling (1,983 configs) plus systematic up-front 8-corner augmentation (1,712 configs).
- Total training rows: 3,695 configurations ($22,170,000$ rays at 6,000 rays/config).
- Validation set: 1,088 configurations ($6,528,000$ rays).
- Held-out test set: 1,089 configurations ($6,534,000$ rays).

## 4. Quantitative Performance
Certified on all 1,089 held-out test configurations:

| Metric | Metric Definition & Protocol | `production_model_v3` Result | Interpretation & Notes |
| :--- | :--- | :--- | :--- |
| **ACE-Protocol $KL$** | 120 width-relative bins on $y \in [-6, 12]$ ($N \ge 5$) | **0.00476 nats median** (p90: 0.0080, max: 0.0635) | Comparable to the $0.00690$ nats published by ACE-Lensing on its separate benchmark grid. |
| **Quantile Adaptive $KL$** | Equal-mass 50-bin Miller-Madow debiased | **0.00179 nats median** (p90: 0.0063) | Stabilized against sparse Poisson tail noise. |
| **Truncated $W_1(\mu)$** | Wasserstein-1 on empirical support $\mu \le 1.05\max\mu_{\text{sample}}$ | **0.00591 $\Delta\mu$ median** (p90: 0.1690) | Mean physical magnification displacement. Full-distribution $W_1$ diverges due to the $\mu^{-2}$ tail. |
| **Kolmogorov-Smirnov ($D_{\text{KS}}$)** | Context-median of $\max \|F_{\text{emp}} - F_{\text{mod}}\|$ | **1.73% median** (p90: 3.66%, max: 7.80%) | Median across test contexts. Context CDF error is bounded per-profile. |
| **Total Variation (TVS)** | Integrated absolute density difference on ACE grid | **3.37% median** (p90: 4.47%, max: 8.48%) | Measured on 1,089 held-out contexts; test-split maximum is 8.48%. |
| **Worst-Case Outlier $KL$** | Full 1,089-config test set maximum ACE $KL$ | **0.0635 nats** | Significantly reduced from legacy v2 (1.0635 nats). |
| **Inference Latency** | Single-evaluation runtime on standard CPU | **~22 ms / eval** | 5.9x faster than legacy multi-body blend architectures. |
| **Tail Handover** | Smooth $C^\infty$ asymptotic relaxation bridge ($h = 1.0/s$) | Asymptotic $p(\mu) \propto \mu^{-2.0000}$ | Exact physical fold caustic scaling without boundary shelf. |
| **Flux Conservation** | Enforced on retained-ray measure | $\langle 1/\mu \rangle = 1.0000 \pm 0.001$ | Boundary-preserving warping via exact Gauss-Legendre calibration. |

### Note on Truncated Wasserstein Distance
Because the model enforces the physical asymptotic power-law tail $p(\mu) \propto \mu^{-2}$, the expectation value $\mathbb{E}[\mu] = \int^\infty \mu\,p(\mu)\,d\mu$ diverges logarithmically. Consequently, the true untruncated Wasserstein-1 distance between the continuous theoretical model and any finite empirical sample is formally infinite. The reported metric $W_1(\mu)$ is the truncated Wasserstein-1 distance computed by conditioning on the empirical support range $\mu \in [\mu_{\text{cut}}, 1.05 \max\mu_{\text{sample}}]$.

## 5. Limitations
- Single-image approximation: Extreme strong-lensing multiple-image micro-interference is outside model scope.
- Sub-cutoff extinction: Densities strictly vanish below the physical empty-beam bound ($\mu < \mu_{\text{cutoff}} = 1/(1+\bar{\kappa})^2$).
- Retention measure: Unit inverse-mean flux $\langle 1/\mu \rangle = 1$ is calibrated and certified on the retained-ray simulation measure.
