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


## 6. Analytic backend (`backend="analytic"`, 2026-10-20)

- **What it emulates.** The semi-analytic compound-Poisson chain of `stochastic_wl_analytic`, draft-paper model
  (`run_full_chain.run(zs, "full")`: r200 halo edges, ODE growth, radiation, $\xi_{\rm lin}$ Gaussian clustering, exact
  moment subhalos, filaments, pseudo-elliptical halos), at the training resolution "aggr2p" (≤ 0.3% of the peak from
  the converged chain). Native quantity: the source-plane PDF.
- **Training data.** 3,903 deterministic PDFs, $z_s\in[0.5,12]$, 10 parameters
  $(z_s, h, \Omega_m, \sigma_8, \Omega_b, n_s, z_{\rm eq}, \Omega_K, w_0, w_a)$, optimised LHS with rejection on the
  dark-energy domain; checkerboard split 1,939 / 982 / 982. Fourier-inversion artifacts removed before training
  (no smoothing inside the trusted range; smooth cutoff and $\mu^{-2}$ tail outside it).
- **Model.**
  - Context: MLP ensemble, 5 members × 3 × 128.
  - Body: NSF flow (zuko; 3 transforms, 2 × 128, 16 bins, Gaussian base) in $u=\ln(y-y_b+0.2)$. Its context is the 10
    parameters plus the predicted (ln s, m/s, y_b). Two seeds, mixture. Final model trained on train + validation,
    stop at epoch 1100.
  - Edge guard: the log-quadratic law below $10^{-8}$ of the peak.
  - Tail: the hybrid $\mu^{-2}$ law, C¹ at the join, ±0.15 smoothstep blend.
- **Test accuracy** (982 configs, once): see README; per $z_s$ band
  0.5–1 / 1–3 / 3–6 / 6–12: max $|\Delta P|$/peak 1.0 / 0.65 / 0.37 / 0.37% (median).
- **Limitations.**
  - $z_s<0.5$ is not covered.
  - The far tail at $z_s\lesssim1$ is only as converged as the chain (10–30% beyond $\ln\mu\approx3$).
  - A small curvature feature at $y=10$ widths (≤ 0.4% in ln P) remains until the next retrain.
  - The model is not the MC engine's: no unit-flux calibration, and different halo truncation.
- **Plane (both backends).** `generate_pdf` defaults to the image plane (random lines of sight, $\langle1/\mu\rangle=1$),
  kept for backward compatibility. For standard sirens (sources at random positions, before detection selection) use
  `plane="source"`; the image-plane PDF is biased toward magnification by about ${\rm Var}(\ln\mu)$.
