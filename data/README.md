# Simulation Datasets (Production Model v3)

This directory contains and documents the datasets used to train, validate, and certify the production single-body normalizing flow emulator (**v3**).

## 1. Dataset Allocation

The production v3 model was trained and evaluated on 6,000 rays per configuration across two Latin Hypercube and corner simulation suites:

- **Base Optimal LHS**:
  - Training: 1,983 configurations
  - Validation: 1,008 configurations
  - Held-out Test: 1,009 configurations
- **Corner Augmentation**:
  - Systematically covers the 8 vertices of the $(\Omega_M, \sigma_8, h)$ parameter space across 8 source redshifts.
  - Training: 1,712 configurations
  - Validation: 80 configurations
  - Held-out Test: 80 configurations

### Summary of Partitions
- **Training Set**: 3,695 configurations ($22,170,000$ rays)
- **Validation Set**: 1,088 configurations ($6,528,000$ rays)
- **Held-Out Test Set**: 1,089 configurations ($6,534,000$ rays)

All 1,089 test configurations remained strictly held out and untouched during training and calibration.

---

## 2. $u$-Space Representation

To enforce the physical Dyer–Roeder (1973) empty-beam cutoff $\mu_{\text{cut}} = 1 / (1 + \bar{\kappa})^2$ by construction, the rays are represented in boundary-adapted coordinates:

$$y = \frac{\ln\mu - m(c)}{s(c)}, \quad y_b(c) = \frac{-2\ln(1 + \bar{\kappa}(c)) - m(c)}{s(c)}$$

$$u = \ln(y - y_b(c) + \delta), \quad \text{with } \delta = 0.05$$

The packaged tensors are stored in `data/u_space/`:
- `train.npz`: training rays ($u$), configuration indices (`idx`), 7D cosmological contexts (`ctx`), and metadata ($m, s, y_b, \bar{\kappa}$)
- `validation.npz`: validation rays, indices, contexts, and metadata
- `test.npz`: held-out test rays, indices, contexts, and metadata for certification

---

## 3. Verification & Asset Download

Checksums for all bundled assets are frozen in `MANIFEST.sha256`. Verify integrity with:

```bash
cd data && shasum -a 256 -c MANIFEST.sha256
```

or using the Python asset manager:

```bash
python -m flumen.data.download
```

To download any missing data arrays from the release repository:

```bash
python -m flumen.data.download --download
```

---

## 4. Calibrated Cosmological Domain

| Parameter | Minimum | Maximum | Description |
| :--- | :--- | :--- | :--- |
| $z_s$ | $0.20$ | $12.00$ | Source redshift |
| $h$ | $0.55$ | $0.80$ | Hubble parameter ($H_0 / 100$) |
| $\Omega_M$ | $0.15$ | $0.50$ | Total matter density |
| $\sigma_8$ | $0.40$ | $1.50$ | Amplitude of matter fluctuations (real-space top-hat) |
| $\Omega_B$ | $0.03$ | $0.07$ | Baryon density |
| $n_s$ | $0.94$ | $0.99$ | Scalar spectral index |
| $z_{\text{eq}} / 1000$ | $3.30$ | $3.50$ | Matter-radiation equality redshift ($z_{\text{eq}} \in [3300, 3500]$) |

Queries outside these boundaries emit a runtime warning and should be treated as extrapolations.
