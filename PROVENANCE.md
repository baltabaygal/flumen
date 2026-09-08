# Provenance

Lineage and change history for the shipped model. For architecture, intended
use, and known limitations see `docs/MODEL_CARD.md`; for machine-readable
qualification metrics see `qualification/QUALIFICATION.json` and the
`qualification/PATCH_*.json` records referenced below.

## Selected checkpoints

- `model/checkpoints/boundary_body.pt` -- the production body ("v1"), a
  warm-start fine-tune.
  - SHA-256: `1a8a07f9deb6937fa47fd98bb9bc3303b1aad4e8265dd35796735dbec2ef4df3`
  - saved epoch: 11
  - best transformed validation NLL: `0.6102788436983607`
  - warm start: epoch-14 boundary body, SHA-256
    `6bb623d307c9d9edbcfd2acf248c7a0d17fd7b2ae9f10f07c1a33407c03e6706`
    (bundled at `training/starting_checkpoints/boundary_body_epoch14.pt`)
  - row allocation: 2.7M reference + 400k augmentation training rows;
    594k reference + 90k augmentation validation rows
  - optimizer: Adam, learning rate `3e-4`, batch size 8192
  - portable reconstruction: `training/train_boundary_finetune_v1.py`
- `model/checkpoints/asymmetric_body_sl0p6.pt` -- an alternative body used by
  `model/composite.py`'s crossing search; reconstructed by
  `training/train_asymmetric.py`.

An intermediate epoch-12 "v2" checkpoint was evaluated and rejected (no
aggregate or target-region KL improvement over v1) and is not shipped.

## Composite (`model/composite.py`)

Combines the flow body with an analytic power-law-tail (POT) extrapolation
above the fitted body/tail crossing, plus a monotonicity-preserving Hermite
bridge across the crossing region. Verified against the source research
implementation: zero plan/mask mismatches and a maximum finite log-density
difference of `0.0` over a 40-context x 500-point reference grid.

## Change log

### 2026-09-03 -- initial qualified bundle
Body (v1 above) + composite qualified together. 15/15 production tests and
18/18 experiment regression tests passed. Test-split (973 configs) KL_y:
median 0.002227, p90 0.004534, p99 0.012211, max 0.030580.

### 2026-09-04 -- shoulder correction
The raw body under-predicts the true density in a narrow band just before
the body/tail crossing (confirmed via held-out validation, independent MC at
several reference cosmologies, and direct visual inspection). Fix: a smooth
multiplicative Gaussian bump centered on the crossing, gated to
`0.4 <= z <= 2.5` (`SHOULDER_CORRECTION_ENABLED`, default `True`). No
checkpoint changed -- `model/composite.py` only. Result: test-split median
KL_y improved 0.002227 -> 0.002205; p90 0.004534 -> 0.004511; p99/max
unchanged to 5 decimals. Details:
`qualification/PATCH_2026-09-04_shoulder_correction.json`.

### 2026-09-07 -- tail-anchor correction, shipped then reverted, then fixed in closed form
An independent audit found a separate deep-tail (8 <= mu < 30) under-prediction
of ~15-16%, largely traced to corner-smoothing eroding the fitted mu=8
survival anchor by ~7.6-7.8%. A gated ramp-based correction
(`TAIL_ANCHOR_CORRECTION_ENABLED`) was shipped, then measured against the
audit's own diagnostic and found counterproductive in practice (it
under-delivered at mu=8 across the training box and briefly suppressed
density while ramping up), so it was reverted to `False`. The code remains
available as an opt-in regression guard.

The underlying erosion was then fixed directly in closed form inside
`pot_tail_logp` (`model/tail.py`): the tail's hazard-rate parameters are
solved so that `ln S(ln 3) = ln S_3` and `ln S(ln 8) = ln S_8` hold to
machine precision, with no post-hoc ramp. Verified monotonic
(`d ln(density)/dx < 0`) across the full reference panel and 100 random
training-box contexts, 0 violations. On the held-out tail audit (845
contexts, 8 <= mu < 30, 0.4 <= z <= 2.5), the pooled sim/model mass ratio
improved from 1.146 to 1.079. Full test-split KL_y median: 0.002211.
Details: `qualification/PATCH_2026-09-07_tail_anchor_reverted.json`.

### 2026-09-07 -- unit-flux calibration
`model/flux.py` gained a `flux_mode="unit"` calibration: a Gauss-Legendre
quadrature enforcing `<1/mu> = 1` exactly, replacing the coarser trapezoid
grid used by `flux_mode="legacy"`. An uncapped version regressed badly at
extreme contexts (median KL ~5x worse at z >= 8), driven by shift magnitude.
Fix: apply the same `FLUX_DELTA_MAX = 0.05` trust-region cap the legacy path
already used. Re-qualified: median/p90/p99/max KL within ~0.1% of legacy in
every z-band, including the previously-worst one. `flux_mode="unit"` is now
the default; `flux_mode="legacy"` remains available. Details:
`qualification/PATCH_2026-09-07_unit_flux.json`.

### 2026-09-07 -- centered C1 Hermite log-density bridge
Replaced the linear density blend across the body/tail crossing with an
exact C1 cubic Hermite bridge in log-density, with a Fritsch-Carlson (1980)
monotonicity limiter (rescales boundary derivatives whenever
`alpha^2 + beta^2 > 9` so the bridge cannot overshoot or develop a local
rise). This removes the visible slope-break ("knee") the previous linear
blend produced at low structure / low redshift. The crossing search
threshold was also extended down to mu=1.7 so low-z crossings near mu~1.85
are no longer missed. Verified: 0 handover slope reversals over the 40
panel contexts; unit-flux conservation preserved; full test-split KL_y
median 0.002201. This is the shipped default.

## Known open items

- A pre-existing (not introduced by any change above) monotonicity
  violation exists at a small number of extreme low-z/high-structure and
  "box lo-struct" corner contexts; see `docs/MODEL_CARD.md` for scope.
- The deep-tail (8 <= mu < 30) under-prediction is real and only partially
  addressed by the closed-form tail-anchor fix above; treat far-tail
  quantiles as approximate. See `docs/MODEL_CARD.md`.
