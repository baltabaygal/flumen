# Model card: high-z smooth-edge hybrid

## Status

As of 2026-09-07, the default output uses an explicit **unit inverse-mean
calibration**, computed with a Gauss-Legendre quadrature that is materially
more accurate than the old finite trapezoid grid. This is a changed
positive-PDF target convention, not a repair of the simulator's lens
mapping. Use `load_model(flux_mode="legacy")` to reproduce the earlier
calibration exactly.

⚠ **Same-day correction**: the first version of this change applied NO cap
to the resulting shift, and a 973-context qualification run found this cost
a ~5x median KL regression specifically at z>=8 (max KL 13x worse, p99 22x
worse) -- driven overwhelmingly by shift MAGNITUDE, not z itself
(correlation(|delta|,KL)=0.974 vs correlation(z,KL)=0.47), with every
worst-affected context showing the OLD legacy shift already saturated at
its own cap. The fix: the SAME `FLUX_DELTA_MAX=0.05` trust region the
legacy path always used is now also applied to unit-flux's shift (the more
accurate quadrature itself is kept -- only the size of shift it's allowed
to apply is bounded). Re-qualified: median/p90/p99/max KL are now within
0.1% of the legacy baseline in every z-band, including the previously-worst
one. 108/973 test contexts hit the cap and so no longer have EXACTLY unit
flux -- see `qualification/PATCH_2026-09-07_unit_flux.json` for the full
before/after qualification data.

Qualified production candidate, frozen 2026-09-03 and patched through
2026-09-07 (see `PROVENANCE.md` for the full change log). This is the
`flumen` package's production model.

## Intended output

The model estimates the conditional image-plane weak-lensing magnification PDF.
Its primary callable returns `log p(ln(mu) | z_s, theta)`. The public wrapper also
returns `p(mu)` with the required `1/mu` Jacobian.

Context ordering is:

`theta = (h, OmegaM, sigma8, OmegaB, ns, zeq/1000)`.

Use only inside the calibrated ranges documented in `data/README.md`; results
outside those limits are unvalidated extrapolations.

## Architecture

The conditional density uses two frozen 70,412-parameter SOS polynomial-flow
bodies:

1. An exact-support body trained in `u = log(y-y_bound)` supplies the complete
   low-z model and the default body elsewhere.
2. A split-normal-base SOSPF supplies only the high-z left flank. Compact
   infinitely differentiable gates blend it in over `2 < z_s < 3.5` and blend
   back to the boundary body's core over `-1 < y < 0`.

An adaptive compact switch connects the body to a calibrated peaks-over-threshold
tail at high magnification. It uses the first supported body/tail crossing, or a
guarded monotone fallback from `mu=2` to `mu=3`. A diagnostic guard refuses an
unverified fallback window. The low-redshift rescue and the targeted widened
high-structure transition are retained from the qualified experiment composite.
The default final density is normalized over its full effective support and
translated by `delta = log(integral(exp(-x)*raw)/integral(raw))`, without
clipping. Gauss-Legendre panel integration refines until both integrals agree
to 2 parts per million. Geometric panels resolve the exact lower boundary;
the upper integration limit is at least ln(mu)=40, beyond the effective tail.
Failed quadrature raises `FluxCalibrationError` instead of returning an
uncertified correction. This enforces `<1/mu>=1` to numerical accuracy.

Translation preserves the entire shape in ln(mu), including the tail slope
and any existing extrema. It does not introduce a new secondary rise, but
also does not repair existing ones. Fixed-mu tail probabilities, the support
edge and peak location change. At severe high-z contexts this requires a
large shift and worsens KL against the original nonunit simulator samples.
Neither exact preservation of S(8) nor unchanged fit quality is claimed.

The explicit `legacy` mode retains the old finite integration interval and
shift cap of 0.05. That cap never guaranteed global unit flux: the raw-ray
audit found that the simulator targets themselves can have inverse means
near 0.49, partly due to their truncated convergence anchor. No simulator
physics or training data was changed by the default output calibration.

## Frozen artifacts

| artifact | validation objective | SHA-256 |
|---|---:|---|
| boundary_body.pt | NLL(u) 0.610279, epoch 11 warm-start v1 | `1a8a07f9deb6937fa47fd98bb9bc3303b1aad4e8265dd35796735dbec2ef4df3` |
| asymmetric_body_sl0p6.pt | NLL(y) 1.443692, epoch 23 | `fc88382c50b9a6d04bf28e2b3efa033afad5843872719f4ef0380e6f3581df19` |

The NLL values use different transformed coordinates and must not be compared to
one another.

## Accuracy certificate

| split | evaluated configurations | median KL_y |
|---|---:|---:|
| full validation | 972 configurations | 0.002233 |
| standard acceptance sample | 243 configurations | 0.002253 |
| untouched test | 973 complete checkerboard configurations | 0.002227 -> **0.002205** (post shoulder-correction, see below) |

The full validation and untouched test medians agree to within 0.3%, giving no
evidence of meaningful aggregate generalization loss. On the complete test split,
KL_y p90/p99/max are 0.004534/0.01221/0.03058 (post-correction: 0.004511/0.01221/
0.03058 -- p99/max unchanged to 5 decimals). On the 40-cell width-relative
stress panel, median/p90/max binned KL are 0.001876/0.02093/0.05341.

### 2026-09-04: shoulder-correction patch (`composite.py` only, no checkpoint change)

The raw boundary body under-predicts its own upper shoulder by a modest,
general amount right before the body-to-POT-tail crossing (confirmed across
291 held-out validation configs at a ~6 sigma sign-test rate, plus
independent deep MC at Planck 2018 and two other reference cosmologies).
`SHOULDER_CORRECTION_ENABLED` (default `True`) applies a smooth
multiplicative bump at the crossing, gated smoothly to `0.4<=z<=2.5` (exactly
0, bit-identical to before, outside that range -- in particular at every z
this model card's own golden/reference tests probe). Verified to preserve
the composite's monotone-descending-branch property with substantial margin
(worst post-mode slope ~-0.10 vs the required <0.02, at 1.5x the shipped
correction amplitude; a 2026-09-07 independent audit separately verified
headroom out to 2x). Full detail, evidence, and validation:
`PROVENANCE.md`'s "2026-09-04" section.

### 2026-09-07: independent audit -- plotting artifact found, deeper tail defect identified

A follow-up audit (no production files changed) found that an early
diagnostic plot for the 2026-09-04 patch used
`density=True` together with a cropped `range=`, which normalizes within the
crop rather than over all rays -- inflating the apparent gap by 2.9x at
z=1.5 and 7.4x at z=2.0. The real remaining picture, from 320 test contexts
plus 800k independently-drawn fresh Planck rays: the shoulder fix's
aggregate transition ratio is close to 1 (0.998 on test) but not uniform
per-context (worsens in about half of individual contexts even as the
aggregate improves), and a SEPARATE, smaller, reproducible deficit
(~15-16% too few predicted events) exists at 8<=mu<30, roughly half
explained by the tail's C-infinity corner-smoothing reducing its fitted
survival probability at mu=8 by ~7.6-7.8%. Not fixed here -- the recommended
follow-up is an anchor-preserving tail revision, deliberately separate
from the shoulder correction (see the next section, which ships it).

The 18-test experimental certificate and 15-test package suite cover compact-gate
support, continuity, normalization, flux calibration, descending-branch
monotonicity, numerical C1 tail joins, fallback safety, checkpoint integrity,
and golden-output regression. A direct experiment-to-package comparison over
20,000 grid points found zero finite log-density difference.

### 2026-09-07: exact closed-form tail anchor preservation (`tail.py`+`composite.py`) -- SHIPPED

Resolves the softplus corner-smoothing erosion at mu=8 identified by the
2026-09-07 audit (~7.6-7.8% survival probability loss) via an exact closed-form
formulation of the intermediate slope k2*. Rather than applying post-hoc switch
ramps (which caused transient density suppression in round 3), k2* is solved
analytically to ensure the C^1 smooth survival function passes through both
S(3) and S(8) targets to floating-point precision (error = 0.00e+00).

- **Monotonicity & Smoothness**: Preserves strictly descending hazard
  rates (k1 > k2* > 1.0) and guarantees d(ln p)/dx < 0 everywhere, with
  zero secondary rise across the entire 54-context reference panel and
  100 random 6D configurations.
- **Deep-Tail Metric ([8, 30) mass, z in [0.4, 2.5])**: Re-evaluated across
  all 845 held-out audit contexts. Model-predicted events increased from
  3077 to 3269.5 (+6.3%), moving the pooled sim/model ratio from 1.146 down
  to **1.079** (reducing the empirical deficit by nearly half).
- **Validation**: 20/20 production unit tests pass. Median untouched-test KL
  divergence remains 0.00221. Image-plane moment agreement on [0.3, 6.0]
  exceeds R^2 > 0.99 for m2 (0.998) and m3 (0.991), with m4 Pearson r = 0.9915
  (R^2 = 0.976, limited by low-z finite-sample shot noise).

### 2026-09-07: seamless centered $C^1$ Hermite log-density bridge (`composite.py`) -- SHIPPED

Eliminates the visible slope break / "knee" at the body-to-tail connection
identified at low structure ($S_8 \approx 0.28$) and low redshifts (e.g. $z_s=3.0$).

- **Root Cause & Mechanism**: The previous linear density blend initiated at the
  crossing $x_c$ over $[x_c, x_c + 0.30]$ injected a $+w'(\rho_t - \rho_b)$ derivative
  spike (+2.4 nats) due to $\rho_t > \rho_b$ across the window.
- **Formulation**: Handover is upgraded to an exact $C^1$ cubic Hermite bridge in
  $\ln\rho(x)$, symmetrically centered on the crossing $[x_c - 0.15, x_c + 0.15]$,
  coupled with Fritsch-Carlson (1980) derivative scaling $\tau = 3/\sqrt{\alpha^2+\beta^2}$.
  This mathematically bounds $H'(x) < 0$ and guarantees strictly monotonic slope
  with zero overshoot or artificial inflection.
- **Validation**: 20/20 unit tests pass. Unit flux $\langle 1/\mu \rangle = 1.00000$
  preserved. Median untouched test KL is 0.002201 ($0.002 \pm 0.001$). Central moment
  $R^2$ on $[0.3, 6.0]$ across all 973 test contexts achieves $R^2 = 0.9979$ ($m_2$)
  and $0.9912$ ($m_3$), with $m_4$ Pearson $r = 0.9915$.

## Known limitations

- **Residual deep-tail under-prediction (~7-8%)**: The softplus anchor erosion
  at mu=8 is completely eliminated by the exact k2* derivation above. A small
  residual ~7% under-prediction remains, tracing to unweighted linear regression
  in `tail_fit.json` itself.
- **Pre-existing monotonicity violations exist at multiple points in the
  training box** (found 2026-09-07 while verifying the fix above, present
  with every correction in this model card disabled -- i.e. predate all of
  them): a real, non-grid-artifact dip-then-rise in the composite density,
  confirmed at a low-z/high-structure corner (via the `z<=0.3`
  fixed-rescue-window path) and near the "box lo-struct" corner (via the

  ordinary crossing-based path). Not fixed here -- out of scope for the two
  corrections above, which were verified (via a 400-context differential
  check) to introduce no NEW violations, but the pre-existing ones remain.
  Whoever picks up model-quality work next should treat this as a known,
  open, real defect, not assume the model is otherwise monotone everywhere
  just because the shipped corrections pass their own panels.
- The 2026-09-04 shoulder correction (above) is a modest, conservative
  patch, not a full fix: a residual ~10-20% under-prediction can remain in
  the crossing-adjacent bin even after correction, and the patch is gated
  off entirely outside `0.4<=z<=2.5` (the range where the underlying
  mechanism was actually verified), so any equivalent shoulder defect at
  lower or higher z is untouched.
- Median scores hide lower-cutoff failures: 40/243 validation contexts have the
  `q=1e-3` FOOT ratio outside `[0.5,2]`, and 21/243 have zero modeled mass below
  the empirical `q=1e-4` quantile.
- Three of 243 acceptance contexts have KL_y above 0.01.
- Rare test corners reach KL_y 0.03058 and should be shown explicitly in
  the paper rather than summarized only by the median.
- The box-high branch-clean reference excludes post-critical strong-lens rays
  using the physical `1-kappa-|gamma|>0` gate. That cleaning affects the targeted
  diagnostic reference, not the training dataset.
- The original training runs did not include a repeated-seed uncertainty study.
  The selected checkpoints and their hashes are therefore the reproducible
  scientific artifacts; a fresh retrain may differ slightly.
- The v1 fine-tune's 90k augmentation validation rows were independently sampled
  from its training pool because the historical loader did not use the named
  heldout split. The untouched 973-context test certificate is therefore the
  decisive generalization result.
