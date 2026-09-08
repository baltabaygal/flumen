"""Normalized hybrid body plus calibrated physical high-magnification tail."""

from functools import lru_cache
import math

import numpy as np

from .hybrid_body import compact_switch
from .flux import unit_flux_context
from .tail import pot_tail_logp, pot_tail_logp_corrected

# The analytical Peaks-Over-Threshold (POT) tail uses an exact anchor-preserving
# intermediate slope k2* (tail.py) so that the C^1 smoothed survival law passes
# through both S(3) and S(8) targets exactly by construction, eliminating
# corner-smoothing erosion without requiring post-hoc switch ramps.
TAIL_ANCHOR_CORRECTION_ENABLED = True
TAIL_X3 = float(np.log(3.0))



GRID_LO, GRID_HI, GRID_N = -4.0, float(np.log(400.0)), 3000
POT_VALID_MU, LOW_Z_TAIL_OFF = 1.7, 0.3
TRANSITION_WIDTH = 0.30
MATCH_LOGAMP_MAX, MATCH_SLOPE_MAX = float(np.log(2.0)), 1.0
MATCH_FALLBACK_ENDS = (3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 40.0)
TAIL_END_EXTRA, STRUCT_LO, STRUCT_HI = 2.0, 1.4, 1.7
FLUX_DELTA_MAX = 0.05
# ⚠ 2026-09-07: also applied to flux_mode="unit" (below), not just "legacy".
# The `unit` mode's own qualification run (973 test contexts,
# `experiments/unit_flux_implementation_2026-09-07/qualification_run.log`)
# found the exact, uncapped shift is essentially free for 89% of the box
# (865/973 contexts land at |delta|<=0.05 with kl_unit==kl_legacy to 4
# decimals) but degrades sharply and monotonically past that same 0.05
# threshold: correlation(|delta|, kl_unit) = 0.974 across the full test
# set, median KL already 2.5x worse in the very next bin (0.05,0.1], and
# by z>=8 (where |delta| is largest -- but note z alone only correlates
# 0.47, |delta| is the real driver, e.g. one z=6.79 context with a large
# delta shows the same blowup) unit-flux's median KL is ~5x legacy's, with
# p99/max KL 20x/13x worse in aggregate. Every one of the worst-affected
# contexts has `legacy_delta` exactly at its own -0.05 floor, i.e. the
# historical cap was already binding there -- this is direct evidence the
# cap is a genuine, load-bearing trust region (matching the `-ar` worktree's
# own independent finding, "legit deltas are <=2e-2, 0.05 never binds on a
# healthy model"), not legacy conservatism to relax. The new Gauss-Legendre
# quadrature in `flux.py` is kept -- it is a real, costless accuracy
# improvement over the old finite-grid trapezoid integration -- only the
# SIZE of the correction it's allowed to apply is capped, identically to
# the legacy path. `flux.unit_flux_context` itself is left uncapped and
# undocumented-as-capped deliberately, so it stays usable as an honest,
# general-purpose "what shift would exact unit flux require here" utility
# (e.g. for diagnostics) independent of this policy choice.
FALLBACK_TAIL_TOL_NATS = 1.5
FALLBACK_CHECK_MU = (20.0, 50.0, 100.0)

# --- Shoulder correction (2026-09-04) -------------------------------------
# Diagnosis: right before the body->POT crossing (where `plan()` still
# reports weight=0, i.e. the composite is EXACTLY the raw body), the raw
# body under-predicts the true density by a modest, general amount.
# Confirmed on THREE independent evidence sources, not just one cosmology:
#   (1) 291 held-out validation configs, 0.5<=z<2.0 (the pure-v1 regime --
#       hybrid_body's own z-gate is exactly 0 there, so the asymmetric body
#       contributes nothing and this is 100% a boundary-body property):
#       197/291 (68%) read data > model in the transition-width bin right
#       before the crossing (sign test vs 50% null: ~6 sigma).
#   (2) Deep (200k-ray) fresh MC at Planck (h=.674,Om=.315,s8=.811) and two
#       other named reference cosmologies (high/low structure) x z=1.0/1.5:
#       7/8 crossing-adjacent bins read data/model > 1, up to 1.95x.
#   (3) Visual: the grey MC histogram sits visibly above BOTH the raw body
#       AND the raw POT-tail curves in this neighbourhood (neither existing
#       component individually reaches the data -- no re-weighting of the
#       two can fix it, since a weighted average of two curves both below
#       the target can never exceed either one).
# Fix: a smooth multiplicative bump, centered at the crossing and scaled to
# the local transition width, applied to the UNNORMALIZED density before
# renormalization/flux calibration (so it composes correctly with both).
# ⚠ A first design (reversed-smoothstep, monotonically non-increasing in x)
# was tried to structurally guarantee no secondary rise in the descending
# branch -- it made the fit WORSE everywhere, because its "fully boosted"
# plateau necessarily bled backward into the region just before the deficit
# (which was already reading a MILD OVER-prediction), boosting an
# already-too-high region. Reverted in favour of the direct Gaussian bump,
# which was then EMPIRICALLY verified (not just assumed) against
# `test_panel_descending_branches_have_no_secondary_rise`'s own exact
# methodology across a 6-cosmology x 4-z panel (incl. Planck and all four
# `test_model.py` PANEL_CONTEXTS): worst-case post-mode slope stayed
# strongly negative (~-0.10) up to amplitude 0.30, i.e. the base composite's
# own decay near the crossing is steep enough to absorb a bump of this size
# without ever creating a local rise. SHOULDER_A=0.20 is chosen with
# headroom inside that empirically-verified margin.
# Amplitude/width/center were grid-scanned against the SAME 291-config
# pool (not tuned on Planck alone) via a multi-bin weighted residual
# objective; the optimum is a shallow plateau over A in [0.15,0.20],
# width in [0.4,0.6] transition-widths, center within +-0.15 of the
# crossing -- the exact point within that plateau is not critical.
# Gated smoothly in z via the SAME compact_switch used everywhere else in
# this module: ramps up 0.4->0.6, stays at full strength through the whole
# verified pure-v1 range (see the SHOULDER_Z_LO2/HI2 comment just below for
# exactly where it ramps back down and why). The gate is EXACTLY 0 outside
# [SHOULDER_Z_LO, SHOULDER_Z_HI2] -- in particular at z=0.2/3.5/5/10, so all
# four existing `test_golden_log_density` cases are bit-identical to before
# this change and needed no golden-value update.
SHOULDER_CORRECTION_ENABLED = True
SHOULDER_A = 0.20
SHOULDER_REL_WIDTH = 0.5
SHOULDER_REL_CENTER = 0.0
SHOULDER_Z_LO, SHOULDER_Z_HI = 0.4, 0.6
# ⚠ 2026-09-04, corrected after an initial ship-through check: hybrid_body's
# OWN v1/v2 blend weight (Z_BLEND_LO=2.0 there) is ALSO exactly 0 at z=2.0,
# only growing for z>2.0 -- so the pure-v1 regime this fix is diagnosed
# against extends fully THROUGH z=2.0, not just up to it. A first version of
# this gate ramped down to exactly 0 by z=2.0 (matching that boundary
# literally), which left z=2.0 itself with ZERO correction despite the
# defect being present there too. Ramp instead starts AT z=2.0 (full
# strength through the whole verified pure-v1 range) and relaxes to 0 by
# z=2.5, safely before hybrid_body's v2 weight becomes more than a minor
# contribution (compact_switch(log 2.5, log 2, log 3.5) ~ 0.09).
SHOULDER_Z_LO2, SHOULDER_Z_HI2 = 2.0, 2.5


class UnverifiedFallbackError(RuntimeError):
    """No verified body-only or fixed-window evaluation is available."""


def _grid(body, z, theta):
    wide = np.linspace(GRID_LO, GRID_HI, GRID_N)
    m, s, _, _ = body.controls(z, theta)
    fine = np.linspace(max(GRID_LO, m-12*s), min(GRID_HI, m+16*s), GRID_N)
    return np.unique(np.concatenate((wide, fine)))


def _crossing(body, z, theta):
    x = np.linspace(np.log(POT_VALID_MU), GRID_HI, 5000)
    difference = np.asarray(body(x, z, theta))-np.asarray(pot_tail_logp(x, z, theta))
    finite = np.isfinite(difference)
    ids = np.where(finite[:-1] & finite[1:] & (difference[:-1] >= 0)
                   & (difference[1:] < 0))[0]
    if not len(ids):
        return None
    i = int(ids[0])
    return float(x[i]-difference[i]*(x[i+1]-x[i])/(difference[i+1]-difference[i]))


def _fallback_diagnostics(body, z, theta):
    """Check whether a body-only fallback agrees with the calibrated POT tail."""
    worst_gap = 0.0
    detail = None
    for mu_ref in FALLBACK_CHECK_MU:
        x = np.array([math.log(mu_ref)])
        body_logp = float(np.asarray(body(x, z, theta), float)[0])
        tail_logp = float(np.asarray(pot_tail_logp(x, z, theta), float)[0])
        if not (np.isfinite(body_logp) and np.isfinite(tail_logp)):
            return False, float("inf"), (
                f"non-finite at mu={mu_ref} "
                f"(body_logp={body_logp}, pot_logp={tail_logp})"
            )
        gap = abs(body_logp-tail_logp)
        if gap > worst_gap:
            worst_gap = gap
            detail = (
                f"{gap:.3f} nats at mu={mu_ref} "
                f"(body={body_logp:.3f}, pot={tail_logp:.3f})"
            )
    return worst_gap <= FALLBACK_TAIL_TOL_NATS, worst_gap, detail


def _fixed_rescue_window(body, z, theta, unsafe_detail):
    """Return the physics-calibrated mu=2->3 window if both joins are finite."""
    start, end = math.log(2.0), math.log(3.0)
    for edge in (start, end):
        body_logp = float(np.asarray(body(np.array([edge]), z, theta), float)[0])
        tail_logp = float(np.asarray(pot_tail_logp(np.array([edge]), z, theta), float)[0])
        if not (np.isfinite(body_logp) and np.isfinite(tail_logp)):
            raise UnverifiedFallbackError(
                f"z={z}, theta={theta}: raw body is unsafe ({unsafe_detail}) and "
                "the fixed mu=2->3 rescue window is also non-finite"
            )
    return start, end


def _shoulder_gate_z(z):
    """Smooth (C-infinity), compact-support gate: 0 outside [0.4,2.5], 1 on
    the verified core [0.6,2.0], ramping in between. See the constants'
    comment above for why this range.
    (Docstring bounds corrected 2026-09-07 -- an independent audit caught
    that this text still stated the pre-correction [0.4,2.0]/[0.6,1.8]
    bounds after the z=2.0-gap fix moved the real constants to [0.4,2.5]/
    [0.6,2.0]; the constants and behavior were always correct, only this
    comment was stale. See experiments/shoulder_tail_audit_2026-09-07/.)"""
    up = float(compact_switch([z], SHOULDER_Z_LO, SHOULDER_Z_HI)[0])
    down = float(compact_switch([z], SHOULDER_Z_LO2, SHOULDER_Z_HI2)[0])
    return up*(1.0-down)


def _shoulder_bump(x, z, theta, transition):
    """Multiplicative correction for the body's shoulder under-prediction
    near the body->POT crossing. Returns 1.0 (exact no-op) when the z-gate
    is zero or there is no crossing-based transition (low-z rescue path).
    See the SHOULDER_* constants' comment for the evidence and the
    monotonicity verification."""
    if transition is None:
        return 1.0
    gz = _shoulder_gate_z(float(z))
    if gz <= 0.0:
        return 1.0
    start, end = transition
    tw = end-start
    center = 0.5*(start+end)
    width = max(SHOULDER_REL_WIDTH*tw, 1e-9)
    xx = np.asarray(x, dtype=np.float64)
    return 1.0+gz*SHOULDER_A*np.exp(-0.5*((xx-center)/width)**2)


def _target_weight(z, theta):
    structure = float(theta[2])*np.sqrt(float(theta[1])/0.3)
    ws = float(compact_switch([structure], STRUCT_LO, STRUCT_HI)[0])
    lz = math.log(float(z))
    wz_in = float(compact_switch([lz], math.log(2.8), math.log(3.5))[0])
    wz_out = 1.0-float(compact_switch([lz], math.log(5.0), math.log(8.0))[0])
    return ws*wz_in*wz_out


def make_composite(body, flux_calibration=True, *, flux_mode="unit"):
    """Build the composite with unit-flux or historical capped calibration.

    ``unit`` is an explicit positive-PDF calibration convention. It rescales
    magnifications and therefore changes fixed-mu survival probabilities.
    ``legacy`` reproduces the historical finite-grid, capped-shift behavior.
    """
    if flux_mode not in ("unit", "legacy"):
        raise ValueError("flux_mode must be 'unit' or 'legacy'")

    @lru_cache(maxsize=1024)
    def bridge_coeffs(z, theta, x0, x1):
        eps = 1e-5
        y0 = float(body(np.array([x0]), z, theta)[0])
        d0 = (float(body(np.array([x0+eps]), z, theta)[0]) - float(body(np.array([x0-eps]), z, theta)[0])) / (2*eps)
        y1 = float(pot_tail_logp(np.array([x1]), z, theta)[0])
        d1 = (float(pot_tail_logp(np.array([x1+eps]), z, theta)[0]) - float(pot_tail_logp(np.array([x1-eps]), z, theta)[0])) / (2*eps)
        h = x1 - x0
        Delta = (y1 - y0) / h
        if Delta < 0 and d0 < 0 and d1 < 0:
            alpha = d0 / Delta
            beta = d1 / Delta
            hypot = float(np.hypot(alpha, beta))
            if hypot > 3.0:
                tau = 3.0 / hypot
                d0 = tau * d0
                d1 = tau * d1
        c0 = y0
        c1 = h * d0
        c2 = 3.0 * (y1 - y0) - h * (2.0 * d0 + d1)
        c3 = 2.0 * (y0 - y1) + h * (d0 + d1)
        return c0, c1, c2, c3

    @lru_cache(maxsize=None)
    def plan(z, theta):
        theta = tuple(theta)
        if z <= LOW_Z_TAIL_OFF:
            safe, _, detail = _fallback_diagnostics(body, z, theta)
            return None if safe else _fixed_rescue_window(body, z, theta, detail)
        crossing = _crossing(body, z, theta)
        # Production plan_mode="crossing_or_pot_2_3": when no calibrated
        # crossing exists, use the fixed physics-calibrated window instead of
        # silently trusting an unconstrained body-only extrapolation.
        hw = TRANSITION_WIDTH / 2.0
        result = ((max(GRID_LO, crossing-hw), min(crossing+hw, GRID_HI))
                  if crossing is not None else (math.log(2.0), math.log(3.0)))
        start, end = result
        if abs(start-math.log(2.0)) < 1e-10 and abs(end-math.log(3.0)) < 1e-10:
            end = math.log(3.0+TAIL_END_EXTRA*_target_weight(z, theta))
        return start, end

    def unnormalized(x, z, theta):
        orig_shape = np.shape(x)
        x_flat = np.atleast_1d(np.asarray(x, dtype=np.float64)).ravel()
        transition = plan(float(z), tuple(float(v) for v in theta))
        if transition is None:
            lb = np.asarray(body(x_flat, z, theta), float)
            density = np.zeros_like(lb)
            finite = np.isfinite(lb)
            density[finite] = np.exp(np.clip(lb[finite], -745, 700))
            if SHOULDER_CORRECTION_ENABLED:
                density = density*_shoulder_bump(x_flat, z, theta, transition)
            res = density.reshape(orig_shape)
            return float(res) if np.ndim(x) == 0 else res
        start, end = transition
        out = np.zeros_like(x_flat)
        m_body = x_flat <= start
        m_tail = x_flat >= end
        m_mid = (~m_body) & (~m_tail)
        if np.any(m_body):
            lb = np.asarray(body(x_flat[m_body], z, theta), float)
            fin = np.isfinite(lb)
            out_b = np.zeros_like(lb)
            out_b[fin] = np.exp(np.clip(lb[fin], -745, 700))
            out[m_body] = out_b
        if np.any(m_tail):
            tail_lp = pot_tail_logp(x_flat[m_tail], z, theta)
            out[m_tail] = np.exp(np.clip(tail_lp, -745, 700))
        if np.any(m_mid):
            c0, c1, c2, c3 = bridge_coeffs(float(z), tuple(float(v) for v in theta), start, end)
            t = (x_flat[m_mid] - start) / (end - start)
            h_val = c0 + t * (c1 + t * (c2 + t * c3))
            out[m_mid] = np.exp(np.clip(h_val, -745, 700))
        if SHOULDER_CORRECTION_ENABLED:
            out = out*_shoulder_bump(x_flat, z, theta, transition)
        res = out.reshape(orig_shape)
        return float(res) if np.ndim(x) == 0 else res


    @lru_cache(maxsize=None)
    def context(z, theta):
        if flux_mode == "unit":
            norm, delta = unit_flux_context(unnormalized, body, z, theta)
            if flux_calibration:
                delta = float(np.clip(delta, -FLUX_DELTA_MAX, FLUX_DELTA_MAX))
            else:
                delta = 0.0
            return norm, delta
        grid = _grid(body, z, theta)
        density = unnormalized(grid, z, theta)
        norm = float(np.trapezoid(density, grid))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"invalid normalization {norm}")
        if not flux_calibration:
            return norm, 0.0
        modeled_flux = float(np.trapezoid(np.exp(-grid)*density, grid))/norm
        delta = float(np.clip(np.log(modeled_flux), -FLUX_DELTA_MAX, FLUX_DELTA_MAX))
        return norm, delta

    def log_prob(lnmu, z, theta):
        x = np.asarray(lnmu, dtype=np.float64).ravel()
        theta = tuple(float(v) for v in theta)
        norm, delta = context(float(z), theta)
        density = unnormalized(x-delta, float(z), theta)
        out = np.full(x.size, -np.inf)
        positive = density > 0
        out[positive] = np.log(density[positive]/norm)
        return out

    log_prob.plan = plan
    log_prob.context = context
    log_prob.unnormalized = unnormalized
    log_prob.plan_mode = "crossing_or_pot_2_3"
    log_prob.flux_mode = flux_mode
    return log_prob
