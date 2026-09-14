"""Composite density evaluator combining single-body flow, boundary cutoff, and C1 Hermite tail bridge."""

from pathlib import Path
import numpy as np
import torch

from .architecture import build_flow
from .flux import calibration_quadrature, compute_normalization_and_flux
from .scale import ContextPredictor

CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "single_body_gauss.pt"


class SingleBodyComposite:
    """Single-body normalizing flow with physical boundary cutoff and smooth C1 mu^-2 tail bridge."""

    def __init__(self, device="cpu", *, checkpoint_path=CHECKPOINT_PATH, flux_mode="unit"):
        self.device = torch.device(device)
        self.flux_mode = flux_mode
        self.predictor = ContextPredictor()
        self._calibration_cache = {}

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.cfg = ckpt.get("cfg", {})
        self.delta = float(ckpt.get("delta", 0.05))
        self.ctx_mean = np.asarray(ckpt["ctx_mean"], dtype=np.float32)
        self.ctx_std = np.asarray(ckpt["ctx_std"], dtype=np.float32)

        # Build and load flow
        self.flow = build_flow(self.cfg, ctx_dim=7).to(self.device)
        self.flow.load_state_dict(ckpt["state"])
        self.flow.eval()

        # Bridge parameters
        self.y_c_rel = 10.0       # handover anchor in normalized widths above median
        self.h_bridge_min = 2.0   # minimum bridge width in y-space
        self.h_factor = 1.0       # bridge width factor in ln(mu)-space: Delta ln(mu) = h_factor
        self.mu_floor = None      # physical magnification floor (None for scale-invariant y_c_rel anchor)
        self.eps_b = 0.02         # boundary safety buffer absorbing finite realization / regression residuals

        # Asymptotic lower cutoff parameters:
        # Gaussian base has natural super-exponential descent, so no artificial taper is needed (u_cut = None).
        # Heavy-tailed bases (e.g. Student-t) activate the quadratic taper to enforce empty-beam extinction.
        if self.cfg.get("base_df") is not None:
            self.u_cut = -0.5
            self.beta_cut = 2.0
        else:
            self.u_cut = None
            self.beta_cut = None

    def _prepare_context(self, z_s, theta):
        """Assemble 7D context vector: (z, h, Om, sigma8, Ob, ns, zeq/1000)."""
        h, om, s8, ob, ns, zeq = theta
        zeq_k = zeq / 1000.0 if zeq > 100.0 else zeq
        ctx_raw = np.array([z_s, h, om, s8, ob, ns, zeq_k], dtype=np.float32)
        ctx_norm = (ctx_raw - self.ctx_mean) / self.ctx_std
        return ctx_norm

    @torch.no_grad()
    def _raw_log_prob_lnmu(self, lnmu_arr, z_s, theta, m, s, y_b):
        """Unnormalized log density with respect to d ln(mu) with C1 Hermite bridge."""
        lnmu_flat = np.atleast_1d(np.asarray(lnmu_arr, dtype=np.float64))
        ctx_norm = self._prepare_context(z_s, theta)
        ctx_tensor = torch.tensor(ctx_norm, dtype=torch.float32, device=self.device).unsqueeze(0)

        y_arr = (lnmu_flat - m) / s
        log_prob = np.full_like(y_arr, -np.inf)

        # Width-adaptive bridge handover:
        # y0 is anchored at y_c_rel widths above the median (or past empty beam).
        # If mu_floor is specified, it is safely capped by the normalizing flow's training domain (u <= 2.6).
        y_cap = (y_b - self.delta) + np.exp(2.6)
        if self.mu_floor is not None:
            y_floor = (np.log(self.mu_floor) - m) / s
            y0 = min(y_cap, max(float(self.y_c_rel), float(y_floor), y_b - self.delta + 1.0))
        else:
            y0 = min(y_cap, max(float(self.y_c_rel), y_b - self.delta + 1.0))
        h = max(self.h_bridge_min, self.h_factor / s)
        y1 = y0 + h

        # Helper to evaluate flow log p_Y at test points
        def eval_flow_log_py(y_pts):
            t = y_pts - y_b + self.delta
            valid = t > 0
            lp_y = np.full_like(y_pts, -np.inf)
            if np.any(valid):
                u_vals = np.log(t[valid])
                u_t = torch.tensor(u_vals, dtype=torch.float32, device=self.device).unsqueeze(-1)
                c_exp = ctx_tensor.expand(len(u_vals), -1)
                lp_u = self.flow(c_exp).log_prob(u_t).cpu().numpy().flatten()
                lp_y[valid] = lp_u - u_vals
            return lp_y

        # Evaluate flow at bridge start y0 and compute numerical slope d0
        eps = 1e-4
        f0 = float(eval_flow_log_py(np.array([y0]))[0])
        lp_plus = float(eval_flow_log_py(np.array([y0 + eps]))[0])
        lp_minus = float(eval_flow_log_py(np.array([y0 - eps]))[0])
        d0 = (lp_plus - lp_minus) / (2.0 * eps)

        # Tail asymptotic slope in y is d1 = -s (giving d ln p / d ln mu = -1 => p(mu) ~ mu^-2)
        d1 = -s

        # Evaluate flow at cutoff anchor u_cut and compute numerical slope d_cut (only if taper active)
        if self.u_cut is not None:
            u_pts = torch.tensor([[self.u_cut], [self.u_cut + eps], [self.u_cut - eps]], dtype=torch.float32, device=self.device)
            c_exp3 = ctx_tensor.expand(3, -1)
            lp_u_pts = self.flow(c_exp3).log_prob(u_pts).cpu().numpy().flatten()
            f_cut = float(lp_u_pts[0] - self.u_cut)
            # Numerical slope in u, floored to 1.0 to guarantee smooth C1 power-law contact at cutoff
            d_cut = max(float((lp_u_pts[1] - (self.u_cut + eps) - (lp_u_pts[2] - (self.u_cut - eps))) / (2.0 * eps)), 1.0)

        # Hermite bridge matching f0, d0, d1 and curvature=0 at y1
        a0 = f0
        a1 = d0
        a3 = (d0 - d1) / (3.0 * h**2)
        a2 = - (d0 - d1) / h
        f1 = a0 + a1 * h + a2 * h**2 + a3 * h**3

        # 1. Body region: y <= y0
        mask_body = (y_arr <= y0)
        t_body = y_arr[mask_body] - y_b + self.delta
        valid_body = t_body > 0
        if np.any(valid_body):
            idx_valid = np.where(mask_body)[0][valid_body]
            t_valid = t_body[valid_body]
            u_valid = np.log(t_valid)

            if self.u_cut is not None:
                # Region 0: Asymptotic cutoff taper (u < u_cut)
                mask_taper = u_valid < self.u_cut
                if np.any(mask_taper):
                    du = u_valid[mask_taper] - self.u_cut
                    lp_y_taper = f_cut + d_cut * du - 0.5 * self.beta_cut * du**2
                    log_prob[idx_valid[mask_taper]] = lp_y_taper - np.log(s)

                # Region 1: Normalizing flow body (u >= u_cut)
                mask_flow = ~mask_taper
                if np.any(mask_flow):
                    u_flow = u_valid[mask_flow]
                    u_tensor = torch.tensor(u_flow, dtype=torch.float32, device=self.device).unsqueeze(-1)
                    c_exp = ctx_tensor.expand(len(u_flow), -1)
                    lp_u = self.flow(c_exp).log_prob(u_tensor).cpu().numpy().flatten()
                    log_prob[idx_valid[mask_flow]] = (lp_u - u_flow) - np.log(s)
            else:
                # Pure normalizing flow evaluation across the entire body (no piecewise taper needed)
                u_tensor = torch.tensor(u_valid, dtype=torch.float32, device=self.device).unsqueeze(-1)
                c_exp = ctx_tensor.expand(len(u_valid), -1)
                lp_u = self.flow(c_exp).log_prob(u_tensor).cpu().numpy().flatten()
                log_prob[idx_valid] = (lp_u - u_valid) - np.log(s)

        # 2. C1 Hermite Bridge region: y0 < y <= y1
        mask_bridge = (y_arr > y0) & (y_arr <= y1)
        if np.any(mask_bridge):
            dy = y_arr[mask_bridge] - y0
            lp_y_bridge = a0 + a1 * dy + a2 * dy**2 + a3 * dy**3
            log_prob[mask_bridge] = lp_y_bridge - np.log(s)

        # 3. Asymptotic tail region: y > y1
        mask_tail = (y_arr > y1)
        if np.any(mask_tail):
            dy_tail = y_arr[mask_tail] - y1
            lp_y_tail = f1 - s * dy_tail
            log_prob[mask_tail] = lp_y_tail - np.log(s)

        return log_prob

    def _calibrate_flux(self, z_s, theta, m, s, y_b):
        """Compute normalization mass and boundary-preserving shift enforcing unit flux."""
        cache_key = (float(z_s), tuple(float(x) for x in theta), self.flux_mode)
        if cache_key in self._calibration_cache:
            return self._calibration_cache[cache_key]

        y_cap = (y_b - self.delta) + np.exp(2.6)
        if self.mu_floor is not None:
            y_floor = (np.log(self.mu_floor) - m) / s
            y0 = min(y_cap, max(float(self.y_c_rel), float(y_floor), y_b - self.delta + 1.0))
        else:
            y0 = min(y_cap, max(float(self.y_c_rel), y_b - self.delta + 1.0))
        h = max(self.h_bridge_min, self.h_factor / s)
        y_c = y0 + h

        x, w = calibration_quadrature(m, s, y_b, delta=self.delta, y_c=y_c, order=16)
        p_raw = np.exp(self._raw_log_prob_lnmu(x, z_s, theta, m, s, y_b))
        valid = np.isfinite(p_raw) & (p_raw > 0)
        x_val = x[valid]
        w_val = w[valid]
        p_val = p_raw[valid]

        mass = float(w_val @ p_val)

        if self.flux_mode != "unit":
            res = (mass, 0.0, 1.0)
            self._calibration_cache[cache_key] = res
            return res

        lnmu_edge = m + s * (y_b - self.delta)
        t_val = (x_val - lnmu_edge) / s
        t0 = 2.0
        phi = t_val / (t_val + t0)
        dphi = (1.0 / s) * (t0 / (t_val + t0)**2)

        inv_0 = float(w_val @ (p_val * np.exp(-x_val)))
        shift = float(np.log(max(inv_0 / mass, 1e-12)))

        mass_cur = mass
        for _ in range(8):
            x_sh = x_val - shift * phi
            jac = 1.0 - shift * dphi
            lp_sh = self._raw_log_prob_lnmu(x_sh, z_s, theta, m, s, y_b)
            p_sh = np.exp(lp_sh) * jac
            mass_cur = float(w_val @ p_sh)
            inv_cur = float(w_val @ (p_sh * np.exp(-x_val)))
            flux_cur = inv_cur / mass_cur
            err = flux_cur - 1.0
            if abs(err) < 1e-7:
                break
            shift = shift + err
            shift = float(np.clip(shift, -0.2, 0.2))

        if len(self._calibration_cache) > 256:
            self._calibration_cache.pop(next(iter(self._calibration_cache)))
        res = (mass_cur, shift, t0)
        self._calibration_cache[cache_key] = res
        return res

    def log_prob_lnmu(self, lnmu, z_s, theta):
        """Log probability density with respect to d ln(mu)."""
        lnmu_arr = np.asarray(lnmu, dtype=np.float64)
        is_scalar = lnmu_arr.ndim == 0
        lnmu_flat = np.atleast_1d(lnmu_arr)

        m, s, y_b_raw, _ = self.predictor.predict(z_s, theta)
        y_b = y_b_raw - self.eps_b
        mass, shift, t0 = self._calibrate_flux(z_s, theta, m, s, y_b)

        lnmu_edge = m + s * (y_b - self.delta)
        valid = lnmu_flat > lnmu_edge
        norm_lp = np.full_like(lnmu_flat, -np.inf)

        if np.any(valid):
            if self.flux_mode == "unit":
                t_val = (lnmu_flat[valid] - lnmu_edge) / s
                phi = t_val / (t_val + t0)
                dphi = (1.0 / s) * (t0 / (t_val + t0)**2)
                lnmu_sh = lnmu_flat[valid] - shift * phi
                jac = np.maximum(1.0 - shift * dphi, 1e-12)
                raw_lp = self._raw_log_prob_lnmu(lnmu_sh, z_s, theta, m, s, y_b)
                norm_lp[valid] = raw_lp + np.log(jac) - np.log(max(mass, 1e-12))
            else:
                raw_lp = self._raw_log_prob_lnmu(lnmu_flat[valid], z_s, theta, m, s, y_b)
                norm_lp[valid] = raw_lp - np.log(max(mass, 1e-12))

        if is_scalar:
            return float(norm_lp[0])
        return norm_lp.reshape(lnmu_arr.shape)

    def pdf_lnmu(self, lnmu, z_s, theta):
        """Probability density p(ln mu) with respect to d ln(mu)."""
        return np.exp(self.log_prob_lnmu(lnmu, z_s, theta))

    def pdf_mu(self, mu, z_s, theta):
        """Probability density p(mu) with respect to d mu."""
        mu_arr = np.asarray(mu, dtype=np.float64)
        is_scalar = mu_arr.ndim == 0
        mu_flat = np.atleast_1d(mu_arr)

        valid = mu_flat > 0
        pdf_vals = np.zeros_like(mu_flat)

        if np.any(valid):
            lnmu = np.log(mu_flat[valid])
            lp_lnmu = self.log_prob_lnmu(lnmu, z_s, theta)
            # p_mu = p_lnmu / mu
            pdf_vals[valid] = np.exp(lp_lnmu) / mu_flat[valid]

        if is_scalar:
            return float(pdf_vals[0])
        return pdf_vals.reshape(mu_arr.shape)
