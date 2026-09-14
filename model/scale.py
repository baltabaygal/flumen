"""Predictor for location m, scale s, and empty-beam bound y_b from cosmological context."""

import json
from pathlib import Path
import numpy as np

CALIB_PATH = Path(__file__).resolve().parent / "calibration" / "context_fits.json"


class ContextPredictor:
    """Predicts m(c), s(c), kbar(c), and y_b(c) given context (z_s, theta).
    
    Uses a calibrated degree-3 polynomial Ridge regression model evaluated in pure NumPy.
    """

    def __init__(self, calib_path=CALIB_PATH):
        with open(calib_path, "r") as f:
            calib = json.load(f)
        self.powers = np.asarray(calib["powers"], dtype=np.int32)
        self.feat_mean = np.asarray(calib["feat_mean"], dtype=np.float64)
        self.feat_std = np.asarray(calib["feat_std"], dtype=np.float64)
        self.coef_log_s = np.asarray(calib["coef_log_s"], dtype=np.float64)
        self.coef_m_over_s = np.asarray(calib["coef_m_over_s"], dtype=np.float64)
        self.coef_log_kb = np.asarray(calib["coef_log_kb"], dtype=np.float64)
        self.coef_yb_direct = np.asarray(calib["coef_yb_direct"], dtype=np.float64)

    def _compute_features(self, z_arr, th_arr):
        """Compute normalized degree-3 polynomial features for context array."""
        z_arr, h, om, s8, ob, ns, zeq = np.broadcast_arrays(z_arr, *th_arr)
        zeq_k = np.where(zeq > 100.0, zeq / 1000.0, zeq)
        # Raw 7D features: [ln z, ln h, ln Om, ln sigma8, Ob, ns, ln zeq_k]
        F = np.column_stack([
            np.log(z_arr),
            np.log(h),
            np.log(om),
            np.log(s8),
            ob,
            ns,
            np.log(zeq_k)
        ])
        # Degree-3 monomials: prod_k (F[:, k] ** powers[j, k])
        X = np.prod(F[:, None, :] ** self.powers[None, :, :], axis=-1)
        Xs = np.column_stack([np.ones((len(X), 1), dtype=np.float64), (X[:, 1:] - self.feat_mean) / self.feat_std])
        return Xs

    def predict(self, z_s, theta):
        """Predict m, s, y_b, kbar for scalar or array z_s and theta (6-tuple)."""
        z_s = np.asarray(z_s, dtype=np.float64)
        is_scalar = z_s.ndim == 0
        z_arr = np.atleast_1d(z_s)
        th_arr = [np.atleast_1d(np.asarray(t, dtype=np.float64)) for t in theta]

        Xs = self._compute_features(z_arr, th_arr)

        pred_s = np.exp(Xs @ self.coef_log_s)
        pred_m = (Xs @ self.coef_m_over_s) * pred_s
        pred_kb = np.exp(Xs @ self.coef_log_kb)
        pred_yb = (-2.0 * np.log1p(pred_kb) - pred_m) / pred_s

        if is_scalar:
            return float(pred_m[0]), float(pred_s[0]), float(pred_yb[0]), float(pred_kb[0])
        return pred_m, pred_s, pred_yb, pred_kb
