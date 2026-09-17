"""Fit degree-3 polynomial Ridge regressions for scale s, location m, anchor kbar, and bound y_b."""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR_DEFAULT = ROOT / "data" / "u_space"
OUT_PATH_DEFAULT = ROOT / "training" / "context_fits_retrained.json"


def raw_inputs_from_context(ctx):
    """Transform context columns to [ln z, ln h, ln Om, ln sigma8, Ob, ns, ln zeq_k]."""
    z = ctx[:, 0]
    h, om, s8, ob, ns, zeq = ctx[:, 1], ctx[:, 2], ctx[:, 3], ctx[:, 4], ctx[:, 5], ctx[:, 6]
    zeq_val = np.where(zeq > 100.0, zeq / 1000.0, zeq)
    return np.column_stack([np.log(z), np.log(h), np.log(om), np.log(s8), ob, ns, np.log(zeq_val)])


def fit_context_predictors(data_dir=DATA_DIR_DEFAULT, out_path=OUT_PATH_DEFAULT):
    data_dir = Path(data_dir)
    out_path = Path(out_path)

    print(f"Loading data from {data_dir}...")
    d_tr = np.load(data_dir / "train.npz")
    d_va = np.load(data_dir / "validation.npz")
    d_te = np.load(data_dir / "test.npz")

    ctx_tr, meta_tr = d_tr["ctx"], d_tr["meta"]
    ctx_va, meta_va = d_va["ctx"], d_va["meta"]
    ctx_te, meta_te = d_te["ctx"], d_te["meta"]

    # Raw 7D transformed inputs
    F_tr = raw_inputs_from_context(ctx_tr)
    F_va = raw_inputs_from_context(ctx_va)
    F_te = raw_inputs_from_context(ctx_te)

    # Degree-3 polynomial expansion (120 terms)
    poly = PolynomialFeatures(degree=3, include_bias=True)
    X_tr = poly.fit_transform(F_tr)
    X_va = poly.transform(F_va)
    X_te = poly.transform(F_te)
    powers = poly.powers_.tolist()

    # Standardize non-bias columns based on training set
    feat_mean = X_tr[:, 1:].mean(axis=0)
    feat_std = X_tr[:, 1:].std(axis=0)
    feat_std[feat_std == 0.0] = 1.0

    X_norm_tr = (X_tr[:, 1:] - feat_mean) / feat_std
    X_norm_va = (X_va[:, 1:] - feat_mean) / feat_std
    X_norm_te = (X_te[:, 1:] - feat_mean) / feat_std

    # Targets
    m_tr, s_tr, yb_tr, kb_tr = meta_tr[:, 0], meta_tr[:, 1], meta_tr[:, 2], meta_tr[:, 3]
    m_te, s_te, yb_te, kb_te = meta_te[:, 0], meta_te[:, 1], meta_te[:, 2], meta_te[:, 3]

    alphas = np.logspace(-4, 3, 25)
    print("Fitting Ridge models with cross-validation...")

    def fit_and_combine(X_train, y_train):
        ridge = RidgeCV(alphas=alphas, fit_intercept=True).fit(X_train, y_train)
        w = np.concatenate([[float(ridge.intercept_)], ridge.coef_])
        return ridge, w

    # 1. Scale s
    ridge_s, w_s = fit_and_combine(X_norm_tr, np.log(s_tr))
    pred_s_te = np.exp(ridge_s.predict(X_norm_te))
    rel_err_s = np.abs(pred_s_te - s_te) / s_te
    print(f"Scale s: median rel err = {np.median(rel_err_s)*100:.2f}%, 95% = {np.percentile(rel_err_s, 95)*100:.2f}%, max = {np.max(rel_err_s)*100:.2f}% (alpha={ridge_s.alpha_:.2e})")

    # 2. Location m (normalized by s)
    ridge_m, w_m = fit_and_combine(X_norm_tr, m_tr / s_tr)
    pred_m_te = ridge_m.predict(X_norm_te) * pred_s_te
    abs_err_m = np.abs(pred_m_te - m_te)
    print(f"Location m: median abs err = {np.median(abs_err_m):.5f}, 95% = {np.percentile(abs_err_m, 95):.5f} (alpha={ridge_m.alpha_:.2e})")

    # 3. Kappa anchor kbar
    ridge_kb, w_kb = fit_and_combine(X_norm_tr, np.log(kb_tr))
    pred_kb_te = np.exp(ridge_kb.predict(X_norm_te))
    rel_err_kb = np.abs(pred_kb_te - kb_te) / kb_te
    print(f"Anchor kbar: median rel err = {np.median(rel_err_kb)*100:.2f}%, 95% = {np.percentile(rel_err_kb, 95)*100:.2f}% (alpha={ridge_kb.alpha_:.2e})")

    # 4. Direct y_b
    ridge_yb, w_yb = fit_and_combine(X_norm_tr, yb_tr)
    pred_yb_dir_te = ridge_yb.predict(X_norm_te)
    pred_yb_der_te = (-2.0 * np.log1p(pred_kb_te) - pred_m_te) / pred_s_te
    print(f"y_b (direct):  median abs err = {np.median(np.abs(pred_yb_dir_te - yb_te)):.4f}")
    print(f"y_b (derived): median abs err = {np.median(np.abs(pred_yb_der_te - yb_te)):.4f}")

    payload = {
        "powers": powers,
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "coef_log_s": w_s.tolist(),
        "coef_m_over_s": w_m.tolist(),
        "coef_log_kb": w_kb.tolist(),
        "coef_yb_direct": w_yb.tolist(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[done] Saved calibrated context predictors to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fit context predictors for flumen v3.")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR_DEFAULT))
    parser.add_argument("--out", type=str, default=str(OUT_PATH_DEFAULT))
    args = parser.parse_args()
    fit_context_predictors(args.data_dir, args.out)


if __name__ == "__main__":
    main()
