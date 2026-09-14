"""Comprehensive test set evaluation module for production_model_v3.

Computes:
1. ACE-Lensing benchmark-compatible KL divergence (cell-integrated, N >= 5)
2. Equal-mass Quantile Adaptive KL divergence with Miller-Madow debiasing
3. Total Variation Distance (TVS)
4. Bin-free Wasserstein-1 distance in physical magnification W_1(mu) and y-space W_1(y)
5. Bin-free Kolmogorov-Smirnov distance (D_KS = max |F_emp - F_mod|)
6. Bin-free Cramér-von Mises distance (CvM)
7. Continuous sample Negative Log-Likelihood (NLL)
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flumen.model import load_model

DATA_PATH = Path("/Users/baltabay/Desktop/gw-wl-emulator/single_body_naive/data/u_space/test.npz")
OUT_PATH = Path(__file__).resolve().parent / "test_summary.json"


def evaluate_test_set(n_max=200, device="cpu", flux_mode="unit", seed=42):
    model = load_model(device=device, flux_mode=flux_mode)

    d = np.load(DATA_PATH)
    u_te, idx_te, ctx_te, meta_te = d["u"], d["idx"], d["ctx"], d["meta"]

    order = np.argsort(idx_te)
    idx_s, u_s = idx_te[order], u_te[order]
    n_cfg = len(ctx_te)
    st = np.searchsorted(idx_s, np.arange(n_cfg))
    en = np.searchsorted(idx_s, np.arange(n_cfg), side="right")

    indices = np.arange(n_cfg)
    if n_max is not None and n_max < n_cfg:
        indices = np.random.default_rng(seed).choice(n_cfg, n_max, replace=False)

    # Metric accumulators
    kl_ace_list = []
    kl_quant_list = []
    tv_list = []
    w1_mu_list = []
    w1_y_list = []
    ks_list = []
    cvm_list = []
    nll_list = []

    # ACE protocol grid: 120 width-relative bins in y in [-6, 12]
    y_edges_ace = np.linspace(-6.0, 12.0, 121)

    print(f"Evaluating {len(indices)} test configurations on ACE-KL, Quantile-KL, W_1, D_KS, and NLL...")
    for idx_count, i in enumerate(indices):
        ui = u_s[st[i]:en[i]].astype(np.float64)
        m_true, s_true, yb_true = meta_te[i, 0], meta_te[i, 1], meta_te[i, 2]
        yi = yb_true + np.exp(ui)
        mui = np.exp(m_true + s_true * yi)
        N = len(yi)

        z_s = float(ctx_te[i, 0])
        theta = (
            float(ctx_te[i, 1]),
            float(ctx_te[i, 2]),
            float(ctx_te[i, 3]),
            float(ctx_te[i, 4]),
            float(ctx_te[i, 5]),
            float(ctx_te[i, 6] * 1000.0),
        )

        # -------------------------------------------------------------
        # 1. ACE-Compatible Protocol KL (cell-integrated, N >= 5)
        # -------------------------------------------------------------
        counts_ace, _ = np.histogram(yi, bins=y_edges_ace)
        valid_ace = counts_ace >= 5
        obs_ace = counts_ace[valid_ace].astype(np.float64)
        obs_ace /= obs_ace.sum()

        pred_ace = []
        for j in np.flatnonzero(valid_ace):
            y_lo = y_edges_ace[j]
            y_hi = y_edges_ace[j + 1]
            grid_lnmu = np.linspace(m_true + s_true * y_lo, m_true + s_true * y_hi, 10)
            p_sub = model.pdf_lnmu(grid_lnmu, z_s=z_s, theta=theta)
            pred_ace.append(float(np.trapezoid(p_sub, grid_lnmu)))
        pred_ace = np.maximum(np.array(pred_ace, dtype=np.float64), 1e-30)
        pred_ace /= pred_ace.sum()

        kl_ace = float(np.sum(obs_ace * np.log(obs_ace / pred_ace)))
        kl_ace_list.append(kl_ace)

        # Total variation on ACE grid
        tv = float(0.5 * np.sum(np.abs(obs_ace - pred_ace)))
        tv_list.append(tv)

        # -------------------------------------------------------------
        # 2. Equal-Mass Quantile Adaptive KL (K = 50, Miller-Madow)
        # -------------------------------------------------------------
        K_bins = 50
        q_edges = np.unique(np.quantile(yi, np.linspace(0, 1, K_bins + 1)))
        K_act = len(q_edges) - 1
        p_q_obs = np.array([np.sum((yi >= q_edges[k]) & (yi <= q_edges[k + 1])) / N for k in range(K_act)])
        p_q_obs /= p_q_obs.sum()

        q_q_pred = []
        for k in range(K_act):
            grid_k = np.linspace(m_true + s_true * q_edges[k], m_true + s_true * q_edges[k + 1], 10)
            p_sub_k = model.pdf_lnmu(grid_k, z_s=z_s, theta=theta)
            q_q_pred.append(float(np.trapezoid(p_sub_k, grid_k)))
        q_q_pred = np.maximum(np.array(q_q_pred, dtype=np.float64), 1e-30)
        q_q_pred /= q_q_pred.sum()

        kl_q_raw = float(np.sum(p_q_obs * np.log(p_q_obs / q_q_pred)))
        kl_q_debiased = max(0.0, kl_q_raw - (K_act - 1) / (2.0 * N))
        kl_quant_list.append(kl_q_debiased)

        # -------------------------------------------------------------
        # 3. Continuous Bin-Free CDF: KS Distance & Wasserstein-1
        # -------------------------------------------------------------
        mu_edge = np.exp(m_true + s_true * (yb_true - 0.02 - 0.05))
        mu_max = float(np.max(mui) * 1.05)
        mu_dense = np.geomspace(mu_edge, mu_max, 1500)
        p_dense = model.pdf_mu(mu_dense, z_s=z_s, theta=theta)
        cdf_dense = np.cumsum(0.5 * (p_dense[:-1] + p_dense[1:]) * np.diff(mu_dense))
        cdf_dense = np.insert(cdf_dense, 0, 0.0)
        if cdf_dense[-1] > 0:
            cdf_dense /= cdf_dense[-1]

        mu_sorted = np.sort(mui)
        u_eval = np.interp(mu_sorted, mu_dense, cdf_dense)
        emp_cdf = (np.arange(1, N + 1)) / N

        # Kolmogorov-Smirnov distance
        ks_dist = float(np.max(np.abs(emp_cdf - u_eval)))
        ks_list.append(ks_dist)

        # Cramér-von Mises distance
        cvm = float(1.0 / (12.0 * N) + np.sum((u_eval - (2.0 * np.arange(1, N + 1) - 1.0) / (2.0 * N)) ** 2) / N)
        cvm_list.append(cvm)

        # Wasserstein-1 distance in physical mu
        target_quantiles = (np.arange(1, N + 1) - 0.5) / N
        model_quantiles = np.interp(target_quantiles, cdf_dense, mu_dense)
        w1_mu = float(np.mean(np.abs(mu_sorted - model_quantiles)))
        w1_mu_list.append(w1_mu)

        # Wasserstein-1 distance in normalized y
        y_target_q = (np.log(model_quantiles) - m_true) / s_true
        y_sorted = np.sort(yi)
        w1_y = float(np.mean(np.abs(y_sorted - y_target_q)))
        w1_y_list.append(w1_y)

        # -------------------------------------------------------------
        # 4. Continuous Ray-by-Ray NLL in y-space
        # -------------------------------------------------------------
        lnmu_rays = m_true + s_true * yi
        lp_rays = model.log_prob_lnmu(lnmu_rays, z_s=z_s, theta=theta)
        lp_y_rays = lp_rays + np.log(s_true)
        nll_list.append(float(-np.mean(lp_y_rays)))

        if (idx_count + 1) % 50 == 0 or (idx_count + 1) == len(indices):
            print(
                f"  [{idx_count + 1:>4}/{len(indices)}] "
                f"KL_ACE med: {np.median(kl_ace_list):.5f} | "
                f"KL_quant med: {np.median(kl_quant_list):.5f} | "
                f"W1(mu) med: {np.median(w1_mu_list):.5f} | "
                f"D_KS med: {np.median(ks_list)*100:.2f}%"
            )

    results = {
        "n_configs": len(indices),
        # 1. ACE-Compatible Protocol KL
        "kl_ace_median": float(np.median(kl_ace_list)),
        "kl_ace_p90": float(np.percentile(kl_ace_list, 90)),
        "kl_ace_p99": float(np.percentile(kl_ace_list, 99)),
        "kl_ace_max": float(np.max(kl_ace_list)),
        "kl_ace_mean": float(np.mean(kl_ace_list)),
        "ace_published_median_kl": 0.0069,
        # 2. Quantile Adaptive KL
        "kl_quant_median": float(np.median(kl_quant_list)),
        "kl_quant_p90": float(np.percentile(kl_quant_list, 90)),
        "kl_quant_max": float(np.max(kl_quant_list)),
        # 3. Total Variation Distance
        "tvs_median": float(np.median(tv_list)),
        "tvs_p90": float(np.percentile(tv_list, 90)),
        "tvs_max": float(np.max(tv_list)),
        # 4. Continuous Wasserstein-1 Distance
        "w1_mu_median": float(np.median(w1_mu_list)),
        "w1_mu_p90": float(np.percentile(w1_mu_list, 90)),
        "w1_mu_max": float(np.max(w1_mu_list)),
        "w1_y_median": float(np.median(w1_y_list)),
        # 5. Continuous Kolmogorov-Smirnov Distance
        "ks_median": float(np.median(ks_list)),
        "ks_p90": float(np.percentile(ks_list, 90)),
        "ks_max": float(np.max(ks_list)),
        # 6. Continuous Cramér-von Mises
        "cvm_median": float(np.median(cvm_list)),
        # 7. Continuous Negative Log-Likelihood
        "nll_y_median": float(np.median(nll_list)),
        "nll_y_mean": float(np.mean(nll_list)),
    }

    print("\n======================= COMPREHENSIVE BENCHMARK SUMMARY =======================")
    print(f"Configurations Evaluated: {results['n_configs']}")
    print("-" * 75)
    print(f"ACE-Protocol KL (nats):       Median = {results['kl_ace_median']:.5f}  (ACE Paper published: 0.00690)")
    print(f"                             90th % = {results['kl_ace_p90']:.5f} | Max = {results['kl_ace_max']:.5f}")
    print(f"Quantile Adaptive KL (nats):  Median = {results['kl_quant_median']:.5f} | 90th % = {results['kl_quant_p90']:.5f}")
    print(f"Total Variation Distance (%): Median = {results['tvs_median']*100:.2f}% | 90th % = {results['tvs_p90']*100:.2f}%")
    print("-" * 75)
    print(f"Wasserstein-1 W_1(mu):        Median = {results['w1_mu_median']:.6f} | 90th % = {results['w1_mu_p90']:.6f}")
    print(f"                              (Mean physical magnification displacement)")
    print(f"Kolmogorov-Smirnov D_KS (%):  Median = {results['ks_median']*100:.2f}% | 90th % = {results['ks_p90']*100:.2f}%")
    print(f"                              (Maximum cumulative probability error)")
    print(f"Ray-by-Ray NLL (y-space):     Median = {results['nll_y_median']:.4f} nats | Mean = {results['nll_y_mean']:.4f} nats")
    print("===============================================================================")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] saved results to {OUT_PATH}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_max", type=int, default=200, help="Number of configs to evaluate (default 200)")
    parser.add_argument("--flux_mode", type=str, default="unit", choices=["unit", "standard"])
    args = parser.parse_args()
    evaluate_test_set(n_max=args.n_max, flux_mode=args.flux_mode)
