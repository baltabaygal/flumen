"""Comprehensive evaluation module for production model v3.

Computes:
1. ACE-Protocol KL divergence: 120 width-relative bins on y in [-6, 12] with N >= 5
2. Equal-mass Quantile Adaptive KL divergence with Miller-Madow debiasing (K = 50)
3. Total Variation Distance (TVS) on the width-relative grid
4. Bin-free truncated Wasserstein-1 distance in physical magnification W_1(mu) on empirical support
   (Note: because p(mu) ~ mu^-2 has a divergent first moment, full-distribution W_1 diverges;
   this metric computes the truncated Wasserstein-1 distance conditioned on mu <= 1.05 * max(mu_sample))
5. Bin-free Kolmogorov-Smirnov distance (D_KS = max |F_emp - F_mod| per context)
6. Bin-free Cramér-von Mises distance (CvM)
7. Continuous sample Negative Log-Likelihood (NLL) in normalized y-space
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flumen.model import load_model


def evaluate_split(
    split="test",
    data_path=None,
    n_max=None,
    device="cpu",
    flux_mode="unit",
    tail_mode="asymptotic",
    seed=42,
    output_path=None,
    rows_output_path=None,
):
    if data_path is None:
        data_path = ROOT / "data" / "u_space" / f"{split}.npz"
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset for split '{split}' not found at {data_path}. "
            f"Run 'python -m flumen.data.download' to verify or obtain datasets."
        )

    model = load_model(device=device, flux_mode=flux_mode, tail_mode=tail_mode)

    d = np.load(data_path)
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
    rows = []

    # ACE protocol grid: 120 width-relative bins in y on [-6, 12]
    y_edges_ace = np.linspace(-6.0, 12.0, 121)

    print(f"Evaluating {len(indices)} configurations ({split} split, flux_mode={flux_mode}, tail_mode={tail_mode})...")
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
            float(ctx_te[i, 6] * 1000.0 if ctx_te[i, 6] < 100.0 else ctx_te[i, 6]),
        )

        # -------------------------------------------------------------
        # 1. ACE-Protocol KL (120 width-relative bins on y, N >= 5)
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
        # 3. Continuous Bin-Free CDF: KS Distance & Truncated Wasserstein-1
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

        # Kolmogorov-Smirnov distance (max CDF error for this context)
        ks_dist = float(np.max(np.abs(emp_cdf - u_eval)))
        ks_list.append(ks_dist)

        # Cramér-von Mises distance
        cvm = float(1.0 / (12.0 * N) + np.sum((u_eval - (2.0 * np.arange(1, N + 1) - 1.0) / (2.0 * N)) ** 2) / N)
        cvm_list.append(cvm)

        # Truncated Wasserstein-1 distance on sample support [mu_cut, 1.05 * mu_max]
        target_quantiles = (np.arange(1, N + 1) - 0.5) / N
        model_quantiles = np.interp(target_quantiles, cdf_dense, mu_dense)
        w1_mu = float(np.mean(np.abs(mu_sorted - model_quantiles)))
        w1_mu_list.append(w1_mu)

        # Truncated Wasserstein-1 distance in normalized y
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
        finite_lp = lp_y_rays[np.isfinite(lp_y_rays)]
        mean_nll = float(-np.mean(finite_lp)) if len(finite_lp) > 0 else float("inf")
        nll_list.append(mean_nll)

        rows.append({
            "split": split,
            "configuration_index": int(i),
            "z_s": z_s,
            "h": theta[0],
            "OmegaM": theta[1],
            "sigma8": theta[2],
            "OmegaB": theta[3],
            "ns": theta[4],
            "zeq": theta[5],
            "kl_ace": kl_ace,
            "kl_quant": kl_q_debiased,
            "tvs": tv,
            "w1_mu_truncated": w1_mu,
            "w1_y": w1_y,
            "d_ks": ks_dist,
            "cvm": cvm,
            "nll_y": mean_nll,
        })

        if (idx_count + 1) % 100 == 0 or (idx_count + 1) == len(indices):
            print(
                f"  [{idx_count + 1:>4}/{len(indices)}] "
                f"KL_ACE med: {np.median(kl_ace_list):.5f} | "
                f"KL_quant med: {np.median(kl_quant_list):.5f} | "
                f"W1_trunc med: {np.median(w1_mu_list):.5f} | "
                f"D_KS med: {np.median(ks_list)*100:.2f}%"
            )

    results = {
        "split": split,
        "n_configs": len(indices),
        # 1. ACE-Protocol KL (120-bin relative-y, N >= 5)
        "kl_ace_median": float(np.median(kl_ace_list)),
        "kl_ace_p90": float(np.percentile(kl_ace_list, 90)),
        "kl_ace_p99": float(np.percentile(kl_ace_list, 99)),
        "kl_ace_max": float(np.max(kl_ace_list)),
        "kl_ace_mean": float(np.mean(kl_ace_list)),
        "ace_published_reference_kl": 0.00690,
        # 2. Quantile Adaptive KL (K = 50, Miller-Madow debiased)
        "kl_quant_median": float(np.median(kl_quant_list)),
        "kl_quant_p90": float(np.percentile(kl_quant_list, 90)),
        "kl_quant_max": float(np.max(kl_quant_list)),
        # 3. Total Variation Distance
        "tvs_median": float(np.median(tv_list)),
        "tvs_p90": float(np.percentile(tv_list, 90)),
        "tvs_max": float(np.max(tv_list)),
        # 4. Truncated Wasserstein-1 Distance (on empirical magnification support)
        "w1_mu_truncated_median": float(np.median(w1_mu_list)),
        "w1_mu_truncated_p90": float(np.percentile(w1_mu_list, 90)),
        "w1_mu_truncated_max": float(np.max(w1_mu_list)),
        "w1_mu_median": float(np.median(w1_mu_list)),
        "w1_y_median": float(np.median(w1_y_list)),
        # 5. Continuous Kolmogorov-Smirnov Distance (per-context max CDF error)
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
    print(f"Configurations Evaluated: {results['n_configs']} ({split} split)")
    print("-" * 75)
    print(f"ACE-Protocol KL (nats):       Median = {results['kl_ace_median']:.5f} (comparable to ACE published 0.00690)")
    print(f"                             90th % = {results['kl_ace_p90']:.5f} | Test Max = {results['kl_ace_max']:.5f}")
    print(f"Quantile Adaptive KL (nats):  Median = {results['kl_quant_median']:.5f} | 90th % = {results['kl_quant_p90']:.5f}")
    print(f"Total Variation Distance (%): Median = {results['tvs_median']*100:.2f}% | Test Max = {results['tvs_max']*100:.2f}%")
    print("-" * 75)
    print(f"Truncated W_1(mu):            Median = {results['w1_mu_truncated_median']:.6f} | 90th % = {results['w1_mu_truncated_p90']:.6f}")
    print(f"                              (Mean magnification error on empirical support mu <= 1.05 * mu_max)")
    print(f"Kolmogorov-Smirnov D_KS (%):  Median = {results['ks_median']*100:.2f}% | 90th % = {results['ks_p90']*100:.2f}% | Test Max = {results['ks_max']*100:.2f}%")
    print(f"                              (Context-median of max cumulative probability error)")
    print(f"Ray-by-Ray NLL (y-space):     Median = {results['nll_y_median']:.4f} nats | Mean = {results['nll_y_mean']:.4f} nats")
    print("===============================================================================")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[done] saved summary to {output_path}")

    if rows_output_path is not None and rows:
        rows_output_path = Path(rows_output_path)
        rows_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rows_output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[done] saved per-context results to {rows_output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate flumen model on validation or test split.")
    parser.add_argument("--split", choices=("test", "validation"), default="test", help="Dataset split to evaluate")
    parser.add_argument("--data_path", type=Path, default=None, help="Explicit path to split .npz file")
    parser.add_argument("--n_max", type=int, default=None, help="Number of configs to evaluate (default: all)")
    parser.add_argument("--flux_mode", type=str, default="unit", choices=["unit", "standard"])
    parser.add_argument("--tail_mode", type=str, default="asymptotic", choices=["asymptotic", "hermite"])
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path for summary results")
    parser.add_argument("--rows_output", type=Path, default=None, help="Output CSV path for per-context rows")
    args = parser.parse_args()

    default_out = ROOT / "evaluation" / "results" / "test_summary.json" if args.split == "test" else ROOT / "evaluation" / "results" / "validation_summary.json"
    out_file = args.output if args.output is not None else default_out

    evaluate_split(
        split=args.split,
        data_path=args.data_path,
        n_max=args.n_max,
        flux_mode=args.flux_mode,
        tail_mode=args.tail_mode,
        output_path=out_file,
        rows_output_path=args.rows_output,
    )


if __name__ == "__main__":
    main()
