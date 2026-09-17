"""Standalone training script for the single-body production normalizing flow (v3).

Trains p_U(u | c) using an SOS polynomial flow with Gaussian base distribution
on the physical empty-beam boundary coordinate u = ln(y - y_b(c) + delta).
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flumen.model.architecture import build_flow

DEV = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


class InMemoryDataset:
    """Pre-loaded u-space tensors for training."""

    def __init__(self, npz_path, delta=0.05, max_configs=None):
        d = np.load(npz_path)
        u, idx, ctx, meta = d["u"], d["idx"], d["ctx"], d["meta"]

        if max_configs is not None and max_configs < len(ctx):
            mask = idx < max_configs
            u, idx = u[mask], idx[mask]
            ctx, meta = ctx[:max_configs], meta[:max_configs]

        if delta > 0:
            u = np.logaddexp(u, np.log(delta).astype(np.float32))

        self.u = torch.from_numpy(u).float().to(DEV)
        self.idx = torch.from_numpy(idx).long().to(DEV)
        self.ctx = torch.from_numpy(ctx).float().to(DEV)
        self.meta = meta

    def batches(self, bs):
        n = self.u.numel()
        perm = torch.randperm(n, device=DEV)
        for k in range(0, n - bs + 1, bs):
            sl = perm[k:k + bs]
            i = self.idx[sl]
            yield self.u[sl].unsqueeze(-1), self.ctx[i]


def train_single_seed(args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n--- Training Seed {seed} ({args.epochs} epochs, bs={args.bs}, lr={args.lr}) ---")
    train_data = InMemoryDataset(os.path.join(args.data_dir, "train.npz"), delta=args.delta, max_configs=args.max_configs)
    val_data = InMemoryDataset(os.path.join(args.data_dir, "validation.npz"), delta=args.delta, max_configs=args.max_configs)

    # Compute context normalization from training split
    ctx_mean = train_data.ctx.mean(dim=0).cpu().numpy()
    ctx_std = train_data.ctx.std(dim=0).cpu().numpy()
    ctx_std[ctx_std == 0] = 1.0

    cmean_t = torch.tensor(ctx_mean, device=DEV)
    cstd_t = torch.tensor(ctx_std, device=DEV)
    train_data.ctx = (train_data.ctx - cmean_t) / cstd_t
    val_data.ctx = (val_data.ctx - cmean_t) / cstd_t

    cfg = {
        "transforms": 3,
        "hidden": 128,
        "depth": 1,
        "degree": 8,
        "polynomials": 5,
        "base_df": args.base_df,
    }

    flow = build_flow(cfg, ctx_dim=7).to(DEV)
    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    best_val_nll = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        flow.train()
        train_loss, n_batches = 0.0, 0
        t0 = time.time()

        for u_batch, c_batch in train_data.batches(args.bs):
            opt.zero_grad(set_to_none=True)
            lp = flow(c_batch).log_prob(u_batch)
            if not torch.isfinite(lp).all():
                lp = lp[torch.isfinite(lp)]
                if lp.numel() == 0:
                    continue
            loss = -lp.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
            train_loss += float(loss.item())
            n_batches += 1

        sched.step()
        train_loss /= max(n_batches, 1)

        # Validation
        flow.eval()
        with torch.no_grad():
            val_loss, val_batches = 0.0, 0
            for u_batch, c_batch in val_data.batches(args.bs):
                lp = flow(c_batch).log_prob(u_batch)
                if not torch.isfinite(lp).all():
                    lp = lp[torch.isfinite(lp)]
                    if lp.numel() == 0:
                        continue
                val_loss += float(-lp.mean().item())
                val_batches += 1
            val_loss /= max(val_batches, 1)

        dt = time.time() - t0
        print(f"Seed {seed} | Epoch {epoch + 1:2d}/{args.epochs:2d} | Train NLL_u: {train_loss:.4f} | Val NLL_u: {val_loss:.4f} | ({dt:.1f}s)")

        if val_loss < best_val_nll:
            best_val_nll = val_loss
            best_state = {k: v.cpu().clone() for k, v in flow.state_dict().items()}

    payload = {
        "state": best_state,
        "state_dict": best_state,
        "cfg": cfg,
        "ctx_mean": ctx_mean,
        "ctx_std": ctx_std,
        "delta": args.delta,
        "seed": seed,
        "best_val_nll": best_val_nll,
    }
    return payload, best_val_nll


def main():
    parser = argparse.ArgumentParser(description="Train production single-body flow (v3).")
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "data" / "u_space"))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--bs", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--base_df", type=float, default=0.0, help="Student-t df; 0.0 for standard Gaussian base")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--max_configs", type=int, default=None, help="Subsample configs for fast pipeline checks")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--out", type=str, default=str(ROOT / "training" / "single_body_retrained.pt"))
    args = parser.parse_args()

    print(f"Device: {DEV} | Data: {args.data_dir} | Seeds: {args.seeds}")
    best_overall_nll = float("inf")
    best_overall_payload = None
    best_seed = None

    for s in args.seeds:
        payload, val_nll = train_single_seed(args, s)
        if val_nll < best_overall_nll:
            best_overall_nll = val_nll
            best_overall_payload = payload
            best_seed = s

    print(f"\n[summary] Best overall validation NLL_u: {best_overall_nll:.4f} (seed {best_seed})")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_overall_payload, out_path)
    print(f"[done] Saved best checkpoint to {out_path}")


if __name__ == "__main__":
    main()
