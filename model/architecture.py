"""Neural network architecture for the production single-body normalizing flow."""

import torch
from torch.distributions import Independent, StudentT
from zuko.flows.polynomial import SOSPF
from zuko.lazy import Unconditional


def set_studentt_base(flow, df=3.0):
    """Swap the flow's base distribution for an independent StudentT."""
    flow.base = Unconditional(
        lambda d, loc, scale: Independent(StudentT(d, loc, scale), 1),
        torch.tensor(float(df)),
        torch.zeros(1),
        torch.ones(1),
        buffer=True,
    )
    return flow


def build_flow(cfg=None, ctx_dim=7):
    """Build the single-body SOSPF flow supporting Gaussian or Student-t base distribution."""
    if cfg is None:
        cfg = {
            "transforms": 3,
            "hidden": 128,
            "depth": 1,
            "degree": 8,
            "polynomials": 5,
        }

    flow = SOSPF(
        features=1,
        context=ctx_dim,
        transforms=cfg.get("transforms", 3),
        hidden_features=[cfg.get("hidden", 128)] * cfg.get("depth", 1),
        degree=cfg.get("degree", 8),
        polynomials=cfg.get("polynomials", 5),
        slope=1e-3,
    )

    base_df = float(cfg.get("base_df", 0.0) or 0.0)
    if base_df > 0.0:
        flow = set_studentt_base(flow, base_df)

    return flow
