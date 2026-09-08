"""Neural architecture required by the frozen SOSPF checkpoints."""

import torch.nn as nn


def build_flow(config: dict) -> nn.Module:
    """Build the one-dimensional conditional SOS polynomial flow."""
    if config.get("family") != "sospf":
        raise ValueError("The production bundle supports only its SOSPF checkpoints")
    from zuko.flows.polynomial import SOSPF

    return SOSPF(
        features=1,
        context=config["context"],
        transforms=config["transforms"],
        hidden_features=[config["hidden"]] * config.get("depth", 2),
        degree=config.get("degree", 4),
        polynomials=config.get("polynomials", 3),
        slope=config.get("slope", 1e-3),
    )
