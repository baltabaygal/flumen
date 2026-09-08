"""Boundary coordinate transform used by the exact-support body."""

import numpy as np


def y_bound_from_edge(y_edge, sigma_edge, k=4.0):
    return y_edge - k * sigma_edge


def y_to_u(y, y_bound):
    difference = np.asarray(y, dtype=np.float64) - np.asarray(y_bound, dtype=np.float64)
    if np.any(difference <= 0):
        raise ValueError(f"{int(np.sum(difference <= 0))} rows lie at/below y_bound")
    return np.log(difference)
