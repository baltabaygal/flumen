"""Per-context robust location and scale of log magnification."""

import numpy as np

from .features import scale_features


def loc_scale(z, theta, fit):
    x = scale_features(z, theta)
    mean = np.asarray(fit["feat_mean"], dtype=np.float64)
    std = np.asarray(fit["feat_std"], dtype=np.float64)
    xs = (x - mean) / std
    s = np.exp(xs @ np.asarray(fit["coef_log_s"], dtype=np.float64))
    m = (xs @ np.asarray(fit["coef_m_s"], dtype=np.float64)) * s
    return m, s


def loc_scale_ctx(context, fit):
    context = np.asarray(context, dtype=np.float64)
    return loc_scale(context[:, 0],
                     tuple(context[:, j] for j in range(1, context.shape[1])), fit)
