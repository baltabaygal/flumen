"""Frozen context feature maps used by the production calibrations."""

import numpy as np


def _cols(z, theta):
    z = np.asarray(z, dtype=np.float64)
    lz = np.log1p(z)
    th = [np.broadcast_to(np.asarray(t, dtype=np.float64), lz.shape) for t in theta]
    return lz, th


def edge_features(z, theta):
    lz, th = _cols(z, theta)
    if len(th) != 6:
        raise ValueError("production theta must contain 6 values")
    h, om, a, ob, ns, ze = th
    cols = ([np.ones_like(lz), lz, lz**2, lz**3]
            + [h, om, a, ob, ns, ze]
            + [h*lz, om*lz, a*lz, ob*lz, ns*lz, ze*lz])
    return np.stack(cols, axis=-1)


def scale_features(z, theta):
    _, th = _cols(z, theta)
    if len(th) != 6:
        raise ValueError("production theta must contain 6 values")
    h, om, a, ob, ns, ze = th
    u = np.log(np.asarray(z, dtype=np.float64))
    ls8, lom, lh = np.log(a), np.log(om), np.log(h)
    cols = ([np.ones_like(u), u, u**2, u**3, u**4]
            + [ls8, lom, lh, ob, ns, ze]
            + [ls8*u, lom*u, lh*u, ob*u, ns*u, ze*u]
            + [ls8*u**2, lom*u**2])
    return np.stack(cols, axis=-1)


def tail_features(z, theta):
    lz, th = _cols(z, theta)
    if len(th) != 6:
        raise ValueError("production theta must contain 6 values")
    h, om, a, ob, ns, ze = th
    one = np.ones_like(lz)
    cols = ([one, lz, lz**2, lz**3]
            + [h, om, a, ob, ns, ze]
            + [h*lz, om*lz, a*lz, ob*lz, ns*lz, ze*lz]
            + [a**2, om*a, a**2*lz, om*a*lz])
    inv2, inv1 = 1.0 / a**2, 1.0 / a
    cols += [inv2, inv1, inv2*lz, inv1*lz, inv2*om, inv2*lz**2]
    return np.stack(cols, axis=-1)
