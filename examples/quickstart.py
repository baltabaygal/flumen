"""Minimal top-level flumen usage: import flumen and call generate_pdf.

Run from the repo root (the directory containing this package):

    conda run -n test python examples/quickstart.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

import flumen

# Fiducial cosmology, only z_s is required.
mu, pdf = flumen.generate_pdf(z_s=1.0)
print(f"peak mu = {mu[pdf.argmax()]:.3f}, integral = {np.trapezoid(pdf, mu):.6f}")

# Override cosmology parameters as keyword arguments.
mu2, pdf2 = flumen.generate_pdf(z_s=2.0, h=0.70, Om=0.28, sigma8=0.90)

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.loglog(mu, pdf, label="z_s=1.0, fiducial")
    ax.loglog(mu2, pdf2, label="z_s=2.0, h=0.70, Om=0.28, sigma8=0.90")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$p(\mu)$")
    ax.legend()
    fig.savefig("flumen_quickstart.png", dpi=150)
    print("wrote flumen_quickstart.png")
except ImportError:
    print("matplotlib not installed; skipping plot")
