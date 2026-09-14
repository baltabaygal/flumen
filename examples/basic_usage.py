import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flumen.model import load_model


model = load_model(device="cpu")
theta = (0.67, 0.30, 0.85, 0.0493, 0.965, 3.402)
mu = np.geomspace(0.7, 20.0, 500)
pdf = model.pdf_mu(mu, z_s=1.0, theta=theta)
print(mu[np.argmax(pdf)], np.trapezoid(pdf, mu))
