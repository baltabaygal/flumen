import numpy as np
import flumen

mu, pdf = flumen.generate_pdf(z_s=1.0)
print("Peak magnification:", mu[np.argmax(pdf)])
print("Mass on this grid:", np.trapezoid(pdf, mu))
