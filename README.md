# FLUMEN

Predict the weak-lensing magnification PDF for a source redshift and cosmology.
The **analytic** model is the default.

```python
import flumen

mu, pdf = flumen.generate_pdf(z_s=1.0)
```

`pdf` is the density with respect to `mu`. To use your own grid:

```python
import numpy as np

mu = np.geomspace(0.7, 20.0, 1000)
mu, pdf = flumen.generate_pdf(z_s=1.0, mu=mu)
```

For a density in log magnification:

```python
lnmu, pdf_lnmu = flumen.generate_pdf_lnmu(z_s=1.0)
```

Change the cosmology with `h`, `Om`, `sigma8`, `Ob`, `ns`, `zeq`, `Ok`, `w0`,
and `wa`. Unspecified parameters use Planck-like values.

```python
mu, pdf = flumen.generate_pdf(z_s=2.0, Om=0.28, sigma8=0.9, w0=-0.9)
```

PDFs use the **source plane** by default. Set `plane="image"` for the image
plane. Use a sufficiently wide and fine grid when integrating the PDF.

The model covers `z_s` from 0.2 to 12; parameter limits are available in
`flumen.TRAINING_RANGE`. Calls outside the range emit a warning. Low-redshift
predictions retain a few-percent uncertainty in derived N10 estimates, and
the region below `z_s=0.5` depends on the halo-edge model.

The older Monte-Carlo model remains available explicitly:

```python
mu, pdf = flumen.generate_pdf(z_s=1.0, backend="mc")
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Baltabay:2026nmz,
    author = "Baltabay, Galymzhan and Vaskonen, Ville",
    title = "{FLUMEN: Neural Emulator of an Advanced Stochastic Weak Lensing Model}",
    eprint = "2609.21806",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "9",
    year = "2026"
}

@article{Baltabay:2026hka,
    author = "Baltabay, Galymzhan and Vaskonen, Ville",
    title = "{Cosmological lensing as a jump process}",
    eprint = "2610.40341",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "10",
    year = "2026"
}
```

MIT license. Model qualifications and provenance are recorded separately in
`docs/MODEL_CARD.md` and `PROVENANCE.md`.
