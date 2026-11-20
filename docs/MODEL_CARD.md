# Analytic model

FLUMEN's default model predicts the source-plane weak-lensing magnification
PDF. It emulates the semi-analytic lensing calculation with r200 halo edges.
`plane="image"` returns the corresponding image-plane PDF.

Since FLUMEN 0.7.0 the default (`mix3filF-v1`) includes the halo-filament
overlap correction: each filament's density is multiplied by 1 - F(M_f, z),
the fraction of its mass not already in halos above 1e7 Msun (excursion-set
walks, closed to the halo mass fraction of each cosmology). This narrows the
PDF: at Planck the source-plane width of ln(mu) drops by about 14% at z_s=1
and 12% at z_s=3. The previous default (`mix3-v1`, filaments not corrected)
is kept and can be loaded with
`flumen.model.load_analytic_model(version="mix3-v1")`.

## Inputs

Required: source redshift `z_s`. Optional cosmology: `h`, `Om`, `sigma8`, `Ob`,
`ns`, `zeq`, `Ok`, `w0`, `wa`. Defaults are Planck-like. Supported parameter
ranges are exposed by `flumen.TRAINING_RANGE`; `z_s` spans 0.2–12. Curvature
and CPL dark energy are supported within the training domain.

## Accuracy and limits

Against its own semi-analytic targets on the validation split, the median KL
is about 1.3e-5 and the median maximum PDF error is 0.52% of the peak (95th
percentile 2.8%). This is about 1.8 times larger than for `mix3-v1` against
its targets. The largest errors are in the sparse low-z_s corner (z_s < 0.4):
median 5.4% of the peak at z_s 0.2-0.3, up to 46% for very narrow peaks
(width about 0.002 in ln mu). These are validation results, not new test
certification. Low-redshift N10 estimates were checked against analytic
references for `mix3-v1` (a few percent, up to about 5%) and have not been
re-checked for this model.

Below z_s=0.5, predictions depend on the halo-edge assumption. Far low-z
tails can be extrapolated. Original guard limitations remain.

The model is normalized in each requested plane. Finite output grids may
omit probability mass. It does not impose unit flux on the physical analytic
model. The optional older Monte-Carlo backend emulates a different calculation.
Both backends include the halo-filament overlap correction (MC since 0.6.0),
but they treat faint filaments differently: the analytic model keeps every
filament, the Monte-Carlo engine drops filaments below its convergence
threshold. Their PDFs differ, as they did before the correction.

Exact weights, export checks, source hashes and training provenance are kept
in the checkpoint metadata and internal freeze/qualification records.
