# Analytic model

FLUMEN's default model predicts the source-plane weak-lensing magnification
PDF. It emulates the semi-analytic lensing calculation with r200 halo edges.
`plane="image"` returns the corresponding image-plane PDF.

## Inputs

Required: source redshift `z_s`. Optional cosmology: `h`, `Om`, `sigma8`, `Ob`,
`ns`, `zeq`, `Ok`, `w0`, `wa`. Defaults are Planck-like. Supported parameter
ranges are exposed by `flumen.TRAINING_RANGE`; `z_s` spans 0.2–12. Curvature
and CPL dark energy are supported within the training domain.

## Accuracy and limits

Validation median KL is about 4.5e-6; median maximum PDF error is 0.30% of
the peak. Sparse parameter corners have larger errors, reaching 26% of the
peak in validation. These are validation results, not new test certification.

Derived low-redshift N10 estimates differ from checked analytic references
by a few percent, reaching about 5% in the smallest-noise comparison.
Below z_s=0.5, predictions depend on the halo-edge assumption. Far low-z
tails can be extrapolated. Original guard limitations remain.

The model is normalized in each requested plane. Finite output grids may
omit probability mass. It does not impose unit flux on the physical analytic
model. The optional older Monte-Carlo backend emulates a different calculation.

Exact weights, export checks, source hashes and training provenance are kept
in the checkpoint metadata and internal freeze/qualification records.
