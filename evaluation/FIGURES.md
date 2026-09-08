# Production figure map

Not checked in — generate on demand with the commands below; both scripts
write into `evaluation/figures/`.

| Figure | Analytical question | Form | Evidence |
|---|---|---|---|
| `generalization_certificate` | Does performance transfer from validation to untouched test data? | Split boxplots plus quantile dot comparison | 243 validation and 973 test configurations |
| `test_parameter_landscape` | Where are the difficult test contexts? | Parameter-space scatter colored by KL_y | All 973 test configurations |
| `representative_pdfs_with_data` | Does the predicted physical PDF follow held-out rays across redshift? | Six small-multiple line-and-dot panels | Six nearest-to-fiducial test configurations, ~20k rays each |
| `representative_pdfs_model_only` | What does the frozen family of model curves look like without point occlusion? | Six small-multiple lines | Same six contexts |
| `full_parameter_atlas_mu_ymin_1e-3_with_data` | Does the final PDF remain smooth and track the simulator across the acceptance panel? | 5×8 physical-magnification line-and-dot atlas | Five reference cosmologies, eight redshifts, 400k-ray fixed references; the historical box-high row is replaced by fresh ordinary-branch Box-midpoint simulations |
| `full_parameter_atlas_mu_ymin_1e-3_model_only` | Is the model family itself free of visible cutoffs, bumps, or splice artifacts? | 5×8 physical-magnification line atlas | The same 40 frozen production curves without point occlusion |

Extended-tail variants of both full atlases are also exported as
`full_parameter_atlas_mu_ymin_1e-5_{with_data,model_only}`. They preserve the
same axes and references while exposing two additional decades of density.

All figures use the frozen `flumen` checkpoints and calibration files.
The full atlas is reproducible with
`python -m flumen.evaluation.plot_full_parameter_atlas`; its fixed
simulator reference is packaged under `evaluation/reference/`.
PNG is the review format; PDF and SVG are vector paper formats.
