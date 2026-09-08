# Bundled simulation data

The base dataset contains 2,055 training, 972 validation, and 973 untouched
checkerboard test configurations. It contains 41,006,144 / 19,389,180 /
19,406,954 rays respectively. The two low-structure augmentations contribute
718 training configurations and 14,359,792 rays. Their `heldout` partitions
were not used as training rows.

The high-structure/high-redshift augmentation contributes 240 training
configurations (4,664,309 rays) and 48 heldout configurations (7,030,138 rays).
The shipped v1 fine-tune drew 400,000 additional training rows from its training
pool and, following the historical run exactly, drew a separate 90,000-row
validation sample from that same pool. Its named heldout partition was not used
for optimization.

Each HDF5 file stores padded `samples/lnmu`, `samples/valid_counts`, source
redshift, and the six cosmological parameters. `zeq` is divided by 1000 before
being passed to the model. `MANIFEST.sha256` freezes the precise files used by
this package.

The calibrated domain covered by the bundled data is approximately:

| input | minimum | maximum |
|---|---:|---:|
| z_s | 0.2 | 11.9930 |
| h | 0.55 | 0.79994 |
| OmegaM | 0.15 | 0.499995 |
| sigma8 | 0.4 | 1.49998 |
| OmegaB | 0.030003 | 0.069997 |
| ns | 0.940004 | 0.989999 |
| zeq/1000 | 3.30003 | 3.49999 |

Treat predictions outside these limits as extrapolations.
