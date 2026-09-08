"""Evaluate width-relative KL_y on a bundled validation or test split."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from flumen.model import load_model
from flumen.model.scale import loc_scale
from flumen.training.data import available_splits, DATASETS, contexts


Y_EDGES = np.linspace(-6.0, 12.0, 121)


def evaluate(split="test", max_configs=None, return_rows=False):
    model = load_model()
    scale_fit = torch.load(
        Path(__file__).resolve().parents[1] / "model/checkpoints/boundary_body.pt",
        map_location="cpu", weights_only=False)["stats"]["scale_fit"]
    rows = []
    for directory in DATASETS:
        datasets = available_splits(directory)
        if split not in datasets:
            continue
        data = datasets[split]
        context = contexts(data)
        ids = range(len(context))
        if max_configs and len(context) > max_configs:
            ids = range(0, len(context), int(np.ceil(len(context)/max_configs)))
        for i in ids:
            x = data["lnmu"][i, :int(data["valid_counts"][i])]
            z, theta = float(context[i, 0]), tuple(context[i, 1:])
            m, s = (float(v) for v in loc_scale(z, theta, scale_fit))
            counts, _ = np.histogram((x-m)/s, Y_EDGES)
            use = counts >= 5
            if use.sum() < 5:
                continue
            observed = counts[use].astype(float); observed /= observed.sum()
            predicted = []
            for lo, hi in zip(Y_EDGES[:-1][use], Y_EDGES[1:][use]):
                grid = np.linspace(m+s*lo, m+s*hi, 12)
                predicted.append(np.trapezoid(model.pdf_lnmu(grid, z, theta), grid))
            predicted = np.maximum(predicted, 1e-300); predicted /= np.sum(predicted)
            kl = float(np.sum(observed*np.log(observed/predicted)))
            rows.append({"dataset": directory.name, "split": split,
                         "configuration_index": int(i), "z": z,
                         "h": float(theta[0]), "OmegaM": float(theta[1]),
                         "sigma8": float(theta[2]), "OmegaB": float(theta[3]),
                         "ns": float(theta[4]), "zeq_over_1000": float(theta[5]),
                         "usable_bins": int(use.sum()), "kl_y": kl})
    z = np.asarray([row["z"] for row in rows]); kl = np.asarray([row["kl_y"] for row in rows])
    result = {"split": split, "configurations": len(rows),
              "median_kl_y": float(np.median(kl)),
              "p90_kl_y": float(np.percentile(kl, 90)),
              "p99_kl_y": float(np.percentile(kl, 99)),
              "max_kl_y": float(np.max(kl)), "bands": {}}
    for lo, hi in ((0, .3), (.3, .7), (.7, 1.5), (1.5, 4), (4, 1e9)):
        selected = (z >= lo) & (z < hi)
        result["bands"][f"[{lo},{hi})"] = {
            "n": int(selected.sum()), "median_kl_y": float(np.median(kl[selected]))}
    return (result, rows) if return_rows else result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rows-output", type=Path)
    args = parser.parse_args()
    result, rows = evaluate(args.split, args.max_configs, return_rows=True)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    if args.rows_output:
        with args.rows_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
