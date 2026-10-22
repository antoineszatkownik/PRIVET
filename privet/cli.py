"""
privet fit  --train train.npy --distance hamming --fit-window 0.01 0.2 --out runs/fit
    stage 1: EVD fit on train->train 1-NN distances, CDF plot to check the window

privet run  --train train.npy --test test.npy --synthetic synth.npy --distance hamming --out runs/audit
    stages 1-3: scores.csv (one row per synthetic sample), summary, plots

Data files: .npy, .csv or .pt, one sample per row.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  

from .plot_utils import plot_cdfs, plot_score_distributions 
from .privet import PRIVET 


def load(path):
    if path is None:
        return None
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".csv"):
        return np.loadtxt(path, delimiter=",")
    if path.endswith(".pt"):
        import torch
        return torch.load(path)
    raise ValueError(f"unsupported file type: {path}")


def main():
    parser = argparse.ArgumentParser(prog="privet", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fit", "run"])
    parser.add_argument("--train", required=True)
    parser.add_argument("--test")
    parser.add_argument("--synthetic")
    parser.add_argument("--distance", default="standard_euclidean",
                        choices=["standard_euclidean", "feature_normalized_euclidean", "hamming"])
    parser.add_argument("--fit-window", nargs=2, type=float, default=[0.01, 0.2])
    parser.add_argument("--fit-method", default="ols", choices=["ols", "mle"])
    parser.add_argument("--score-fraction", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=None,
                        help="flag scored samples with delta_p (log10_p_train without --test) "
                             "<= threshold (default None)")
    parser.add_argument("--loo-correction", action="store_true",
                        help="reference size N_train - 1 for the train->train fit")
    parser.add_argument("--upper-tail", default="extrapolate", choices=["extrapolate", "empirical"])
    parser.add_argument("--test-reference", default="train_law", choices=["train_law", "test_law"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Stage 1
    privet = PRIVET(load(args.train), args.device, *args.fit_window,
                    distance=args.distance, fit_method=args.fit_method,
                    loo_correction=args.loo_correction)
    law = privet.law
    print(f"EVD fit: {law.family} ({law.fit_method}), window {privet.partition_fit} -> "
          f"d in [{law.window[0]:.4g}, {law.window[1]:.4g}]; criterion (lower wins): "
          + ", ".join(f"{k} {v:.4g}" for k, v in law.criterion.items()))
    if args.command == "fit":
        plot_cdfs(privet)
        plt.savefig(out / "cdfs.png", dpi=200, bbox_inches="tight")
        print(f"check {out / 'cdfs.png'}")
        return

    # Stage 2
    scores = privet.compute_scores(load(args.synthetic), load(args.test),
                                   score_fraction=args.score_fraction, threshold=args.threshold,
                                   upper_tail=args.upper_tail, test_reference=args.test_reference)
    scores.to_csv(out / "scores.csv", index=False, float_format="%.6g")

    # Stage 3
    summary = privet.summary(scores)
    text = json.dumps(summary, indent=2)
    print(text)
    (out / "summary.json").write_text(text + "\n")

    plot_cdfs(privet, scores)
    plt.savefig(out / "cdfs.png", dpi=200, bbox_inches="tight")
    plot_score_distributions(scores, args.threshold)
    plt.savefig(out / "score_distributions.png", dpi=200, bbox_inches="tight")
    print(f"written to {out}/: scores.csv, summary.json, cdfs.png, "
          "score_distributions.png")


if __name__ == "__main__":
    main()
