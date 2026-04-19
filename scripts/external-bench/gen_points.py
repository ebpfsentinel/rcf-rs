#!/usr/bin/env python3
"""Emit a deterministic CSV dataset of D-dim f64 points.

Used as the shared input for every external-bench runner so the
comparison is on identical bytes.  Deterministic via a caller-
supplied seed — NumPy's default RNG is bit-reproducible across
platforms for the same seed.

Usage:
    python3 gen_points.py --n 10000 --dim 16 --seed 2026 > data.csv
"""

import argparse
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--outlier-frac",
        type=float,
        default=0.01,
        help="Fraction of points drawn from a distant outlier mode.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_outliers = int(args.n * args.outlier_frac)
    n_normal = args.n - n_outliers

    # Normals: tight cluster around the origin, stddev 0.1.
    normals = rng.normal(loc=0.0, scale=0.1, size=(n_normal, args.dim))
    # Outliers: shifted mean 5.0, stddev 0.2.
    outliers = rng.normal(loc=5.0, scale=0.2, size=(n_outliers, args.dim))

    data = np.vstack([normals, outliers])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_outliers)])

    # Two-phase layout: first `n_normal` rows are clean (suitable
    # for warm-up), last `n_outliers` rows are the anomaly probes.
    # Downstream drivers that want a shuffled stream can re-permute
    # on their side, but the canonical layout is "train → eval"
    # so the warm phase is guaranteed anomaly-free.

    writer = sys.stdout
    writer.write("label," + ",".join(f"d{i}" for i in range(args.dim)) + "\n")
    for row, label in zip(data, labels):
        writer.write(f"{int(label)}," + ",".join(f"{v:.17g}" for v in row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
