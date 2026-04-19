#!/usr/bin/env python3
"""Benchmark scikit-learn's `IsolationForest` on the shared dataset.

IsolationForest is *not* RCF — it is batch-only and trains on a
fixed window — but it is the canonical tree-based outlier baseline
for apples-to-apples comparison of speed + detection quality.

Usage:
    pip install --user scikit-learn numpy
    python3 bench_sklearn_iforest.py --input data.csv --trees 100
"""

import argparse
import csv
import time

import numpy as np
from sklearn.ensemble import IsolationForest


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    rows = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            labels.append(int(row[0]))
            rows.append([float(x) for x in row[1:]])
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int8)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    # Reuse the manual trapezoidal routine to avoid a sklearn.metrics
    # dependency divergence between rrcf / sklearn runners.
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    total_pos = labels_sorted.sum()
    total_neg = len(labels_sorted) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5
    tp = 0
    fp = 0
    prev_tpr = 0.0
    prev_fpr = 0.0
    auc_val = 0.0
    for y in labels_sorted:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        auc_val += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr = tpr
        prev_fpr = fpr
    return auc_val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.5,
        help="Fraction of the dataset used for `.fit`; rest is scored.",
    )
    args = parser.parse_args()

    X, labels = load_csv(args.input)
    n, d = X.shape
    split = int(n * args.train_frac)
    print(f"points={n} dim={d} trees={args.trees} train={split}")

    # sklearn's IsolationForest is NumPy/Cython-backed; the hot
    # loop auto-vectorises via BLAS SIMD. `n_jobs=-1` actually
    # regresses at this batch size (joblib task-spawn overhead
    # exceeds the win on 100 trees x 10k points) — keep the
    # default single-threaded config which measures the real
    # cost of the vectorised path.
    model = IsolationForest(n_estimators=args.trees, random_state=2026)
    t0 = time.perf_counter_ns()
    model.fit(X[:split])
    fit_ns = time.perf_counter_ns() - t0

    t0 = time.perf_counter_ns()
    # `decision_function` returns higher = more normal; invert so
    # higher = more anomalous, matching rrcf / rcf-rs conventions.
    scores = -model.decision_function(X)
    score_ns = time.perf_counter_ns() - t0

    print(f"  fit            = {fit_ns / 1e6:.2f} ms")
    print(f"  score(all)     = {score_ns / 1e6:.2f} ms")
    print(f"  per-op score   = {score_ns / n:.0f} ns")
    print(f"  scores_per_s   = {n * 1e9 / score_ns:.0f}")
    print(f"  auc            = {auc(scores, labels):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
