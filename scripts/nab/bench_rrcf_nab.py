#!/usr/bin/env python3
"""Run `rrcf` on the NAB `realKnownCause` subset and report per-
file + aggregate AUC, same protocol as `tests/nab.rs` (8-lag
temporal embedding, 15 % warm fraction, frozen baseline).

Usage:
    ./scripts/nab/fetch.sh /opt/nab
    python3 scripts/nab/bench_rrcf_nab.py --nab /opt/nab
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import rrcf

LAG = 8
WARM_FRAC = 0.15
TREES = 100
SAMPLE = 256


def load_csv(path):
    with open(path) as fh:
        fh.readline()
        rows = [line.rstrip().split(",") for line in fh]
    ts = [r[0][:19] for r in rows]
    vals = np.asarray([float(r[1]) for r in rows])
    return ts, vals


def labels_from_windows(timestamps, windows):
    out = np.zeros(len(timestamps), dtype=np.int8)
    for s, e in windows:
        s19 = s[:19]
        e19 = e[:19]
        for i, t in enumerate(timestamps):
            if s19 <= t <= e19:
                out[i] = 1
    return out


def lag_embed(values, lag):
    n = len(values)
    emb = np.lib.stride_tricks.sliding_window_view(values, lag)
    # emb shape: (n - lag + 1, lag). Row i corresponds to values[i:i+lag].
    return emb


def auc(scores, labels):
    order = np.argsort(-scores)
    y = labels[order]
    tp = 0
    fp = 0
    total_pos = int(y.sum())
    total_neg = len(y) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5
    auc_val = 0.0
    prev_tpr = 0.0
    prev_fpr = 0.0
    for yi in y:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        auc_val += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr = tpr
        prev_fpr = fpr
    return auc_val


def score_file(timestamps, values, windows):
    if len(values) < 2 * LAG:
        return None
    emb = lag_embed(values, LAG)
    embed_ts = timestamps[LAG - 1 :]
    labels = labels_from_windows(embed_ts, windows)
    warm_end = int(len(emb) * WARM_FRAC)

    forest = []
    index = 0
    for _ in range(TREES):
        forest.append(rrcf.RCTree())

    # Warm — feed first slice, no scoring.
    for p in emb[:warm_end]:
        for t in forest:
            if len(t.leaves) >= SAMPLE:
                t.forget_point(index - SAMPLE)
            t.insert_point(p, index=index)
        index += 1

    # Score — frozen baseline.
    scores = np.zeros(len(emb) - warm_end)
    for idx, p in enumerate(emb[warm_end:]):
        codisp = 0.0
        # rrcf codisp requires the point to be in the tree. Insert
        # then remove on each scoring step to mimic a probe.
        for t in forest:
            t.insert_point(p, index=-1)
            codisp += t.codisp(-1)
            t.forget_point(-1)
        scores[idx] = codisp / TREES
    return auc(scores, labels[warm_end:]), int(labels[warm_end:].sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nab", required=True, help="path to cloned NAB")
    args = parser.parse_args()

    nab = Path(args.nab)
    data_dir = nab / "data/realKnownCause"
    windows = json.load(open(nab / "labels/combined_windows.json"))

    files = sorted(p for p in data_dir.glob("*.csv"))
    weighted_sum = 0.0
    total = 0
    for csv in files:
        key = f"realKnownCause/{csv.name}"
        ts, vals = load_csv(csv)
        w = windows.get(key, [])
        result = score_file(ts, vals, w)
        if result is None:
            continue
        a, pos = result
        print(f"  {a:.3f}  pos={pos:<6}  {csv.name}")
        weighted_sum += a * pos
        total += pos
    if total > 0:
        print(f"aggregate weighted AUC: {weighted_sum / total:.3f}")


if __name__ == "__main__":
    sys.exit(main())
