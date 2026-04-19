#!/usr/bin/env python3
"""Run `rrcf` on the TSB-AD multivariate track and report per-file
+ per-dataset + aggregate AUC. Same protocol as
`tests/tsb_ad_m.rs`: per-dim z-score on the upstream `tr_<N>`
train split, frozen-baseline codisp scoring, EMA smoothing.

Runtime caveat: rrcf codisp inserts + removes a probe in each
tree per scored point. A 200k-row file × 100 trees is
prohibitive, so we cap the eval stream to `--max-eval` rows with
uniform stride downsampling. Reservoir warm-up stays full
to preserve the baseline.

Usage:
    scripts/tsb_ad/fetch.sh /tmp/tsb-ad
    python3 scripts/tsb_ad/bench_rrcf_tsb_ad_m.py \
        --dir /tmp/tsb-ad/TSB-AD-M
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rrcf

TREES = 100
SAMPLE = 256
SMOOTH_ALPHA = 0.02
MIN_POSITIVES = 5
FILENAME_TR = re.compile(r"_tr_(\d+)_")


def parse_meta(path):
    m = FILENAME_TR.search(path.name)
    if not m:
        return None
    train_end = int(m.group(1))
    parts = path.stem.split("_")
    dataset = parts[1] if len(parts) > 1 else "unknown"
    return dataset, train_end


def load_csv(path):
    with open(path) as fh:
        header = fh.readline().rstrip()
        dim = header.count(",")
        data = []
        labels = []
        for line in fh:
            cols = line.rstrip().split(",")
            data.append([float(c) for c in cols[:dim]])
            labels.append(int(float(cols[dim]) >= 0.5))
    return np.asarray(data, dtype=np.float64), np.asarray(labels, dtype=np.int8), dim


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


def stride_subsample(indices, max_rows):
    if len(indices) <= max_rows:
        return indices
    stride = max(1, len(indices) // max_rows)
    return indices[::stride][:max_rows]


def score_file(data, labels, dim, train_end, max_eval):
    n = len(labels)
    if n <= train_end + 1:
        return None
    means = data[:train_end].mean(axis=0)
    stds = data[:train_end].std(axis=0)
    stds = np.where(stds < 1e-9, 1e-9, stds)
    z = (data - means) / stds

    forest = [rrcf.RCTree() for _ in range(TREES)]
    index = 0
    for p in z[:train_end]:
        for t in forest:
            if len(t.leaves) >= SAMPLE:
                t.forget_point(index - SAMPLE)
            t.insert_point(p, index=index)
        index += 1

    eval_idx = np.arange(train_end, n)
    eval_idx = stride_subsample(eval_idx, max_eval)
    raw_scores = np.zeros(len(eval_idx))
    eval_labels = labels[eval_idx]
    for out_i, src_i in enumerate(eval_idx):
        p = z[src_i]
        codisp = 0.0
        for t in forest:
            t.insert_point(p, index=-1)
            codisp += t.codisp(-1)
            t.forget_point(-1)
        raw_scores[out_i] = codisp / TREES

    smoothed = np.empty_like(raw_scores)
    if len(raw_scores) > 0:
        acc = raw_scores[0]
        for i, s in enumerate(raw_scores):
            acc = SMOOTH_ALPHA * s + (1.0 - SMOOTH_ALPHA) * acc
            smoothed[i] = acc

    pos = int(eval_labels.sum())
    return auc(smoothed, eval_labels), pos


def _one_file(csv_path_str, max_eval, skip_dim_above):
    csv = Path(csv_path_str)
    meta = parse_meta(csv)
    if meta is None:
        return None
    dataset, train_end = meta
    data, labels, dim = load_csv(csv)
    if dim > skip_dim_above:
        return (csv.name, dataset, dim, None, None, True)
    result = score_file(data, labels, dim, train_end, max_eval)
    if result is None:
        return (csv.name, dataset, dim, None, None, False)
    a, pos = result
    return (csv.name, dataset, dim, a, pos, False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="path to extracted TSB-AD-M dir")
    parser.add_argument(
        "--max-eval",
        type=int,
        default=5000,
        help="cap eval rows per file (uniform stride subsample above this)",
    )
    parser.add_argument(
        "--skip-dim-above",
        type=int,
        default=66,
        help="skip files with feature dim strictly greater than this",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="parallel worker processes (rrcf cannot thread due to GIL + tree mutation)",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    files = sorted(root.glob("*.csv"))
    if not files:
        print(f"no CSVs under {root}", file=sys.stderr)
        sys.exit(3)

    per_dataset = defaultdict(lambda: [0.0, 0, 0])
    weighted_sum = 0.0
    weighted_pos = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_one_file, str(p), args.max_eval, args.skip_dim_above)
            for p in files
        ]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            name, dataset, dim, a, pos, was_skipped = res
            if was_skipped:
                skipped += 1
                continue
            if a is None:
                continue
            print(f"  {a:.3f}  D={dim:<3}  pos={pos:<7}  {dataset:<12}  {name}", flush=True)
            if pos >= MIN_POSITIVES:
                per_dataset[dataset][0] += a * pos
                per_dataset[dataset][1] += pos
                per_dataset[dataset][2] += 1
                weighted_sum += a * pos
                weighted_pos += pos

    print("\nper-dataset AUC (weighted by positives):")
    for ds in sorted(per_dataset):
        s, p, f = per_dataset[ds]
        if p == 0:
            continue
        print(f"  {s / p:.3f}  files={f:<3}  pos={p:<7}  {ds}")
    if weighted_pos > 0:
        print(
            f"\naggregate weighted AUC: {weighted_sum / weighted_pos:.3f} "
            f"across {sum(v[2] for v in per_dataset.values())} files / "
            f"{weighted_pos} positives (rrcf, max-eval={args.max_eval})"
        )
    if skipped:
        print(f"skipped {skipped} file(s) with D > {args.skip_dim_above}")


if __name__ == "__main__":
    sys.exit(main())
