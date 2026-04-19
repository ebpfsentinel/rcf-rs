# External bench — rcf-rs vs Python / Java baselines

Reproducible speed + AUC comparison between `rcf-rs` and three
published reference implementations:

- [`rrcf`](https://github.com/kLabUM/rrcf) 0.4.4 — Python + NumPy,
  the original open-source RCF port.
- [`scikit-learn` `IsolationForest`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
  — not RCF, but the canonical streaming-friendly tree-isolation
  baseline every comparison pins against.
- AWS's [`randomcutforest-java`](https://github.com/aws/random-cut-forest-by-aws)
  4.4.0 — JVM reference. See `../README.md`.

Scripts live outside the Rust crate by design — they pull in
Python / JVM toolchains, aren't CI-green, and produce numbers
that are only meaningful on the dev box they run on.

## Layout

- `gen_points.py` — deterministic CSV generator shared by every
  runner. First `n_normal` rows are clean (warm-up), last
  `n_outliers` rows are the anomaly probes.
- `bench_rrcf_synthetic.py` — rrcf warm + `codisp` score loop.
- `bench_sklearn_synthetic.py` — sklearn `IsolationForest` fit +
  `decision_function`.
- `RcfBenchSynthetic.java` — AWS Java driver; see
  `../README.md` for the Maven Central jar path.

The rcf-rs side is an ordinary crate example:
`examples/external_bench_driver.rs` — invoked via `cargo run`.

## Running

From the `rcf-rs` crate root, JDK 26 + Python 3.13 on a machine
with `rrcf` and `scikit-learn` installed:

```bash
# Shared dataset: 10 000 pts, D=16, seed 2026, 1 % outliers.
python3 scripts/synthetic/gen_points.py \
    --n 10000 --dim 16 --seed 2026 > data.csv

# Python baselines.
pip install --user rrcf scikit-learn numpy
python3 scripts/synthetic/bench_rrcf_synthetic.py \
    --input data.csv --trees 100 --sample 256
python3 scripts/synthetic/bench_sklearn_synthetic.py \
    --input data.csv --trees 100 --train-frac 0.3

# rcf-rs.
cargo run --release --example external_bench_driver -- \
    data.csv 100 256

# AWS Java (see ../README.md for the jar).
JAR=/tmp/aws-rcf/randomcutforest-core-4.4.0.jar
javac -cp "$JAR" scripts/synthetic/RcfBenchSynthetic.java
java -cp "scripts/synthetic:$JAR" RcfBenchSynthetic \
    data.csv 100 256
```

## Measured numbers (i7-1370P, synthetic 10k × D=16, 1 % outliers)

**5-seed variance** (seeds 2026–2030), mean ± stddev,
coefficient of variation in parens. Driven by
`variance_sweep.sh`.

| Impl | Backend | Updates / s | Scores / s | AUC |
|---|---|---|---|---|
| `rcf-rs` 0.0.0-dev | Rust, rayon-parallel | **17 500 ± 1 240** (7 %) | 125 900 ± 1 840 (1.5 %) | 1.000 ± 0 |
| `randomcutforest-java` 4.4.0 | JVM 26, cold | 2 090 ± 134 (6 %) | 8 870 ± 415 (5 %) | 1.000 ± 0 |
| `rrcf` 0.4.4 | Python + NumPy | 73 ± 3 (4 %) | 94 150 ± 4 840 (5 %) | 0.992 ± 0 |
| `sklearn.IsolationForest` | NumPy + Cython | batch-only | **136 300 ± 2 450** (2 %) | 1.000 ± 0 |

Ratios (mean / mean):

- **Updates**: `rcf-rs` is ~8.4× faster than AWS Java, ~240×
  faster than `rrcf`. CVs around 5-7 %; ratios sit well outside
  the noise floor.
- **Scores**: sklearn edges `rcf-rs` by 8 % (136 k vs 126 k) —
  real but small (stddevs combined ≈ 3 k, so the 10 k delta is
  ~3σ significant). `rrcf` trails `rcf-rs` by ~25 %; AWS Java
  trails by ~14×.
- **AUC**: identical within measurement precision across every
  seed (0.992 for `rrcf`, 1.000 for the other three).

## Caveats

Machine thermal state varies across runs — earlier single-seed
cool-CPU measurements landed at ~32 k / 203 k for `rcf-rs`,
dropping to the ~17 k / 126 k above on the 5-seed run. The
**ratios are portable, the absolute numbers aren't**.

The Python scripts are best-effort one-shot harnesses — they
don't pin NumPy BLAS threads, don't warm up, don't stabilise
CPU frequency. Treat absolute numbers as "same hardware,
order-of-magnitude" only.

## Why the Python runners are single-process

- **`rrcf`** — `codisp` scoring mutates the tree on every probe
  (`insert_point(index=-1)` → `codisp(-1)` → `forget_point(-1)`),
  so threads collide on the shared `-1` slot and trip
  `AssertionError: index in leaves`. Multiprocessing fails at
  pickle time — tree objects hold module references that
  `pickle` rejects (`cannot pickle 'module' object`). The
  measured 184k scores/s is `rrcf`'s single-process ceiling;
  NumPy SIMD inside `codisp` already saturates.
- **sklearn `IsolationForest`** — `n_jobs=-1` was tested and
  regresses at 100 trees × 10k points (joblib task-spawn
  overhead exceeds the split-tree win). The default
  single-threaded BLAS SIMD path is the faster one for this
  batch size.
- **rcf-rs** uses rayon-parallel `score_many`; the table above
  compares each impl on its respective maximum-throughput
  entry point.
