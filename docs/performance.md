# Performance

Criterion (`cargo bench`) — wall-clock mean point estimate,
`mimalloc` pinned globally. Two bench files:

- `benches/forest_throughput.rs` — core ops (insert, score,
  attribution).
- `benches/extended.rs` — bulk, early-term, forensic, tenant.

```bash
cargo bench                                            # full
cargo bench -- --sample-size 10 --measurement-time 2   # quick
```

## Reference hardware

| | |
|---|---|
| CPU | Intel Core i7-1370P (13th gen), 14C/20T, L3 24 MiB |
| Memory | 32 GB DDR5 |
| Kernel | Linux 6.17 |
| Allocator | mimalloc 0.1 (pinned in bench harness) |
| Compiler | rustc 1.95 stable |

Absolute values scale with CPU / memory bandwidth; *ratios*
(parallel speedup, early-term savings, tenant scaling) are the
portable signal.

## Caveats

- **Cross-group variance**: `b.iter()` mutates a persistent
  forest and criterion chooses batch sizes per-op, so reservoir
  state + per-iter overhead drift across bench groups. Trust
  ratios within a group, not absolute numbers across groups.
- **Parallel ceiling**: `score_many` plateaus at ~6× speedup on
  a 14-core host — memory-bandwidth-bound past L3 working set.

## Core ops

`(trees=100, sample=256, D=16)` after split-typed-arena refactor
(persistence v4):

| Workload | Time | Throughput |
|---|---|---|
| `forest_update` | ~23 µs | 43k/s |
| `forest_score` | ~23 µs | 43k/s |
| `forest_attribution` | ~31 µs | 32k/s |

Refactor delta vs pre-v4 at the same config: update −28 %,
score −10 %, attribution −37 %. Leaf-arena memory −90 %
(~320 B → ~40 B per slot).

Other `(trees, samples, D)` tuples: criterion HTML report
(`target/criterion/`).

## Bulk batch scoring

`D=16`, forest `(100, 256)`:

| Batch size | `score_many` (parallel) | Serial loop | Speedup |
|---|---|---|---|
| 64 | 440 µs | 2.19 ms | 5.0× |
| 512 | 3.17 ms | 19.5 ms | 6.1× |
| 4096 | 24.1 ms | 146 ms | 6.0× |

## Early-termination

`D=16`, forest `(100, 256)`, single probe:

| Path | Time |
|---|---|
| `score` (parallel ensemble) | 36 µs |
| `score_early_term` threshold=0.02 (tight) | 59 µs |
| `score_early_term` threshold=0.20 (loose, stops ~20 trees) | 8.4 µs |

Loose threshold → 4.3× speedup on baseline-dominated traffic;
tight threshold loses to parallel `score` (sequential walk
rarely short-circuits).

## Forensic baseline

`forensic_baseline`:

| `(trees, samples, D)` | Time |
|---|---|
| `(100, 256, 4)` | 68 µs |
| `(100, 256, 16)` | 79 µs |
| `(100, 1024, 16)` | 315 µs |

Cost ≈ `O(live_points × D)` Welford sweep — `sample_size` ×4
→ ×4 time, dim cost marginal.

## Tenant pool at scale

`tenant_pool`, each tenant `D=4` / `(50, 64)`, warmed 128 samples:

| N | `similarity_matrix` | `score_across_tenants` | `most_similar_top5` |
|---|---|---|---|
| 32 | 48 µs | 136 µs | 0.70 µs |
| 128 | 131 µs | 456 µs | 2.2 µs |
| 512 | 1.48 ms | 6.69 ms | 9.1 µs |

Scaling `N=32→512` (16× tenants):
- `similarity_matrix` O(N²) parallelised: 31× (not 256× —
  rayon fan-out hides quadratic until core saturation).
- `score_across_tenants` O(N) parallelised: 49×.
- `most_similar_top5` O(N·log k) bounded heap: 13×.

## External baselines (synthetic)

Input: 10k points, `D=16`, 1 % outliers, 30 % warm / 70 % eval,
frozen baseline. Each impl on its idiomatic fast path
(rcf-rs rayon / rrcf single-process / sklearn NumPy-Cython SIMD
/ AWS Java cold JVM). **5-seed variance** (seeds 2026–2030),
mean ± stddev, coefficient of variation in parens.

| Impl | Backend | Updates/s | Scores/s | AUC |
|---|---|---|---|---|
| `rcf-rs` 0.0.0-dev | Rust, rayon-parallel | **17 500 ± 1 240** (7 %) | 125 900 ± 1 840 (1.5 %) | 1.000 ± 0 |
| `randomcutforest-java` 4.4.0 | JVM 26, cold | 2 090 ± 134 (6 %) | 8 870 ± 415 (5 %) | 1.000 ± 0 |
| `rrcf` 0.4.4 | Python + NumPy | 73 ± 3 (4 %) | 94 150 ± 4 840 (5 %) | 0.992 ± 0 |
| `sklearn.IsolationForest` | NumPy + Cython | batch-only | **136 300 ± 2 450** (2 %) | 1.000 ± 0 |

Ratios (mean/mean):

- **Updates**: rcf-rs is ~8.4× faster than AWS Java, ~240× faster
  than rrcf. CVs around 5-7 % on all impls; the ratios sit well
  outside the noise floor.
- **Scores**: sklearn edges rcf-rs by 8 % (136k vs 126k) — a
  real but small gap (stddevs combined ≈ 3k, so the 10k delta
  is ~3σ significant). rrcf trails rcf-rs by ~25 %; AWS Java
  trails by ~14×.
- **AUC**: identical within measurement precision across every
  seed (0.992 for rrcf, 1.000 for the other three).

Noise sources documented: machine thermal state varies across
runs — single-seed numbers from an earlier cool-CPU session
landed at ~32k/203k for rcf-rs, dropping to ~17k/126k on this
run. The **ratios are portable, the absolute numbers aren't**.

Reproduce the sweep:

```bash
scripts/external-bench/variance_sweep.sh /tmp/aws-rcf/randomcutforest-core-4.4.0.jar
```

## Detection quality — NAB `realKnownCause`

Two scoring APIs, two use cases:

- **`RandomCutForest::score()`** — isolation-depth, never
  mutates the forest, rayon-parallel, eBPF-hot-path friendly.
  On NAB: **0.719** aggregate AUC after the lag=32 + zscore +
  smooth(0.02) pipeline.
- **`RandomCutForest::score_codisp()`** — probe-based (insert,
  walk leaf→root accumulating `max(sibling.mass /
  subtree.mass)`, remove). Matches rrcf / AWS Java scoring
  semantic. ~30× slower; intended for SOC triage / forensic
  replay. On NAB: **0.776** aggregate AUC, beats both rrcf
  and AWS Java.

Same embedding pipeline (32-lag → warm-phase z-score → EMA
α = 0.02), 15 % warm, 100 trees × 256 sample.

| File | rcf-rs `score()` | rcf-rs `score_codisp()` | rrcf | AWS Java |
|---|---|---|---|---|
| `ambient_temperature_system_failure` | 0.813 | 0.813 | 0.734 | 0.786 |
| `cpu_utilization_asg_misconfiguration` | 0.953 | 0.939 | 0.849 | 0.906 |
| `ec2_request_latency_system_failure` | 0.709 | 0.739 | 0.481 | 0.482 |
| `machine_temperature_system_failure` | 0.578 | 0.666 | 0.880 | 0.883 |
| `nyc_taxi` | 0.698 | 0.721 | 0.571 | 0.540 |
| `rogue_agent_key_hold` | 0.145 | 0.692 | 0.535 | 0.633 |
| `rogue_agent_key_updown` | 0.633 | 0.721 | 0.657 | 0.542 |
| **weighted aggregate** | 0.719 | **0.776** | 0.748 | 0.757 |

### Hyperparameter ablation

`examples/nab_ablation.rs` on the same corpus:

| Config | Aggregate AUC |
|---|---|
| baseline (lag=8, raw score) | 0.615 |
| lag=32 | 0.665 |
| lag=32 + diff | 0.640 |
| lag=32 + zscore | 0.683 |
| lag=32 + smooth(0.1) | 0.687 |
| lag=32 + zscore + smooth(0.05) | 0.718 |
| **lag=32 + zscore + smooth(0.02)** | **0.719** |
| lag=64 + zscore + smooth(0.05) | 0.672 |
| trcf-online D=32 | 0.320 |
| probe-score D=8 (naive hack) | 0.330 |
| **codisp D=32 + zscore + smooth(0.02)** | **0.776** |

- **Longer embedding** (lag=32) captures wider context — +0.050.
- **Z-score per warm-phase dim stddev** compensates NAB's scale
  variance (CPU % vs taxi counts vs temperatures) — RCF cuts are
  range-weighted, so un-normalised inputs let one dim dominate.
  +0.018 on top of lag=32.
- **EMA smoothing of raw scores** (α ≈ 0.02, half-life ~35 steps)
  cuts per-point noise without losing the wide-window shape.
  +0.036 on top of lag=32 + zscore.
- **Differencing** (first-diff of lag values) regresses — NAB's
  contextual signal lives in absolute values, not rate-of-change.
- **TRCF online** regresses catastrophically (0.72 → 0.32): the
  EMA threshold adapts UP during the multi-day anomaly windows
  and stops flagging them. Frozen baseline is the right paradigm
  for NAB's wide-window labels.
- **Probe-based naive hack** (`update_indexed → score → delete`)
  tanks AUC (0.330) — post-insert `score` ranks the freshly-
  inserted probe as seen. Proper codisp is `score_codisp()`:
  walks from inserted leaf → root accumulating
  `max(sibling.mass / subtree.mass)`, then deletes the probe.

### Two-API split

`score()` (isolation-depth, non-mutating, parallel) and
`score_codisp()` (probe-based, mutating, sequential per tree)
serve different use cases. `score()` is the eBPF hot-path
default; `score_codisp()` is for SOC triage / forensic replay
where the extra ~30× latency is acceptable for the +0.057 AUC
gain. On NAB, `score_codisp()` (0.776) leads both rrcf (0.748)
and AWS Java (0.757).

- `tests/detection_quality.rs` pins synthetic-corpus regression
  guards: AUC > 0.95 on separable clusters, > 0.90 on transition.
- `tests/nab.rs` pins NAB aggregate floor at 0.70.

Reproduce:

```bash
git clone --depth 1 https://github.com/numenta/NAB.git /opt/nab
RCF_NAB_PATH=/opt/nab cargo test --test nab --all-features -- --ignored --nocapture
python3 scripts/nab/bench_rrcf_nab.py --nab /opt/nab
java -cp ".:/tmp/aws-rcf/randomcutforest-core-4.4.0.jar" RcfBenchNab /opt/nab
```
