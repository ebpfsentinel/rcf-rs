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

Input: 10k points, `D=16`, 1 % outliers, seed 2026, 30 % warm
/ 70 % eval, frozen baseline. Each impl on its idiomatic fast
path (rcf-rs rayon / rrcf single-process / sklearn
NumPy-Cython SIMD / AWS Java cold JVM).

| Impl | Backend | Updates/s | Scores/s | AUC |
|---|---|---|---|---|
| `rcf-rs` 0.0.0-dev | Rust, rayon-parallel | **32.5k** | **203k** | 1.000 |
| `randomcutforest-java` 4.4.0 | JVM 26, cold | 3.9k | 21k | 1.000 |
| `rrcf` 0.4.4 | Python + NumPy | 0.15k | 184k | 0.992 |
| `sklearn.IsolationForest` | NumPy + Cython | batch ≈ 48k/s | 234k | 1.000 |

- rcf-rs inserts 10× faster than AWS Java, 220× faster than rrcf.
- Score throughput within 15 % across all four on fast paths.
- sklearn `IsolationForest` is batch-only (no streaming update).
- rrcf parallelism unusable (thread-unsafe `codisp`, unpicklable
  trees — see `scripts/external-bench/README.md`).

Reproduce:

```bash
python3 scripts/external-bench/gen_points.py --n 10000 --dim 16 --seed 2026 > data.csv
python3 scripts/external-bench/bench_rrcf.py --input data.csv --trees 100 --sample 256
python3 scripts/external-bench/bench_sklearn_iforest.py --input data.csv --trees 100 --train-frac 0.3
cargo run --release --example external_bench_driver -- data.csv 100 256
# AWS Java — see scripts/external-bench/README-aws-java.md
```

## Detection quality — NAB `realKnownCause`

Protocol: 32-lag temporal embedding, 15 % warm, 100 trees × 256
sample, frozen baseline, AUC via trapezoidal rule against
`combined_windows.json`.

| File | rcf-rs | rrcf | AWS Java |
|---|---|---|---|
| `ambient_temperature_system_failure` | 0.702 | 0.734 | **0.786** |
| `cpu_utilization_asg_misconfiguration` | 0.812 | 0.849 | **0.906** |
| `ec2_request_latency_system_failure` | **0.601** | 0.481 | 0.482 |
| `machine_temperature_system_failure` | 0.578 | 0.880 | **0.883** |
| `nyc_taxi` | **0.701** | 0.571 | 0.540 |
| `rogue_agent_key_hold` | 0.274 | 0.535 | **0.633** |
| `rogue_agent_key_updown` | **0.585** | 0.657 | 0.542 |
| **weighted aggregate** | 0.665 | 0.748 | **0.757** |

### Hyperparameter ablation (baseline = lag=8)

`examples/nab_ablation.rs` on the same corpus:

| Config | Aggregate AUC | vs baseline |
|---|---|---|
| baseline (lag=8) | 0.615 | — |
| lag=16 | 0.650 | +0.035 |
| **lag=32** | **0.665** | **+0.050** |
| trees=200 | 0.611 | −0.004 |
| sample=512 | 0.582 | −0.033 |
| iaf=0.125 | 0.618 | +0.003 |
| warm=0.30 | 0.586 | −0.029 |

- Longer embedding is the one free win — more temporal context
  absorbs NAB's wide contextual shifts.
- Bigger capacity (trees=200, sample=512) regresses: larger
  reservoir swallows the outliers into the baseline faster.
- rrcf and AWS Java still top rcf-rs by 0.08–0.09 points
  because both use **probe-based scoring** (insert probe →
  query displacement → remove). rcf-rs's isolation-depth
  `score()` never mutates the forest, is ~18× faster per probe.
  A naive `update_indexed → score → delete` approximation was
  tested — it tanks AUC to 0.33 (the post-insert `score` path
  ranks the probe as seen → low anomaly). Proper `codisp`
  requires walking from the inserted leaf back to root and
  summing `sibling_mass / subtree_size` per level — not
  reachable through the current public API.
- `tests/detection_quality.rs` pins synthetic-corpus regression
  guards: AUC > 0.95 on separable clusters, > 0.90 on
  transition anomalies.
- `tests/nab.rs` pins NAB aggregate floor at 0.60.

Reproduce:

```bash
git clone --depth 1 https://github.com/numenta/NAB.git /opt/nab
RCF_NAB_PATH=/opt/nab cargo test --test nab --all-features -- --ignored --nocapture
python3 scripts/nab/bench_rrcf_nab.py --nab /opt/nab
java -cp ".:/tmp/aws-rcf/randomcutforest-core-4.4.0.jar" RcfBenchNab /opt/nab
```
