# Performance

Criterion benches (`cargo bench`), wall-clock mean point estimate
on `x86_64` with `mimalloc` pinned globally. Two bench files:

- `benches/forest_throughput.rs` — core ops (insert, score,
  attribution) across the `(trees, samples, D)` matrix.
- `benches/extended.rs` — value-add APIs: bulk, early-term,
  forensic, tenant.

Quick run with smaller sample: `cargo bench -- --sample-size 10
--warm-up-time 1 --measurement-time 2`. Full run (default
criterion config): `cargo bench`.

## Reference hardware

The numbers below were captured on:

- **CPU**: Intel Core i7-1370P (13th gen) —
  14 cores / 20 threads, L3 = 24 MiB
- **Memory**: 32 GB DDR5
- **Kernel**: Linux 6.17
- **Allocator**: mimalloc 0.1 pinned globally in the bench harness
- **Compiler**: rustc 1.95 stable

Absolute numbers scale with CPU generation / frequency /
memory-bandwidth — the *ratios* between ops (parallel speedup,
early-term savings, tenant-count scaling) are the portable signal.
Re-run on target hardware before committing SLO budgets.

## Measurement methodology caveats

- **Cross-group variance**: do not compare absolute numbers across
  benches that run at different points of the `cargo bench` run.
  Each bench function mutates a persistent forest through its
  `b.iter()` body, and criterion chooses batch sizes based on
  per-op cost — so the reservoir state + per-iter overhead drift
  between groups. Trust *ratios* inside a group; suspect
  cross-group comparisons.
- **Parallel ceiling**: `score_many` plateaus at ~6× speedup on
  a 14-core host. Per-point work is memory-bandwidth-bound once
  the cache working set exceeds L3; more cores do not help past
  that point. Known target for future arena-layout work.
- **No external comparison yet**: no side-by-side vs AWS's
  `randomcutforest-java`, `rrcf` (Python), or Isolation Forest
  baselines. Tracked under future work.
- **No external-dataset detection-quality measurement here**: this
  file measures speed. Detection quality on public corpora
  (`NAB` / Yahoo S5 / Numenta) is not covered; see Future work.
  `tests/detection_quality.rs` does report **AUC**, **score
  separation ratio**, and **precision / recall at top-K** on
  synthetic ground-truth streams (cluster + outliers, transition
  anomalies) — regression-guards the core quality claim and pins
  AUC > 0.95 on separable data, > 0.90 on transition data.

## Core ops (`forest_throughput`)

Post split-typed-arena refactor (persistence v4) — see the
**Arena split** entry in *Done* below.

| Workload | `(trees, samples, D)` | Time |
|---|---|---|
| `forest_update` | `(100, 256, 16)` | ~23 µs |
| `forest_score` | `(100, 256, 16)` | ~23 µs |
| `forest_attribution` | `(100, 256, 16)` | ~31 µs |

At `(100, 256, 16)`: ~43k inserts/s and ~43k scores/s
single-thread-equivalent.

Measured via `cargo bench --bench forest_throughput`. Absolute
values under the other `(trees, samples, D)` tuples are available
in the criterion HTML report (`target/criterion/`) — not pinned
here since cross-bench variance on the laptop-class reference
hardware exceeds the typical row-to-row delta (see
*Measurement methodology caveats* above).

## Bulk batch scoring

`bulk_scoring` bench group, `D=16`, forest `(100, 256)`, batches
of random probes:

| Batch size | `score_many` (par) | Serial for-loop | Speedup |
|---|---|---|---|
| 64 | 439.64 µs | 2.19 ms | 5.0× |
| 512 | 3.17 ms | 19.48 ms | 6.1× |
| 4096 | 24.14 ms | 145.81 ms | 6.0× |

Speedup saturates around 6× as rayon task-scheduling amortises
then the ceiling is set by per-probe memory bandwidth.

## Early-termination scoring

`early_term` bench group, `D=16`, forest `(100, 256)`, single
probe:

| Path | Time |
|---|---|
| `score` (full parallel ensemble) | 36.21 µs |
| `score_early_term`, `threshold=0.02` (tight, rarely stops) | 58.73 µs |
| `score_early_term`, `threshold=0.20` (loose, stops ~20 trees) | 8.41 µs |

Tight threshold is slower than plain `score` because it walks
trees sequentially and rarely short-circuits — the parallel
ensemble wins when ambiguity forces a full traversal. Loose
threshold gives a **~4.3× speedup** on baseline-dominated traffic
where most points stop early.

## Forensic baseline

`forensic_baseline` bench group, `D` and `sample_size` swept:

| `(trees, samples, D)` | Time |
|---|---|
| `(100, 256, 4)` | 68.30 µs |
| `(100, 256, 16)` | 78.55 µs |
| `(100, 1024, 16)` | 315.07 µs |

Cost is dominated by the `O(live_points × D)` Welford sweep over
the union of tenant reservoirs. Quadrupling `sample_size` → ~4×
slower. Per-dim cost is marginal vs. the iteration overhead.

## External baselines

`scripts/external-bench/` wires `rcf-rs`, `rrcf` (Python/NumPy)
and scikit-learn's `IsolationForest` on identical input:
10 000 points, `D = 16`, 1 % outlier fraction, seed 2026.
Warm phase = 30 % (clean normal only), eval phase = 70 % (mixed).
Frozen-baseline scoring — no updates during eval.

Throughput normalised to each implementation's idiomatic fast
path: `rcf-rs` uses the parallel `score_many` (rayon across
points + trees); `rrcf` runs its native Python loop; `sklearn`
uses NumPy/Cython SIMD on a batch `decision_function` call.
`n_jobs=-1` on `IsolationForest` regresses at this batch size
(joblib task-spawn overhead) so the default single-threaded
path is reported — it already saturates on vectorised SIMD.

| Impl | Backend | Updates / s | Scores / s | AUC |
|---|---|---|---|---|
| `rcf-rs` 0.0.0-dev | native Rust, rayon-parallel | **32.5k** | **203k** | 1.000 |
| `rrcf` 0.4.4 | pure Python + NumPy | 0.15k | 184k | 0.992 |
| `sklearn.IsolationForest` | NumPy + Cython, batch-fit | batch-fit ≈ 48k/s | 234k | 1.000 |

Commentary:
- **Streaming updates**: `rcf-rs` inserts ~220× faster than
  `rrcf`. `rrcf`'s per-insert cost is dominated by Python object
  churn on every `insert_point` call. `sklearn`'s IsolationForest
  is batch-only — no streaming-update API — so its "Updates/s"
  entry is the batch `.fit()` cost amortised per training point.
- **Score throughput**: all three land in the same order of
  magnitude once parallelism / vectorisation are on.
  `sklearn` edges out on pure score-batch because its per-point
  decision is a BLAS-vectorised matrix walk; `rcf-rs`'s
  `score_many` stays competitive and is within 15 % of the
  sklearn batch number.
- **Detection quality**: all three sit within 1 % of perfect on
  the synthetic separable corpus — as expected; the interesting
  comparison is NAB below.

Re-run the matrix on your hardware with:

```bash
python3 scripts/external-bench/gen_points.py --n 10000 --dim 16 --seed 2026 > data.csv
python3 scripts/external-bench/bench_rrcf.py --input data.csv --trees 100 --sample 256
python3 scripts/external-bench/bench_sklearn_iforest.py --input data.csv --trees 100 --train-frac 0.3
./scripts/external-bench/run_rcf_rs.sh data.csv 100 256
```

## Detection quality on public corpora (NAB `realKnownCause`)

Same embedding protocol for both implementations: 8-lag temporal
embedding, 15 % warm fraction, 100 trees × 256 sample. Scoring
semantics differ slightly (see commentary below) — that is itself
the interesting data point.

| File | `rcf-rs` AUC | `rrcf` AUC |
|---|---|---|
| `ambient_temperature_system_failure` | 0.604 | **0.734** |
| `cpu_utilization_asg_misconfiguration` | 0.749 | **0.849** |
| `ec2_request_latency_system_failure` | **0.525** | 0.481 |
| `machine_temperature_system_failure` | 0.584 | **0.880** |
| `nyc_taxi` | **0.588** | 0.571 |
| `rogue_agent_key_hold` | 0.379 | **0.535** |
| `rogue_agent_key_updown` | 0.544 | **0.657** |
| **weighted aggregate** | 0.615 | **0.748** |

Commentary:

- `rrcf` wins on 5 / 7 files and the weighted aggregate. The gap
  (~13 absolute points) is driven by scoring semantics: `rrcf`'s
  `codisp` temporarily inserts the probe into every tree and
  queries its *collusive displacement* (how much mass would move
  if the point were removed). `rcf-rs`'s `score()` walks the
  existing tree and scores by expected isolation depth — never
  mutates the forest. Both are valid RCF scoring conventions;
  `codisp` is closer to the "AWS paper" reference semantic and
  trades throughput (insert-then-remove per probe = ~18× slower
  at scoring) for accuracy on context-sensitive anomalies.
- NAB is not RCF's strongest home turf — anomaly windows are
  wide contextual shifts where time-aware detectors (HTM, LSTM)
  typically land in the 0.75–0.85 range. Both `rcf-rs` and
  `rrcf` are within the published RCF ballpark for the corpus.
- **Action item**: wiring a `codisp`-style scoring path on top of
  `rcf-rs` is a tracked future item — the current `score()` API
  is frozen for the 0.1 release. Expected gain on NAB ≈ +0.10
  aggregate AUC based on the `rrcf` delta.

Run both via:

```bash
./scripts/nab/fetch.sh /opt/nab
RCF_NAB_PATH=/opt/nab \
    cargo test --test nab --all-features -- --ignored --nocapture
python3 scripts/nab/bench_rrcf_nab.py --nab /opt/nab
```

## Tenant pool at scale

`tenant_pool` bench group, each tenant `D=4` / `(50, 64)`, warmed
with 128 samples:

| N tenants | `similarity_matrix` | `score_across_tenants` | `most_similar_top5` |
|---|---|---|---|
| 32 | 48.16 µs | 135.61 µs | 698.78 ns |
| 128 | 131.26 µs | 455.59 µs | 2.24 µs |
| 512 | 1.48 ms | 6.69 ms | 9.06 µs |

Observations:
- `similarity_matrix` is `O(N²)` on EMA-stat pairs, parallelised
  via rayon — N=32→512 gives ~31× (not 256×) because the parallel
  fan-out hides the quadratic cost up to core-count saturation.
- `score_across_tenants` is `O(N)` — one `score_only` per tenant,
  parallelised; N=32→512 gives ~49× for 16× more tenants (the
  extra ~3× beyond linear is rayon scheduling overhead at larger
  fan-outs).
- `most_similar_top5` is `O(N · log top_n)` via bounded
  `BinaryHeap`; N=32→512 gives ~13× for 16× more tenants —
  sub-linear because the fixed-size heap caps per-iter work.

## Future work

- **`codisp` scoring path** — `rrcf` beats `rcf-rs` by ~0.13
  aggregate AUC on NAB thanks to its insert-then-remove probe
  protocol (see *Detection quality on public corpora* above).
  A `score_codisp(&point)` entry point would close that gap at
  the cost of ~18× scoring latency. Tracked as future work; the
  0.1 API is intentionally frozen on the faster isolation-depth
  score.
- **Yahoo S5 / Wikipedia pageviews** — Yahoo S5 is
  licence-gated (requires registration, no redistribution);
  Wikipedia pageviews has no ground-truth anomaly labels.
  Neither is a viable target for an open-source crate.
- **Arena compression beyond the split** — the split-typed
  arenas shipped in v4 still pay ~2×300 B per resident
  `InternalData` at `D = 16`. Further wins would come from a
  DFS-packed internal arena (parent-before-children with `u16`
  deltas) — risky: RCF inserts re-shuffle the tree shape at
  `O(log n)`, and a DFS layout forces `O(N)` restructuring on
  mid-tree insertion. Not pursued for now.
- **AVX-512 `f64x8`** — not actionable on stable Rust without
  relaxing `#![forbid(unsafe_code)]`. `wide 0.7` ships `f64x4`
  only; `std::simd` `f64x8` is nightly. Workaround: build with
  `RUSTFLAGS="-C target-cpu=native"` so LLVM widens the existing
  `f64x4` lanes to AVX-512 via auto-vectorisation when the host
  supports it — no code change needed.

### Done (previously listed here)

- **No-alloc scoring** — `RandomCutForest::score_many_with(points, cb)`
  invokes a caller-supplied closure per score, no intermediate
  `Vec`. See `tests/bulk_scoring.rs` for coverage.
- **Arena split (persistence v4)** — `NodeStore` arenas split
  into typed `Vec<Option<InternalData<D>>>` + `Vec<Option<LeafData>>`
  (was `Vec<Option<Node<D>>>` enum with worst-case-sized slots).
  Leaf slot size drops from ~320 B to ~40 B at `D = 16` — cuts
  per-forest leaf-arena memory by ~90 %. Measured at
  `(100, 256, 16)`: `forest_attribution` -37 %, `forest_update`
  -28 %, `forest_score` -10 %.
