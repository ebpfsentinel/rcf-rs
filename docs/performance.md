# Performance

Criterion (`cargo bench`) — wall-clock mean point estimate,
`mimalloc` pinned globally. Five bench harnesses split across
the workspace:

- `core/benches/forest_throughput.rs` — core ops (insert, score,
  attribution, codisp batched + loop).
- `core/benches/extended.rs` — bulk, early-term, forensic, tenant,
  stateless codisp, thresholded process, delete.
- `core/benches/modules.rs` — 23 groups: shingled forest, matrix
  profile (STOMP), t-digest / histogram, feature / meta drift,
  ADWIN, SPOT/DSPOT, Fisher, dynamic-dim forest, drift-aware
  shadow swap, companion primitives (OnlineStats / CountMinSketch
  / HyperLogLog / SpaceSaving / BloomFilter / Normalizer /
  PerFeatureEwma / PerFeatureCusum), and five explain / triage
  extras (group_scores / attribution_stability / score_ci /
  bootstrap / persistence).
- `triage/benches/modules.rs` — LSH clustering, Platt calibration,
  SAGE explanations, AlertClusterer (cosine), FeedbackStore.
- `hotpath/benches/modules.rs` — UpdateSampler, PrefixRateCap,
  bounded MPSC channel.

```bash
cargo bench --workspace                                                     # full
cargo bench -p anomstream-core --bench modules                              # single crate harness
cargo bench --workspace -- --sample-size 10 --measurement-time 2            # quick
cargo bench -p anomstream-core --bench modules -- per_feature_ewma/         # single group
```

## Reference hardware

|           |                                                    |
| --------- | -------------------------------------------------- |
| CPU       | Intel Core i7-1370P (13th gen), 14C/20T, L3 24 MiB |
| Memory    | 32 GB DDR5                                         |
| Kernel    | Linux 6.17                                         |
| Allocator | mimalloc 0.1 (pinned in bench harness)             |
| Compiler  | rustc 1.95 stable                                  |

Absolute values scale with CPU / memory bandwidth; _ratios_
(parallel speedup, early-term savings, tenant scaling) are the
portable signal.

## Caveats

- **Cross-group variance**: `b.iter()` mutates a persistent
  forest and criterion chooses batch sizes per-op, so reservoir
  state + per-iter overhead drift across bench groups. Trust
  ratios within a group, not absolute numbers across groups.
- **Cross-session variance**: numbers in this doc were captured on
  a warm CPU (governor `performance`, ambient IDE running). The
  same benches on a cool-CPU / powersave session landed 25-30 %
  faster on the single-probe `forest_update` / `forest_score`
  hot path. Ratios (fusion savings, parallel speedup, early-term
  ratio, tenant-scaling slopes) are stable across sessions and
  the portable signal.
- **Parallel ceiling**: `score_many` plateaus at ~5-6× speedup on
  a 14-core host — memory-bandwidth-bound past L3 working set.

## Core ops

`(trees=100, sample=256, D=16)`, single-seed warm run, `mimalloc`
pinned, `--sample-size 50 --warm-up-time 3 --measurement-time 8`:

| Workload                              | Time      | Throughput |
| ------------------------------------- | --------- | ---------- |
| `forest_update`                       | **34 µs** | ~29 k/s    |
| `forest_score`                        | **34 µs** | ~29 k/s    |
| `forest_attribution`                  | **45 µs** | ~22 k/s    |
| `forest_score_and_attribution`        | **47 µs** | ~21 k/s    |
| `forest_split_score_then_attribution` | **79 µs** | ~13 k/s    |

The fused `score_and_attribution` walk is **~41 % faster** than
calling `score` + `attribution` back-to-back (single traversal
instead of two, `47 / 79 ≈ 0.59`). The fused bbox SIMD kernel
(`total_probability_of_cut`) saves one pass over `min`/`max` loads
per internal node. Post split-typed-arena refactor (persistence
v4) leaf arena memory is −90 % (~320 B → ~40 B per slot).

Other `(trees, samples, D)` tuples below:

| Config           | `forest_update` | `forest_score` | `forest_attribution` |
| ---------------- | --------------- | -------------- | -------------------- |
| `(50, 128, 16)`  | 29 µs           | 25 µs          | —                    |
| `(100, 256, 4)`  | 29 µs           | 30 µs          | 35 µs                |
| `(100, 256, 16)` | 34 µs           | 34 µs          | 45 µs                |
| `(100, 256, 64)` | 104 µs          | 42 µs          | 88 µs                |
| `(200, 512, 16)` | 55 µs           | 52 µs          | —                    |

Criterion HTML report lives at `target/criterion/`.

## Bulk batch scoring

`D=16`, forest `(100, 256)`:

| Batch size | `score_many` (parallel) | Serial loop | Speedup |
| ---------- | ----------------------- | ----------- | ------- |
| 64         | 360 µs                  | 2.11 ms     | 5.9×    |
| 512        | 3.73 ms                 | 17.1 ms     | 4.6×    |
| 4096       | 28.6 ms                 | 137 ms      | 4.8×    |

### Codisp batched scoring

Probe-based codisp batched API (`score_codisp_many`) pre-inserts
probes into every tree, shared-walk amortises per-tree leaf →
root descent, rayon across trees:

| Batch K | `score_codisp_many` | `score_codisp` loop | Speedup |
| ------- | ------------------- | ------------------- | ------- |
| 16      | 1.76 ms             | 2.39 ms             | 1.4×    |
| 64      | 6.59 ms             | 9.58 ms             | 1.5×    |

Gain caps at ~1.5× because insert/delete mutation phases still
scale with `K × num_trees`; only the walk phase benefits from
shared-walk. For **frozen-baseline batched codisp** on any batch
size, prefer `score_codisp_stateless_many` (single-probe stateless
codisp mapped in parallel — no reservoir mutation, no `O(K)`
saturation limit).

### Memory-bandwidth plateau @ ~6× / 14 C

`score_many` tops out at ~6× rayon speedup on the 14-core / 20-thread
reference host — once the tree arena for 100 trees × 256 samples (~6 MB
at `D = 16`) fits in L3 but the per-probe working set thrashes L1/L2,
all workers compete for the same LLC → DRAM channel and further
threads contend rather than scale. Two avenues have been explored:

- **Cache-aware probe reordering** — explored via a shipped-then-
  removed `score_many_locality_sorted` variant that quantised
  leading dims into a Morton-lite bucket key and sorted the batch
  before dispatching. Benchmarked at `k = 1024`, `D = 16`,
  correlated cluster: plain `score_many` 5.10 ms, sorted variant
  5.69 ms — the `O(N log N)` sort + double-gather overhead beat
  the cache-locality gain on uniformly-random batches. Reverted
  in favour of letting callers re-order their own batches if
  they have a workload that benefits.
- **Packed Cut (`dim: u8` + `value: f32`, 8 B vs 16 B)** — **not
  shipped**. Halves the per-internal-node cut footprint, which
  would improve L1 fit at `D = 16`, but `value: f32` changes the
  isolation-depth boundary on points near a cut — bit-level
  divergence from the f64-cut scorer. Correctness audit + full
  NAB/TSB-AD AUC regression required before wiring. Documented
  here so the decision is explicit, not lost to future archaeology.

## Early-termination

`D=16`, forest `(100, 256)`, single probe:

| Path                                                       | Time    |
| ---------------------------------------------------------- | ------- |
| `score` (parallel ensemble)                                | 33 µs   |
| `score_early_term` threshold=0.02 (tight)                  | 36 µs   |
| `score_early_term` threshold=0.20 (loose, stops ~20 trees) | 4.99 µs |

Loose threshold → 6.6× speedup on baseline-dominated traffic;
tight threshold matches parallel `score` (sequential walk
rarely short-circuits).

## Forensic baseline

`forensic_baseline`:

| `(trees, samples, D)` | Time  |
| --------------------- | ----- |
| `(100, 256, 4)`       | 13 µs |
| `(100, 256, 16)`      | 16 µs |
| `(100, 1024, 16)`     | 64 µs |

Cost ≈ `O(live_points × D)` Welford sweep — `sample_size` ×4
→ ×4 time (close to linear on the current run), dim cost
grows ~1.3× over 4 → 16 (rayon hides per-dim cost until L1
pressure sets in).

## Tenant pool at scale

`tenant_pool`, each tenant `D=4` / `(50, 64)`, warmed 128 samples:

| N   | `similarity_matrix` | `score_across_tenants` | `most_similar_top5` |
| --- | ------------------- | ---------------------- | ------------------- |
| 32  | 37 µs               | 126 µs                 | 0.30 µs             |
| 128 | 99 µs               | 470 µs                 | 1.12 µs             |
| 512 | 586 µs              | 2.69 ms                | 5.01 µs             |

Scaling `N=32→512` (16× tenants):

- `similarity_matrix` O(N²) parallelised: ~16× (rayon fan-out
  hides quadratic until core saturation).
- `score_across_tenants` O(N) parallelised: 21×.
- `most_similar_top5` O(N·log k) bounded heap: 17×.

## Stateless codisp (frozen-baseline batched)

`(100, 256, D=16)`, walks root → leaf along stored cuts, no
reservoir mutation, rayon across trees:

| Workload                              | Time     |
| ------------------------------------- | -------- |
| `score_codisp_stateless` single probe | 29 µs    |
| `score_codisp_stateless_many` k=16    | 108 µs   |
| `score_codisp_stateless_many` k=64    | 312 µs   |
| `score_codisp_stateless_many` k=256   | 1.07 ms  |

Key results:

- **Stateless is ~1.1× faster than non-mutating `score()` single
  probe** (29 µs vs 34 µs) — stored-cut walk skips the EMA /
  reservoir update cost that `score()` amortises into the
  shared forest state.
- **Stateless batched beats mutating batched by ~21×**:
  `score_codisp_many` @ k=64 lands at 6.59 ms (above, in the
  throughput table), stateless @ k=64 at 312 µs. Mutating pays
  per-probe insert + walk + delete and serialises on the
  reservoir; stateless parallelises cleanly. This ratio is
  why NAB evaluation dropped from 12.6 s → 1.09 s after
  switching the eval path from `score_codisp` → stateless.

## TRCF adaptive threshold pipeline

`ThresholdedForest::process` — one call per point, full
pipeline (update + score + EMA + verdict):

| Workload                               | Time   |
| -------------------------------------- | ------ |
| `thresholded_process` `(100, 256, 16)` | 73 µs  |

Approximately `update (34 µs) + score (34 µs)` plus EMA +
tdigest + threshold logic — `73 µs ≈ 68 + 5` overhead.

## Forest delete

`RandomCutForest::delete` — reservoir eviction + bbox teardown
per tree. Measured via round-trip `update_indexed → delete`:

| Workload                                   | Time   |
| ------------------------------------------ | ------ |
| `update_indexed + delete` `(100, 256, 16)` | 115 µs |

~3.4× more expensive than an update alone — bbox recomputation
up the path and split-arena slot release dominate. Pair with
`update_indexed` when a probe-based workflow (e.g. codisp)
needs to restore the reservoir; otherwise rely on the
reservoir's own eviction (cheaper, amortised).

## Hot-path ingress primitives

Per-call overhead on the classifier hot path — picosecond to
nanosecond territory:

| Workload                                           | Time     | Throughput |
| -------------------------------------------------- | -------- | ---------- |
| `UpdateSampler::accept_stride` keep=8              | 28 ns    | ~36 M/s    |
| `UpdateSampler::accept_hash` (unkeyed) keep=8      | 14 ns    | ~73 M/s    |
| `UpdateSampler::accept_hash` (keyed) keep=8        | 15 ns    | ~67 M/s    |
| `PrefixRateCap::check_and_record` 100/1s           | 11.6 ns  | ~86 M/s    |
| `PrefixRateCap::check_and_record` 8-thread contended | ~9 µs/op (foreground) | contention floor |
| `update_channel::try_enqueue` cap=4096 (+ drain)   | 487 ns   | ~2.1 M/s   |
| `metrics::default_sink()` shared-Arc clone         | 12 ns    | ~85 M/s    |

- **Keyed vs unkeyed hash**: murmur-mix finaliser costs ~1.2 ns
  — negligible vs the atomic fetch-add on the accepted/rejected
  counters.
- **`accept_hash` faster than `accept_stride`**: skips the
  counter atomic; admission decision is a multiply + mod.
- **Channel throughput** is bounded by the `sync_channel`
  lock, not by try_send itself. 2.1 M/s per producer is more
  than enough for typical TC/XDP hot paths (~1-10 M pkt/s at
  10 Gbps, with multiple producer clones fanning out).
- **`PrefixRateCap` 11.6 ns** reflects the post-hardening rollover
  path (Acquire `load` of `window_start_ms` short-circuits the
  common "window still valid" case before any CAS runs) plus
  batched metrics dispatch (1 sink call per `METRICS_BATCH_SIZE
  = 64` ops vs per call). Down from 18 ns pre-batching (-36 %).
  The `compare_exchange_weak` loop only triggers once per window
  roll, not once per packet.
- **Cache-line padding**: `PrefixRateCap` buckets are
  `#[repr(C, align(64))]` (16 KiB total vs 1 KiB unpadded) so
  concurrent `fetch_add`s on different buckets do not bounce a
  shared cache line through MOESI/MESI. The 8-thread contended
  bench measures the residual cross-thread cost on the shared
  `window_start_ms` CAS — without padding, the bucket false
  sharing adds another order of magnitude.
- **Batched metrics**: `record_batched` accumulates per-counter
  increments locally and emits to the `Arc<dyn MetricsSink>`
  every 64 ops. At 12.5 Mpps the savings vs per-call dispatch
  are ≈157 ms / s of CPU clock per producer; in-process atomic
  counters stay bit-exact every call. Drain residue at shutdown
  via `flush_metrics()`.

## Streaming quantiles / histograms

| Workload                             | Time    | Throughput |
| ------------------------------------ | ------- | ---------- |
| `TDigest::record`                    | 42 ns   | ~24 M/s    |
| `TDigest::quantile(0.99)` after 100k | 57 ns   | query-only |
| `ScoreHistogram::record` default     | 4.9 ns  | ~205 M/s   |

TDigest amortises centroid compaction into the 10×-compression
buffer flush — per-call cost is an unsorted-buffer push + a
rare flush. ScoreHistogram is a bin-index + array increment,
essentially free.

## Drift detectors

| Workload                                   | Time    | Throughput |
| ------------------------------------------ | ------- | ---------- |
| `MetaDriftDetector::observe` (CUSUM)       | 8.0 ns  | ~125 M/s   |
| `FeatureDriftDetector::observe` D=16/10bin | 85 ns   | ~12 M/s    |
| `FeatureDriftDetector::psi()` D=16/10bin   | 1.17 µs | query-only |
| `AdwinDetector::update` cap=4096           | 26.3 µs | ~38 k/s    |

ADWIN's O(N) prefix-sum scan dominates at cap=4096. Caveat:
most deployments use score-stream ADWIN, where "one update per
alert" makes the 26 µs cost irrelevant. Per-packet ADWIN is
**not** hot-path material — use `MetaDriftDetector` (8 ns) on
the score stream instead.

## SOC layer (clustering, calibration, ensemble)

| Workload                                    | Time    | Throughput |
| ------------------------------------------- | ------- | ---------- |
| `LshAlertClusterer::hash_divector` D=16     | 73 ns   | ~13.7 M/s  |
| `LshAlertClusterer::observe` D=16           | 95 ns   | ~10.5 M/s  |
| `AlertClusterer::observe` D=16 window=32    | 771 ns  | ~1.3 M/s   |
| `PotDetector::record` post-freeze           | 43 ns   | ~23 M/s    |
| `PotDetector::p_value` post-freeze          | 8.2 ns  | ~122 M/s   |
| `PlattCalibrator::fit` 2048 samples         | 1.75 ms | offline    |
| `PlattCalibrator::calibrate` single score   | 25.6 ns | ~39 M/s    |
| `ensemble::fisher_combine` k=8              | 43 ns   | ~24 M/s    |
| `ensemble::fisher_combine` k=32             | 161 ns  | ~6.2 M/s   |
| `ensemble::fisher_combine` k=128            | 642 ns  | ~1.6 M/s   |
| `FeedbackStore::label` capacity=256         | 571 ns  | ~1.8 M/s   |
| `FeedbackStore::adjust` 512 labels, D=16    | 8.6 µs  | ~116 k/s   |
| `AuditChain::append` D=4 (HMAC-SHA256 + postcard) | 426 ns | ~2.3 M/s |
| `verify_audit_chain` 256 entries D=4        | 121 µs  | ~470 ns/entry |

- **LSH clustering**: `observe` = `hash_divector` + HashMap
  bucket increment. After the `String` → `u128` FNV-1a fold
  swap, `hash_divector` is 73 ns (−82 % vs the prior
  format-built hex string) and `observe` is 95 ns (−80 %).
  Zero heap allocations per alert.
- **Cosine `AlertClusterer`** now ~8× slower than LSH at
  window=32; scan cost grows linearly with window size vs O(1)
  bucket lookup for LSH. Prefer LSH at MSSP volume (>10k
  alerts/min).
- **SPOT p_value** is ~5× faster than `record`: recording
  updates the tdigest + Welford peak stats; querying is a
  closed-form GPD survival on cached γ, σ.
- **Platt fit** is offline (one call per calibration window);
  `calibrate` is 26 ns — σ(A·s + B) plus two floats.
- **`fisher_combine` scales linearly** at ~5 ns per p-value
  (Kahan-compensated sum + χ² survival tail).
- **FeedbackStore adjust** scales with stored label count
  (Gaussian-kernel sum): 512 labels × 16 dims → 8.6 µs.
  Label ingestion itself is 571 ns (single push + counter).

## Shingled forest

`ShingledForest::update_scalar` + `score_scalar` on the
embedded shingle (scalar stream → `D=16` sliding window):

| Workload                                   | Time    |
| ------------------------------------------ | ------- |
| `update_scalar` + `score_scalar` D=16      | 71 µs   |

≈ forest update (34 µs) + forest score (34 µs) + shingle ring
push. The ring-buffer shingle itself is free; the cost is the
downstream forest ops on the embedded vector.

## Matrix profile (STOMP)

`MatrixProfile::compute` — exact batch time-series discord / motif
via the diagonal dot-product recurrence. `O(n²)` time, `O(n)`
memory.

| Workload                                     | Time    |
| -------------------------------------------- | ------- |
| `compute` n=1024, window=32                  | 3.1 ms  |
| `compute` n=2048, window=64                  | 12.3 ms |
| `compute` n=4096, window=128                 | 49.3 ms |
| `discord_topk(5)` cached n=2048, window=64   | 27.8 µs |

- **Scaling matches `O(n²)`**: doubling `n` + doubling `m` raises
  cost ~4× (`n=1024→2048` 3.1→12.3 ms; `n=2048→4096` 12.3→49.3 ms).
  Window length affects only the first-column seed and the
  per-row constant — the diagonal recurrence is `O(1)` per cell.
- **`discord_topk(5)` is trivial** once the profile is cached
  (one sort + exclusion-aware walk). Running STOMP up front is
  the real cost; reuse the `MatrixProfile` for all downstream
  queries.
- Use as a forensic complement to `ShingledForest`: online path
  flags a region, STOMP gives the exact shape-level discord
  inside it.

## Dynamic dim + drift-aware wrappers

| Workload                                       | Time   |
| ---------------------------------------------- | ------ |
| `DynamicForest::update` active=8 / MAX_D=16    | 32 µs  |
| `DriftAwareForest::update` no shadow           | 34 µs  |
| `DriftAwareForest::update` with active shadow  | 80 µs  |

- **DynamicForest zero-pad overhead** is tiny (32 µs vs 34 µs
  native D=16 update) — the active-8 forest traverses shallower
  trees, recovering most of the padding cost.
- **DriftAwareForest no-shadow path** matches native forest
  (34 µs = native update), confirming the wrapper has no
  always-on cost.
- **With active shadow**: 2.4× overhead as expected — primary
  and shadow run sequentially in the same thread at bench time.

## SAGE Shapley attribution

`SageEstimator::explain` D=16, K=64 permutations (≈1024 forest
scores per explain call):

| Workload                                         | Time     |
| ------------------------------------------------ | -------- |
| `SageEstimator::explain` D=16, K=64, `(50, 128)` | 40.3 ms  |

Per-probe cost scales as `K · D × forest_score_cost`. SOC
triage / forensic replay territory, **not** per-alert real-time.

## Per-feature streaming detectors

`D=16`, per-dim accumulators, no forest involvement. These
primitives were promoted from the downstream enterprise ML layer
into `anomstream-core` for OSS reuse.

| Workload                                          | Time     | Throughput  |
| ------------------------------------------------- | -------- | ----------- |
| `OnlineStats::update` (Welford, hot)              | 5.6 ns   | ~180 M/s    |
| `OnlineStats::update` (cold, 32-sample loop)      | 75 ns    | ~14 M/s     |
| `OnlineStats::variance` read                      | 0.21 ns  | ~4.8 G/s    |
| `OnlineStats::std_dev` read                       | 0.20 ns  | ~5.0 G/s    |
| `Normalizer<16>::transform` MinMax                | 15.3 ns  | ~65 M/s     |
| `Normalizer<16>::transform` ZScore                | 8.0 ns   | ~125 M/s    |
| `Normalizer<16>::transform` None (identity)       | 3.5 ns   | ~286 M/s    |
| `Normalizer<16>::fit` 1024 samples                | 6.5 µs   | per-batch   |
| `CountMinSketch::increment` w=2048 d=4            | 65 ns    | ~15 M/s     |
| `CountMinSketch::estimate` w=2048 d=4             | 61 ns    | ~16 M/s     |
| `CountMinSketch::reset` w=2048 d=4                | 808 ns   | per-reset   |
| `BloomFilter::insert_bytes` n=1k, p=0.01          | 22.6 ns  | ~44 M/s     |
| `BloomFilter::insert_hash` n=100k, p=0.01         | 16.8 ns  | ~60 M/s     |
| `BloomFilter::contains_bytes` n=100k, p=0.01      | 28.8 ns  | ~35 M/s     |
| `BloomFilter::union` two n=10k, p=0.01 filters    | 260 ns   | per-merge   |
| `HyperLogLog::add_bytes` p=12                     | 11.5 ns  | ~87 M/s     |
| `HyperLogLog::add_hash` p=12                      | 2.7 ns   | ~370 M/s    |
| `HyperLogLog::estimate` after 100k adds, p=12     | 18.6 µs  | query-only  |
| `HyperLogLog::merge` two p=12 sketches            | 1.64 µs  | per-merge   |
| `SpaceSaving::observe` tracked-key hot, K=128     | 9.7 ns   | ~103 M/s    |
| `SpaceSaving::observe` evict path, K=128          | 310 ns   | ~3.2 M/s    |
| `SpaceSaving::top_k(10)` from 1024 distinct keys  | 5.6 µs   | per-query   |
| `PerFeatureEwma<16>::observe` (warmed)            | 129 ns   | ~7.8 M/s    |
| `PerFeatureEwma<16>::observe` (spike path)        | 96 ns    | ~10 M/s     |
| `PerFeatureEwma<16>::observe` (cold/warmup)       | 1.50 µs  | batch-only  |
| `PerFeatureCusum<16>::observe` below threshold    | 68 ns    | ~15 M/s     |
| `PerFeatureCusum<16>::observe` alert trip         | 85 ns    | ~12 M/s     |
| `PerFeatureCusum<16>::observe` stable ref         | 40 ns    | ~25 M/s     |

Insights:

- **OnlineStats variance/std_dev reads at 0.2 ns** — fully
  inlined, dominated by a single divide. `update` at 5.6 ns
  covers the full Welford recurrence (two subs, one mul, two
  adds, one fdiv).
- **Normalizer None (3.5 ns) = memcpy cost** — baseline for
  the other strategies. MinMax (15 ns) adds the range clamp;
  ZScore (8 ns) is between (no clamp, just centre + scale).
- **CMS increment ≈ estimate** because both paths hash the
  key twice over 4 rows → same cost. `reset` is a zero-fill
  over 2048 × 4 × 8 B = 64 KiB → 808 ns ≈ 80 GiB/s memset
  (hitting DRAM on the first pass; L3 for warm).
- **SpaceSaving hot ≪ evict (9.7 ns vs 310 ns)**: tracked-key
  path is one HashMap lookup + counter increment; evict path
  pays `O(K)` linear min-scan across 128 entries. Under
  heavy-hitter workloads the table stabilises so the hot path
  dominates. `top_k(10)` over 1024 entries (5.6 µs) is the
  query-time rank cost — run out-of-band, not per-packet.
- **BloomFilter insert_hash (18.5 ns) < insert_bytes (22.6 ns)**:
  the `_hash` path skips the single `SipHash` call, leaving just
  `k = 7` word-bank writes. `contains` costs ≈ `insert` under
  the miss-path (has to probe every hash before it can return
  `false`); positive lookups short-circuit on the first unset
  bit. `union` is a pure bitwise OR across the bit bank —
  scales with `m`, not `n`.
- **PerFeatureEwma spike < warmed**: the zero-variance branch
  (returns `f64::MAX` immediately) skips the sqrt — visible
  in the 33 ns gap vs the warmed normal path.
- **PerFeatureCusum stable ref (40 ns) < below threshold (68 ns)**:
  stable skips the `max(0, …)` branches entirely once the
  cumulative sums settle to zero.

## Explain + triage additions

Five bench groups covering explain / triage primitives on the
core surface:

| Workload                                           | Time    |
| -------------------------------------------------- | ------- |
| `FeatureGroups::group_scores` D=16, 3 groups       | 39.0 µs |
| `RandomCutForest::attribution_stability` D=16      | 55.4 µs |
| `RandomCutForest::score_with_confidence` D=16      | 58.4 µs |
| `RandomCutForest::bootstrap` 4096 pts, `(50, 128)` | 187.8 ms |
| `RandomCutForest::to_bytes` `(100, 256, D=16)`     | 5.98 ms |
| `RandomCutForest::from_bytes` `(100, 256, D=16)`   | 7.66 ms |

- **`group_scores`** ≈ `attribution` + O(D) post-reduce; the
  39 µs cost is essentially the attribution walk with a tiny
  per-group sum at the end.
- **`attribution_stability`** ≈ 1.2× `attribution`: walks the
  forest once to collect per-tree DiVectors, folds per-dim
  variance in a second pass. Cheaper than naive "run
  attribution N times and variance the result".
- **`score_with_confidence`** ≈ 1.7× `score`: same walk plus
  per-tree score stddev accumulator. Always non-parallel
  (stderr needs per-tree outputs in order).
- **`bootstrap` 4096 pts / (50t × 128s, D=16)**: ~188 ms
  → ~22 k pts/s on the reduced-size forest. Scales linearly
  with point count; use `(100, 256)` config for a ~4× cost.
- **Persistence roundtrip**: 6 ms to serialise a warmed
  `(100, 256, D=16)` forest (2.6 MB payload at postcard's
  compact encoding), 7.7 ms to deserialise (bincode tree
  rehydration dominates the deserialise path).

## External baselines (synthetic)

Input: 10k points, `D=16`, 1 % outliers, 30 % warm / 70 % eval,
frozen baseline. Each impl on its idiomatic fast path
(anomstream-core rayon / rrcf single-process / sklearn NumPy-Cython SIMD
/ AWS Java cold JVM). **5-seed variance** (seeds 2026–2030),
mean ± stddev, coefficient of variation in parens.

| Impl                                   | Backend              | Updates/s                   | Scores/s                  | AUC       |
| -------------------------------------- | -------------------- | --------------------------- | ------------------------- | --------- |
| `anomstream-core` 0.0.0-dev, `score()`          | Rust, rayon-parallel | **31 500** (single seed)    | **197 900** (single seed) | 1.000     |
| `anomstream-core` 0.0.0-dev, `score_codisp()`   | Rust, parallel walk  | — (per-probe insert/delete) | 8 150 (single seed)       | 1.000     |
| `anomstream-core` 0.0.0-dev, `score()` (5-seed) | Rust, rayon-parallel | 17 500 ± 1 240 (7 %)        | 125 900 ± 1 840 (1.5 %)   | 1.000 ± 0 |
| `randomcutforest-java` 4.4.0           | JVM 26, cold         | 2 090 ± 134 (6 %)           | 8 870 ± 415 (5 %)         | 1.000 ± 0 |
| `rrcf` 0.4.4                           | Python + NumPy       | 73 ± 3 (4 %)                | 94 150 ± 4 840 (5 %)      | 0.992 ± 0 |
| `sklearn.IsolationForest`              | NumPy + Cython       | batch-only                  | 136 300 ± 2 450 (2 %)     | 1.000 ± 0 |

Ratios (mean/mean):

- **Updates**: anomstream-core is ~8.4× faster than AWS Java, ~240× faster
  than rrcf. CVs around 5-7 % on all impls; the ratios sit well
  outside the noise floor.
- **Scores (fast path)**: sklearn edges anomstream-core `score()` by 8 %
  (136k vs 126k) — real but small (stddevs combined ≈ 3k, so the
  10k delta is ~3σ significant). rrcf trails anomstream-core by ~25 %;
  AWS Java trails by ~14×.
- **Scores (codisp path)**: anomstream-core `score_codisp()` mutates the
  forest per probe (insert → walk leaf→root → delete). Post the
  rayon-per-tree parallel refactor (walk + delete now fan out
  across trees) it hits ~8 k probes/s at `(100, 256, D=16)` —
  ~25× slower than the isolation-depth `score()` fast path.
  Matches AWS Java `getAnomalyScore` / rrcf `codisp()` semantic;
  use it for SOC triage / forensic replay,
  not the eBPF hot path.
- **AUC**: identical within measurement precision across every
  seed (0.992 for rrcf, 1.000 for the other three).

Noise sources documented: machine thermal state varies across
runs — single-seed numbers from an earlier cool-CPU session
landed at ~32k/203k for anomstream-core, dropping to ~17k/126k on this
run. The **ratios are portable, the absolute numbers aren't**.

Reproduce the sweep:

```bash
scripts/synthetic/variance_sweep.sh /tmp/aws-rcf/randomcutforest-core-4.4.0.jar
```

## Detection quality — NAB `realKnownCause`

Three scoring APIs, three trade-offs:

- **`RandomCutForest::score()`** — isolation-depth, non-mutating,
  rayon-parallel, eBPF-hot-path friendly.
  On NAB: **0.719** aggregate AUC after the lag=32 + zscore +
  smooth(0.02) pipeline.
- **`RandomCutForest::score_codisp()`** — probe-based (insert,
  walk leaf→root accumulating `max(sibling.mass /
subtree.mass)`, remove). Matches rrcf / AWS Java scoring
  semantic. ~30× slower than `score()` and **mutates the
  reservoir** — insertion evicts baseline points the following
  `delete` cannot restore, so long eval streams drift away from
  the frozen warm-phase baseline. On NAB: **0.776** aggregate
  AUC, beats rrcf (0.748) and AWS Java (0.757).
- **`RandomCutForest::score_codisp_stateless()`** — root → leaf
  walk along stored cuts, accumulates `max(sibling_mass /
subtree_mass)` per level without inserting the probe. Preserves
  the frozen-baseline promise exactly, takes `&self`, rayon-
  parallel across trees. On NAB: **0.763** aggregate AUC
  (**~0.013 shy of mutating codisp, ~0.044 above `score()`**).
  Runtime for the 7-file corpus: **1.09 s parallel** — 12× faster
  than the mutating variant.

Same embedding pipeline (32-lag → warm-phase z-score → EMA
α = 0.02), 15 % warm, 100 trees × 256 sample. `tests/nab.rs`
runs the 7-file corpus in parallel via rayon `par_iter` over
files — each file owns an independent forest. Full run
(both variants, parallel file iter) completes in ~12 s.

| File                                   | `score()` | `score_codisp()` | `score_codisp_stateless()` | rrcf  | AWS Java  |
| -------------------------------------- | --------- | ---------------- | -------------------------- | ----- | --------- |
| `ambient_temperature_system_failure`   | **0.813** | **0.813**        | 0.793                      | 0.734 | 0.786     |
| `cpu_utilization_asg_misconfiguration` | 0.953     | **0.969**        | 0.963                      | 0.849 | 0.906     |
| `ec2_request_latency_system_failure`   | **0.709** | 0.706            | 0.621                      | 0.481 | 0.482     |
| `machine_temperature_system_failure`   | 0.578     | **0.817**        | 0.815                      | 0.880 | 0.883     |
| `nyc_taxi`                             | **0.698** | 0.636            | 0.623                      | 0.571 | 0.540     |
| `rogue_agent_key_hold`                 | 0.145     | 0.198            | 0.181                      | 0.535 | **0.633** |
| `rogue_agent_key_updown`               | **0.633** | 0.579            | 0.563                      | 0.657 | 0.542     |
| **weighted aggregate**                 | 0.719     | **0.776**        | 0.763                      | 0.748 | 0.757     |

### Hyperparameter ablation

`examples/nab_ablation.rs` on the same corpus:

| Config                                  | Aggregate AUC |
| --------------------------------------- | ------------- |
| baseline (lag=8, raw score)             | 0.615         |
| lag=32                                  | 0.665         |
| lag=32 + diff                           | 0.640         |
| lag=32 + zscore                         | 0.683         |
| lag=32 + smooth(0.1)                    | 0.687         |
| lag=32 + zscore + smooth(0.05)          | 0.718         |
| **lag=32 + zscore + smooth(0.02)**      | **0.719**     |
| lag=64 + zscore + smooth(0.05)          | 0.672         |
| trcf-online D=32                        | 0.320         |
| probe-score D=8 (naive hack)            | 0.330         |
| **codisp D=32 + zscore + smooth(0.02)** | **0.776**     |

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
java -cp "scripts/nab:/tmp/aws-rcf/randomcutforest-core-4.4.0.jar" RcfBenchNab /opt/nab
```

## Detection quality — TSB-AD-M (multivariate)

TSB-AD-M (TheDatumOrg, 2024): 200 multivariate series across 16
source datasets, per-point binary labels, native multivariate
(no lag embedding). Pipeline: per-dim z-score on the upstream
`tr_<N>` train split, frozen-baseline scoring, EMA-smooth
α = 0.02. Forest `(100, 256)`, seed `2026`. Const-generic
whitelist `{2, 3, 7, 8, 9, 12, 16, 17, 18, 19, 25, 29, 31, 38, 51,
55, 66}` covers **192 / 200 files (96 %)**; the eight D=248 files
are skipped. `tests/tsb_ad_m.rs` runs the corpus in parallel via
rayon `par_iter` over files. Runtime on reference hardware:
~3 min `score()`, ~6 min `score_codisp()` (stride-subsampled to
50 k eval rows / file), ~3 min `score_codisp_stateless()` on the
**full** eval stream.

Per-dataset ROC-AUC (weighted by positive count) against
`randomcutforest-java` 4.4.0 on the same corpus. rrcf 0.4.4 was
benched with the same protocol
(`scripts/tsb_ad/bench_rrcf_tsb_ad.py`, parallel across files)
but wall-time is prohibitive on the full corpus — ~3–4 h at
14 workers / `--max-eval 1500`. Numbers are left for the reader
to reproduce; the script is provided for reproducibility.

| Source dataset         | Files             | `score()` | `score_codisp()` | `score_codisp_stateless()` | AWS Java  |
| ---------------------- | ----------------- | --------- | ---------------- | -------------------------- | --------- |
| Genesis                | 1                 | 0.968     | **0.991**        | **0.994**                  | 0.982     |
| SMAP                   | 27                | 0.803     | **0.823**        | 0.716                      | 0.805     |
| SMD                    | 22                | 0.618     | **0.760**        | 0.752                      | 0.806     |
| MSL                    | 16                | **0.705** | 0.746            | 0.599                      | 0.762     |
| SVDB                   | 31                | 0.692     | 0.737            | **0.779**                  | 0.757     |
| LTDB                   | 5                 | 0.601     | 0.755            | **0.758**                  | 0.755     |
| Exathlon               | 27                | 0.491     | 0.894            | **0.996**                  | 0.865     |
| MITDB                  | 13                | 0.597     | **0.678**        | 0.603                      | 0.660     |
| PSM                    | 1                 | 0.608     | 0.595            | **0.613**                  | 0.611     |
| CATSv2                 | 6                 | **0.580** | 0.547            | 0.496                      | 0.547     |
| CreditCard             | 1                 | 0.589     | 0.679            | 0.658                      | **0.693** |
| Daphnet                | 1                 | 0.309     | 0.885            | **0.926**                  | 0.944     |
| GECCO                  | 1                 | 0.412     | 0.523            | **0.753**                  | 0.594     |
| GHL                    | 25                | 0.454     | 0.461            | **0.570**                  | 0.419     |
| OPPORTUNITY            | 8 (skipped D=248) | —         | —                | —                          | 0.298     |
| SWaT                   | 2                 | 0.282     | **0.825**        | 0.715                      | 0.825     |
| TAO                    | 13                | 0.451     | 0.453            | **0.487**                  | 0.471     |
| **aggregate weighted** | **192 / 200**     | 0.583     | **0.768**        | 0.751                      | 0.753     |

- **anomstream-core `score()`** — isolation depth, rayon-parallel, full
  eval scan. Same fast API eBPFsentinel ships on the hot path.
  `tests/tsb_ad_m.rs::tsb_ad_m_aggregate_auc_above_floor` pins
  the aggregate floor at 0.55 — regression guard, not a quality
  claim.
- **anomstream-core `score_codisp()`** — probe-based codisp walk (leaf → root,
  `max(sibling.mass / subtree.mass)`), sequential per tree,
  mutates the reservoir per probe. Stride-subsampled to 50 000
  eval rows per file (const `CODISP_MAX_EVAL`). Directly
  comparable to the AWS Java / rrcf semantic; leads aggregate
  **0.768** vs AWS Java 0.753.
  `tests/tsb_ad_m.rs::tsb_ad_m_codisp_aggregate_auc_above_floor`.
- **anomstream-core `score_codisp_stateless()`** — root → leaf walk along
  stored cuts, max `sibling_mass / subtree_mass` per level, no
  reservoir mutation. Takes `&self` → rayon-parallel across
  trees. Covers the **full** eval stream (no stride) and
  preserves the frozen-baseline semantic across long runs.
  Aggregate **0.751** — ~0.017 below the drift-affected mutating
  codisp but within measurement noise of AWS Java (0.753) and
  the only variant that scales past the `CODISP_MAX_EVAL` cap.
  `tests/tsb_ad_m.rs::tsb_ad_m_codisp_stateless_aggregate_auc_above_floor`.
- **AWS Java `getAnomalyScore()`** — codisp-like, stride-
  subsampled to 50 000 eval rows per file (essentially full-scan
  for 95 % of the corpus). Covers all 200 files including the
  eight D=248 OPPORTUNITY series the const-generic whitelist
  skips.

Caveats:

- **Plain point-wise ROC-AUC**; the official TSB-AD leaderboard
  ranks on **VUS-PR** (Paparrizos et al. 2022) which integrates
  range-based precision / recall across a sliding window. VUS-PR
  is now available in-crate — see the `vus_pr` module and
  `examples/tsb_ad_m_eval.rs` for a range-aware alternative to
  this point-wise AUC regression guard.
- **RCF is classical by design** — transformer-based SOTA
  (TimesNet, Anomaly Transformer) outscores every impl here on
  heavy-physics datasets (SWaT, Daphnet, GECCO) where the anomaly
  signature lives in higher-order cross-channel structure. RCF
  stays competitive on Genesis / SMAP / MSL / SVDB where per-dim
  statistical drift dominates — closer to eBPFsentinel's
  production feature mix (rate, ratio, entropy, cardinality).

Reproduce:

```bash
scripts/tsb_ad/fetch.sh /tmp/tsb-ad
RCF_TSB_AD_M_PATH=/tmp/tsb-ad/TSB-AD-M \
    cargo test --release --test tsb_ad_m --all-features -- --ignored --nocapture
python3 scripts/tsb_ad/bench_rrcf_tsb_ad.py \
    --dir /tmp/tsb-ad/TSB-AD-M --max-eval 1500 --workers "$(nproc)"
javac -cp /tmp/aws-rcf-central/randomcutforest-core-4.4.0.jar \
    scripts/tsb_ad/RcfBenchTsbAd.java
java -cp scripts/tsb_ad:/tmp/aws-rcf-central/randomcutforest-core-4.4.0.jar \
    RcfBenchTsbAd /tmp/tsb-ad/TSB-AD-M 50000
```
