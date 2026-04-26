# Features

`anomstream-core` is a **streaming anomaly detection toolkit**.
Three detector families compose with a shared substrate of streaming
stats, explanation, triage, and hot-path primitives — Random Cut Forest
is one detector family among several, drawn in wherever isolation-depth
scoring makes sense.

## Crate layout

Most entries in this catalogue live in `anomstream-core`; a handful
live in sibling crates (`anomstream-triage`, `anomstream-hotpath`)
so their SemVer can evolve at their own cadence:

| Section | Owning crate |
|---|---|
| Multivariate anomaly detectors | `anomstream-core` |
| Per-feature univariate detectors | `anomstream-core` |
| Score-level drift & regime change | `anomstream-core` |
| Streaming stats & sketches | `anomstream-core` |
| Forest scoring operations | `anomstream-core` |
| Explanation & triage → `SeverityBands`, `DiVector`, `AttributionStability`, `ForensicBaseline`, `FeatureGroups` | `anomstream-core` |
| Explanation & triage → `SageEstimator`, `PlattCalibrator` | **`anomstream-triage`** |
| SOC & ops → `AlertClusterer`, `LshAlertClusterer`, `FeedbackStore`, `AlertRecord` | **`anomstream-triage`** |
| Training & retention | `anomstream-core` |
| Persistence | `anomstream-core` |
| Observability → `MetricsSink` + metric names table | `anomstream-core` |
| Hot-path integration (eBPF ingress) → `UpdateSampler`, `PrefixRateCap`, `channel` | **`anomstream-hotpath`** |
| Security & threat model | `anomstream-core` |
| Multi-tenancy | `anomstream-core` |
| Quality | `anomstream-core` |

Consumers using the [`anomstream`](../meta/) meta-crate see every
type re-exported under a single import path regardless of the owning
member crate. Direct-dependency consumers import from the owning
crate explicitly.

## Catalogue

This doc catalogues every public module grouped by the same
taxonomy as the README:

1. [Multivariate anomaly detectors](#multivariate-anomaly-detectors)
2. [Per-feature univariate detectors](#per-feature-univariate-detectors)
3. [Score-level drift & regime change](#score-level-drift--regime-change)
4. [Streaming stats & sketches](#streaming-stats--sketches)
5. [Forest scoring operations](#forest-scoring-operations)
6. [Explanation & triage](#explanation--triage)
7. [SOC & ops](#soc--ops)
8. [Training & retention](#training--retention)
9. [Persistence](#persistence)
10. [Observability](#observability)
11. [Hot-path integration (eBPF ingress)](#hot-path-integration-ebpf-ingress)
12. [Security & threat model](#security--threat-model)
13. [Multi-tenancy](#multi-tenancy)
14. [Quality](#quality)

Driven by eBPFsentinel Enterprise needs but kept detector-agnostic —
none of the modules are required to use any other.

## Multivariate anomaly detectors

Operate on the joint `[f64; D]` distribution. RCF is the headline
but the wrappers (shingled / runtime-dim / shadow swap / pool) let
the same forest ride many shapes of workload.

### `RandomCutForest<D>` — bare RCF

Const-generic per-point dimensionality, enforces AWS SageMaker
hyperparameter bounds, score averaged across trees, reservoir
sampling without replacement. Built via `ForestBuilder<D>`.

Hot-path API: `update`, `update_indexed`, `score`, `attribution`,
`delete`, `delete_by_value`.

Source: `src/forest/`, `src/config.rs`.

Examples: `examples/quickstart.rs` (minimal 30-line demo),
`examples/streaming.rs` (CSV stdin loop),
`examples/attribution_explain.rs` (top-dim breakdown),
`examples/with_mimalloc.rs` (allocator swap).

### `ThresholdedForest<D>` — adaptive threshold (TRCF)

Tracks an EMA of the anomaly-score stream and derives a
continuously updated `mean + z · stddev` threshold. Callers
receive an `AnomalyGrade` (`score` + `threshold` + `grade ∈ [0, 1]`
+ `is_anomaly` + `ready`) instead of comparing a raw score against
a magic constant. Inspired by AWS's `TRCF` in
`randomcutforest-parkservices`, kept light (no short/long-term
duality, no near-threshold heuristics).

Built via `ThresholdedForestBuilder<D>`. API: `process`,
`process_indexed`, `score_only`, `current_threshold`, `stats`,
`reset_stats`.

**Two threshold modes** (see `ThresholdMode`):

- `ZSigma { z_factor }` (default, back-compat): `threshold =
  max(min_threshold, mean + z_factor × stddev)` on the EMA stats.
  Good for Gaussian-like scores (lag-embedded streams with
  symmetric noise).
- `Quantile { p }`: `threshold = tdigest.quantile(p)` of the
  streaming score distribution. Robust to the right-skew of
  isolation-depth scores; calibrates directly on the caller's
  alert-rate budget (`p = 0.99` ≈ 1 % firing rate in steady
  state; `0.999` ≈ 0.1 %). Uses the shipped `TDigest` primitive
  — no extra deps. Enable via `.quantile_threshold(p)` on the
  builder.

Source: `src/thresholded/`.

Example: `examples/thresholded.rs`.

### `ShingledForest<D>` — scalar-stream temporal wrapper

`ShingledForest<D>` wraps a bare `RandomCutForest<D>` with a ring
buffer of the last `D` scalars. Each `update_scalar(x)` shifts the
window and emits a fresh `[f64; D]` to the forest, turning a
scalar stream into a `D`-dim feature vector that captures
temporal autocorrelation. This is the fix for the NAB
`rogue_agent_key_hold` = 0.145 / SWaT = 0.282 failures caused by
isolation depth blind-spotting contextual temporal anomalies
(dwell / drop / frequency shift): the scalar value stays in
baseline range but the shingled subsequence sits far from every
baseline subsequence in the `D`-dim shingle space.

API: `update_scalar` (mutates ring + forest once ring is full),
`score_scalar` (non-mutating query with `value` as newest slot),
`attribution_scalar` (per-lag-index DiVector),
`score_codisp_stateless_scalar` (drift-free codisp over the
shingle). `current_shingle`, `is_warmed`, `reset_ring` for
diagnostics + lifecycle.

Built via `ShingledForestBuilder<D>` — same knobs as
`ForestBuilder` (`num_trees`, `sample_size`, `seed`, `time_decay`);
the const-generic `D` **is** the shingle size.

Types: `ShingledForest<D>`, `ShingledForestBuilder<D>`.

Source: `src/shingled.rs`.

Example: `examples/shingled.rs` (periodic sine baseline + three
injected contextual anomalies — dwell / drop / frequency shift).

### `MatrixProfile` — STOMP time-series discord / motif

Batch complement to `ShingledForest`: where the shingled forest is
online, approximate and tree-based, the matrix profile is offline,
exact and distance-based. Given a univariate series `T[0..n]` and
a window length `m`, STOMP computes `P[i]` = z-normalised
Euclidean distance from `T[i..i+m]` to its nearest non-trivial
neighbour (a small exclusion zone around `i` is skipped so
near-duplicates of the query don't win). `argmax P` is the
discord (least-similar subsequence — shape anomaly); `argmin P`
is the motif (most-repeated shape).

STOMP runs in `O(n²)` time with `O(n)` memory via the diagonal
dot-product recurrence
`QT[i, j] = QT[i-1, j-1] + T[i+m-1]·T[j+m-1] - T[i-1]·T[j-1]`,
piggy-backing on pre-computed sliding means / standard deviations.
Practical working set: 1 K samples at `m = 32` in ≈ 3 ms; 4 K at
`m = 128` in ≈ 49 ms on a modern core. Use it to confirm /
localise a shape anomaly flagged by the online path, or to scan
a captured rolling window when exactness matters more than latency.

Default exclusion zone is `ceil(window / 4)` — the standard
matrix-profile convention.

Gated behind `std`.

API: `compute(series, window, exclusion_zone)` (the `None` default
picks the convention above) + `discord() -> (usize, f64)` +
`discord_topk(k) -> Vec<(usize, f64)>` (exclusion-aware
suppression so multiple emitted positions cannot cluster inside
one anomalous region) + `motif() -> (usize, f64)` + `profile()` /
`profile_index()` accessors + `window` / `exclusion_zone` / `len`
/ `is_empty`.

Types: `MatrixProfile`, `MATRIX_PROFILE_MIN_WINDOW`.

References:

1. Y. Zhu, Z. Zimmerman, N. Senobari, C. Yeh, G. Funning,
   A. Mueen, P. Brisk, E. Keogh, *Matrix Profile II: Exploiting a
   Novel Algorithm and GPUs to Break the One Hundred Million
   Barrier for Time Series Motifs and Joins*, ICDM 2016.
2. C. Yeh, Y. Zhu, L. Ulanova, N. Begum, Y. Ding, H. A. Dau,
   D. F. Silva, A. Mueen, E. Keogh, *Matrix Profile I: All Pairs
   Similarity Joins for Time Series*, ICDM 2016.

Source: `src/matrix_profile.rs`.

### `DynamicForest<MAX_D>` — runtime-dim variant

`DynamicForest<MAX_D>` wraps `RandomCutForest<MAX_D>` behind a
runtime-sized input API (`&[f64]` of caller-declared `active_dim
≤ MAX_D`). Zero-pads shorter inputs to `MAX_D` — RCF's range-
weighted cut sampling naturally skips zero-range dims so the
pad contributes nothing to scores or attributions. Targets MSSP
/ heterogeneous multi-tenant deployments where const-generic
`D` blocks a single monomorphisation across tenants.

Hot-path callers with a known-fixed dim keep using the bare
`RandomCutForest<D>` — it's faster (fewer runtime checks, better
inlining). `DynamicForest` is the escape hatch.

Types: `DynamicForest<MAX_D>`.

Source: `src/dynamic_forest.rs`.

### `DriftAwareForest<D>` — shadow swap on drift

Facade around a live `RandomCutForest<D>` with an optional
shadow. Swap policy via `DriftRecoveryConfig`: `min_primary_age`
guards against flap-loops, `shadow_warmup` controls when the
shadow becomes the new primary. Call `on_drift()` to spawn a
shadow (trigger lives outside — route from `AdwinDetector`,
`FeatureDriftDetector`, or `MetaDriftDetector`). `update` feeds
both primary and shadow; once the shadow hits `shadow_warmup` the
swap is atomic on the next `update` tick. `score` always reads
from the primary (stable baseline until the swap lands).

Types: `DriftAwareForest<D>`, `DriftRecoveryConfig`.

Source: `src/drift_aware.rs`.

Example: `examples/drift_recovery.rs` (2-regime synthetic stream
with ADWIN-triggered shadow spawn + atomic swap after warmup).

### `TenantForestPool<K, D>` — per-tenant isolation

One `ThresholdedForest` per tenant key, bounded LRU eviction,
lazy instantiation via a factory closure. A baseline shock on
tenant A does not affect tenant B's adaptive threshold. Std-only.

API: `process`, `score_only`, `attribution`, `peek`, `get`,
`get_mut`, `insert`, `remove`, `clear`, `iter`, `iter_mut`,
`tenants`, `evict_lru`, `evict_idle`.

`evict_lru` sheds on capacity pressure (replace oldest when full).
`evict_idle(ttl)` sheds on wall-clock staleness (evict every tenant
whose last access is older than `ttl`) — intended for SaaS / MSSP
with thousands of intermittent tenants where LRU alone would
preserve dormant entries. Orthogonal paths; both emit
`rcf_tenant_evictions_total`, TTL path also emits
`rcf_tenant_idle_evictions_total`.

`readiness_summary()` returns a `ReadinessSummary`
(`resident / warming / ready / capacity / tenants_{created,evicted}_lifetime`)
for `/healthz` / `/readyz` endpoints — zero-allocation `O(resident)`
scan, no metrics-sink plumbing required. Helpers `readiness_ratio()`,
`is_fully_ready()`, `is_at_capacity()` give one-line health gates.

Source: `src/pool/`.

Example: `examples/tenant_pool.rs`.

## Per-feature univariate detectors

One accumulator per dimension — answer *which* feature drifted.
Complementary to multivariate detectors: the forest catches joint
anomalies the per-feature detectors miss, the per-feature
detectors attribute drift the forest reduces to a single scalar.

### `PerFeatureEwma<D>` — parallel EWMA z-score

`D` parallel univariate EWMA accumulators track per-dim mean +
variance with geometric decay `α`. After a warmup budget,
`observe(&[f64; D])` reports per-feature z-scores and the max
across dims. Policy-free — returns raw z-scores; caller maps
`max_z` to alert severity via `SeverityBands` or a custom rule.

Scoring runs *before* the accumulator update so the current
observation is judged against the prior distribution (textbook
EWCD / EWMA-Z convention) — without this a large step folds into
the mean in the same tick and the alert is missed.

EWMA recurrence:

```text
mean_t     ← α·x_t + (1 − α)·mean_{t−1}
variance_t ← α·(x_t − mean_{t−1})² + (1 − α)·variance_{t−1}
```

First observation seeds `mean_0 = x_0`, `variance_0 = 0`.

API: `new(PerFeatureEwmaConfig { alpha, warmup_samples })` +
`observe(&[f64; D]) → Option<PerFeatureEwmaResult<D>>` +
`is_warmed_up()` + `reset()` + `accumulators()` read-only
snapshot.

Types: `PerFeatureEwma<D>`, `PerFeatureEwmaConfig`,
`PerFeatureEwmaResult<D>`, `EwmaAccumulator`.

Reference: J. S. Hunter, *The Exponentially Weighted Moving
Average*, JQT 18(4), 1986.

Source: `src/per_feature_ewma.rs`.

### `PerFeatureCusum<D>` — parallel two-sided CUSUM

`D` parallel univariate CUSUMs track positive and negative
cumulative sums of the deviation from a reference mean. Alerts
when either side exceeds the threshold `h` — detects **sustained**
mean shifts that an EWMA adapts to and stops reporting (slow-ramp
DDoS, gradual leak).

CUSUM recurrence (per dim):

```text
S+ ← max(0, S+ + (x − μ₀ − k))
S− ← max(0, S− − (x − μ₀ + k))
alert when S+ > h  (increase)  or  S− > h  (decrease)
```

`k` is the slack (allowable drift, typical `0.5·σ`), `h` is the
threshold (typical `4·σ`), `μ₀` is the reference mean (auto-
learned on the first observation unless overridden via
`set_reference(&[f64; D])`).

API: `observe(&[f64; D]) → PerFeatureCusumResult<D>` (always
returns — no warmup gate, just empty alerts until trip) +
`active_drifts()` (count of dims in drift) + `reset()`.

Orthogonal to `MetaDriftDetector` (scalar CUSUM on the score
stream): this module is per-feature CUSUM on **raw observations**
so the caller can answer *which feature drifted and in which
direction*. Use both — they serve different triage paths.

Types: `PerFeatureCusum<D>`, `PerFeatureCusumConfig`,
`PerFeatureCusumResult<D>`, `PerFeatureCusumAlert`,
`DriftDirection`, `PerFeatureCusumAccumulator`.

References: E. S. Page, *Continuous Inspection Schemes*,
Biometrika 41, 1954. D. M. Hawkins & D. H. Olwell, *Cumulative
Sum Charts and Charting for Quality Improvement*, Springer, 1998.

Source: `src/per_feature_cusum.rs`.

### `FeatureDriftDetector<D>` — PSI / KL drift

Pins a baseline per-dim histogram, folds live traffic into a mirror
histogram with identical bin edges, and reports Population
Stability Index (`Σ (Q − P) · ln(Q/P)`) and KL divergence
`D_KL(Q || P)` per feature. CUSUM on the score stream catches the
detector *re-centring*; PSI on the features catches the data
*itself drifting*. Industry thresholds wired into
`DriftLevel::{Stable, Watch, Alert}` (`< 0.10`, `0.10..0.25`,
`≥ 0.25`). `argmax_psi()` pins the offending dim;
`reset_production()` starts a fresh monitoring window without
invalidating the baseline.

Types: `FeatureDriftDetector`, `DriftLevel`, `PSI_WATCH_THRESHOLD`,
`PSI_ALERT_THRESHOLD`.

Source: `src/feature_drift.rs`.

## Score-level drift & regime change

Operate on a scalar anomaly-score stream (or any univariate
signal). Catch baseline re-centring, regime change, and
heavy-tail shifts that the detectors above miss once they've
adapted.

### `MetaDriftDetector` — two-sided CUSUM on score stream

Runs a two-sided CUSUM on the anomaly-score stream. Fires
`DriftKind::Upward` on a sustained climb (baseline drift) and
`DriftKind::Downward` on a sustained decline — strictly more
sensitive to small persistent shifts than a `μ + 3σ` gate.
Standalone type; plug any score-like stream in. `reset()` clears
the accumulators while keeping the EMA reference, `reset_stats()`
clears everything.

Types: `CusumConfig`, `DriftKind`, `DriftVerdict`,
`MetaDriftDetector`.

Source: `src/meta_drift.rs`.

Example: `examples/meta_drift.rs`.

### `AdwinDetector` — adaptive windowing

Streaming change-point detector (Bifet & Gavaldà, SIAM SDM 2007).
Bounded ring buffer of the last `N` observations, per-update
`O(N)` scan over every split point with a Hoeffding bound
`ε_cut`; flags drift and drops the older sub-window on fire.
Confidence `δ`, window cap `N`, and observed stream amplitude
`range` are caller-configured. Use on the score stream (or any
per-step scalar) for an adaptive drift trigger with automatic
window sizing — strictly more sensitive than a fixed-window mean
test. Drives `DriftAwareForest` shadow spawn when paired.

Types: `AdwinDetector`, `ADWIN_DEFAULT_DELTA`,
`ADWIN_DEFAULT_WINDOW`.

Source: `src/adwin.rs`.

### `PotDetector` — SPOT / DSPOT univariate peaks-over-threshold

Streaming Peaks-Over-Threshold per-dimension anomaly detector
(Siffer et al., KDD 2017). For each feature dim it tracks a
running quantile `u` via the shipped `TDigest`, fits a Generalised
Pareto Distribution (GPD) to the peak excesses `(x − u) | x > u`
by method-of-moments, and returns the tail survival probability as
a p-value in `(0, 1]`. SPOT mode freezes the quantile after warm
via `freeze_baseline`; DSPOT mode keeps the digest drifting for
non-stationary streams.

Composition: run one `PotDetector` per feature dim, collect
per-dim p-values, pipe through `fisher_combine`. Joint p below
`1e-3` → anomaly. Orthogonal to RCF — the SPOT bank catches
per-dim marginal drift that isolation depth misses on
heterogeneously-distributed multivariate features (architect
review targets TSB-AD-M AUC lift 0.583 → 0.80+).

Types: `PotDetector`, `SPOT_DEFAULT_QUANTILE`, `SPOT_DEFAULT_ALERT_P`.

Source: `src/univariate_spot.rs`.

Example: `examples/univariate_spot_bank.rs` (4-dim baseline +
single-dim and all-dim outlier probes, joint p-value reported).

### `fisher_combine` — Fisher p-value combination

`fisher_combine(&[p_values]) -> f64` combines K independent
p-values into a joint anomaly score via Fisher 1932:
`T = −2 Σ ln(p_i) ~ χ²(2K)` → survival returned. Uses the
closed-form `χ²` survival for even dof.

Types: `fisher_combine`, `chi_squared_survival_even`.

Source: `src/ensemble.rs`.

## Streaming stats & sketches

Bounded-memory summaries reused across detectors — updating in
`O(1)` or `O(log n)` per observation, with read-time queries cheap
enough to run on every detection tick. Policy-free primitives;
detectors consume them internally, callers consume them directly
for dashboards or custom logic.

### Capacity planning — safe defaults

Right-sizing each sketch for a target working-set cardinality
costs a handful of `f64` arithmetic ops up front and saves
gigabytes of wasted counter bank at run time. The table below
captures the canonical sizing for the `1 000 000`-distinct-key
regime typical of day-scale NDR ingest; scale proportionally for
larger feeds.

| Sketch | Params for 1 M keys | Per-instance memory | Accuracy bound | Constructor |
|---|---|---|---|---|
| `CountMinSketch` | `width = 2048`, `depth = 4` | ~64 KiB counter bank | `ε ≈ 1.33 × 10⁻³` (`e/w`), `δ ≈ 1.83 × 10⁻²` (`(1/e)^d`) | `CountMinSketch::new(2048, 4)` |
| `HyperLogLog` | `precision = 12` | 4 KiB register bank | ≈ 1.625 % relative std error | `HyperLogLog::with_default_precision()` |
| `BloomFilter` | `capacity = 1_000_000`, `fpr = 0.01` | ~1.2 MiB bit bank (`m ≈ 9 585 059`, `k = 7`) | 1 % false-positive rate, 0 false negatives | `BloomFilter::new(1_000_000, 0.01)` |
| `SpaceSaving<K>` | `capacity = 1024` | ~32 KiB (16-byte keys × table × 2 for open-addressing) | Guaranteed retention of keys with true frequency `> N / K`; worst-case overestimate `≤ N / K` | `SpaceSaving::new(1024)` |

Rules of thumb:

- **Bloom at target FPR `p`**: bit bank ≈ `-1.44 · n · log₂(p)`
  bits. Doubling `n` costs 2× memory; halving `p` costs ~+1 bit
  per key.
- **CMS**: memory is `width · depth · 8 B`. Pick `width` from the
  additive-error budget `ε · N`; add rows until `δ` fits your
  confidence floor (usually 4–8 already gets you `(1/e)⁸ ≈
  3 × 10⁻⁴`).
- **HLL** is cardinality-insensitive by design — `p = 12` holds
  to `~10⁹` distinct elements at 1.6 % std error; bump to
  `p = 14` (16 KiB) when sub-percent error matters.
- **`SpaceSaving`** keeps exactly `K` entries — `K = 2 · target_topk`
  is a safe cushion; the theoretical guarantee is tight at `K =
  target_topk / ε` for relative-error `ε`.

### `OnlineStats` — Welford mean + variance

Numerically stable single-pass variance (Welford 1962 / Knuth
TAOCP vol. 2 §4.2.2) — updates `(mean, M2)` in place so the
returned sample variance is equivalent to a two-pass algorithm
without storing the stream.

Shared by the normalizer, per-feature CUSUM / EWMA detectors, and
any downstream consumer that needs a cheap streaming
`(mean, std_dev)` summary. `O(1)` per sample, read-time
`variance()` / `std_dev()` both `O(1)`. Layer warmup / decay /
per-feature arrays on top.

API: `new()` + `update(value: f64)` + `variance()` + `std_dev()`
+ Default.

Types: `OnlineStats`.

Reference: B. P. Welford, *Note on a Method for Calculating
Corrected Sums of Squares and Products*, Technometrics 4(3),
1962.

Source: `src/online_stats.rs`.

### `TDigest` — streaming quantile digest

`TDigest` is a streaming quantile estimator (Dunning 2019) with
sub-percent accuracy on tail quantiles (p99, p99.9) where
`ScoreHistogram`'s fixed bins lose resolution. `record(x)` is
`O(1)` amortised, `quantile(q)` is `O(δ)` where `δ` is the
compression parameter (default `100`). Merge via `merge(&other)`
for multi-shard aggregation; `postcard`-serialisable under the
`serde` feature for persistence.

Uses scale function 1 (`k_1(q) = (δ / 2π) · asin(2q − 1)`) — the
canonical choice for uniform error across the quantile range.

Types: `TDigest`, `Centroid`, `TDIGEST_DEFAULT_COMPRESSION`.

Source: `src/tdigest.rs`.

### `ScoreHistogram` — fixed-bin histogram

Standalone fixed-bin histogram for score / grade / CUSUM
streams. `record(value)`, `bins()`, `bin_edges()`, `total()`,
`underflow()` / `overflow()` / `non_finite()`, `merge(&other)`,
`percentile(p)`, `reset()`. Export to Grafana / Prometheus
without re-deriving per-minute aggregates in the monitoring
pipeline.

Types: `HistogramConfig`, `ScoreHistogram`.

Source: `src/histogram.rs`.

Example: `examples/observability.rs`.

### `BloomFilter` — probabilistic set membership

Word-packed bit bank of size `m` paired with `k` hash probes per
key. Sized from `(n, p)` — expected insert count `n` and target
false-positive rate `p` — via the optimal formulas
`m = ⌈−n · ln(p) / (ln 2)²⌉` and `k = round((m/n) · ln 2)`. At the
design load the filter meets its `p` budget; zero false negatives
hold unconditionally.

Typical IOC-lookup sizing at `p = 0.01`: 10 K entries → ≈ 12 KiB
and 7 hashes, 100 K → ≈ 120 KiB and 7 hashes, 1 M → ≈ 1.2 MiB
and 7 hashes. The `k = 7` pattern is characteristic of `p = 0.01`
and fixed across feed sizes.

Hashing uses a single `SipHash` pass (`DefaultHasher`) split into
two 64-bit lanes and combined by the Kirsch-Mitzenmacher
double-hashing trick `g_i(x) = h1 + i·h2 (mod m)` — halves the
hashing cost with no accuracy penalty at practical filter
parameters.

Gated behind `std`.

API: `new(capacity, fpr)` + `with_capacity(capacity)` (`fpr = 0.01`)
+ `with_params(num_bits, num_hashes)` (exact control) +
`insert<T: Hash>(&T)` + `insert_bytes(&[u8])` + `insert_hash(u64,
u64)` (escape hatch) + matching `contains*` variants + `union(&Self)`
(bitwise OR for cross-shard merge) + `reset()` + accessors
(`num_bits`, `num_hashes`, `memory_bytes`, `total_added`,
`effective_fpr`, `is_empty`).

Types: `BloomFilter`, `BLOOM_DEFAULT_FPR`, `BLOOM_MAX_HASHES`.

References:

1. B. Bloom, *Space/Time Trade-offs in Hash Coding with Allowable
   Errors*, CACM 13(7), 1970.
2. A. Kirsch, M. Mitzenmacher, *Less Hashing, Same Performance:
   Building a Better Bloom Filter*, ESA 2006.

Source: `src/bloom.rs`.

### `CountMinSketch` — probabilistic frequency sketch

`d` pairwise-independent hash rows over `w` counters. Each
`increment(key, c)` updates `d` cells (one per row); each
`estimate(key)` returns the minimum across the `d` cells the key
hashed to, guaranteeing `estimate(x) ≤ true_count(x) + ε·N` with
probability `1 − δ`, where `ε = e/w` and `δ = (1/e)^d` (Cormode
& Muthukrishnan 2005).

Default (`w=2048`, `d=4`): `ε ≈ 1.33·10⁻³`, `δ ≈ 1.83·10⁻²`,
memory `~64 KB` — enough headroom for per-flow or per-source
heavy-hitter counting over a multi-million-key stream.

Gated behind `std` because the row hashes rely on
`std::hash::DefaultHasher` (SipHash 1-3). The rest of the crate's
`no_std + alloc` surface is unaffected.

API: `new(width, depth)` + `increment(&[u8], u64)` (saturating)
+ `estimate(&[u8]) → u64` + `total()` + `reset()` +
`memory_bytes()` + `width()` / `depth()` accessors.

Types: `CountMinSketch`.

Reference: G. Cormode, S. Muthukrishnan, *An Improved Data Stream
Summary: The Count-Min Sketch and its Applications*, Journal of
Algorithms 55(1), 2005.

Source: `src/count_min_sketch.rs`.

### `HyperLogLog` — probabilistic distinct-count sketch

`m = 2^p` register bank. Each `add(x)` hashes `x` to 64 bits,
uses the top `p` bits as a register index and counts the leading
zeros of the remaining `64 − p` bits. Per-register state is
`max(zeros + 1)` across every element routed there. Cardinality
is recovered by a harmonic mean of `2^(-register)` with an `α_m`
bias correction, plus small-range linear counting when many
registers are still empty.

Memory is `m` bytes (one `u8` per register); typical `p=12`
gives 4 KiB + ≈1.625 % standard error, `p=14` → 16 KiB + 0.81 %,
`p=16` → 64 KiB + 0.40 %.

Gated behind `std` (uses `std::hash::DefaultHasher` — SipHash).

API: `new(p)` + `with_default_precision()` (`p=12`) +
`add<T: Hash>(&mut self, v: &T)` + `add_bytes(&mut self, &[u8])`
+ `add_hash(&mut self, u64)` (escape hatch for keyed hashers) +
`estimate(&self) -> u64` + `merge(&mut self, &Self)` (cross-shard
aggregation — per-register max) + `reset()` + accessors
(`register_count`, `precision`, `memory_bytes`, `total_added`).

Types: `HyperLogLog`, `HLL_DEFAULT_PRECISION`,
`HLL_MIN_PRECISION`, `HLL_MAX_PRECISION`.

References:

1. P. Flajolet, É. Fusy, O. Gandouet, F. Meunier, *HyperLogLog:
   the analysis of a near-optimal cardinality estimation
   algorithm*, AofA 2007.
2. S. Heule, M. Nunkesser, A. Hall, *HyperLogLog in Practice:
   Algorithmic Engineering of a State of the Art Cardinality
   Estimation Algorithm*, EDBT 2013.

Source: `src/hyperloglog.rs`.

### `SpaceSaving<K>` — deterministic top-K heavy hitters

Bounded table of at most `K` `(key, estimate, error)` entries.
On each observation:

1. Key already tracked → increment its `estimate`.
2. Key new, table below capacity → insert with `estimate = 1`,
   `error = 0`.
3. Key new, table full → evict the current minimum entry `m`,
   insert the new key with `estimate = m.estimate + 1` and
   `error = m.estimate`.

Guarantees (Metwally 2005):

- Every key with true frequency `> N/K` is retained.
- `estimate − error ≤ true_count ≤ estimate` for every tracked
  key. `error_bound()` returns the `N/K` worst-case overestimate.

Complements `CountMinSketch`: CMS is probabilistic per-key
frequency, `SpaceSaving` is deterministic top-K under an `O(K)`
memory cap. Typical `K = 128` costs ≈ 4 KiB for 16-byte keys
(IPv6 addresses / flow-hash tuples). Per-observe cost is `O(1)`
on the tracked-key path, `O(K)` on the evict path (linear scan
for the minimum).

Gated behind `std` (uses `std::collections::HashMap`).

API: `new(capacity)` + `with_default_capacity()` (`K = 128`) +
`observe(K)` + `observe_weighted(K, u64)` (byte-count heavy
hitters) + `estimate(&K) -> Option<HeavyHitterEntry>` +
`top_k(n) -> Vec<HeavyHitter<K>>` (ranked descending) + `iter()`
+ `reset()` + accessors (`capacity`, `len`, `is_empty`, `total`,
`error_bound`).

Types: `SpaceSaving<K>`, `HeavyHitter<K>`, `HeavyHitterEntry`,
`SPACE_SAVING_DEFAULT_CAPACITY`.

References:

1. A. Metwally, D. Agrawal, A. El Abbadi, *Efficient Computation
   of Frequent and Top-k Elements in Data Streams*, ICDT 2005.

Source: `src/space_saving.rs`.

### `Normalizer<D>` — per-feature min-max / z-score

Rescales a `D`-dimensional point into `[0, 1]` (MinMax) or
`(value − μ) / σ` (ZScore) given per-dimension `NormParams`
learned from a batch or loaded from a saved baseline. The `None`
strategy is an identity transform kept so the normalizer can sit
in a detection pipeline before fit data arrives.

Policy-free — the lib stores params and applies the math; caller
decides when to refit, when to swap, and what the source of the
samples is.

API: `identity(strategy)` (const fn) +
`fit(strategy, &[[f64; D]])` (two-pass: mean/min/max pass then
variance pass) + `transform(&[f64; D]) → [f64; D]`. Fields
`pub strategy` + `pub params: [NormParams; D]` so callers can
hand-tune individual dims after a fit.

MinMax clamps to `[0, 1]` and guards zero-range (returns `0.5`);
ZScore guards zero-variance (returns `0.0`).

Types: `Normalizer<D>`, `NormParams`, `NormStrategy`.

Source: `src/normalize.rs`.

## Forest scoring operations

RCF-specific scoring entry points beyond the bare `score()`. All
layered on `RandomCutForest<D>` (most also on `ThresholdedForest`
and `TenantForestPool`).

### Bulk batch scoring

`score_many(&[[f64; D]])` / `score_many_early_term(&[…], cfg)` /
`attribution_many(&[…])`. Under `parallel`, rayon parallelises
across points on top of the per-tree parallelism — 5-6× speedup
on backfill / SOC replay (see `docs/performance.md`). First error
aborts the batch. Available on `RandomCutForest`,
`ThresholdedForest`, and `TenantForestPool` (per-tenant;
absent-tenant returns `None` on read paths).

`score_many_with(points, |i, score| …)` — callback variant that
skips the intermediate `Vec<AnomalyScore>` allocation. Serial by
design so the closure needs no `Send + Sync` bound; preferred for
hot ingest paths that stream results directly into a writer /
histogram / alert pipeline.

Example: `examples/bulk_scoring.rs`.

### Score with confidence interval

`RandomCutForest::score_with_confidence(&point)` returns a
`ScoreWithConfidence` packaging the mean score plus per-tree
dispersion (`stddev`, `stderr`, `trees_evaluated`). SOC threshold
tuning benefits from knowing how tightly the ensemble agrees: a
`2.1 ± 0.05` verdict is qualitatively different from `2.1 ± 0.8`.
Call `ci95()` for the 95 % Gaussian CI, `ci(z)` for any custom
factor (99 % → `2.576`, 90 % → `1.645`). Always walks every tree;
use `score_early_term` when tail latency matters instead.

Types: `ScoreWithConfidence`, `DEFAULT_CI_Z_FACTOR` (`1.96`).

Source: `src/score_ci.rs`.

### Probe-based codisp scoring

`RandomCutForest::score_codisp(&point)` inserts the probe, walks
leaf → root in every tree accumulating
`max(sibling.mass / subtree.mass)` per level, then deletes the
probe. Matches the rrcf / AWS Java `codisp` semantic — captures
contextual displacement better than pure isolation depth on
wide-window anomalies. ~25× slower than `score()` post the
rayon-per-tree parallel walk + delete refactor; mutates the
reservoir per probe (baseline drifts on long eval streams, see
stateless variant below). On NAB `realKnownCause` lifts
aggregate AUC from 0.719 (`score()`) to 0.776 — beats rrcf
(0.748) and AWS Java (0.757).

`score_codisp_many(&[points])` — batched variant with per-tree
shared-walk cache + rayon across trees. Pre-inserts every probe,
walks once per unique leaf, bulk-deletes. Saturates the
reservoir past batch ≥ sample_size → `EmptyForest`.

`score_codisp_stateless(&point)` / `score_codisp_stateless_many`
— **drift-free** codisp estimate via root → leaf descent along
stored cuts, `max(sibling_mass / subtree_mass)` per depth, zero
reservoir mutation. Takes `&self`, parallel across trees,
preserves the frozen baseline exactly. Aggregate AUC 0.763 on
NAB, 0.751 on TSB-AD-M — ~0.01-0.02 below the mutating variant
but 12× faster on NAB (1.09 s full corpus vs 12.6 s). Preferred
for long eval streams.

Source: `src/forest/random_cut_forest.rs`, `src/tree/random_cut_tree.rs`.

### Fused score + attribution

`RandomCutForest::score_and_attribution(&point)` returns
`(AnomalyScore, DiVector)` from a **single tree walk** instead
of two — ~40 % faster than calling `score` + `attribution`
back-to-back. Uses `ScoreAttributionVisitor` which folds both
accumulators in one pass; scalar accumulator + per-dim
attribution split share the same `blend × damp` computation.

Types: `ScoreAttributionVisitor`.

Source: `src/visitor/combined.rs`.

### Early-termination scoring

`score_early_term(&point, config)` walks trees sequentially and
breaks as soon as the running per-tree mean has converged —
`stderr / |mean|` below `EarlyTermConfig::confidence_threshold`
after `min_trees`. Cuts wall-clock latency on
baseline-dominated traffic. `EarlyTermScore` reports
`trees_evaluated` / `trees_available` / `stderr` /
`early_stopped`. Sequential by design — use plain `score` when
full parallelism is preferred.

Types: `EarlyTermConfig`, `EarlyTermScore`.

Source: `src/early_term.rs`.

Example: `examples/early_term.rs`.

### Trimmed-mean ensemble scoring

`RandomCutForest::score_trimmed(&point, trim_fraction)` sorts
per-tree scores, drops the top + bottom `trim_fraction` fraction,
averages the middle. Robust against single-tree poisoning: an
attacker who manages to move a minority of trees' scores to the
extreme tails sees their contribution trimmed from the ensemble
mean. Typical `trim_fraction` values: `0.10` (10 %/10 %) or
`0.25` (quartile trim). `trim_fraction = 0.0` matches `score()`.

### Cross-tenant what-if

`TenantForestPool::score_across_tenants(&point)` pipes the same
probe through every resident tenant's detector and returns a
`Vec<(K, AnomalyGrade)>` sorted by descending grade. Warming-up
tenants are dropped. Intended for MSSP / threat-intel lateral
scans. Read-only — no tenant creation, no state mutation. Under
`parallel`, rayon fans out one `score_only` per tenant
(`O(N)` cost, near-linear at ~6.7 ms for 512 tenants).

Example: `examples/cross_tenant.rs`.

## Explanation & triage

*Why* did this point score high, and *how confident* is the
answer? These primitives turn scalar scores into analyst-
actionable narratives.

### Group-score decomposition

Declare named semantic groups over dim indices once
(e.g. `rate`, `payload`, `cardinality`). Every
`group_scores(&point, &groups)` returns per-group contributions
plus `top_group()` driver + `coverage()` (fraction of the raw
attribution mass the groups cover, catches gaps/overlaps). More
actionable than a 14-entry `DiVector`. Available on all three
detector types.

Types: `FeatureGroup`, `FeatureGroups`, `FeatureGroupsBuilder`,
`GroupScores`.

Source: `src/group_score.rs`.

Example: `examples/group_scores.rs`.

### `AttributionStability` — inter-tree dispersion

`attribution_stability(&point)` exposes per-dim variance /
stddev across trees alongside the mean. `confidence(dim)` is
`1 / (1 + CV)`; `argmax_weighted()` picks the dim with highest
`mean × confidence` — demotes dims where the forest disagreed.
`argmax_mean()` = classic `DiVector::argmax` for comparison.

Types: `AttributionStability`.

Source: `src/attribution_stability.rs`.

Example: `examples/attribution_stability.rs`.

### `SageEstimator<D>` — SAGE Shapley attribution

`SageEstimator<D>` (in `anomstream_core::sage`) — Monte-Carlo
permutation-sampling Shapley estimator (Covert et al. NeurIPS
2020). Accounts for feature interactions the marginal per-dim
`DiVector` ignores: when two dims jointly signal anomaly but
neither alone does, Shapley distributes the score contribution
across both. Caller supplies a baseline point (warm-phase mean
or synthetic null); estimator samples `K` random permutations,
computes each dim's marginal contribution as it joins the
coalition, averages. Cost `O(K · D)` forest scores per probe
— batch / forensic replay, not hot-path.

Types: `SageEstimator<D>`, `SageExplanation<D>`,
`SAGE_DEFAULT_PERMUTATIONS`, `SAGE_DEFAULT_SEED`.

Source: `src/sage.rs`.

### `PlattCalibrator` — probability calibration

`PlattCalibrator::fit(&[(score, label)], config)` fits a 1-D
sigmoid (Platt 1999 / Lin-Lin-Weng 2007) that maps raw scores to
`P(anomaly | score) ∈ [0, 1]`. Stable Newton-Raphson with
backtracking line search, target smoothing keeps
label-homogeneous sets numerically safe. `calibrate(score)` /
`calibrate_many(&scores)` at inference. Serde roundtrippable.
Intended for audit-defensible alerting policies (SOC2 / NIS2)
where a raw score is meaningless in compliance paperwork.

Online update via `PlattCalibrator::update_online(score, label,
lr)` applies one SGD step on the logistic loss per observation
— refine the fit as SOC feedback accumulates without re-running
the batch Newton-Raphson solver.

Types: `PlattCalibrator`, `PlattFitConfig`.

Source: `src/calibrator.rs`.

Example: `examples/calibrator.rs`.

### `SeverityBands` / `Severity` — ordinal classification

`SeverityBands` + `Severity` enum classify raw anomaly scores
into ordinal labels (`Normal` / `Low` / `Medium` / `High` /
`Critical`). Defaults match eBPFsentinel Enterprise ml-detection
(`2.0 / 3.0 / 4.0 / 5.0`). `Severity: Ord` lets callers route
alerts via `sev >= Severity::High`. Methods on both
`AnomalyScore::severity(&bands)` (bare forest) and
`AnomalyGrade::severity(&bands)` (TRCF). `SeverityBands::new`
validates `0 ≤ low < medium < high < critical`;
`SeverityBands::classify(score)` maps non-finite → `Normal`
(NaN-safe).

Types: `Severity`, `SeverityBands`.

Source: `src/severity.rs`.

Example: `examples/severity.rs`.

### `ForensicBaseline<D>` — imputation-like post-hoc baseline

`forensic_baseline(&point)` answers *"what would this dim have
looked like if the point were normal?"*. Returns
`ForensicBaseline<D>` with per-dim `expected` / `stddev` /
`delta` / `zscore` / `live_points` against every live sample in
the forest's reservoirs, plus `argmax_abs_zscore()`. Baseline
returned in raw caller coordinates — the internal `feature_scales`
transform is inverted. Great for SOC triage: alert table can
display *observed* vs *normal expected* per dim.

Types: `ForensicBaseline`.

Source: `src/forensic.rs`.

Example: `examples/forensic.rs`.

## SOC & ops

Turn a stream of alerts into something a human analyst can triage
without drowning — dedup, label, audit, retain.

### `AlertClusterer<K, D>` — cosine similarity dedup

`AlertClusterer<K, D>` groups near-duplicate `AlertRecord`s in a
sliding window so SOC dashboards see one cluster summary per
incident instead of hundreds of individual rows. Similarity is
cosine on the flattened attribution `DiVector` (`high ⧺ low`) —
two alerts with the same dominant driver and magnitude profile
cluster; unrelated attributions stay apart. Window is pruned
automatically on every `observe` call (or explicitly via
`prune_stale(now_ms)`).

Returns `ClusterDecision::{NewCluster(idx), Joined(idx)}` so
upstream can gate SIEM-write on `NewCluster` only and keep the
hot path cheap. Emits `rcf_alerts_observed_total`,
`rcf_alert_clusters_new_total`, `rcf_alert_clusters_joined_total`,
`rcf_alert_clusters_pruned_total` counters and the
`rcf_alert_clusters_active` gauge.

Active-cluster pool is hard-capped at `DEFAULT_MAX_CLUSTERS =
16 384` (override via `with_max_clusters`); exceeding the cap
LRU-evicts the oldest cluster. Per-cluster
`contributing_tenants` rolodex is hard-capped at
`MAX_TENANTS_PER_CLUSTER = 32` with FIFO eviction so an attacker
rotating synthetic tenant keys against a single cluster cannot
grow the membership vector unboundedly. The default `K = String`
clones the tenant identity once per distinct tenant joining a
cluster — at MSSP scale (10 k+ alerts / s, churning tenants)
prefer `AlertClusterer::<u64, D>` (or `Arc<str>` for refcount-
bump clones) to drop the heap traffic.

Types: `AlertClusterer`, `AlertCluster`, `ClusterDecision`,
`DEFAULT_MAX_CLUSTERS`, `MAX_TENANTS_PER_CLUSTER`.

Source: `src/alert_cluster.rs`.

### `LshAlertClusterer` — LSH-based dedup

`LshAlertClusterer` (`anomstream_triage::lsh_cluster`) — quantises
every per-dim signed attribution into a 16-bucket symbol and folds
the per-dim bucket sequence into a `u128` via FNV-1a. `O(1)`
lookup via `HashMap<u128, u64>`. Complement to the
cosine-similarity `AlertClusterer` — LSH scales to MSSP-volume
alert streams where pairwise cosine is too slow. Mirrors the
TLSH spirit (Oliver et al. 2013) without bigram-frequency
overhead.

Each instance draws a fresh 128-bit hash secret from
`getrandom` at construction (`LshAlertClusterer::new` /
`default_lsh`), pre- and post-XOR'd into the FNV state so an
offline-precomputed collision set against one clusterer cannot
be replayed against another (defends against
`AML.T0020`-style alert-deduplication crafting).
`LshAlertClusterer::with_seed(buckets, attr_cap, seed)` exposes
a deterministic variant for snapshot replay; **do not** hard-code
a constant seed in production.

Types: `LshAlertClusterer`, `LshClusterDecision`.

Source: `src/lsh_cluster.rs`.

### `FeedbackStore<D>` — SOC-label-driven score adjustment

`FeedbackStore<D>` (in `anomstream_core::feedback`) — bounded
ledger of analyst-labelled points. Das et al., *Incorporating
Feedback into Tree-based Anomaly Detection*, `arXiv:1708.09441`.
API: `label(point, FeedbackLabel::Benign | Confirmed)`,
`adjust(probe, raw_score) -> adjusted`. The adjustment adds a
Gaussian-kernel-weighted sum of every stored label's sign to the
raw score — Benign labels pull nearby probes **down**, Confirmed
labels push nearby probes **up**. Non-mutating on the forest
side: the hot-path `score()` is untouched, adjustment is an
additive bias layer the caller applies post-score.

Lifetime defaults: `capacity = 512`, `sigma = 1.0`, `strength =
1.0`. FIFO eviction on capacity pressure. Caller-configured
`capacity` is hard-capped at `MAX_CAPACITY = 65 536` so a
hostile config cannot drive the allocator into pressure with
`usize::MAX`. Clamps adjusted score to `≥ 0`.

`adjust(probe, raw_score)` is `O(D · L)` per call (where `L` is
the live label count): walks every stored label and computes a
`D`-element squared distance + one `exp` per label. At
`MAX_CAPACITY` and `D = 64`, that is ≈ 1 ms per call on a modern
core — fine for SOC triage cadence (one call per alert), **not**
for hot-path per-packet adjustment. High-`D` (`D ≥ 1024`) /
high-`L` (`L ≥ 16 384`) deployments should prune `capacity`,
shard per tenant, or pair with a spatial index.

Types: `FeedbackStore<D>`, `FeedbackLabel`,
`FEEDBACK_DEFAULT_CAPACITY`, `FEEDBACK_DEFAULT_SIGMA`,
`FEEDBACK_DEFAULT_STRENGTH`, `MAX_CAPACITY`.

Source: `src/feedback.rs`.

Example: `examples/feedback_adjust.rs` (baseline forest + benign
label on an outlier probe, adjusted score drops from 2.23 → 1.23
on the exact probe and from 2.35 → 1.36 on a nearby one).

### `AlertRecord<K, D>` — audit trail (NIS2 / SOC2)

`AlertRecord<K, D>` packages every analytic output (`score`,
`grade`, `attribution`, `baseline`, `severity`) plus provenance
(`tenant`, `timestamp_ms`, `point`) into one serialisable struct.
Build via `AlertRecord::from_forest` (bare RCF) or
`AlertRecord::from_thresholded` (TRCF); chain `with_severity` to
attach a band. Versioned through `ALERT_RECORD_VERSION` so
incompatible schema changes surface at decode time.
`#[serde(deny_unknown_fields)]` on `AlertRecord`,
`AlertRecordShadow`, and `AlertContext` rejects splice-field
attacks where a newer producer adds fields and an older consumer
silently drops them. Emit to a SIEM / object-store / WORM log
via any `serde` sink — postcard for compact per-event bytes,
JSON for self-describing records.

`AlertRecord` is an **operational log**, not a tamper-evident
audit trail — the `serde` roundtrip carries no integrity check
beyond the version prefix. For SOC2 CC6 / NIS2 / PCI-DSS 10.5
audit-trail requirements, wrap each emission in `AuditChain`
(below).

Types: `AlertRecord`, `AlertContext`, `ALERT_RECORD_VERSION`.

Source: `src/audit.rs`.

### `AuditChain<K, D>` — tamper-evident HMAC envelope

`anomstream-triage` ships an HMAC-SHA256 audit chain behind the
`audit-integrity` feature. Each emitted record lands in an
`AuditChainEntry { record, seq, prev_tag, tag }` where `tag =
HMAC-SHA256(key, u64_le(seq) || prev_tag || postcard(record))`,
chaining `prev_tag` from the previous entry's tag so reordering
or deletion breaks the linkage. `verify_chain(entries, key,
genesis_prev)` walks the chain end-to-end with constant-time tag
comparisons (via the `subtle` crate); `AuditChain::with_genesis`
resumes appending against persisted chain state.

Threat model and key-management caveats (key rotation, at-rest
encryption, cross-chain replay) live in `docs/threat_model.md`
under T5.

Types: `AuditChain`, `AuditChainEntry`, `verify_audit_chain`,
`AUDIT_CHAIN_GENESIS_PREV`, `AUDIT_CHAIN_TAG_LEN`,
`AUDIT_CHAIN_MIN_KEY_LEN`. Pulls `hmac` + `sha2` + `subtle` (all
no-default-features) and is std-only.

Source: `src/audit_chain.rs`.

## Training & retention

### Cold-start bootstrap

`bootstrap(points)` replays historical points from an upstream
TSDB (Prometheus / Loki / parquet dump) before live traffic is
switched on — eliminates the per-restart warmup coverage hole.
Non-finite rows are skipped and tallied in the returned
`BootstrapReport`. Available on `RandomCutForest`,
`ThresholdedForest`, and `TenantForestPool` (per-tenant).

Types: `BootstrapReport`.

Source: `src/bootstrap.rs`.

Example: `examples/bootstrap.rs`.

### Point timestamps + retention

`update_at(point, ts)` / `update_indexed_at(point, ts)` /
`process_at(point, ts)` / `process_indexed_at(point, ts)` tag
each fresh observation with a caller-supplied `u64` (epoch-ms,
sequence number…). `delete_before(cutoff)` retracts every older
point. `point_timestamp(idx)`, `oldest_timestamp()`,
`newest_timestamp()`, `tracked_timestamps()` for diagnostics.
Reservoir eviction cleans the side-map automatically. Per-tenant
on the pool. Intended for GDPR / NIS2 data retention and
forensic replay of a specific period.

Example: `examples/retention.rs`.

### Explicit retraction

`update_indexed` returns the fresh `point_idx`; `delete(idx)` /
`delete_by_value(&point)` remove a previously-observed point.
`process_indexed` on TRCF returns `(usize, AnomalyGrade)`.
Intended for SOC-driven false-positive retractions without
waiting for natural reservoir eviction. Pool variants do **not**
auto-create tenants on retraction.

Example: `examples/delete_and_scales.rs`.

### Per-dim feature scales

`ForestBuilder::feature_scales([f64; D])` pre-scales every caller
point before it reaches the forest hot paths. Use when dims have
wildly different dynamic ranges (packet-rate, protocol ratios,
entropy, cardinality); pass `1 / stddev[d]` per dim for
unit-variance normalisation. Mean-centre upstream for full
z-score. For richer pre-processing pipelines, `Normalizer<D>`
(see [Streaming stats & sketches](#streaming-stats--sketches))
supports `MinMax` / `ZScore` / `None` strategies with a batch
learner.

### Cold-start warmup (reservoir)

`initial_accept_fraction` ramps the reservoir admission
probability over the first `initial_accept_fraction · sample_size`
offers so the very first points do not dominate the sample. `1.0`
default disables the gate; AWS `CompactSampler` uses `0.125`.

## Persistence

### Warm reload

Serde-gated `to_bytes` / `from_bytes` (postcard binary) and
`to_json` / `from_json` on both `RandomCutForest` and
`ThresholdedForest`. Std-gated `to_path` / `from_path` /
`to_json_path` / `from_json_path` apply an atomic write
discipline (`tmp + fsync + rename`) — a mid-save crash leaves
the previous snapshot intact. Versioned envelope
(`PERSISTENCE_VERSION` / `THRESHOLDED_PERSISTENCE_VERSION`);
incompatible versions reject up front before any third-party
deserialiser runs.

Source: `src/persistence.rs`.

Example: `examples/warm_reload.rs`.

## Observability

### `MetricsSink` trait

`with_metrics_sink(Arc<dyn MetricsSink>)` on every detector
streams counters / gauges / histogram observations. `NoopSink`
is the zero-cost default. `TestSink` records events in-memory
for tests. Callers plug their own Prometheus / StatsD / OTel
adapter against the trait.

Canonical metric names (`metrics::names::*`):

| Type | Name | Source |
|---|---|---|
| counter | `rcf_updates_total` | every `RandomCutForest::update` |
| counter | `rcf_deletes_total` | every `delete` that actually removed a point |
| counter | `rcf_attribution_total` | every successful `attribution` |
| counter | `rcf_rejected_nan_total` | per-call NaN/±inf rejection, data-quality signal |
| counter | `rcf_early_term_stopped_total` | `score_early_term` short-circuits |
| counter | `rcf_process_total` | every `ThresholdedForest::process` |
| counter | `rcf_anomalies_fired_total` | `process` whose verdict was `is_anomaly` |
| counter | `rcf_drift_fires_total` | aggregate CUSUM fire (up + down) |
| counter | `rcf_drift_up_total` | CUSUM upward drift fires |
| counter | `rcf_drift_down_total` | CUSUM downward drift fires |
| counter | `rcf_tenant_evictions_total` | pool eviction (LRU + TTL, aggregate) |
| counter | `rcf_tenant_idle_evictions_total` | TTL-driven eviction subset |
| counter | `rcf_tenant_created_total` | pool factory invocation (fresh tenant) |
| counter | `rcf_bootstrap_points_total` | bootstrap-ingested points |
| counter | `rcf_bootstrap_skipped_total` | bootstrap points skipped (non-finite) |
| counter | `rcf_alerts_observed_total` | every `AlertClusterer::observe` |
| counter | `rcf_alert_clusters_new_total` | new cluster opened |
| counter | `rcf_alert_clusters_joined_total` | alert merged into existing cluster |
| counter | `rcf_alert_clusters_pruned_total` | cluster dropped by window prune |
| gauge | `rcf_alert_clusters_active` | active clusters in `AlertClusterer` |
| counter | `rcf_feature_drift_observed_total` | every `FeatureDriftDetector::observe` |
| gauge | `rcf_feature_drift_max_psi` | max per-dim PSI on `psi()` call |
| gauge | `rcf_forest_trees` | tree count of a forest |
| gauge | `rcf_threshold_current` | TRCF adaptive threshold |
| gauge | `rcf_ema_mean` | TRCF score-stream EMA mean |
| gauge | `rcf_ema_stddev` | TRCF score-stream EMA stddev |
| gauge | `rcf_observations_seen` | TRCF EMA observation count (warmup progress) |
| gauge | `rcf_tenants_resident` | live tenants in pool |
| gauge | `rcf_tenant_capacity` | configured pool capacity |
| histogram | `rcf_score` | raw anomaly score per scored point |
| histogram | `rcf_grade` | graded verdict `[0, 1]` per processed point |
| histogram | `rcf_drift_s_high` | CUSUM upward accumulator |
| histogram | `rcf_drift_s_low` | CUSUM downward accumulator |
| histogram | `rcf_early_term_trees` | trees walked per `score_early_term` |
| counter | `rcf_hot_path_sampler_accepted_total` | `UpdateSampler::accept_*` admitted |
| counter | `rcf_hot_path_sampler_rejected_total` | `UpdateSampler::accept_*` rejected |
| counter | `rcf_hot_path_queue_enqueued_total` | `UpdateProducer::try_enqueue` landed |
| counter | `rcf_hot_path_queue_dropped_total` | `UpdateProducer::try_enqueue` dropped on full |
| counter | `rcf_hot_path_prefix_admitted_total` | `PrefixRateCap::check_and_record` admitted |
| counter | `rcf_hot_path_prefix_capped_total` | `PrefixRateCap::check_and_record` capped |
| counter | `rcf_drift_aware_swaps_total` | `DriftAwareForest` shadow → primary swap |
| counter | `rcf_drift_aware_on_drift_total` | `DriftAwareForest::on_drift` actually spawned a shadow |
| gauge | `rcf_drift_aware_shadow_active` | 1.0 while shadow warming, 0.0 otherwise |
| counter | `rcf_adwin_observed_total` | `AdwinDetector::update` folded a finite value |
| counter | `rcf_adwin_drift_fires_total` | `AdwinDetector::update` detected drift |
| counter | `rcf_lsh_alerts_observed_total` | `LshAlertClusterer::observe` call |
| counter | `rcf_lsh_clusters_new_total` | LSH alert opened a new bucket |
| counter | `rcf_lsh_clusters_joined_total` | LSH alert merged into existing bucket |
| gauge | `rcf_lsh_clusters_active` | distinct active LSH cluster hashes |
| counter | `rcf_feedback_labels_observed_total` | `FeedbackStore::label` accepted |
| counter | `rcf_feedback_labels_benign_total` | label verdict `Benign` |
| counter | `rcf_feedback_labels_confirmed_total` | label verdict `Confirmed` |
| counter | `rcf_spot_observations_total` | `PotDetector::record` folded a finite value |
| counter | `rcf_spot_peaks_total` | value above the SPOT/DSPOT threshold `u` |

Every detector exposing these ships a `.with_metrics_sink(Arc<dyn
MetricsSink>)` chain-style builder; default is `NoopSink`.

Metric names keep the `rcf_` prefix for continuity with existing
deployments that pre-date the toolkit-framing scope expansion.

Source: `src/metrics.rs`.

## Hot-path integration (eBPF ingress)

### `UpdateSampler` + `channel` MPSC split

`anomstream_core::hot_path::UpdateSampler` drops low-value updates
before any RCF work. Two decision modes:

- `accept_stride()` — monotonic counter, keeps `1 / keep_every_n`
  offers deterministically.
- `accept_hash(flow_hash)` — per-flow sampling. Preserves
  baseline shape per flow rather than slicing any single flow.

Build via `UpdateSampler::new(keep)` (unkeyed, deterministic
admission — back-compatible), **`UpdateSampler::new_keyed(keep)`**
(128-bit secret from `getrandom`, murmur3 keyed mix applied
before the modulo), or
**`UpdateSampler::new_keyed_with_seeds(keep, k1, k2)`**
(caller-supplied seeds, for restricted environments where
`getrandom` is unavailable — embedded boot, chroot without
`/dev/urandom`, `wasm32-unknown-unknown`). Keyed sampler defends
against the reservoir-poisoning spray (MITRE ATLAS `AML.T0020`).

`update_channel::<D>(capacity)` returns `(UpdateProducer<D>,
UpdateConsumer<D>)` — bounded MPSC on
`std::sync::mpsc::sync_channel`. Capacity is validated
`1..=MAX_CHANNEL_CAPACITY` (`1 << 20` slots) at construction;
caller cannot OOM the allocator with `usize::MAX` or silently
drop every offer with `0`. The non-panicking
`try_update_channel` Result-returning variant surfaces the bound
as `RcfError::InvalidConfig`. Clone the producer per classifier
thread; hand the consumer to a dedicated updater thread.
`try_enqueue` is non-blocking; on queue-full it drops +
increments `dropped_total` (ops signal for "classifier outpaces
updater"). The classifier scores against the
previous-generation forest snapshot while the updater drains at
its own cadence.

`MetricsSink` dispatch on `accept_*` / `try_enqueue` /
`check_and_record` is **batched every `METRICS_BATCH_SIZE = 64`
hot-path calls** — line-rate load no longer pays the per-call
`Arc<dyn>` vtable cost (≈13 ns × 12.5 Mpps ≈ 160 ms / s of
clock burned on dispatch). The in-process atomic counters
(`accepted_total`, `enqueued_total`, `admitted_total`, …) stay
bit-exact every call; the sink view lags by ≤ 64 increments.
Call `flush_metrics()` on shutdown to drain the residue.

Types: `UpdateSampler`, `UpdateProducer<D>`, `UpdateConsumer<D>`,
`MAX_CHANNEL_CAPACITY`, `METRICS_BATCH_SIZE`. Functions:
`update_channel`, `update_channel_with_sink`,
`try_update_channel`, `try_update_channel_with_sink`.

Source: `src/lib.rs` (anomstream-hotpath crate).

### `PrefixRateCap` — per-prefix admission rate cap

`PrefixRateCap::new(cap: NonZeroU32, window: NonZeroU64)` bounds
how many admissions a single source-prefix hash bucket can push
into the reservoir within a rolling window. Both arguments are
`NonZero` typed so the previous footgun pair (`window_ms == 0`
panicked, `cap_per_window == 0` silently disabled the cap) is
impossible to express; use `PrefixRateCap::disabled(window)` for
the explicit always-admit mode.

Fixed 256-bucket counter sketch, lock-free `check_and_record`.
Each bucket is **cache-line padded** (`#[repr(C, align(64))]`,
16 KiB total vs 1 KiB unpadded) so concurrent `fetch_add`s on
different buckets do not bounce a shared cache line through
MOESI/MESI — multi-core throughput stays close to the single
thread cost. The bench
`hot_path_prefix_cap/check_and_record_contended_8threads`
quantifies the gain.

Soft over-admission window of ≈ `4 × cap_per_window` per bucket
per window under heavy concurrent load (the `fetch_add` +
cap-comparison sequence is two distinct atomics — design the
operator-facing cap with that slack baked in by passing
`cap_per_window = ceiling / 4`).

Second defence line (alongside the keyed `UpdateSampler`)
against reservoir-poisoning floods from a single compromised
source — documented in `docs/threat_model.md`.

Types: `PrefixRateCap`.

## Security & threat model

See [`docs/threat_model.md`](threat_model.md) for the full
adversarial threat model covering reservoir poisoning,
evasion via contextual shift, model extraction, and classifier-
side resource exhaustion — with the MITRE ATLAS technique IDs
and the defences shipped in-crate for each.

### SemVer hygiene (v1)

Public enums expected to grow carry `#[non_exhaustive]` so additive
variants do not require a major version bump — callers must use a
`_` wildcard arm. Enums tagged: `DriftLevel`, `DriftDirection`,
`DriftKind`, `Severity`, `ThresholdMode`, `ClusterDecision`,
`FeedbackLabel`, `LshClusterDecision`.

Public structs with `pub` fields carry `#[non_exhaustive]` so
additional fields can be added without breaking downstream literal
construction: `RcfConfig`, `AlertRecord`, `UpdateSampler`,
`UpdateProducer`, `UpdateConsumer`, `PrefixRateCap`. Assemble
`AlertRecord` through `AlertRecord::new(...)` /
`AlertRecord::from_forest(...)` / `AlertRecord::from_thresholded(...)`
rather than a struct literal.

Every `.score*()` on `RandomCutForest`, every `ForestBuilder` /
`ShingledForestBuilder` / `ThresholdedForestBuilder` build method,
and every `.observe()` / `.update()` / `.record()` /
`.process()` / `.adjust()` / `.explain()` returning a non-unit
verdict is `#[must_use = "…"]`. Drops in hot paths must use
`let _ = detector.observe(x);` explicitly.

### v1 polish batch

Grab-bag of pre-release cleanups driven by the audit:

- **Cross-crate inlining hints.** Hot sketch helpers —
  `BloomFilter::{insert,insert_bytes,insert_hash,contains,
  contains_bytes,contains_hash,combined_index,set_bit,get_bit}`,
  `HyperLogLog::{add_hash,add_bytes}`, `CountMinSketch::{increment,
  estimate,hash_to_col}`, `SpaceSaving::{observe,observe_weighted}`
  — carry `#[inline]` so `rustc` inlines them across the
  anomstream-core / anomstream-hotpath crate boundary. Measured:
  **bloom `contains_bytes` -11 %** (32.0 → 28.8 ns), **`insert_hash`
  -12 %** (18.5 → 16.8 ns). Other sketches stay flat within
  noise — they were already reachable through mono-generic
  dispatch that rustc inlines regardless.
- **Bounded `Debug` on large stateful types.** `CountMinSketch`
  and `RandomCutForest<D>` now implement a manual `fmt::Debug`
  that prints `width / depth / memory` and `num_trees /
  sample_size / live_points` summaries respectively. The derived
  impls were dumping up to tens of MiB of counter / tree state
  on a single `{:?}` call — a trap for anyone inadvertently
  logging `debug!("{forest:?}")`.
- **`MatrixProfile` window cap.** `MAX_WINDOW = 10_000`; the
  first-column seed is `O(n · m)` and doubling `m` doubles the
  cost, so a runaway `window` turned forensic calls into
  compute bombs. The public docstring now carries a
  `# Complexity` note with practical wall-clock numbers.
- **`AlertClusterer` tenant key.** Docstring updated to
  recommend `AlertClusterer::<u64, D>` for deployments that
  carry numeric tenant identifiers; the default `K = String`
  stays for JSON ergonomics but incurs a string-hash + heap
  allocation per tenant-set insert.
- **`deny.toml`: MPL-2.0 removed from the blanket allow list.**
  Static-link redistribution from an Apache-2.0 product into a
  proprietary binary stays safe only when MPL source is
  unmodified; consumers needing per-crate exceptions add them
  to the `exceptions` list with a written justification.
- **`MetricsSink` name stability.** Module-level docstring on
  `metrics::names` now carries an explicit `SemVer` guarantee:
  identifiers and their string values never change across patch
  / minor releases, so Prometheus / Grafana queries referencing
  them survive every non-major bump. Explains why
  `inc_counter` takes `&str` rather than an enum
  (integrator-supplied dynamic names).
- **Facade `SemVer` scope documented.** `meta/src/lib.rs`
  module doc draws the line between the committed public surface
  (the catalogue of ~80 types printed in `README.md` / this file)
  and the glob-reachable transitive set. Consumers who want
  strict compile-time pinning should import from member crates
  directly.

### Supply-chain + build hygiene

- **`RcfError::InvalidConfig(Box<str>)`** — the hot-path variant
  drops from 24 B to 16 B on 64-bit targets by switching the
  payload from `String` to `Box<str>`. The `Display` impl,
  `.contains(..)` checks, and every existing callsite keep
  working via `Deref<Target = str>`.
  `SerializationFailed` / `DeserializationFailed` stay `String`
  because their emission sites always allocate a fresh message
  from upstream errors — forcing a `String → Box<str>` hop would
  spend the saved 8 B and more.
- **Intra-doc links are strict.** Every member-crate doc link
  uses an absolute `[\`crate::X\`]` path or an in-scope
  identifier, so rustdoc resolves cleanly under both the member
  and the facade namespaces. The former blanket `#![allow(
  rustdoc::broken_intra_doc_links)]` is gone; CI runs
  `RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links"` and fails
  on new breakage.
- **CI MSRV + minimal-versions pin.** `dtolnay/rust-toolchain` is
  now version-pinned (`@1.12.0`) so a compromised upstream push
  cannot swap the toolchain download under CI. A new
  `minimal_versions` job runs `cargo +nightly update -Z
  minimal-versions && cargo check --workspace --all-features`;
  loose `foo = "1"` requirements that rely on features only
  added in `1.x.y+n` trip this job.
- **`deny.toml` acknowledges dev-only `getrandom 0.4`.** The
  duplicate comes exclusively from `proptest → tempfile →
  rusty-fork`, all dev-dependencies; the release graph still
  links a single `getrandom 0.3`. Revisit when proptest ships an
  update pinning tempfile on the same major as `rand`.

### Calibration + feedback quality

- **Platt skew fallback** — `PlattCalibrator::fit` now detects
  class imbalance greater than `DEFAULT_SKEW_THRESHOLD = 100` and
  skips the Newton-Raphson path whose Hessian goes near-singular
  at that regime. Skew cases initialise `(a, b)` from the smoothed
  prior (`a = 0`, `b = ln((N− + 1) / (N+ + 1))`) and then run
  `DEFAULT_SKEW_SGD_EPOCHS = 16` passes of online logistic SGD at
  `DEFAULT_SKEW_SGD_LR = 0.01`. The returned calibrator carries a
  `high_skew()` flag so SOC dashboards can surface the fallback
  condition — extreme imbalance is typically a labelling-pipeline
  regression the calibrator cannot fix on its own. Regression
  test `severe_skew_triggers_sgd_fallback_and_still_separates`
  pins the 200:1 separability floor.
- **`PlattCalibrator::update_online` gradient sign fix** — the
  SGD step subtracts `lr · (y − p) · s` (derived from
  `dL/da = (y − p) · s` for the Platt parameterisation
  `p = 1 / (1 + exp(a · s + b))`). The earlier code flipped the
  sign and drifted the calibrator in the ascent direction under
  sustained online updates. Behavioural break for callers relying
  on the previous wrong direction; pre-v1 so intentional.
- **`FeedbackStore::adjust` kernel-weighted mean** — the
  Gaussian-kernel-weighted sum is now normalised by the sum of
  kernel weights, yielding a bias in `[-1, 1]` regardless of
  stored-label count. Previously `500` Confirmed labels at a
  probe added `500 · strength` to the raw score; now they add at
  most `strength`. Mixed labels interpolate — `k` Confirmed / `m`
  Benign at a probe give bias `(k − m) / (k + m)`. Regression
  tests pin both the hard-bound and the mixed-label interpolation
  semantics.

### Default sink allocation

`metrics::default_sink()` (used by every `UpdateSampler::new` /
`update_channel` / detector constructor) now clones a
process-wide `LazyLock<Arc<dyn MetricsSink>>` instead of building
a fresh `Arc::new(NoopSink)` per call. First call initialises
the static; every subsequent call is a single relaxed-atomic
refcount bump — measured **12 ns** on a modern core. No
observable behavioural change (the noop sink is stateless); the
shared-Arc identity is enforced by a regression test
(`Arc::ptr_eq`).

### Hotpath correctness

- **Sampler key seeding** — `UpdateSampler::new_keyed` always
  forces the `mix_k1` multiplier odd via `| 1`. The previous
  code had a zero-check that, on the 2⁻⁶⁴ chance of `getrandom`
  returning exactly zero, swapped in a publicly-known constant;
  the current code degrades to `1` instead (still odd, still a
  valid multiplicative bijection, non-deterministic for attackers).
- **PrefixRateCap rollover** — window reset is a
  `compare_exchange_weak` loop with `AcqRel` on success and
  `Acquire` on the rollover-side load. Happens-before order
  guarantees that the bucket zero-fill completes before any peer
  thread's subsequent `fetch_add`. Closes the soft-over-admission
  window the earlier `load` → `compare_exchange` code had, with
  no perf regression (measured **8.9 ns** per `check_and_record`,
  -60 % vs the earlier code — the `Acquire` load short-circuits
  the common "window still valid" case).
- **Counter semantics** — every `*_total` counter
  (`UpdateSampler::accepted_total`, `UpdateProducer::enqueued`,
  `PrefixRateCap::admitted_total`, etc.) is a plain
  `AtomicU64::fetch_add(1, Relaxed)`. Atomic `fetch_add` is
  wrapping by definition — `overflow-checks` does not apply to
  atomic operations. At 10 Gpps sustained load a `u64` wraps in
  ~58 years; export cadence is an ops choice, not a correctness
  requirement.
- **`channel` rename** — the free function that built the MPSC
  pair is now `anomstream_hotpath::update_channel` (previously
  `channel`); the old name shadowed `std::sync::mpsc::channel`
  in use-star imports. Paired
  `update_channel_with_sink(capacity, sink)`.

### Release discipline

CI is split across two workflows:

- `.github/workflows/ci.yml` — fires **weekly** (Monday 03:00 UTC)
  and on `workflow_dispatch`. Runs fmt, clippy, `no_std` checks,
  feature-matrix build, workspace tests on stable + MSRV, doc,
  `cargo bench --no-run`, examples, `cargo audit`, `cargo deny`,
  `cargo machete`, CycloneDX SBOM. Does **not** fire per-commit —
  the weekly cadence catches supply-chain drift without burning
  runner minutes on noise.
- `.github/workflows/release.yml` — fires on every `v*` tag push
  and on `workflow_dispatch`. Enforces a stricter gate before
  publication is considered ready:
  1. Workspace version is not the `0.0.0-dev` development
     sentinel.
  2. Git tag matches `workspace.package.version` (`v<version>`).
  3. Full fmt / clippy / tests / doc.
  4. `cargo audit` + `cargo deny check` clean.
  5. `cargo publish --dry-run` for every member in dependency
     order (`core → triage → hotpath → anomstream`).

The workflow gates readiness; `cargo publish` itself is still run
manually by a maintainer with `CARGO_REGISTRY_TOKEN`.

### Hot-path allocation scrub

Three hot-path sites are now alloc-free:

- `FeedbackStore<D>` — label storage switched from `Vec` to
  `VecDeque`. Oldest-first eviction on capacity pressure is
  `O(1)` via `pop_front`; the old `Vec::remove(0)` memmoved the
  full ring per label (`O(capacity)`), a measurable cost at the
  default 512-cap under sustained SOC labelling.
- `RandomCutForest::score_trimmed` — per-call `Vec::with_capacity`
  scratch replaced by a `thread_local!` `RefCell<Vec<f64>>`
  reused across calls. Per-packet scoring on an NDR hot path no
  longer allocates. Falls back to a fresh `Vec` under `no_std`
  where `thread_local!` is unavailable.
- `LshAlertClusterer::hash_divector` / `observe` — bucket key
  changed from a `format!`-built hex `String` to a `u128`
  FNV-1a fold of the per-dim quantised buckets. Measured:
  hash_divector 430 ns → 73 ns (−82 %), observe 480 ns → 95 ns
  (−80 %). Zero heap allocations per-alert on the MSSP hot
  path. Breaking API: `observe` now returns `(u128,
  LshClusterDecision)` and `cluster_size` takes `u128` by value.

### Allocation caps (DoS hardening)

Caller-supplied sizing parameters on types that allocate
proportionally to their inputs are bounded by `MAX_*` constants
at construction so an attacker-controlled
`new(width, depth, …)` call cannot force an OOM:

- `CountMinSketch::new(width, depth)` — `width ≤ MAX_WIDTH` (262 144)
  and `depth ≤ MAX_DEPTH` (16). Caps worst-case allocation at
  ~32 MiB of counter table. `new` now returns `RcfResult<Self>`.
- `BloomFilter::with_params(num_bits, num_hashes)` —
  `num_bits ≤ MAX_NUM_BITS` (1 Gibit = 128 MiB of bit bank),
  `num_hashes ≤ MAX_HASHES` (64). Deserialize path enforces the
  same cap via `BloomFilterShadow`'s `TryFrom`.
- `SageEstimator::new(baseline, permutations, seed)` —
  `permutations ≤ MAX_PERMUTATIONS` (65 536). Each permutation
  triggers `D + 1` forest scores; the cap bounds the worst-case
  `explain()` cost at ~4 s for `D = 16` on a modern core.
- `AlertClusterer` — active-cluster pool bounded by
  `DEFAULT_MAX_CLUSTERS` (16 384) and overridable via
  `with_max_clusters(max)`. On cap hit, the oldest cluster
  (smallest `last_seen_ms`) is LRU-evicted before a new one opens.
  Protects against adversarial high-cardinality attribution
  streams that would otherwise keep opening fresh clusters
  within `window_ms`.

### Serde deserialization hardening

All stateful public types that derive `Deserialize` route through
a private `<Name>Shadow` struct with plain derive, and a
`TryFrom<Shadow>` impl that re-runs the constructor's invariant
checks before a live value escapes. Attacker-controlled
`postcard::from_bytes(bytes)` therefore cannot produce a
`BloomFilter` with `num_hashes > MAX_HASHES`, a `HyperLogLog`
whose register bank length disagrees with `precision`, a
`ScoreHistogram` with NaN bounds, an `RcfConfig` outside the AWS
SageMaker bounds, a `PerFeatureCusumAccumulator` seeded with
NaN/Inf, or an `AlertRecord` with a mismatched
`ALERT_RECORD_VERSION`. Types hardened this way:
`BloomFilter`, `HyperLogLog`, `HistogramConfig`, `ScoreHistogram`,
`RcfConfig`, `PerFeatureCusumAccumulator`, `PerFeatureCusumConfig`,
`AlertRecord`. Adversarial-deserialize regression tests live
alongside the roundtrip tests in each module.

## Multi-tenancy

### Tenant similarity index

`TenantForestPool::similarity_matrix(min_obs)` /
`most_similar(&key, top_n, min_obs)` return pairwise tenant
similarity ∈ `(0, 1]` computed on each tenant's TRCF
score-stream EMA (`mean`, `stddev`). Similarity =
`exp(-sqrt(Δmean² + Δstddev²))`. Intended for SaaS deployments
with thousands of tenants — identify clusters for tiered
alerting or spot a tenant whose baseline drifts away from its
peer group. Tenants below `min_obs` samples are excluded.

`similarity_matrix` parallelises the pair enumeration under
`parallel`; `most_similar` uses a bounded `BinaryHeap<top_n>` for
`O(N · log top_n)` — microsecond-scale at 512 tenants (~9 µs).

Example: `examples/tenant_similarity.rs`.

## Quality

### AWS SageMaker conformance

`tests/aws_conformance.rs` asserts the hyperparameter bounds,
score-averaging semantics and reservoir behaviour against the
published AWS RCF spec. Regression-guards the wire-level
compatibility story whenever the crate's internals move.

### Detection-quality regression guards

`tests/detection_quality.rs` pins AUC floors on synthetic
corpora (separable clusters > 0.95, transition > 0.90).
`tests/nab.rs` (ignored by default, needs `RCF_NAB_PATH`) pins
the NAB `realKnownCause` weighted aggregate at 0.70 via the
lag=32 + zscore + smooth(0.02) pipeline.
`tests/tsb_ad_m.rs` (ignored, needs `RCF_TSB_AD_M_PATH`) exercises
the native multivariate path across 192 / 200 files spanning
D ∈ {2 .. 66}. Two test functions: `tsb_ad_m_aggregate_auc_above_floor`
runs the fast `score()` path, `tsb_ad_m_codisp_aggregate_auc_above_floor`
runs `score_codisp()` stride-subsampled to 50 k eval rows per file
for direct parity with the AWS Java / rrcf codisp semantic. Both
pin a 0.55 aggregate floor. See `docs/performance.md` for the
per-source breakdown.

### Quality metrics — VUS-PR (`vus_pr` module)

`vus_pr` + `vus_pr_with_buffer` + `range_auc_pr` implement the
Volume-Under-Surface PR metric (Paparrizos VLDB 2022) — a
threshold-free, length-aware AUC-PR for time-series anomaly
detection. For each buffer size `l ∈ [0, L]` the metric inflates
the anomaly mask by `l` positions on both sides (precision side)
and dilates the prediction set by `l` positions (recall side),
computes `RangeAUCPR(l)`, then trapezoidally integrates over `l`.
Rewards detectors whose scores peak *near* the true anomaly
instead of exactly on it — the key weakness of point-wise AUC on
TS workloads.

Default `DEFAULT_MAX_BUFFER = 100` matches the typical average
anomaly length on TSB-AD-M. Complexity `O(n · L²)` for the full
surface; use `range_auc_pr(_, _, l)` when only a single `l` is
needed.

Types: `vus_pr`, `vus_pr_with_buffer`, `range_auc_pr`,
`VUS_PR_DEFAULT_MAX_BUFFER`.

Source: `src/vus_pr.rs`.

### `TsbAdMDataset` — TSB-AD-M CSV loader

Dependency-free CSV loader for the TSB-AD-M benchmark dumps (Liu
& Paparrizos NeurIPS 2024). Parses the header, finds the `Label`
column (any case-insensitive match, any position), loads the
remaining columns as `f64` features, and exposes `.features`,
`.labels`, `.feature_headers`, plus `column(c)` for univariate
projections.

`TsbAdMDataset::load_csv(path)` reads from disk;
`TsbAdMDataset::parse_csv(&str)` handles in-memory buffers.
Accepts both integer (`0` / `1`) and float (`0.0` / `1.0`) label
encodings.

Types: `TsbAdMDataset`.

Source: `src/tsb_ad_m.rs`. Paired example:
`examples/tsb_ad_m_eval.rs` (loads one CSV → `DynamicForest<128>`
50/50 split → prints VUS-PR).

### Property-based fuzz suite

`tests/fuzz_properties.rs` runs eight adversarial properties via
`proptest` on stable: postcard + JSON roundtrip preserve scores,
non-finite inputs (`NaN` / ±inf / subnormals) are rejected cleanly
without poisoning forest state, `score_many` is bit-exact with
the serial `score` loop, `delete_by_value` shrinks live count by
exactly the number of matches, TRCF `process` never panics on
arbitrary input, `score_across_tenants` output is sorted
descending, `forensic_baseline` stays within the observed
per-dim `[min, max]`. Ships a regression file on failure so CI
pins the offending input.

### Benchmark suites

Five bench harnesses across the workspace — all pin `mimalloc`
globally and measure wall-clock with criterion:

- `core/benches/forest_throughput.rs` — core ops (insert, score,
  attribution, codisp batched + loop) over the
  `(trees, samples, D)` matrix.
- `core/benches/extended.rs` — bulk scoring, early-term, forensic,
  tenant similarity / cross-tenant, stateless codisp, TRCF
  process, delete.
- `core/benches/modules.rs` — 23 groups: shingled forest, matrix
  profile (STOMP), quantiles (t-digest + histogram), drift
  detectors (feature / meta / ADWIN / SPOT-DSPOT / Fisher),
  per-feature EWMA/CUSUM, sketches (CountMinSketch / HyperLogLog /
  SpaceSaving / BloomFilter), normalize, dynamic + drift-aware
  forests, plus closing-pass adds (group_scores,
  attribution_stability, score_ci, bootstrap, persistence).
- `triage/benches/modules.rs` — LSH clustering, Platt calibration,
  SAGE explanations, cosine alert clustering, feedback store.
- `hotpath/benches/modules.rs` — UpdateSampler, PrefixRateCap,
  bounded MPSC channel.

Run the whole matrix with `cargo bench --workspace`, or target a
single crate with `cargo bench -p anomstream-core --bench modules`.
See [`performance.md`](performance.md) for the current reference
numbers on `x86_64`.
