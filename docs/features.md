# Features

Modules layered on top of the bare `RandomCutForest<D>`. Each is
driven by an eBPFsentinel Enterprise need; none are strictly
required to use the core RCF scorer. All modules live under
`src/` and are re-exported at crate root (see `src/lib.rs`).

## Core detectors

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

Source: `src/thresholded/`.

Example: `examples/thresholded.rs`.

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

## Scoring variants

### Bulk batch scoring

`score_many(&[[f64; D]])` / `score_many_early_term(&[…], cfg)` /
`attribution_many(&[…])`. Under `parallel`, rayon parallelises
across points on top of the per-tree parallelism — 5-6× speedup
on backfill / SOC replay (see `docs/performance.md`). First error
aborts the batch. Available on `RandomCutForest`,
`ThresholdedForest`, and `TenantForestPool` (per-tenant;
absent-tenant returns `None` on read paths).

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

### Cross-tenant what-if

`TenantForestPool::score_across_tenants(&point)` pipes the same
probe through every resident tenant's detector and returns a
`Vec<(K, AnomalyGrade)>` sorted by descending grade. Warming-up
tenants are dropped. Intended for MSSP / threat-intel lateral
scans. Read-only — no tenant creation, no state mutation. Under
`parallel`, rayon fans out one `score_only` per tenant
(`O(N)` cost, near-linear at ~6.7 ms for 512 tenants).

Example: `examples/cross_tenant.rs`.

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
z-score.

### Cold-start warmup (reservoir)

`initial_accept_fraction` ramps the reservoir admission
probability over the first `initial_accept_fraction · sample_size`
offers so the very first points do not dominate the sample. `1.0`
default disables the gate; AWS `CompactSampler` uses `0.125`.

## Explainability

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

### Attribution stability

`attribution_stability(&point)` exposes per-dim variance /
stddev across trees alongside the mean. `confidence(dim)` is
`1 / (1 + CV)`; `argmax_weighted()` picks the dim with highest
`mean × confidence` — demotes dims where the forest disagreed.
`argmax_mean()` = classic `DiVector::argmax` for comparison.

Types: `AttributionStability`.

Source: `src/attribution_stability.rs`.

Example: `examples/attribution_stability.rs`.

### Imputation-like forensic baseline

`forensic_baseline(&point)` answers *"what would this dim have
looked like if the point were normal?"*. Returns
`ForensicBaseline<D>` with per-dim `expected` / `stddev` /
`delta` / `zscore` / `live_points` against every live sample in
the forest's reservoirs, plus `argmax_abs_zscore()`. Baseline
returned in raw caller coordinates — the internal `feature_scales`
transform is inverted. Great for SOC triage: alert table can
display *observed* vs *normal expected* per dim.

Source: `src/forensic.rs`.

Example: `examples/forensic.rs`.

## Drift & calibration

### Meta-drift CUSUM

`MetaDriftDetector` runs a two-sided CUSUM on the anomaly-score
stream. Fires `DriftKind::Upward` on a sustained climb (baseline
drift) and `DriftKind::Downward` on a sustained decline —
strictly more sensitive to small persistent shifts than a
`μ + 3σ` gate. Standalone type; plug any score-like stream in.
`reset()` clears the accumulators while keeping the EMA reference,
`reset_stats()` clears everything.

Types: `CusumConfig`, `DriftKind`, `DriftVerdict`,
`MetaDriftDetector`.

Source: `src/meta_drift.rs`.

Example: `examples/meta_drift.rs`.

### Severity bands

`SeverityBands` + `Severity` enum classify raw anomaly scores into
ordinal labels (`Normal` / `Low` / `Medium` / `High` / `Critical`).
Defaults match eBPFsentinel Enterprise ml-detection
(`2.0 / 3.0 / 4.0 / 5.0`). `Severity: Ord` lets callers route
alerts via `sev >= Severity::High`. Methods on both
`AnomalyScore::severity(&bands)` (bare forest) and
`AnomalyGrade::severity(&bands)` (TRCF).

Types: `Severity`, `SeverityBands`.

Source: `src/severity.rs`.

Example: `examples/severity.rs`.

### Alert clustering / dedup

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

Types: `AlertClusterer`, `AlertCluster`, `ClusterDecision`.

Source: `src/alert_cluster.rs`.

### Audit trail (NIS2 / SOC2)

`AlertRecord<K, D>` packages every analytic output (`score`,
`grade`, `attribution`, `baseline`, `severity`) plus provenance
(`tenant`, `timestamp_ms`, `point`) into one serialisable struct.
Build via `AlertRecord::from_forest` (bare RCF) or
`AlertRecord::from_thresholded` (TRCF); chain `with_severity` to
attach a band. Versioned through `ALERT_RECORD_VERSION` so
incompatible schema changes surface at decode time. Emit to a
SIEM / object-store / WORM log via any `serde` sink — postcard
for compact per-event bytes, JSON for self-describing records.

Types: `AlertRecord`, `AlertContext`, `ALERT_RECORD_VERSION`.

Source: `src/audit.rs`.

### Calibrated probability (Platt scaling)

`PlattCalibrator::fit(&[(score, label)], config)` fits a 1-D
sigmoid (Platt 1999 / Lin-Lin-Weng 2007) that maps raw scores to
`P(anomaly | score) ∈ [0, 1]`. Stable Newton-Raphson with
backtracking line search, target smoothing keeps
label-homogeneous sets numerically safe. `calibrate(score)` /
`calibrate_many(&scores)` at inference. Serde roundtrippable.
Intended for audit-defensible alerting policies (SOC2 / NIS2)
where a raw score is meaningless in compliance paperwork.

Types: `PlattCalibrator`, `PlattFitConfig`.

Source: `src/calibrator.rs`.

Example: `examples/calibrator.rs`.

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

Source: `src/metrics.rs`.

### `ScoreHistogram`

Standalone fixed-bin histogram for score / grade / CUSUM
streams. `record(value)`, `bins()`, `bin_edges()`, `total()`,
`underflow()` / `overflow()` / `non_finite()`, `merge(&other)`,
`percentile(p)`, `reset()`. Export to Grafana / Prometheus
without re-deriving per-minute aggregates in the monitoring
pipeline.

Types: `HistogramConfig`, `ScoreHistogram`.

Source: `src/histogram.rs`.

Example: `examples/observability.rs`.

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

`benches/forest_throughput.rs` (core insert/score/attribution
sweep over the `(trees, samples, D)` matrix) and
`benches/extended.rs` (bulk, early-term, forensic, tenant
similarity/cross-tenant) pin `mimalloc` globally and measure
wall-clock with criterion. See `docs/performance.md` for the
current reference numbers on `x86_64`.
