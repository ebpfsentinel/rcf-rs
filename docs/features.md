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

### Runtime-dim wrapper (`DynamicForest<MAX_D>`)

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

### Shapley attribution via SAGE estimator

`SageEstimator<D>` (in `rcf_rs::sage`) — Monte-Carlo
permutation-sampling Shapley estimator (Covert et al. NeurIPS
2020). Accounts for feature interactions the marginal per-dim
`DiVector` ignores: when two dims jointly signal anomaly but
neither alone does, Shapley distributes the score contribution
across both. Caller supplies a baseline point (warm-phase mean
or synthetic null); estimator samples `K` random permutations,
computes each dim's marginal contribution as it joins the
coalition, averages. Cost `O(K · D)` forest scores per probe
— batch / forensic replay, not hot-path.

Types: `SageEstimator<D>`, `SageExplanation<D>`.

Source: `src/sage.rs`.

### LSH-based alert clustering

`LshAlertClusterer` (in `rcf_rs::lsh_cluster`) — quantises every
per-dim attribution value into a 4-bit symbol and uses the
concatenated hex string as the bucket key. O(1) lookup via
`HashMap<String, u64>`. Complement to the cosine-similarity
`AlertClusterer` — LSH scales to MSSP-volume alert streams where
pairwise cosine is too slow. Mirrors the TLSH spirit (Oliver et
al. 2013) without bigram-frequency overhead.

Types: `LshAlertClusterer`, `LshClusterDecision`.

Source: `src/lsh_cluster.rs`.

### SOC feedback ingestion

`FeedbackStore<D>` (in `rcf_rs::feedback`) — bounded ledger of
analyst-labelled points. Das et al., *Incorporating Feedback
into Tree-based Anomaly Detection*, `arXiv:1708.09441`. API:
`label(point, FeedbackLabel::Benign | Confirmed)`,
`adjust(probe, raw_score) -> adjusted`. The adjustment adds a
Gaussian-kernel-weighted sum of every stored label's sign to
the raw score — Benign labels pull nearby probes **down**,
Confirmed labels push nearby probes **up**. Non-mutating on the
forest side: the hot-path `score()` is untouched, adjustment is
an additive bias layer the caller applies post-score.

Lifetime defaults: `capacity = 512`, `sigma = 1.0`, `strength =
1.0`. FIFO eviction on capacity pressure. Clamps adjusted score
to `≥ 0`.

Types: `FeedbackStore<D>`, `FeedbackLabel`.

Source: `src/feedback.rs`.

Example: `examples/feedback_adjust.rs` (baseline forest + benign
label on an outlier probe, adjusted score drops from 2.23 → 1.23
on the exact probe and from 2.35 → 1.36 on a nearby one).

### Drift recovery — shadow forest + ADWIN

`AdwinDetector` (in `rcf_rs::adwin`) — streaming change-point
detector (Bifet & Gavaldà, SIAM SDM 2007). Bounded ring buffer of
the last `N` observations, per-update O(N) scan over every split
point with a Hoeffding bound `ε_cut`; flags drift and drops the
older sub-window on fire. Confidence `δ`, window cap `N`, and
observed stream amplitude `range` are caller-configured. Use on
the score stream (or any per-step scalar) for an adaptive drift
trigger with automatic window sizing — strictly more sensitive
than a fixed-window mean test.

`DriftAwareForest<D>` (in `rcf_rs::drift_aware`) — facade around
a live `RandomCutForest<D>` with an optional shadow. Swap policy
via `DriftRecoveryConfig`: `min_primary_age` guards against
flap-loops, `shadow_warmup` controls when the shadow becomes the
new primary. Call `on_drift()` to spawn a shadow (trigger lives
outside — route from `AdwinDetector`, `FeatureDriftDetector`, or
`MetaDriftDetector`). `update` feeds both primary and shadow;
once the shadow hits `shadow_warmup` the swap is atomic on the
next `update` tick. `score` always reads from the primary
(stable baseline until the swap lands).

Types: `AdwinDetector`, `DriftAwareForest<D>`, `DriftRecoveryConfig`.

Source: `src/adwin.rs`, `src/drift_aware.rs`.

Example: `examples/drift_recovery.rs` (2-regime synthetic stream
with ADWIN-triggered shadow spawn + atomic swap after warmup).

### Univariate SPOT detector bank + Fisher combiner

`PotDetector` (in `rcf_rs::univariate_spot`) — streaming
Peaks-Over-Threshold per-dimension anomaly detector (Siffer et
al., KDD 2017). For each feature dim it tracks a running quantile
`u` via the shipped `TDigest`, fits a Generalised Pareto
Distribution (GPD) to the peak excesses `(x − u) | x > u` by
method-of-moments, and returns the tail survival probability as
a p-value in `(0, 1]`. SPOT mode freezes the quantile after warm
via `freeze_baseline`; DSPOT mode keeps the digest drifting for
non-stationary streams.

`fisher_combine(&[p_values]) -> f64` (in `rcf_rs::ensemble`) —
combines K independent p-values into a joint anomaly score via
Fisher 1932: `T = −2 Σ ln(p_i) ~ χ²(2K)` → survival returned.
Uses the closed-form `χ²` survival for even dof.

Composition: run one `PotDetector` per feature dim, collect
per-dim p-values, pipe through `fisher_combine`. Joint p below
`1e-3` → anomaly. Orthogonal to RCF — the SPOT bank catches
per-dim marginal drift that isolation depth misses on
heterogeneously-distributed multivariate features (architect
review targets TSB-AD-M AUC lift 0.583 → 0.80+).

Types: `PotDetector`, `fisher_combine`, `chi_squared_survival_even`.

Source: `src/univariate_spot.rs`, `src/ensemble.rs`.

Example: `examples/univariate_spot_bank.rs` (4-dim baseline +
single-dim and all-dim outlier probes, joint p-value reported).

### Internal shingling (`ShingledForest<D>`)

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

### Feature-distribution drift (PSI / KL)

`FeatureDriftDetector<D>` pins a baseline per-dim histogram, folds
live traffic into a mirror histogram with identical bin edges, and
reports Population Stability Index (`Σ (Q − P) · ln(Q/P)`) and KL
divergence `D_KL(Q || P)` per feature. CUSUM on the score stream
catches the detector *re-centring*; PSI on the features catches
the data *itself drifting*. Industry thresholds wired into
`DriftLevel::{Stable, Watch, Alert}` (`< 0.10`, `0.10..0.25`,
`≥ 0.25`). `argmax_psi()` pins the offending dim; `reset_production()`
starts a fresh monitoring window without invalidating the baseline.

Types: `FeatureDriftDetector`, `DriftLevel`, `PSI_WATCH_THRESHOLD`,
`PSI_ALERT_THRESHOLD`.

Source: `src/feature_drift.rs`.

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

Online update via `PlattCalibrator::update_online(score, label,
lr)` applies one SGD step on the logistic loss per observation
— refine the fit as SOC feedback accumulates without re-running
the batch Newton-Raphson solver.

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

Source: `src/metrics.rs`.

### T-digest streaming quantiles

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

## Hot-path integration (eBPF ingress)

### `UpdateSampler` + `channel` MPSC split

`rcf_rs::hot_path::UpdateSampler` drops low-value updates before
any RCF work. Two decision modes:

- `accept_stride()` — monotonic counter, keeps `1 / keep_every_n`
  offers deterministically.
- `accept_hash(flow_hash)` — per-flow sampling. Preserves
  baseline shape per flow rather than slicing any single flow.

Build via `UpdateSampler::new(keep)` (unkeyed, deterministic
admission — back-compatible) or **`UpdateSampler::new_keyed(keep)`**
(128-bit secret from `getrandom`, murmur3 keyed mix applied
before the modulo). Keyed sampler defends against the
reservoir-poisoning spray (MITRE ATLAS `AML.T0020`) where an
attacker steers their flow hash into the admitted residue class.

`channel::<D>(capacity)` returns `(UpdateProducer<D>,
UpdateConsumer<D>)` — bounded MPSC on `std::sync::mpsc::sync_channel`.
Clone the producer per classifier thread; hand the consumer to a
dedicated updater thread. `try_enqueue` is non-blocking; on
queue-full it drops + increments `dropped_total` (ops signal
for "classifier outpaces updater"). The classifier scores
against the previous-generation forest snapshot while the
updater drains at its own cadence.

Types: `UpdateSampler`, `UpdateProducer<D>`, `UpdateConsumer<D>`.

Source: `src/hot_path.rs`.

### `PrefixRateCap` — per-prefix admission rate cap

`rcf_rs::hot_path::PrefixRateCap::new(cap_per_window, window_ms)`
bounds how many admissions a single source-prefix hash bucket
can push into the reservoir within a rolling window. Fixed
256-bucket counter sketch, lock-free `check_and_record`.
Second defence line (alongside the keyed `UpdateSampler`)
against reservoir-poisoning floods from a single compromised
source — documented in `docs/threat_model.md`.

Types: `PrefixRateCap`.

### Trimmed-mean ensemble scoring

`RandomCutForest::score_trimmed(&point, trim_fraction)` sorts
per-tree scores, drops the top + bottom `trim_fraction` fraction,
averages the middle. Robust against single-tree poisoning: an
attacker who manages to move a minority of trees' scores to the
extreme tails sees their contribution trimmed from the ensemble
mean. Typical `trim_fraction` values: `0.10` (10 %/10 %) or
`0.25` (quartile trim). `trim_fraction = 0.0` matches `score()`.

## Security & threat model

See [`docs/threat_model.md`](threat_model.md) for the full
adversarial threat model covering reservoir poisoning,
evasion via contextual shift, model extraction, and classifier-
side resource exhaustion — with the MITRE ATLAS technique IDs
and the defences shipped in-crate for each.

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
