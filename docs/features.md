# Features

Modules layered on top of the bare `RandomCutForest<D>`. Each is
driven by an eBPFsentinel Enterprise need; none are strictly
required to use the core RCF scorer.

## `ThresholdedForest` — adaptive threshold

Tracks an EMA of the anomaly-score stream and derives a continuously
updated `mean + z · stddev` threshold. Callers receive an
`AnomalyGrade` (`is_anomaly` + `grade ∈ [0, 1]`) instead of comparing
a raw score against a magic constant. Inspired by AWS's `TRCF` in
`randomcutforest-parkservices`, kept light (no short/long-term
duality, no near-threshold heuristics).

Example: `examples/thresholded.rs`.

## `TenantForestPool<K, D>` — per-tenant isolation

One `ThresholdedForest` per tenant key, bounded LRU eviction, lazy
instantiation via a factory closure. Baseline shock on tenant A
does not affect tenant B's adaptive threshold.

Example: `examples/tenant_pool.rs`.

## Cold-start bootstrap

`bootstrap(points)` replays historical points from an upstream TSDB
(Prometheus / Loki / parquet) before live traffic is switched on —
eliminates per-restart warmup coverage hole. Non-finite rows are
skipped and counted in `BootstrapReport`.

Example: `examples/bootstrap.rs`.

## Warm reload

`to_path` / `from_path` (binary via `postcard`) and
`to_json_path` / `from_json_path` on both `RandomCutForest` and
`ThresholdedForest`. Atomic write (`tmp + fsync + rename`) — a
mid-save crash leaves the previous snapshot intact.

Example: `examples/warm_reload.rs`.

## Group-score decomposition

Declare named semantic groups over dim indices once (e.g. `rate`,
`payload`, `cardinality`). Every `group_scores(&point, &groups)`
returns per-group contributions + `top_group()` driver — more
actionable than a 14-entry `DiVector`.

Example: `examples/group_scores.rs`.

## Attribution stability

`attribution_stability(&point)` exposes per-dim variance across
trees alongside the mean. `confidence(dim)` is
`1 / (1 + CV)`; `argmax_weighted()` picks the dim with highest
`mean × confidence` — demotes dims where the forest disagreed.

Example: `examples/attribution_stability.rs`.

## Meta-drift CUSUM

`MetaDriftDetector` runs a two-sided CUSUM on the anomaly-score
stream. Fires `DriftKind::Upward` on a sustained climb (baseline
drift) and `DriftKind::Downward` on a sustained decline — strictly
more sensitive to small persistent shifts than a `μ + 3σ` gate.

Example: `examples/meta_drift.rs`.

## Explicit retraction

`update_indexed` returns the fresh `point_idx`; `delete(idx)` /
`delete_by_value(&point)` remove a previously-observed point.
Intended for SOC-driven false-positive retractions without waiting
for reservoir eviction.

Example: `examples/delete_and_scales.rs`.

## Per-dim feature scales

`feature_scales([f64; D])` pre-scales every caller point before it
reaches the forest hot paths. Use when dims have wildly different
dynamic ranges (packet-rate, protocol ratios, entropy, cardinality);
pass `1 / stddev[d]` per dim for unit-variance normalisation.

## Bulk batch scoring

`score_many` / `score_many_early_term` / `attribution_many` — rayon
parallelises across points on top of the per-tree parallelism.
5-8× speedup on backfill / SOC replay workloads vs a serial
for-loop.

Example: `examples/bulk_scoring.rs`.

## Point timestamps + retention

`update_at(point, ts)` / `process_at(point, ts)` tag each fresh
observation with a caller-supplied `u64` (epoch-ms, sequence).
`delete_before(cutoff)` retracts every older point. Per-tenant on
the pool. Intended for GDPR / NIS2 retention windows and forensic
replay of a specific period.

Example: `examples/retention.rs`.

## Early-termination scoring

`score_early_term(&point, &cfg)` walks trees sequentially and
breaks as soon as the running per-tree mean has converged tightly
enough (`stderr / |mean|` below the configured relative threshold).
Cuts wall-clock latency on baseline-dominated traffic. `EarlyTermScore`
reports how many trees were actually walked.

Example: `examples/early_term.rs`.

## Imputation-like forensic baseline

`forensic_baseline(&point)` answers *"what would this dim have
looked like if the point were normal?"*. Returns
`ForensicBaseline<D>` with per-dim `expected` / `stddev` /
`delta` / `zscore` against every live sample in the forest's
reservoirs, plus `argmax_abs_zscore()` picking the most-anomalous
dim. Baseline is returned in raw caller coordinates — the internal
`feature_scales` transform is inverted. Great for SOC triage:
alert table can display *observed* vs *normal expected* per dim.

Example: `examples/forensic.rs`.

## Calibrated probability

`PlattCalibrator::fit(&[(score, label)])` fits a 1-D sigmoid
(Platt 1999 / Lin-Lin-Weng 2007) that maps raw scores to
`P(anomaly | score) ∈ [0, 1]`. Stable Newton-Raphson with
backtracking line search, target smoothing keeps label-homogeneous
sets numerically safe. Fits persist via serde for reuse at
inference time. Intended for audit-defensible alerting policies
(SOC2 / NIS2) where a raw score is meaningless in compliance
paperwork.

Example: `examples/calibrator.rs`.

## Observability

`with_metrics_sink(Arc<dyn MetricsSink>)` on every detector streams
counters / gauges / histograms into Prometheus / StatsD / OTel.
`NoopSink` is the zero-cost default. `ScoreHistogram` provides
in-process bin + percentile export.

Example: `examples/observability.rs`.

## Cold-start warmup

`initial_accept_fraction` ramps the reservoir admission probability
over the first `initial_accept_fraction · sample_size` offers so
the very first points do not dominate the sample. `1.0` default
disables the gate; AWS `CompactSampler` uses `0.125`.
