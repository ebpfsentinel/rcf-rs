# rcf-rs

Pure Rust Random Cut Forest for streaming anomaly detection.

`rcf-rs` implements the Random Cut Forest algorithm from Guha et al.
(ICML 2016) and is conformant with the
[AWS SageMaker RCF specification](https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html):
reservoir sampling without replacement, random cuts weighted by per-dimension
range, anomaly score averaged across trees, hyperparameter bounds matching
the AWS reference (`feature_dim`, `num_trees`, `num_samples_per_tree`).

> **Status**: under active development — APIs are unstable until v0.1.0.

Optional **adaptive-threshold layer** (`ThresholdedForest`) tracks an
EMA of the anomaly-score stream and derives a continuously updated
threshold `mean + z · stddev`. Callers receive a graded verdict
(`is_anomaly` + `grade ∈ [0, 1]`) instead of a raw score to compare
against a magic constant — see `examples/thresholded.rs`.

Optional **cold-start warmup** (`initial_accept_fraction`) ramps the
reservoir's admission probability over the first
`initial_accept_fraction · sample_size` offers so the very first
points do not dominate the sample. `1.0` (default) disables the gate;
AWS's `CompactSampler` uses `0.125`. Configure via
`ForestBuilder::initial_accept_fraction`.

**Warm reload**: both `RandomCutForest` and `ThresholdedForest` expose
`to_path` / `from_path` (binary) and `to_json_path` / `from_json_path`
(JSON) helpers. Writes go through a tmp-then-rename dance with
`fsync` so a crash mid-save leaves the previous snapshot intact. Pair
with periodic checkpointing so restarts resume exactly where they left
off — see `examples/warm_reload.rs`.

**Per-tenant pool** (`TenantForestPool<K, D>`) keeps one
`ThresholdedForest` per tenant key with bounded LRU eviction and a
factory closure for lazy instantiation. A baseline shock on tenant
A does not move tenant B's adaptive threshold — each tenant owns its
own EMA and reservoir. Walk the pool with `iter` to snapshot every
tenant to disk; reload by iterating a snapshot directory and calling
`insert` on a fresh pool. See `examples/tenant_pool.rs`.

**Cold-start bootstrap** (`bootstrap(points)`) replays historical
points from an upstream store (Prometheus / Loki / parquet dump) into
a detector before live traffic is switched on — eliminating the
per-restart warmup coverage hole. Available on `RandomCutForest`,
`ThresholdedForest`, and `TenantForestPool` (per-tenant). Non-finite
rows are skipped and tallied in the returned `BootstrapReport` so
gappy TSDB queries don't sink the restart. See
`examples/bootstrap.rs`.

**Group-score decomposition** (`group_scores(&point, &FeatureGroups)`)
turns the per-dim `DiVector` into a compact per-group breakdown —
declare semantic groups like `rate`, `payload`, `cardinality` once
at build time and every score call returns a `GroupScores` with the
contribution of each group plus a `top_group()` driver. Available on
all three detector types. Great for SOC triage: *"alert driven by
payload, not by rate"* is more actionable than a ranked 14-dim
vector. See `examples/group_scores.rs`.

**Attribution stability** (`attribution_stability(&point)`) exposes
the per-dim variance / stddev of the attribution across trees on top
of the usual mean. Pairs with a `confidence(dim) ∈ [0, 1]` helper
derived from the coefficient of variation — `argmax_weighted()` picks
the dim whose `mean × confidence` is highest, demoting dims where the
forest disagreed. An alert whose driver dim has low confidence is
likely a lucky coincidence, not a stable signal. Available on all
three detector types. See `examples/attribution_stability.rs`.

**Meta-drift CUSUM** (`MetaDriftDetector`) runs a two-sided CUSUM
change-point detector on the anomaly-score stream — the orthogonal
question to "is this point anomalous?" becomes "is the score
distribution itself shifting?" Fires `DriftKind::Upward` on a
sustained score climb (baseline drift) and `DriftKind::Downward` on a
sustained decline (quiescence / desensitisation), strictly more
sensitive to small persistent shifts than a `μ + 3σ` gate. Standalone
type — plug any score-like stream in. See `examples/meta_drift.rs`.

**Explicit retraction** (`update_indexed` / `delete` /
`delete_by_value`) lets callers track the `point_idx` of each fresh
observation and remove it later — typically to handle SOC-driven
false-positive retractions without waiting for the reservoir to evict
the point naturally. Available on `RandomCutForest`,
`ThresholdedForest` (plus `process_indexed`), and `TenantForestPool`.
The pool variants do not auto-create tenants on retraction.

**Per-dim feature scales** (`feature_scales([f64; D])`) pre-scale
every caller point before it reaches the forest's hot paths. Use to
rebalance dims with wildly different dynamic ranges (packet-rate in
`[10², 10⁶]`, protocol ratios in `[0, 1]`, entropy in `[0, 8]` bits) —
without scaling, random cuts are dominated by the widest dim. Pass
`1 / stddev[d]` per dim for unit-variance normalisation; mean-centre
upstream if full z-score is needed. See
`examples/delete_and_scales.rs`.

## Quickstart

```rust,ignore
use rcf_rs::{ForestBuilder, AnomalyScore};

// Build a forest with the AWS-default hyperparameters. Per-point
// dimensionality is pinned at the type level via the const-generic
// `D` parameter — `4` here.
let mut forest = ForestBuilder::<4>::new()
    .num_trees(100)
    .sample_size(256)
    .seed(42)
    .build()?;

// Stream points through the forest.
for point in stream_of_points {
    forest.update(point)?;
    let score: AnomalyScore = forest.score(&point)?;
    if f64::from(score) > 1.5 {
        println!("anomaly: score={score}");
    }
}
# Ok::<(), rcf_rs::RcfError>(())
```

## Algorithm

`rcf-rs` follows the original paper:

> Sudipto Guha, Nina Mishra, Gourav Roy, Okke Schrijvers.
> "Robust Random Cut Forest Based Anomaly Detection on Streams."
> *International Conference on Machine Learning*, pp. 2712–2721. 2016.

Reservoir sampling without replacement is from:

> Byung-Hoon Park, George Ostrouchov, Nagiza F. Samatova, Al Geist.
> "Reservoir-based random sampling with replacement from data stream."
> *SIAM International Conference on Data Mining*, pp. 492–496. 2004.

## AWS SageMaker conformance

| AWS specification | `rcf-rs` mapping |
|---|---|
| `feature_dim ∈ [1, 10000]` | const-generic `D`, validated by `ForestBuilder::build` |
| `num_trees ∈ [50, 1000]`, default `100` | enforced by `ForestBuilder` |
| `num_samples_per_tree ∈ [1, 2048]`, default `256` | enforced by `ForestBuilder` |
| `time_decay = 0.1 / sample_size` | resolved by `ForestBuilder`; pass `.time_decay(0.0)` to disable |
| Reservoir sampling without replacement | `sampler::ReservoirSampler` |
| Score = average across trees | `forest::RandomCutForest::score` |
| Anomaly threshold `≥ 3σ` from mean | caller responsibility |

Beyond the AWS spec, `rcf-rs` adds a `ThresholdedForest` on top of the
bare forest — an EMA-driven adaptive `μ + z · σ` threshold inspired
by AWS's `randomcutforest-parkservices` TRCF, kept deliberately light
(no short/long-term duality, no near-threshold heuristics).

## Cargo features

| Feature | Default | Effect |
|---|---|---|
| `std` | ✅ | Standard library support (future `no_std` planned) |
| `parallel` | ✅ | Per-tree parallel insert/score/attribution via `rayon` |
| `serde` | ✅ | Forest state serialisation |
| `postcard` | ✅ | Versioned binary persistence helpers via `postcard` (implies `serde`). Replaced `bincode` in format v2 after RustSec flagged `bincode` as unmaintained |
| `serde_json` | ❌ | JSON helpers (implies `serde`) |

The production profile (`parallel` + `serde` + `postcard`) is enabled
by default because the intended deployment target — a long-running
streaming agent that parallelises across cores and persists its
forest across restarts — always needs all three. Opt out with
`default-features = false` for embedded / mono-thread / no-persistence
scenarios.

### `parallel` and the dedicated thread pool

Enable the `parallel` feature to run the per-tree work across rayon
workers. By default the global rayon pool is used; pin a dedicated
pool (and isolate this forest from the rest of the application's
rayon workload) via `ForestBuilder::num_threads`:

```rust,ignore
let forest = ForestBuilder::<16>::new()
    .num_trees(100)
    .sample_size(256)
    .num_threads(4)              // dedicated 4-worker pool
    .build()?;
```

`num_threads` is only honoured with `--features parallel`; without it
the field is recorded in the config but ignored at runtime.

## Performance

### Bench matrix (`forest_throughput`)

Latest run (`cargo bench --features parallel`), times reported as the
mean point estimate:

| Workload | `(trees, samples, D)` | Time |
|---|---|---|
| `forest_update` | `(50, 128, 16)` | 35.91 µs |
| `forest_update` | `(100, 256, 4)` | 31.89 µs |
| `forest_update` | `(100, 256, 16)` | 47.98 µs |
| `forest_update` | `(100, 256, 64)` | 104.93 µs |
| `forest_update` | `(200, 512, 16)` | 84.91 µs |
| `forest_score` | `(50, 128, 16)` | 26.60 µs |
| `forest_score` | `(100, 256, 4)` | 37.08 µs |
| `forest_score` | `(100, 256, 16)` | 38.88 µs |
| `forest_score` | `(100, 256, 64)` | 46.62 µs |
| `forest_score` | `(200, 512, 16)` | 67.05 µs |
| `forest_attribution` | `(100, 256, 4)` | 72.21 µs |
| `forest_attribution` | `(100, 256, 16)` | 131.26 µs |
| `forest_attribution` | `(100, 256, 64)` | 150.39 µs |

At `(100, 256, 16)` this is **~21k inserts/sec**, **~26k scores/sec**
single-thread-equivalent on a 4-core box, with both metrics scaling
sub-linearly down to single-core because each operation already
parallelises across trees.

A `forest_tuning_dim16` group sweeps `(num_trees, sample_size)` at the
AWS-default `D = 16` so callers can pick a precision/latency tradeoff:

| `(num_trees, sample_size)` | `update` | `score` |
|---|---|---|
| `(50, 64)` | 32.44 µs | 27.71 µs |
| `(50, 128)` | 35.98 µs | 27.97 µs |
| `(50, 256)` | 43.30 µs | 30.41 µs |
| `(100, 64)` | 36.85 µs | 35.13 µs |
| `(100, 128)` | 41.78 µs | 37.41 µs |
| `(100, 256)` | 50.75 µs | 37.61 µs |

## Minimum Supported Rust Version

`rcf-rs` requires Rust **1.93** or later, edition 2024.

## License

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE).

Contributions submitted to this repository are licensed under the same terms,
without any additional terms or conditions.
