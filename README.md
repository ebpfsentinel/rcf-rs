# anomstream

A composable Rust toolkit for **streaming anomaly detection**. Multiple detector families (multivariate, per-feature, score-level) plus the primitives needed to turn them into production pipelines — streaming stats, normalisation, probability calibration, alert clustering, feedback loops, SOC triage, hot-path ingress.

Among the detectors: an AWS-conformant Random Cut Forest implementation (Guha et al. ICML 2016) — one of several detectors.

Powers the ML detection pipeline of the **eBPFsentinel Enterprise** NDR agent; designed to be reused anywhere a stream of high-dim observations needs online scoring.

## Scope

**In scope** — streaming, bounded-memory, online-update primitives:

- Multivariate anomaly detection (Random Cut Forest and variants)
- Per-feature drift detectors (EWMA z-score, two-sided CUSUM, PSI / KL)
- Score-level drift + regime-change (meta CUSUM, ADWIN, SPOT / DSPOT)
- Streaming stats + sketches (Welford `OnlineStats`, t-digest, histograms, Count-Min Sketch, `HyperLogLog`, Space-Saving top-K, Bloom filter)
- Normalisation (`Normalizer<D>` — min-max / z-score / identity)
- Explanation + triage (per-dim attribution, SAGE Shapley, Platt calibration, severity bands, alert clustering, SOC feedback, audit trail, forensic baseline)
- Hot-path ingress (sampler, rate cap, bounded MPSC channel, pluggable metrics sink)

**Out of scope** — intentionally absent to keep the library focused:

- Protocol parsers, IP-centric trackers, L7 intelligence
- ONNX / torch runtimes, supervised model training
- Rule synthesis, policy engines
- Density estimation, forecasting, GLAD variant, near-neighbour list, feature-completion `impute()` (the RCF "imputation" idea is repurposed as a SOC-triage `forensic_baseline` helper).

The Random Cut Forest implementation inside the toolkit is a focused port of the 2016 paper — not an attempt to match every feature of AWS's `randomcutforest-by-aws`.

### Catalogue

**Multivariate anomaly detectors** — operate on the joint `[f64; D]` distribution

- `RandomCutForest<D>` — AWS-conformant aggregate root (Guha 2016)
- `ThresholdedForest<D>` — adaptive threshold wrapper (TRCF)
- `ShingledForest` — scalar-stream temporal wrapper over the forest
- `DynamicForest` — runtime-dim variant
- `DriftAwareForest` — shadow-swap recovery when a drift detector fires
- `TenantForestPool` — bounded per-tenant forest pool with LRU eviction

**Per-feature univariate detectors** — one accumulator per dimension, finds _which_ feature drifted

- `PerFeatureEwma<D>` — parallel univariate EWMA z-score detector
- `PerFeatureCusum<D>` — parallel two-sided CUSUM change-point detector
- `FeatureDriftDetector<D>` — PSI / KL distributional drift on raw features

**Score-level drift + regime change** — operate on a scalar anomaly-score stream

- `MetaDriftDetector` — two-sided CUSUM on the score stream
- `AdwinDetector` — adaptive windowing (Bifet 2007)
- `PotDetector` — SPOT / DSPOT univariate Peaks-Over-Threshold (Siffer 2017)
- `fisher_combine` — combine `k` independent p-values into one test statistic

**Streaming stats + sketches** — bounded-memory summaries reused across detectors

- `OnlineStats` — Welford streaming mean + variance
- `TDigest` — Dunning streaming quantile digest
- `ScoreHistogram` — fixed-bin score histogram
- `CountMinSketch` — probabilistic frequency sketch (std-gated)
- `HyperLogLog` — probabilistic distinct-count / cardinality sketch (std-gated)
- `SpaceSaving<K>` — deterministic `O(K)`-memory top-K heavy hitters (std-gated, complements `CountMinSketch`)
- `BloomFilter` — probabilistic set-membership for IOC lookup (std-gated, zero false negatives, tunable FPR)
- `Normalizer<D>` — per-feature `MinMax` / `ZScore` / `None` transforms (with `fit(&[[f64; D]])` learner)

**Explanation + triage**

- `DiVector` + `FeatureGroups` — per-dim and per-group attribution
- `AttributionStability` — inter-tree dispersion + confidence
- `SageEstimator<D>` — SAGE Shapley attribution (Covert 2020)
- `PlattCalibrator` — batch + online-SGD probability calibration
- `SeverityBands` / `Severity` — ordinal severity classification

**SOC + ops**

- `AlertClusterer` / `LshAlertClusterer` — cosine + LSH alert dedup
- `FeedbackStore` — SOC-label-driven score adjustment
- `AuditRecord` — immutable alert envelope
- `ForensicBaseline` — post-hoc distance-to-sample summary

**Hot-path ingress**

- `hot_path::UpdateSampler` / `PrefixRateCap` / `channel` — stride - hash + keyed sampler, 256-bucket atomic counter sketch, bounded MPSC channel for classifier/updater thread split
- `MetricsSink` — pluggable telemetry (`NoopSink` + your own impl)

See [docs/features.md](docs/features.md) for the full module catalogue with per-feature rationale.

## Crate layout

The toolkit ships as a Cargo workspace with four members. Consumers can depend on the `anomstream` meta-crate for the simple case or pick individual members when minimising dep graph matters.

| Crate | Role | Publish |
|---|---|---|
| [`anomstream`](meta/) | Facade — feature-gated re-exports of the three members. **Primary public-facing crate.** | `anomstream` |
| [`anomstream-core`](core/) | Detectors + streaming primitives + cross-cut contracts (`MetricsSink`, `SeverityBands`, `ForestSnapshot`). Targets SemVer 1.0 first. | `anomstream-core` |
| [`anomstream-triage`](triage/) | SOC-opinionated higher-level layer — Platt, SAGE, alert clustering, feedback store, audit record. Depends on core. | `anomstream-triage` |
| [`anomstream-hotpath`](hotpath/) | Opinionated eBPF-style ingress primitives — `UpdateSampler`, `PrefixRateCap`, bounded MPSC `channel`. Depends on core. | `anomstream-hotpath` |

Three consumption patterns:

```toml
# Most consumers — single dep, all layers, default features.
[dependencies]
anomstream = "0.0.0-dev"

# Core-only — detectors + primitives, minimal dep graph.
[dependencies]
anomstream = { version = "0.0.0-dev", default-features = false, features = ["core", "std", "parallel", "serde"] }

# Fine-grained — pick members directly when you want per-member SemVer tracking.
[dependencies]
anomstream-core   = { version = "0.0.0-dev", features = ["parallel", "serde"] }
anomstream-triage = { version = "0.0.0-dev" }
```

## Quickstart

Two examples — one per detector family — to show the toolkit is more than its forest.

### Multivariate: Random Cut Forest

```rust,ignore
use anomstream::ForestBuilder;

let mut forest = ForestBuilder::<4>::new()
    .num_trees(100)
    .sample_size(256)
    .seed(42)
    .build()?;

for point in stream_of_points {
    forest.update(point)?;
    let score = forest.score(&point)?;
    if f64::from(score) > 1.5 {
        eprintln!("anomaly: {score}");
    }
}
# Ok::<(), anomstream::RcfError>(())
```

### Per-feature: two-sided CUSUM change-point

```rust,ignore
use anomstream::{PerFeatureCusum, PerFeatureCusumConfig};

let mut det = PerFeatureCusum::<4>::new(PerFeatureCusumConfig {
    slack: 0.5,
    threshold: 5.0,
});

for point in stream_of_points {
    let result = det.observe(&point);
    for alert in &result.alerts {
        eprintln!(
            "drift on feature {} ({:?}) magnitude {:.2}",
            alert.feature_index, alert.direction, alert.magnitude
        );
    }
}
```

The two detectors compose: feed the forest's scalar score into `MetaDriftDetector` for score-level regime change; run `PerFeatureCusum` alongside for per-feature attribution; wrap everything in `ThresholdedForest` for adaptive alerting.

## Algorithms

Each detector cites the paper it implements. Representative references by family:

- **Random Cut Forest** — Guha, Mishra, Roy, Schrijvers, *Robust Random Cut Forest Based Anomaly Detection on Streams*, ICML 2016. Reservoir sampling without replacement: Park, Ostrouchov, Samatova, Geist — SIAM SDM 2004.
- **EWMA** — Hunter, *The Exponentially Weighted Moving Average*, JQT 18(4), 1986.
- **CUSUM** — Page, *Continuous Inspection Schemes*, Biometrika 41, 1954. Two-sided variant: Hawkins & Olwell, 1998.
- **ADWIN** — Bifet, *Learning from Time-Changing Data with Adaptive Windowing*, SIAM SDM 2007.
- **SPOT / DSPOT** — Siffer et al., *Anomaly Detection in Streams with Extreme Value Theory*, KDD 2017.
- **t-digest** — Dunning, *Computing Extremely Accurate Quantiles using t-Digests*, 2019.
- **Count-Min Sketch** — Cormode & Muthukrishnan, JoA 55(1), 2005.
- **HyperLogLog** — Flajolet, Fusy, Gandouet, Meunier — AofA 2007. *HyperLogLog in Practice*: Heule, Nunkesser, Hall, EDBT 2013.
- **Space-Saving** — Metwally, Agrawal, El Abbadi, *Efficient Computation of Frequent and Top-k Elements in Data Streams*, ICDT 2005.
- **Bloom filter** — Bloom, *Space/Time Trade-offs in Hash Coding with Allowable Errors*, CACM 13(7), 1970. Double-hashing: Kirsch & Mitzenmacher, *Less Hashing, Same Performance*, ESA 2006.
- **SAGE** — Covert, Lundberg, Lee, *Understanding Global Feature Contributions Through Additive Importance Measures*, NeurIPS 2020.
- **Welford variance** — Welford, Technometrics 4(3), 1962.

The Random Cut Forest implementation conforms to the AWS `SageMaker` hyperparameter bounds (`feature_dim`, `num_trees`, `num_samples_per_tree`, `time_decay`) — enforced at build time.

Details: [docs/conformance.md](docs/conformance.md).

## Features

| Cargo feature | Default | Role                                                     |
| ------------- | ------- | -------------------------------------------------------- |
| `std`         | ✅      | Standard library support                                 |
| `parallel`    | ✅      | Per-tree / batch parallelism via `rayon` (implies `std`) |
| `serde`       | ✅      | State serialisation                                      |
| `postcard`    | ✅      | Compact binary persistence (implies `serde`)             |
| `serde_json`  | ❌      | JSON persistence (implies `serde`)                        |

### `no_std` + `alloc`

`default-features = false` drops the runtime layer (MPSC channel, tenant pool, drift-aware shadow swap, ADWIN, LSH clustering, SAGE, SPOT/DSPOT, feedback store, shingled forest, dynamic forest, `CountMinSketch`, `HyperLogLog`, `SpaceSaving`, `BloomFilter`) and leaves the core forest + trees + reservoir sampler + thresholded layer + meta / feature drift detectors +
t-digest + alert clusterer + bootstrap + calibrator + forensic baseline + audit record + severity bands + companion primitives (`OnlineStats`, `Normalizer<D>`, `PerFeatureEwma<D>`, `PerFeatureCusum<D>`) running under `#![no_std]` with `alloc`. Transcendentals (`ln`, `sqrt`, `exp`, …) route through `num-traits`

- `libm`; hashing-dependent code paths fall back to `alloc::collections::BTreeMap`.

```toml
[dependencies]
anomstream = { version = "…", default-features = false }
# Optional: serde persistence under no_std
anomstream = { version = "…", default-features = false, features = ["serde"] }
```

The `no_std` configuration is gated in CI (`cargo check --no-default-features` + `--features serde`).

## Performance

See [docs/performance.md](docs/performance.md) for the full criterion bench matrix. Benches are split across the three member crates: `cargo bench -p anomstream-core --bench modules` (detectors + primitives), `cargo bench -p anomstream-triage --bench modules` (Platt, SAGE, LSH), `cargo bench -p anomstream-hotpath --bench modules` (sampler, rate cap, channel).

## License

[Apache-2.0](LICENSE). Contributions under the same licence.
