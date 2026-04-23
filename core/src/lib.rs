//! `anomstream-core` — core detectors + streaming primitives + cross-cut contracts.
//!
//! This crate is the math-first floor of the
//! [`anomstream`](https://crates.io/crates/anomstream) workspace.
//! It implements the Random Cut Forest (RCF) algorithm from Guha et al.
//! (ICML 2016) conformant with the
//! [AWS SageMaker RCF specification][aws-rcf]: reservoir sampling without
//! replacement (Park et al. 2004), random cuts weighted by per-dimension
//! range, anomaly score averaged across trees, and hyperparameter bounds
//! matching the AWS reference (`feature_dim`, `num_trees`,
//! `num_samples_per_tree`).
//!
//! Beyond the core forest, the crate ships a set of **companion
//! primitives** — per-feature drift detectors, normalisers, streaming
//! stats, frequency sketches — reused across detection pipelines so
//! callers can compose `RandomCutForest` + `PerFeatureEwma` +
//! `PerFeatureCusum` + `FeatureDriftDetector` + `Normalizer` + …
//! without reimplementing the underlying math.
//!
//! # Workspace charter
//!
//! `anomstream-core` = *detectors + streaming primitives + cross-cut
//! contracts*. Three categories of cross-cutting contracts live here
//! (not in a sibling crate) because every downstream layer depends
//! on them:
//!
//! - [`metrics::MetricsSink`] — telemetry trait consumed by every
//!   detector, the hot-path sampler, and the triage pipeline.
//! - [`severity::Severity`] + [`severity::SeverityBands`] —
//!   classification vocabulary used by both the bare forest
//!   ([`domain::AnomalyScore::severity`]) and the triage layer's
//!   `AlertRecord` / `AlertClusterer`.
//! - [`forest::ForestSnapshot`] — read-only health view that lets
//!   downstream triage consume forest state without reaching into
//!   reservoir internals.
//!
//! The scope is kept deliberately tight: streaming multivariate
//! anomaly primitives only; no protocol parsers, no ONNX runtimes,
//! no IP-centric trackers, no SOC-specific triage opinion. Those
//! belong in [`anomstream-triage`](https://crates.io/crates/anomstream-triage)
//! and [`anomstream-hotpath`](https://crates.io/crates/anomstream-hotpath).
//!
//! # Architecture
//!
//! ```text
//! +-----------------+        +------------------+
//! |  ForestBuilder  |        |  RcfConfig       |
//! +--------+--------+        +---------+--------+
//!          |                           |
//!          v                           v
//!   +------+---------------------------+------+
//!   |          RandomCutForest                |   <- aggregate root
//!   |                                         |
//!   |   PointStore (ring buffer, repository)  |
//!   |                                         |
//!   |   trees: Vec<(RandomCutTree, Sampler)>  |
//!   +-+---------------------+-----------------+
//!     |                     |
//!     v                     v
//! +---+---------+   +-------+--------+
//! | RandomCutTr |   | ReservoirSampl |
//! +-------------+   +----------------+
//!     |
//!     v  via Visitor trait
//! +---+----------------+    +-----------------------+
//! | ScalarScoreVisitor |    | AttributionVisitor    |
//! +--------------------+    +-----------------------+
//! ```
//!
//! Modules are organised in layered fashion: `domain` (value objects),
//! `tree` (storage + cut tree), `sampler` (reservoir), `visitor` (scoring
//! strategies), `forest` (aggregate root), `thresholded` (adaptive
//! threshold layer on top of the forest), `pool` (bounded per-tenant
//! detector pool with LRU eviction), `bootstrap` (cold-start replay
//! helpers for restart resumption from an upstream TSDB),
//! `group_score` (named-group decomposition of the per-dim
//! attribution vector), `attribution_stability` (inter-tree
//! dispersion + confidence for attribution), and `meta_drift`
//! (two-sided CUSUM change-point detector over the score stream).
//! The `persistence` module is gated behind the `serde` feature;
//! `pool` is gated behind `std`.
//!
//! # Companion primitives
//!
//! Reusable streaming primitives that compose with the forest:
//!
//! | Module | Purpose |
//! |---|---|
//! | [`online_stats`] | Welford streaming mean + variance |
//! | [`count_min_sketch`] | Probabilistic frequency sketch (`std`-gated) |
//! | [`normalize`] | `MinMax` / `ZScore` / `None` per-feature transforms |
//! | [`per_feature_ewma`] | Parallel univariate EWMA z-score detector |
//! | [`per_feature_cusum`] | Parallel two-sided CUSUM change-point detector |
//! | [`severity`] | Ordinal severity bands + classification |
//!
//! The companion layer is policy-free — detectors return raw
//! statistics (z-scores, CUSUM magnitudes, min-max transforms);
//! callers map them to alert severity via [`SeverityBands`] or a
//! custom rule. `per_feature_cusum` intentionally co-exists with
//! [`meta_drift::MetaDriftDetector`] (scalar CUSUM on the score
//! stream): use the per-feature variant for attribution, the
//! meta variant for score-level regime change.
//!
//! # Example
//!
//! ```ignore
//! use anomstream_core::ForestBuilder;
//!
//! let mut forest = ForestBuilder::<4>::new()
//!     .num_trees(100)
//!     .sample_size(256)
//!     .seed(42)
//!     .build()?;
//!
//! for point in stream_of_points {
//!     forest.update(point.clone())?;
//!     let score = forest.score(&point)?;
//!     if f64::from(score) > 1.5 {
//!         eprintln!("anomaly: {score}");
//!     }
//! }
//! # Ok::<(), anomstream_core::RcfError>(())
//! ```
//!
//! # Conformance
//!
//! `anomstream-core` enforces the AWS `SageMaker` hyperparameter bounds at build time:
//!
//! | Parameter | Range | Default |
//! |---|---|---|
//! | `feature_dim` | `[1, 10000]` | required |
//! | `num_trees` | `[50, 1000]` | `100` |
//! | `num_samples_per_tree` | `[1, 2048]` | `256` |
//! | `time_decay` | `[0, 1]` | `0.1 / sample_size` (tracks AWS Java `CompactSampler`; pass `0.0` to disable recency bias) |
//!
//! See the crate's `README.md` for the full conformance matrix and the
//! comparison against `krcf` and the AWS Java port.
//!
//! # References
//!
//! 1. Sudipto Guha, Nina Mishra, Gourav Roy, Okke Schrijvers. "Robust Random
//!    Cut Forest Based Anomaly Detection on Streams." *International
//!    Conference on Machine Learning*, pp. 2712–2721. 2016.
//! 2. Byung-Hoon Park, George Ostrouchov, Nagiza F. Samatova, Al Geist.
//!    "Reservoir-based random sampling with replacement from data stream."
//!    *SIAM International Conference on Data Mining*, pp. 492–496. 2004.
//! 3. AWS `SageMaker` RCF reference.
//!
//! [aws-rcf]: https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
// The crate-level restriction deny on `unwrap_used` / `panic` keeps
// the public prod paths free of panic-on-unwrap; unit tests inside
// `#[cfg(test)]` modules use `.unwrap()` and explicit `panic!` as
// fail-fast idioms and opt out here.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::panic))]

extern crate alloc;

#[cfg(feature = "std")]
pub mod adwin;
pub mod attribution_stability;
#[cfg(feature = "std")]
pub mod bloom;
pub mod bootstrap;
pub mod config;
#[cfg(feature = "std")]
pub mod count_min_sketch;
pub mod domain;
#[cfg(feature = "std")]
pub mod drift_aware;
#[cfg(feature = "std")]
pub mod dynamic_forest;
pub mod early_term;
#[cfg(feature = "std")]
pub mod ensemble;
pub mod error;
pub mod feature_drift;
pub mod forensic;
pub mod forest;
pub mod group_score;
pub mod histogram;
#[cfg(feature = "std")]
pub mod hyperloglog;
pub mod meta_drift;
pub mod metrics;
pub mod normalize;
pub mod online_stats;
pub mod per_feature_cusum;
pub mod per_feature_ewma;
#[cfg(feature = "serde")]
pub mod persistence;
#[cfg(feature = "std")]
pub mod pool;
pub mod sampler;
pub mod score_ci;
#[cfg(feature = "serde")]
pub mod serde_util;
pub mod severity;
#[cfg(feature = "std")]
pub mod shingled;
#[cfg(feature = "std")]
pub mod space_saving;
pub mod tdigest;
pub mod thresholded;
pub mod tree;
#[cfg(feature = "std")]
pub mod univariate_spot;
pub mod visitor;

#[cfg(feature = "std")]
pub use adwin::{
    AdwinDetector, DEFAULT_DELTA as ADWIN_DEFAULT_DELTA, DEFAULT_WINDOW_CAP as ADWIN_DEFAULT_WINDOW,
};
pub use attribution_stability::AttributionStability;
#[cfg(feature = "std")]
pub use bloom::{
    BloomFilter, DEFAULT_FALSE_POSITIVE_RATE as BLOOM_DEFAULT_FPR, MAX_HASHES as BLOOM_MAX_HASHES,
};
pub use bootstrap::BootstrapReport;
pub use config::{ForestBuilder, RcfConfig};
#[cfg(feature = "std")]
pub use count_min_sketch::CountMinSketch;
pub use domain::{AnomalyScore, BoundingBox, Cut, DiVector, Point};
#[cfg(feature = "std")]
pub use drift_aware::{DriftAwareForest, DriftRecoveryConfig};
#[cfg(feature = "std")]
pub use dynamic_forest::DynamicForest;
pub use early_term::{EarlyTermConfig, EarlyTermScore};
#[cfg(feature = "std")]
pub use ensemble::{chi_squared_survival_even, fisher_combine};
pub use error::{RcfError, RcfResult};
pub use feature_drift::{DriftLevel, FeatureDriftDetector};
pub use forensic::ForensicBaseline;
pub use forest::{ForestSnapshot, PointStore, RandomCutForest};
pub use group_score::{FeatureGroup, FeatureGroups, FeatureGroupsBuilder, GroupScores};
pub use histogram::{HistogramConfig, ScoreHistogram};
#[cfg(feature = "std")]
pub use hyperloglog::{
    DEFAULT_PRECISION as HLL_DEFAULT_PRECISION, HyperLogLog, MAX_PRECISION as HLL_MAX_PRECISION,
    MIN_PRECISION as HLL_MIN_PRECISION,
};
pub use meta_drift::{CusumConfig, DriftKind, DriftVerdict, MetaDriftDetector};
pub use metrics::{MetricsSink, NoopSink};
pub use normalize::{NormParams, NormStrategy, Normalizer};
pub use online_stats::OnlineStats;
pub use per_feature_cusum::{
    DriftDirection, PerFeatureCusum, PerFeatureCusumAccumulator, PerFeatureCusumAlert,
    PerFeatureCusumConfig, PerFeatureCusumResult,
};
pub use per_feature_ewma::{
    EwmaAccumulator, PerFeatureEwma, PerFeatureEwmaConfig, PerFeatureEwmaResult,
};
#[cfg(feature = "std")]
pub use pool::{ReadinessSummary, TenantForestPool};
pub use sampler::{ReservoirSampler, SamplerOp};
pub use score_ci::{DEFAULT_Z_FACTOR as DEFAULT_CI_Z_FACTOR, ScoreWithConfidence};
pub use severity::{Severity, SeverityBands};
#[cfg(feature = "std")]
pub use shingled::{ShingledForest, ShingledForestBuilder};
#[cfg(feature = "std")]
pub use space_saving::{
    DEFAULT_CAPACITY as SPACE_SAVING_DEFAULT_CAPACITY, HeavyHitter, HeavyHitterEntry, SpaceSaving,
};
pub use tdigest::{Centroid, DEFAULT_COMPRESSION as TDIGEST_DEFAULT_COMPRESSION, TDigest};
pub use thresholded::{
    AnomalyGrade, EmaStats, ThresholdMode, ThresholdedConfig, ThresholdedForest,
    ThresholdedForestBuilder,
};
pub use tree::{
    InternalData, LeafData, NodeRef, NodeStore, NodeView, NodeViewMut, PointAccessor, RandomCutTree,
};
#[cfg(feature = "std")]
pub use univariate_spot::{
    DEFAULT_ALERT_P as SPOT_DEFAULT_ALERT_P, DEFAULT_QUANTILE as SPOT_DEFAULT_QUANTILE, PotDetector,
};
pub use visitor::{AttributionVisitor, ScalarScoreVisitor, ScoreAttributionVisitor, Visitor};
