//! Pure Rust Random Cut Forest for streaming anomaly detection.
//!
//! `rcf-rs` implements the Random Cut Forest (RCF) algorithm from Guha et al.
//! (ICML 2016) and is conformant with the
//! [AWS SageMaker RCF specification][aws-rcf]: reservoir sampling without
//! replacement (Park et al. 2004), random cuts weighted by per-dimension
//! range, anomaly score averaged across trees, and hyperparameter bounds
//! matching the AWS reference (`feature_dim`, `num_trees`,
//! `num_samples_per_tree`).
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
//! # Example
//!
//! ```ignore
//! use rcf_rs::ForestBuilder;
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
//! # Ok::<(), rcf_rs::RcfError>(())
//! ```
//!
//! # Conformance
//!
//! `rcf-rs` enforces the AWS `SageMaker` hyperparameter bounds at build time:
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

pub mod alert_cluster;
pub mod attribution_stability;
pub mod audit;
pub mod bootstrap;
pub mod calibrator;
pub mod config;
pub mod domain;
pub mod early_term;
pub mod error;
pub mod forensic;
pub mod forest;
pub mod group_score;
pub mod histogram;
pub mod meta_drift;
pub mod metrics;
#[cfg(feature = "serde")]
pub mod persistence;
#[cfg(feature = "std")]
pub mod pool;
pub mod sampler;
pub mod score_ci;
#[cfg(feature = "serde")]
pub(crate) mod serde_util;
pub mod severity;
pub mod thresholded;
pub mod tree;
pub mod visitor;

pub use alert_cluster::{AlertCluster, AlertClusterer, ClusterDecision};
pub use attribution_stability::AttributionStability;
pub use audit::{ALERT_RECORD_VERSION, AlertContext, AlertRecord};
pub use bootstrap::BootstrapReport;
pub use calibrator::{PlattCalibrator, PlattFitConfig};
pub use config::{ForestBuilder, RcfConfig};
pub use domain::{AnomalyScore, BoundingBox, Cut, DiVector, Point};
pub use early_term::{EarlyTermConfig, EarlyTermScore};
pub use error::{RcfError, RcfResult};
pub use forensic::ForensicBaseline;
pub use forest::{PointStore, RandomCutForest};
pub use group_score::{FeatureGroup, FeatureGroups, FeatureGroupsBuilder, GroupScores};
pub use histogram::{HistogramConfig, ScoreHistogram};
pub use meta_drift::{CusumConfig, DriftKind, DriftVerdict, MetaDriftDetector};
pub use metrics::{MetricsSink, NoopSink};
#[cfg(feature = "std")]
pub use pool::{ReadinessSummary, TenantForestPool};
pub use sampler::{ReservoirSampler, SamplerOp};
pub use score_ci::{DEFAULT_Z_FACTOR as DEFAULT_CI_Z_FACTOR, ScoreWithConfidence};
pub use severity::{Severity, SeverityBands};
pub use thresholded::{
    AnomalyGrade, EmaStats, ThresholdedConfig, ThresholdedForest, ThresholdedForestBuilder,
};
pub use tree::{Node, NodeRef, NodeStore, PointAccessor, RandomCutTree};
pub use visitor::{AttributionVisitor, ScalarScoreVisitor, Visitor};
