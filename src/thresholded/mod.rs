//! Adaptive-threshold layer on top of [`crate::RandomCutForest`].
//!
//! Where the bare forest returns a raw anomaly score in `[0, ∞)`,
//! [`ThresholdedForest`] tracks the running distribution of those
//! scores and emits a graded verdict — `is_anomaly: bool`,
//! `grade ∈ [0, 1]`, and the `threshold` in effect at observation
//! time. Callers no longer have to hand-pick a magic threshold per
//! deployment: the detector adapts to the traffic it sees.
//!
//! Inspired by the AWS *Thresholded Random Cut Forest* (TRCF)
//! facility in `randomcutforest-parkservices`, but intentionally
//! lighter: only the adaptive μ + z·σ threshold over an EMA of the
//! score stream, without the short/long-term duality or the
//! near-threshold heuristics of the full TRCF.
//!
//! # Example
//!
//! ```ignore
//! use rcf_rs::ThresholdedForestBuilder;
//!
//! let mut detector = ThresholdedForestBuilder::<4>::new()
//!     .num_trees(100)
//!     .sample_size(256)
//!     .z_factor(3.0)
//!     .min_observations(32)
//!     .seed(42)
//!     .build()?;
//!
//! for packet in stream_of_feature_vectors {
//!     let verdict = detector.process(packet)?;
//!     if verdict.is_anomaly() {
//!         eprintln!(
//!             "anomaly: grade={:.2} score={} threshold={:.3}",
//!             verdict.grade(),
//!             verdict.score(),
//!             verdict.threshold(),
//!         );
//!     }
//! }
//! # Ok::<(), rcf_rs::RcfError>(())
//! ```

pub mod config;
pub mod detector;
pub mod grade;
pub mod stats;

pub use config::{
    DEFAULT_MIN_OBSERVATIONS, DEFAULT_MIN_THRESHOLD, DEFAULT_QUANTILE, DEFAULT_SCORE_DECAY,
    DEFAULT_Z_FACTOR, ThresholdMode, ThresholdedConfig, ThresholdedForestBuilder,
};
pub use detector::ThresholdedForest;
pub use grade::AnomalyGrade;
pub use stats::EmaStats;
