//! Configuration + builder for [`crate::ThresholdedForest`].
//!
//! [`ThresholdedConfig`] holds the parameters that govern the adaptive
//! threshold layer on top of the underlying [`crate::RandomCutForest`]:
//!
//! | Field | Role | Default |
//! |---|---|---|
//! | `z_factor` | Multiplier on the score stddev used to derive the threshold (`mean + z · stddev`). | `3.0` |
//! | `score_decay` | EMA smoothing factor for the running mean/variance of the anomaly scores. | `0.01` |
//! | `min_observations` | Samples required before the detector emits a non-warmup verdict. | `32` |
//! | `min_threshold` | Absolute floor on the adaptive threshold — prevents a near-zero stddev from firing on trivial jitter. | `1.0` |
//!
//! The builder mirrors [`crate::ForestBuilder`] so forest and threshold
//! parameters can be tuned side-by-side in one fluent chain.

use alloc::format;

use crate::config::ForestBuilder;
use crate::error::{RcfError, RcfResult};
use crate::thresholded::detector::ThresholdedForest;

/// Default `z_factor` — 3 standard deviations above the running mean,
/// matching the AWS `SageMaker` RCF guidance ("scores beyond 3σ are
/// considered anomalous"). Only meaningful under
/// [`ThresholdMode::ZSigma`].
pub const DEFAULT_Z_FACTOR: f64 = 3.0;

/// Default streaming quantile used when
/// [`ThresholdMode::Quantile`] is selected — `0.99` lets 1 % of
/// scores cross the threshold in steady state, matching the typical
/// SOC alert-rate budget.
pub const DEFAULT_QUANTILE: f64 = 0.99;

/// Default EMA smoothing factor on the anomaly-score stream. `0.01`
/// corresponds to an effective memory window of ~100 points.
pub const DEFAULT_SCORE_DECAY: f64 = 0.01;

/// Default minimum observations before the detector emits a
/// non-warmup verdict.
pub const DEFAULT_MIN_OBSERVATIONS: u64 = 32;

/// Default absolute floor on the adaptive threshold.
pub const DEFAULT_MIN_THRESHOLD: f64 = 1.0;

/// Which statistic drives the adaptive threshold. Isolation-depth
/// scores are right-skewed and heavy-tailed (not Gaussian), so the
/// `μ + z·σ` form systematically over-flags during baseline calm
/// periods and under-flags during drift. [`ThresholdMode::Quantile`]
/// uses a streaming `TDigest` of the score distribution and thresholds
/// on the chosen tail percentile — closer to the caller's actual
/// alert-rate budget (e.g. `p = 0.99` ≈ 1 % firing rate).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ThresholdMode {
    /// Classic `mean + z · stddev` on the EMA of the score stream.
    /// Back-compat default; keep this mode for Gaussian-like scores
    /// (lag-embedded streams with symmetric noise).
    ZSigma {
        /// Multiplier on the EMA stddev.
        z_factor: f64,
    },
    /// Streaming quantile threshold — `threshold = TDigest.quantile(p)`
    /// of observed scores. Robust to the isolation-depth right-skew;
    /// calibrates directly on the caller's alert-rate budget. `p`
    /// must be in `(0, 1)`; typical values are `0.99` / `0.999`.
    Quantile {
        /// Quantile used as the threshold. Higher `p` means a
        /// stricter threshold (fewer firings).
        p: f64,
    },
}

impl Default for ThresholdMode {
    fn default() -> Self {
        Self::ZSigma {
            z_factor: DEFAULT_Z_FACTOR,
        }
    }
}

/// Validated configuration of the adaptive-threshold layer.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThresholdedConfig {
    /// Multiplier on the score stddev used to derive the adaptive
    /// threshold when [`Self::threshold_mode`] is
    /// [`ThresholdMode::ZSigma`]. Kept as a top-level field for
    /// back-compat — callers constructing via struct literal get
    /// the legacy behaviour without opt-in. Ignored under
    /// [`ThresholdMode::Quantile`].
    pub z_factor: f64,
    /// Selects whether the threshold is driven by the EMA's
    /// `mean + z·σ` or by a streaming quantile of the score
    /// distribution. Defaults to [`ThresholdMode::ZSigma`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub threshold_mode: ThresholdMode,
    /// EMA smoothing factor on the score stream. Must be in `(0, 1]`.
    pub score_decay: f64,
    /// Samples required before the detector stops emitting
    /// warming-up verdicts.
    pub min_observations: u64,
    /// Absolute floor on the adaptive threshold.
    pub min_threshold: f64,
}

impl Default for ThresholdedConfig {
    fn default() -> Self {
        Self {
            z_factor: DEFAULT_Z_FACTOR,
            threshold_mode: ThresholdMode::default(),
            score_decay: DEFAULT_SCORE_DECAY,
            min_observations: DEFAULT_MIN_OBSERVATIONS,
            min_threshold: DEFAULT_MIN_THRESHOLD,
        }
    }
}

impl ThresholdedConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when any field is outside
    /// its accepted range: `z_factor` must be finite and positive,
    /// `score_decay` finite and in `(0, 1]`, `min_threshold` finite
    /// and non-negative.
    pub fn validate(&self) -> RcfResult<()> {
        match self.threshold_mode {
            ThresholdMode::ZSigma { z_factor } => {
                if !z_factor.is_finite() || z_factor <= 0.0 {
                    return Err(RcfError::InvalidConfig(format!(
                        "z_factor must be finite and > 0, got {z_factor}"
                    )));
                }
            }
            ThresholdMode::Quantile { p } => {
                if !p.is_finite() || !(0.0..1.0).contains(&p) || p <= 0.0 {
                    return Err(RcfError::InvalidConfig(format!(
                        "Quantile p must be in (0.0, 1.0), got {p}"
                    )));
                }
            }
        }
        // The legacy `z_factor` field is still validated so callers
        // building via struct literal (without touching
        // `threshold_mode`) still get the strictness they used to.
        if !self.z_factor.is_finite() || self.z_factor <= 0.0 {
            return Err(RcfError::InvalidConfig(format!(
                "z_factor must be finite and > 0, got {}",
                self.z_factor
            )));
        }
        if !self.score_decay.is_finite() || self.score_decay <= 0.0 || self.score_decay > 1.0 {
            return Err(RcfError::InvalidConfig(format!(
                "score_decay must be in (0.0, 1.0], got {}",
                self.score_decay
            )));
        }
        if !self.min_threshold.is_finite() || self.min_threshold < 0.0 {
            return Err(RcfError::InvalidConfig(format!(
                "min_threshold must be finite and >= 0, got {}",
                self.min_threshold
            )));
        }
        Ok(())
    }
}

/// Fluent builder for [`ThresholdedForest`].
///
/// Wraps a [`ForestBuilder`] so callers configure the underlying
/// forest and the threshold layer in one chain:
///
/// ```
/// use rcf_rs::ThresholdedForestBuilder;
///
/// let detector = ThresholdedForestBuilder::<4>::new()
///     .num_trees(50)
///     .sample_size(64)
///     .z_factor(3.0)
///     .seed(42)
///     .build()
///     .unwrap();
/// assert_eq!(detector.forest().num_trees(), 50);
/// ```
#[derive(Debug, Clone)]
pub struct ThresholdedForestBuilder<const D: usize> {
    /// Forest layer builder (forwarded to through explicit methods).
    forest: ForestBuilder<D>,
    /// Threshold layer configuration under construction.
    thresholded: ThresholdedConfig,
}

impl<const D: usize> Default for ThresholdedForestBuilder<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> ThresholdedForestBuilder<D> {
    /// Start a new builder with AWS-conformant forest defaults and
    /// the threshold defaults described in [`ThresholdedConfig`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            forest: ForestBuilder::<D>::new(),
            thresholded: ThresholdedConfig::default(),
        }
    }

    /// Override the number of trees in the underlying forest.
    #[must_use]
    pub fn num_trees(mut self, n: usize) -> Self {
        self.forest = self.forest.num_trees(n);
        self
    }

    /// Override the per-tree reservoir size of the underlying forest.
    #[must_use]
    pub fn sample_size(mut self, s: usize) -> Self {
        self.forest = self.forest.sample_size(s);
        self
    }

    /// Override the reservoir time-decay factor of the underlying
    /// forest (biases the reservoir toward recent points).
    #[must_use]
    pub fn time_decay(mut self, d: f64) -> Self {
        self.forest = self.forest.time_decay(d);
        self
    }

    /// Pin the forest RNG seed for reproducible runs.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.forest = self.forest.seed(seed);
        self
    }

    /// Request a dedicated rayon thread pool for the forest's parallel
    /// paths. Requires the `parallel` cargo feature. See
    /// [`ForestBuilder::num_threads`].
    #[must_use]
    pub fn num_threads(mut self, n: usize) -> Self {
        self.forest = self.forest.num_threads(n);
        self
    }

    /// Override the warmup admission fraction forwarded to each
    /// per-tree reservoir. Forwarded to
    /// [`ForestBuilder::initial_accept_fraction`].
    #[must_use]
    pub fn initial_accept_fraction(mut self, f: f64) -> Self {
        self.forest = self.forest.initial_accept_fraction(f);
        self
    }

    /// Set per-dimension multiplicative feature scales on the
    /// underlying forest. Forwarded to
    /// [`ForestBuilder::feature_scales`]. See that method for
    /// semantics and validation rules.
    #[must_use]
    pub fn feature_scales(mut self, scales: [f64; D]) -> Self {
        self.forest = self.forest.feature_scales(scales);
        self
    }

    /// Override the threshold's z-factor. Implies
    /// [`ThresholdMode::ZSigma`] — mutually exclusive with
    /// [`Self::quantile_threshold`]; the last call wins.
    #[must_use]
    pub fn z_factor(mut self, z: f64) -> Self {
        self.thresholded.z_factor = z;
        self.thresholded.threshold_mode = ThresholdMode::ZSigma { z_factor: z };
        self
    }

    /// Drive the threshold from a streaming quantile of the score
    /// distribution instead of the Gaussian `μ + z·σ`. `p` is the
    /// target tail quantile — `0.99` budgets ~1 % alert rate in
    /// steady state, `0.999` ~0.1 %. Mutually exclusive with
    /// [`Self::z_factor`]; the last call wins.
    #[must_use]
    pub fn quantile_threshold(mut self, p: f64) -> Self {
        self.thresholded.threshold_mode = ThresholdMode::Quantile { p };
        self
    }

    /// Override the EMA smoothing factor on the anomaly-score stream.
    #[must_use]
    pub fn score_decay(mut self, d: f64) -> Self {
        self.thresholded.score_decay = d;
        self
    }

    /// Override the number of samples the detector requires before
    /// emitting a non-warmup verdict.
    #[must_use]
    pub fn min_observations(mut self, n: u64) -> Self {
        self.thresholded.min_observations = n;
        self
    }

    /// Override the absolute floor on the adaptive threshold.
    #[must_use]
    pub fn min_threshold(mut self, t: f64) -> Self {
        self.thresholded.min_threshold = t;
        self
    }

    /// Read-only access to the forest-layer builder.
    #[must_use]
    pub fn forest_builder(&self) -> &ForestBuilder<D> {
        &self.forest
    }

    /// Read-only access to the threshold-layer configuration.
    #[must_use]
    pub fn thresholded_config(&self) -> &ThresholdedConfig {
        &self.thresholded
    }

    /// Validate every parameter and build the detector.
    ///
    /// # Errors
    ///
    /// Propagates [`ForestBuilder::build`] errors and
    /// [`ThresholdedConfig::validate`] errors.
    pub fn build(self) -> RcfResult<ThresholdedForest<D>> {
        self.thresholded.validate()?;
        let forest = self.forest.build()?;
        ThresholdedForest::<D>::from_parts(forest, self.thresholded)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Defaults compared bit-exactly against the module constants.
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        ThresholdedConfig::default().validate().unwrap();
    }

    #[test]
    fn default_config_fields_match_constants() {
        let c = ThresholdedConfig::default();
        assert_eq!(c.z_factor, DEFAULT_Z_FACTOR);
        assert_eq!(c.score_decay, DEFAULT_SCORE_DECAY);
        assert_eq!(c.min_observations, DEFAULT_MIN_OBSERVATIONS);
        assert_eq!(c.min_threshold, DEFAULT_MIN_THRESHOLD);
    }

    fn cfg(z: f64, decay: f64, min_obs: u64, min_thr: f64) -> ThresholdedConfig {
        ThresholdedConfig {
            z_factor: z,
            threshold_mode: ThresholdMode::ZSigma { z_factor: z },
            score_decay: decay,
            min_observations: min_obs,
            min_threshold: min_thr,
        }
    }

    #[test]
    fn validate_rejects_non_finite_z_factor() {
        assert!(
            cfg(f64::NAN, DEFAULT_SCORE_DECAY, 1, 0.0)
                .validate()
                .is_err()
        );
        assert!(
            cfg(f64::INFINITY, DEFAULT_SCORE_DECAY, 1, 0.0)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_rejects_non_positive_z_factor() {
        assert!(cfg(0.0, DEFAULT_SCORE_DECAY, 1, 0.0).validate().is_err());
        assert!(cfg(-1.0, DEFAULT_SCORE_DECAY, 1, 0.0).validate().is_err());
    }

    #[test]
    fn validate_rejects_score_decay_outside_range() {
        assert!(cfg(DEFAULT_Z_FACTOR, 0.0, 1, 0.0).validate().is_err());
        assert!(cfg(DEFAULT_Z_FACTOR, 1.5, 1, 0.0).validate().is_err());
        assert!(cfg(DEFAULT_Z_FACTOR, f64::NAN, 1, 0.0).validate().is_err());
    }

    #[test]
    fn validate_rejects_negative_min_threshold() {
        assert!(
            cfg(DEFAULT_Z_FACTOR, DEFAULT_SCORE_DECAY, 1, -0.001)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn builder_defaults_pass_validation() {
        let b = ThresholdedForestBuilder::<4>::new();
        b.thresholded_config().validate().unwrap();
        b.forest_builder().config().validate().unwrap();
    }

    #[test]
    fn builder_overrides_apply_to_both_layers() {
        let b = ThresholdedForestBuilder::<4>::new()
            .num_trees(150)
            .sample_size(128)
            .z_factor(2.5)
            .score_decay(0.05)
            .min_observations(10)
            .min_threshold(0.5)
            .initial_accept_fraction(0.125)
            .seed(7);
        assert_eq!(b.forest_builder().config().num_trees, 150);
        assert_eq!(b.forest_builder().config().sample_size, 128);
        assert_eq!(b.forest_builder().config().seed, Some(7));
        assert!((b.forest_builder().config().initial_accept_fraction - 0.125).abs() < f64::EPSILON);
        assert_eq!(b.thresholded_config().z_factor, 2.5);
        assert_eq!(b.thresholded_config().score_decay, 0.05);
        assert_eq!(b.thresholded_config().min_observations, 10);
        assert_eq!(b.thresholded_config().min_threshold, 0.5);
    }

    #[test]
    fn builder_build_validates_forest_layer() {
        let err = ThresholdedForestBuilder::<4>::new()
            .num_trees(10)
            .build()
            .unwrap_err();
        assert!(matches!(err, RcfError::InvalidConfig(_)));
    }

    #[test]
    fn builder_build_validates_threshold_layer() {
        let err = ThresholdedForestBuilder::<4>::new()
            .z_factor(-1.0)
            .build()
            .unwrap_err();
        assert!(matches!(err, RcfError::InvalidConfig(_)));
    }
}
