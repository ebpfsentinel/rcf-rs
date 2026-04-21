//! Forest configuration and the [`ForestBuilder`] entry point.
//!
//! [`RcfConfig`] enforces the AWS `SageMaker` Random Cut Forest
//! hyperparameter bounds at validation time. Callers should construct
//! a forest through [`ForestBuilder`] rather than instantiating
//! [`RcfConfig`] directly so the builder picks AWS-conformant defaults.
//!
//! Per-point dimensionality is encoded at the type level as the
//! `D` const-generic on [`ForestBuilder`] / [`crate::RandomCutForest`]
//! so the bounding-box and per-tree node storage live on the stack
//! and the compiler can vectorise the hot tree-traversal loops.

use alloc::format;
use alloc::vec::Vec;

use crate::error::{RcfError, RcfResult};
use crate::forest::random_cut_forest::RandomCutForest;

/// AWS lower bound for `feature_dim`.
pub const MIN_DIMENSION: usize = 1;
/// AWS upper bound for `feature_dim`.
pub const MAX_DIMENSION: usize = 10_000;
/// AWS lower bound for `num_trees`.
pub const MIN_NUM_TREES: usize = 50;
/// AWS upper bound for `num_trees`.
pub const MAX_NUM_TREES: usize = 1_000;
/// AWS default for `num_trees`.
pub const DEFAULT_NUM_TREES: usize = 100;
/// AWS lower bound for `num_samples_per_tree`.
pub const MIN_SAMPLE_SIZE: usize = 1;
/// AWS upper bound for `num_samples_per_tree`.
pub const MAX_SAMPLE_SIZE: usize = 2_048;
/// AWS default for `num_samples_per_tree`.
pub const DEFAULT_SAMPLE_SIZE: usize = 256;
/// Scaling numerator used to derive the default time-decay factor
/// from `sample_size`: `default_time_decay = TIME_DECAY_NUMERATOR /
/// sample_size`. `0.1` matches the AWS Java `CompactSampler` default
/// and gives an effective reservoir "half-life" of a handful of
/// reservoirs-worth of input — enough recency bias to track baseline
/// drift on a streaming agent over hours / days without losing the
/// uniform-sampling character on each individual window.
pub const TIME_DECAY_NUMERATOR: f64 = 0.1;
/// Default time-decay resolved against [`DEFAULT_SAMPLE_SIZE`] —
/// `0.1 / 256 ≈ 3.9 × 10⁻⁴`. Prefer [`default_time_decay_for`] when
/// the sample size differs from the default.
// Cast is precision-safe: 256 fits in f64's mantissa exactly.
#[allow(clippy::cast_precision_loss)]
pub const DEFAULT_TIME_DECAY: f64 = TIME_DECAY_NUMERATOR / DEFAULT_SAMPLE_SIZE as f64;

/// Compute the default time-decay for a given `sample_size`:
/// `0.1 / sample_size`, clamped to `0.0` for `sample_size == 0`
/// (which is caught separately by [`RcfConfig::validate`]).
#[must_use]
pub fn default_time_decay_for(sample_size: usize) -> f64 {
    if sample_size == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    {
        TIME_DECAY_NUMERATOR / sample_size as f64
    }
}
/// Default warmup admission fraction — `1.0` disables the gate and
/// matches the classic reservoir behaviour. Set below `1.0` (AWS
/// uses `0.125`) to ramp admission during the cold-start period.
pub const DEFAULT_INITIAL_ACCEPT_FRACTION: f64 = 1.0;

/// Validated forest hyperparameters (dimension is encoded separately
/// at the type level).
///
/// # Examples
///
/// ```
/// use rcf_rs::{ForestBuilder, RcfConfig};
///
/// let builder = ForestBuilder::<4>::new()
///     .num_trees(50)
///     .sample_size(64);
/// let cfg: &RcfConfig = builder.config();
/// assert_eq!(cfg.num_trees, 50);
/// cfg.validate().unwrap();
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RcfConfig {
    /// Number of trees in the forest (`num_trees`).
    pub num_trees: usize,
    /// Maximum reservoir size per tree (`num_samples_per_tree`).
    pub sample_size: usize,
    /// Time-decay factor applied to reservoir sampling weights. A
    /// value of `0.0` restores strict uniform sampling; positive
    /// values bias the reservoir toward recent points. Default
    /// resolved by [`ForestBuilder`] is `0.1 / sample_size`, matching
    /// the AWS Java `CompactSampler` reference.
    pub time_decay: f64,
    /// Optional deterministic seed; `None` falls back to entropy.
    pub seed: Option<u64>,
    /// Optional dedicated rayon thread pool size for the `parallel`
    /// cargo feature. `None` means "use rayon's global pool"
    /// (configurable via the `RAYON_NUM_THREADS` env var). `Some(n)`
    /// builds a per-forest [`rayon::ThreadPool`] of `n` workers so
    /// callers can isolate this forest from the rest of the
    /// application's rayon workload. Ignored without `parallel`.
    pub num_threads: Option<usize>,
    /// Warmup admission fraction forwarded to every per-tree
    /// [`crate::ReservoirSampler`]. See that type's module-level docs
    /// for semantics. `1.0` disables the gate; smaller values ramp
    /// admission during the cold-start period so the reservoir is
    /// less dominated by the first few stream entries.
    #[cfg_attr(feature = "serde", serde(default = "default_initial_accept_fraction"))]
    pub initial_accept_fraction: f64,
    /// Optional per-dimension multiplicative weights applied to every
    /// point before it reaches the forest's hot paths (`update`,
    /// `score`, `attribution`, `bootstrap`, `delete_by_value`). Length
    /// must match the forest's compile-time dimension `D`.
    ///
    /// Intended for per-feature scale normalisation: when different
    /// input dimensions have wildly different dynamic ranges
    /// (packet-rate in `[10², 10⁶]`, protocol-mix ratios in `[0, 1]`,
    /// entropy in `[0, 8]` bits), a naive random cut weights each
    /// dimension by its raw range. Pre-scaling with `1 / stddev[d]`
    /// recovers a unit-variance input space where every dim pulls
    /// its weight. For full z-score normalisation the caller should
    /// still mean-centre upstream — `feature_scales` is a weight, not
    /// a full affine transform.
    ///
    /// `None` keeps the classic "forest sees the raw caller point"
    /// behaviour. The field is `#[serde(default)]` so old snapshots
    /// deserialise without migration.
    #[cfg_attr(feature = "serde", serde(default))]
    pub feature_scales: Option<Vec<f64>>,
}

/// Serde default for [`RcfConfig::initial_accept_fraction`] so payloads
/// persisted before the warmup knob existed deserialise with the
/// gate disabled.
#[cfg(feature = "serde")]
#[must_use]
fn default_initial_accept_fraction() -> f64 {
    DEFAULT_INITIAL_ACCEPT_FRACTION
}

impl RcfConfig {
    /// Validate the configuration against the AWS hyperparameter
    /// bounds. The forest's compile-time dimension `D` is checked
    /// separately via [`Self::validate_dimension`] so non-const
    /// callers can apply the AWS bounds without instantiating a
    /// generic.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] with the offending
    /// parameter when any bound is violated.
    pub fn validate(&self) -> RcfResult<()> {
        if !(MIN_NUM_TREES..=MAX_NUM_TREES).contains(&self.num_trees) {
            return Err(RcfError::InvalidConfig(format!(
                "num_trees {} out of [{}, {}]",
                self.num_trees, MIN_NUM_TREES, MAX_NUM_TREES
            )));
        }
        if !(MIN_SAMPLE_SIZE..=MAX_SAMPLE_SIZE).contains(&self.sample_size) {
            return Err(RcfError::InvalidConfig(format!(
                "sample_size {} out of [{}, {}]",
                self.sample_size, MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE
            )));
        }
        if !self.time_decay.is_finite() || !(0.0..=1.0).contains(&self.time_decay) {
            return Err(RcfError::InvalidConfig(format!(
                "time_decay {} out of [0.0, 1.0]",
                self.time_decay
            )));
        }
        if let Some(n) = self.num_threads
            && n == 0
        {
            return Err(RcfError::InvalidConfig(
                "num_threads must be > 0 when set; use None to fall back to rayon's global pool"
                    .into(),
            ));
        }
        if !self.initial_accept_fraction.is_finite()
            || self.initial_accept_fraction <= 0.0
            || self.initial_accept_fraction > 1.0
        {
            return Err(RcfError::InvalidConfig(format!(
                "initial_accept_fraction {} out of (0.0, 1.0]",
                self.initial_accept_fraction
            )));
        }
        if let Some(scales) = &self.feature_scales {
            for (i, s) in scales.iter().enumerate() {
                if !s.is_finite() || *s <= 0.0 {
                    return Err(RcfError::InvalidConfig(format!(
                        "feature_scales[{i}] must be finite and > 0, got {s}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validate the declared [`RcfConfig::feature_scales`] against a
    /// target per-point dimension `d`. When `feature_scales` is
    /// `None`, the check is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::DimensionMismatch`] when the scales vector
    /// length does not equal `d`.
    pub fn validate_feature_scales_dimension(&self, d: usize) -> RcfResult<()> {
        if let Some(scales) = &self.feature_scales
            && scales.len() != d
        {
            return Err(RcfError::DimensionMismatch {
                expected: d,
                got: scales.len(),
            });
        }
        Ok(())
    }

    /// Validate the compile-time dimension `D` against the AWS
    /// `feature_dim` bounds. Called by [`ForestBuilder::build`] so
    /// every user-facing entry point gates on the AWS limits.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `D` is outside
    /// `[MIN_DIMENSION, MAX_DIMENSION]`.
    pub fn validate_dimension(dimension: usize) -> RcfResult<()> {
        if !(MIN_DIMENSION..=MAX_DIMENSION).contains(&dimension) {
            return Err(RcfError::InvalidConfig(format!(
                "dimension {dimension} out of [{MIN_DIMENSION}, {MAX_DIMENSION}]"
            )));
        }
        Ok(())
    }
}

/// Fluent builder for [`RandomCutForest`].
///
/// Defaults: `num_trees = 100`, `sample_size = 256`,
/// `time_decay = 0.1 / sample_size` (matches AWS Java
/// `CompactSampler`; call [`Self::time_decay`] with `0.0` to recover
/// strict uniform sampling), RNG seeded from entropy.
///
/// `D` is the per-point dimensionality. Callers pin it at construction
/// via turbofish: `ForestBuilder::<4>::new()`.
///
/// # Examples
///
/// ```
/// use rcf_rs::ForestBuilder;
///
/// let mut forest = ForestBuilder::<4>::new()
///     .num_trees(50)
///     .sample_size(64)
///     .seed(42)
///     .build()
///     .expect("AWS-conformant config");
/// forest.update([0.0, 0.0, 0.0, 0.0]).expect("dim matches");
/// ```
#[derive(Debug, Clone)]
pub struct ForestBuilder<const D: usize> {
    /// Working configuration mutated by the fluent builder methods
    /// and validated when [`ForestBuilder::build`] runs.
    config: RcfConfig,
    /// Whether the caller has explicitly overridden `time_decay` via
    /// [`Self::time_decay`]. When `false`, [`Self::sample_size`] and
    /// [`Self::build`] resolve `time_decay` from the current
    /// `sample_size` so the AWS `0.1 / sample_size` default tracks
    /// any reservoir-size override the caller applies.
    time_decay_explicit: bool,
}

impl<const D: usize> Default for ForestBuilder<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> ForestBuilder<D> {
    /// Start a new builder for `D`-dimensional points.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: RcfConfig {
                num_trees: DEFAULT_NUM_TREES,
                sample_size: DEFAULT_SAMPLE_SIZE,
                time_decay: default_time_decay_for(DEFAULT_SAMPLE_SIZE),
                seed: None,
                num_threads: None,
                initial_accept_fraction: DEFAULT_INITIAL_ACCEPT_FRACTION,
                feature_scales: None,
            },
            time_decay_explicit: false,
        }
    }

    /// Override the number of trees.
    #[must_use]
    pub fn num_trees(mut self, n: usize) -> Self {
        self.config.num_trees = n;
        self
    }

    /// Override the per-tree reservoir size. When `time_decay` has
    /// not been explicitly set via [`Self::time_decay`], the default
    /// `0.1 / sample_size` is re-resolved against the new value so
    /// the effective recency bias stays consistent with AWS's
    /// `CompactSampler` formula.
    #[must_use]
    pub fn sample_size(mut self, s: usize) -> Self {
        self.config.sample_size = s;
        if !self.time_decay_explicit {
            self.config.time_decay = default_time_decay_for(s);
        }
        self
    }

    /// Override the sampler time-decay factor. Pass `0.0` to disable
    /// recency bias and recover strict uniform reservoir sampling.
    /// Once called, the builder stops auto-resolving `time_decay`
    /// from subsequent [`Self::sample_size`] changes — the caller's
    /// choice wins.
    #[must_use]
    pub fn time_decay(mut self, d: f64) -> Self {
        self.config.time_decay = d;
        self.time_decay_explicit = true;
        self
    }

    /// Pin the RNG seed for reproducible runs.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Build a dedicated rayon thread pool of size `n` for this
    /// forest's parallel score / attribution / update paths.
    /// Requires the `parallel` cargo feature. When unset (default)
    /// rayon's global pool is used.
    #[must_use]
    pub fn num_threads(mut self, n: usize) -> Self {
        self.config.num_threads = Some(n);
        self
    }

    /// Override the warmup admission fraction forwarded to each
    /// per-tree reservoir. See
    /// [`crate::ReservoirSampler`] module-level docs for semantics.
    /// `1.0` (the default) disables the gate; AWS's `CompactSampler`
    /// uses `0.125`.
    #[must_use]
    pub fn initial_accept_fraction(mut self, f: f64) -> Self {
        self.config.initial_accept_fraction = f;
        self
    }

    /// Set per-dimension multiplicative weights applied to every
    /// point before it reaches the forest's hot paths. See
    /// [`RcfConfig::feature_scales`] for semantics. Pass the exact
    /// `[f64; D]` array — length is checked against the builder's
    /// compile-time `D` at [`Self::build`] time, contents are
    /// checked for finiteness and positivity by
    /// [`RcfConfig::validate`].
    #[must_use]
    pub fn feature_scales(mut self, scales: [f64; D]) -> Self {
        self.config.feature_scales = Some(scales.to_vec());
        self
    }

    /// Drop any previously-set `feature_scales` — returns the builder
    /// to the unweighted state. Useful for tests that want to clear a
    /// shared template's scale vector before building a specialised
    /// forest.
    #[must_use]
    pub fn clear_feature_scales(mut self) -> Self {
        self.config.feature_scales = None;
        self
    }

    /// Read-only access to the config under construction.
    #[must_use]
    pub fn config(&self) -> &RcfConfig {
        &self.config
    }

    /// Per-point dimensionality (compile-time `D`).
    #[must_use]
    pub const fn dimension(&self) -> usize {
        D
    }

    /// Validate the config and instantiate the forest.
    ///
    /// # Errors
    ///
    /// Forwards [`RcfConfig::validate`] errors and propagates any
    /// failure from the underlying [`RandomCutForest`] constructor.
    pub fn build(self) -> RcfResult<RandomCutForest<D>> {
        RcfConfig::validate_dimension(D)?;
        self.config.validate()?;
        self.config.validate_feature_scales_dimension(D)?;
        RandomCutForest::<D>::from_config(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n: usize, s: usize, td: f64) -> RcfConfig {
        RcfConfig {
            num_trees: n,
            sample_size: s,
            time_decay: td,
            seed: None,
            num_threads: None,
            initial_accept_fraction: DEFAULT_INITIAL_ACCEPT_FRACTION,
            feature_scales: None,
        }
    }

    #[test]
    fn validate_default_passes() {
        let c = cfg(DEFAULT_NUM_TREES, DEFAULT_SAMPLE_SIZE, DEFAULT_TIME_DECAY);
        c.validate().unwrap();
    }

    #[test]
    fn validate_dimension_rejects_zero() {
        assert!(matches!(
            RcfConfig::validate_dimension(0).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn validate_dimension_rejects_above_max() {
        assert!(RcfConfig::validate_dimension(10_001).is_err());
    }

    #[test]
    fn validate_dimension_accepts_at_max() {
        RcfConfig::validate_dimension(10_000).unwrap();
    }

    #[test]
    fn validate_rejects_num_trees_below_min() {
        assert!(cfg(49, 256, 0.0).validate().is_err());
    }

    #[test]
    fn validate_accepts_num_trees_at_bounds() {
        cfg(50, 256, 0.0).validate().unwrap();
        cfg(1000, 256, 0.0).validate().unwrap();
    }

    #[test]
    fn validate_rejects_num_trees_above_max() {
        assert!(cfg(1001, 256, 0.0).validate().is_err());
    }

    #[test]
    fn validate_rejects_sample_size_zero() {
        assert!(cfg(100, 0, 0.0).validate().is_err());
    }

    #[test]
    fn validate_accepts_sample_size_at_bounds() {
        cfg(100, 1, 0.0).validate().unwrap();
        cfg(100, 2048, 0.0).validate().unwrap();
    }

    #[test]
    fn validate_rejects_sample_size_above_max() {
        assert!(cfg(100, 2049, 0.0).validate().is_err());
    }

    #[test]
    fn validate_rejects_negative_time_decay() {
        assert!(cfg(100, 256, -0.01).validate().is_err());
    }

    #[test]
    fn validate_rejects_time_decay_above_one() {
        assert!(cfg(100, 256, 1.01).validate().is_err());
    }

    #[test]
    fn validate_rejects_non_finite_time_decay() {
        assert!(cfg(100, 256, f64::NAN).validate().is_err());
        assert!(cfg(100, 256, f64::INFINITY).validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_num_threads() {
        let mut c = cfg(100, 256, 0.0);
        c.num_threads = Some(0);
        assert!(matches!(
            c.validate().unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn validate_accepts_some_num_threads() {
        let mut c = cfg(100, 256, 0.0);
        c.num_threads = Some(4);
        c.validate().unwrap();
    }

    #[test]
    fn validate_accepts_default_num_threads_none() {
        let c = cfg(100, 256, 0.0);
        assert_eq!(c.num_threads, None);
        c.validate().unwrap();
    }

    #[test]
    fn builder_num_threads_sets_field() {
        let b = ForestBuilder::<4>::new().num_threads(8);
        assert_eq!(b.config().num_threads, Some(8));
    }

    #[test]
    fn validate_accepts_initial_accept_fraction_at_bounds() {
        let mut c = cfg(100, 256, 0.0);
        c.initial_accept_fraction = 0.001;
        c.validate().unwrap();
        c.initial_accept_fraction = 1.0;
        c.validate().unwrap();
    }

    #[test]
    fn validate_rejects_initial_accept_fraction_out_of_range() {
        let mut c = cfg(100, 256, 0.0);
        c.initial_accept_fraction = 0.0;
        assert!(c.validate().is_err());
        c.initial_accept_fraction = -0.1;
        assert!(c.validate().is_err());
        c.initial_accept_fraction = 1.01;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_finite_initial_accept_fraction() {
        let mut c = cfg(100, 256, 0.0);
        c.initial_accept_fraction = f64::NAN;
        assert!(c.validate().is_err());
        c.initial_accept_fraction = f64::INFINITY;
        assert!(c.validate().is_err());
    }

    #[test]
    fn builder_initial_accept_fraction_sets_field() {
        let b = ForestBuilder::<4>::new().initial_accept_fraction(0.125);
        assert!((b.config().initial_accept_fraction - 0.125).abs() < f64::EPSILON);
    }

    #[test]
    fn builder_defaults_initial_accept_fraction_to_one() {
        let b = ForestBuilder::<4>::new();
        assert!((b.config().initial_accept_fraction - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn builder_defaults_match_aws() {
        let b = ForestBuilder::<8>::new();
        assert_eq!(b.dimension(), 8);
        assert_eq!(b.config().num_trees, 100);
        assert_eq!(b.config().sample_size, 256);
        assert!(
            (b.config().time_decay - TIME_DECAY_NUMERATOR / 256.0).abs() < f64::EPSILON,
            "default time_decay should resolve to 0.1 / sample_size, got {}",
            b.config().time_decay
        );
        assert_eq!(b.config().seed, None);
    }

    #[test]
    fn builder_sample_size_override_rescales_default_time_decay() {
        let b = ForestBuilder::<4>::new().sample_size(128);
        // No explicit time_decay — should auto-resolve to 0.1/128.
        assert!(
            (b.config().time_decay - TIME_DECAY_NUMERATOR / 128.0).abs() < f64::EPSILON,
            "sample_size(128) should rescale default to 0.1 / 128, got {}",
            b.config().time_decay,
        );
    }

    #[test]
    fn builder_explicit_time_decay_sticks_across_sample_size_override() {
        let b = ForestBuilder::<4>::new().time_decay(0.05).sample_size(128);
        // Explicit override must not be clobbered by later sample_size.
        assert!((b.config().time_decay - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn builder_sample_size_override_before_time_decay() {
        let b = ForestBuilder::<4>::new().sample_size(128).time_decay(0.05);
        // Explicit override applied after sample_size wins too.
        assert!((b.config().time_decay - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn builder_time_decay_zero_still_accepted() {
        let b = ForestBuilder::<4>::new().time_decay(0.0);
        assert!(b.config().time_decay.abs() < f64::EPSILON);
        b.build().expect("time_decay=0 must still build");
    }

    #[test]
    fn default_time_decay_for_zero_sample_size_is_zero() {
        assert!(default_time_decay_for(0).abs() < f64::EPSILON);
    }

    #[test]
    fn default_time_decay_for_default_sample_size_matches_constant() {
        assert!(
            (default_time_decay_for(DEFAULT_SAMPLE_SIZE) - DEFAULT_TIME_DECAY).abs() < f64::EPSILON,
        );
    }

    #[test]
    fn builder_overrides_apply() {
        let b = ForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(64)
            .time_decay(0.05)
            .seed(42);
        assert_eq!(b.config().num_trees, 50);
        assert_eq!(b.config().sample_size, 64);
        assert!((b.config().time_decay - 0.05).abs() < f64::EPSILON);
        assert_eq!(b.config().seed, Some(42));
    }

    #[test]
    fn builder_build_validates() {
        let err = ForestBuilder::<4>::new().num_trees(10).build().unwrap_err();
        assert!(matches!(err, RcfError::InvalidConfig(_)));
    }
}
