//! CUSUM change-point detector over the anomaly-score stream.
//!
//! The per-point detectors in this crate ([`crate::RandomCutForest`]
//! / [`crate::ThresholdedForest`]) answer *"is this point
//! anomalous?"*. [`MetaDriftDetector`] answers the orthogonal
//! second-order question *"is the distribution of anomaly scores
//! itself shifting?"* by running a two-sided CUSUM (Page 1954) over
//! the score stream.
//!
//! The distinction matters operationally:
//!
//! - a **single high score** is an individual anomaly — a potential
//!   incident to triage;
//! - a **sustained upward trend** in scores is baseline drift — the
//!   threshold needs to adapt, not an incident to page;
//! - a **sustained downward trend** is a quiescence event or a
//!   detector becoming less sensitive over time.
//!
//! The existing [`crate::ThresholdedForest`] already tracks the
//! score stream's EMA mean and variance — but CUSUM is strictly more
//! sensitive than a `μ + zσ` gate for *small* persistent shifts,
//! which is the signature of drift. CUSUM fires on many consecutive
//! mild-deviation scores where a sigma-band test misses entirely.
//!
//! # Algorithm
//!
//! Two-sided tabular CUSUM, parameterised by:
//!
//! - `allowance_k` (slack in σ units, default `0.5`): how much
//!   expected noise the detector absorbs before a shift starts
//!   accumulating.
//! - `threshold_h` (detection bound in σ units, default `5.0`): how
//!   far the accumulator can drift from zero before firing. Higher
//!   `h` = slower to fire, fewer false positives.
//!
//! ```text
//! S⁺[t] = max(0, S⁺[t−1] + (x[t] − μ) − k·σ)   // upward-shift accumulator
//! S⁻[t] = max(0, S⁻[t−1] + (μ − x[t]) − k·σ)   // downward-shift accumulator
//! fire  = S⁺ > h·σ (upward) | S⁻ > h·σ (downward)
//! ```
//!
//! `μ` and `σ` are maintained by the built-in [`EmaStats`] the
//! detector owns — the CUSUM update uses the *previous* values so a
//! fresh observation cannot influence the reference it is compared
//! against.
//!
//! # Reset protocol
//!
//! Callers that react to a drift event should invoke
//! [`MetaDriftDetector::reset`] afterwards so the CUSUM accumulators
//! start over from zero. The EMA stats are kept — they *are* the new
//! reference the post-drift distribution will be measured against.
//! Use [`MetaDriftDetector::reset_stats`] to discard both the
//! accumulators and the EMA, e.g. after a config change or a major
//! regime switch.

use alloc::format;

use crate::error::{RcfError, RcfResult};
use crate::thresholded::EmaStats;

/// Default slack (`k`) in σ units. The CUSUM absorbs noise up to
/// `k · stddev` before a shift starts accumulating.
pub const DEFAULT_ALLOWANCE_K: f64 = 0.5;
/// Default detection bound (`h`) in σ units. The CUSUM accumulator
/// must exceed `h · stddev` before firing.
pub const DEFAULT_THRESHOLD_H: f64 = 5.0;
/// Default minimum observations before emitting a non-warmup verdict.
pub const DEFAULT_MIN_OBSERVATIONS: u64 = 32;
/// Default EMA smoothing factor on the score stream.
pub const DEFAULT_DECAY: f64 = 0.01;

/// Direction of a detected drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DriftKind {
    /// The score stream shifted upward — sustained higher anomaly
    /// scores than the running mean. Typical in baseline drift where
    /// traffic gets slightly "weirder" over time.
    Upward,
    /// The score stream shifted downward — sustained lower anomaly
    /// scores. Typical when the detector has adapted to what was
    /// previously anomalous, or when traffic quiesces.
    Downward,
}

/// Validated configuration of the CUSUM layer.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CusumConfig {
    /// Slack in σ units before a shift starts accumulating.
    pub allowance_k: f64,
    /// Detection bound in σ units.
    pub threshold_h: f64,
    /// Samples required before the detector emits a non-warmup
    /// verdict.
    pub min_observations: u64,
    /// EMA smoothing factor on the score stream. Must be in `(0, 1]`.
    pub decay: f64,
}

impl Default for CusumConfig {
    fn default() -> Self {
        Self {
            allowance_k: DEFAULT_ALLOWANCE_K,
            threshold_h: DEFAULT_THRESHOLD_H,
            min_observations: DEFAULT_MIN_OBSERVATIONS,
            decay: DEFAULT_DECAY,
        }
    }
}

impl CusumConfig {
    /// Validate every parameter.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when any field is outside
    /// its accepted range: `allowance_k` and `threshold_h` must be
    /// finite and non-negative (`allowance_k == 0` is legal — no
    /// slack), `decay` must be finite and in `(0, 1]`.
    pub fn validate(&self) -> RcfResult<()> {
        if !self.allowance_k.is_finite() || self.allowance_k < 0.0 {
            return Err(RcfError::InvalidConfig(format!(
                "allowance_k must be finite and >= 0, got {}",
                self.allowance_k
            )));
        }
        if !self.threshold_h.is_finite() || self.threshold_h <= 0.0 {
            return Err(RcfError::InvalidConfig(format!(
                "threshold_h must be finite and > 0, got {}",
                self.threshold_h
            )));
        }
        if !self.decay.is_finite() || self.decay <= 0.0 || self.decay > 1.0 {
            return Err(RcfError::InvalidConfig(format!(
                "decay must be in (0.0, 1.0], got {}",
                self.decay
            )));
        }
        Ok(())
    }
}

/// Verdict emitted by a single [`MetaDriftDetector::observe`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DriftVerdict {
    /// Upward CUSUM accumulator `S⁺` after the observation.
    pub s_high: f64,
    /// Downward CUSUM accumulator `S⁻` after the observation.
    pub s_low: f64,
    /// Absolute detection threshold in effect (`threshold_h · stddev`).
    pub threshold: f64,
    /// EMA mean of the score stream before the observation was folded
    /// in — the reference the CUSUM compared against.
    pub mean: f64,
    /// EMA stddev of the score stream before the observation.
    pub stddev: f64,
    /// Whether the detector had enough observations to emit a
    /// meaningful verdict (`observations >= min_observations` and
    /// `stddev > 0`). When `false`, [`Self::drift`] is always `None`.
    pub ready: bool,
    /// Direction of the drift, or `None` when neither accumulator
    /// exceeded the threshold.
    pub drift: Option<DriftKind>,
}

/// CUSUM drift detector.
///
/// Feed anomaly scores through [`Self::observe`] one by one; the
/// detector returns a [`DriftVerdict`] every time. Reset the
/// accumulators via [`Self::reset`] once the caller has acted on a
/// drift event; reset the whole state (including the reference EMA)
/// via [`Self::reset_stats`] after a config change.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetaDriftDetector {
    /// Validated CUSUM config.
    config: CusumConfig,
    /// Running EMA of the score stream used for μ / σ.
    stats: EmaStats,
    /// Upward cumulative sum accumulator.
    s_high: f64,
    /// Downward cumulative sum accumulator.
    s_low: f64,
    /// Observability sink — emits the CUSUM accumulators + fire
    /// counter. Defaults to [`crate::NoopSink`].
    #[cfg(feature = "std")]
    #[cfg_attr(
        feature = "serde",
        serde(skip, default = "crate::metrics::default_sink")
    )]
    metrics: std::sync::Arc<dyn crate::metrics::MetricsSink>,
}

impl MetaDriftDetector {
    /// Build a detector with the supplied config.
    ///
    /// # Errors
    ///
    /// Forwards [`CusumConfig::validate`] and [`EmaStats::new`]
    /// errors.
    pub fn new(config: CusumConfig) -> RcfResult<Self> {
        config.validate()?;
        let stats = EmaStats::new(config.decay)?;
        Ok(Self {
            config,
            stats,
            s_high: 0.0,
            s_low: 0.0,
            #[cfg(feature = "std")]
            metrics: crate::metrics::default_sink(),
        })
    }

    /// Install a [`crate::MetricsSink`] — emits
    /// `rcf_drift_s_high` / `rcf_drift_s_low` histograms per call
    /// and increments `rcf_drift_fires_total` on fire verdicts.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_metrics_sink(
        mut self,
        sink: std::sync::Arc<dyn crate::metrics::MetricsSink>,
    ) -> Self {
        self.metrics = sink;
        self
    }

    /// Read-only handle to the installed sink.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn metrics_sink(&self) -> &std::sync::Arc<dyn crate::metrics::MetricsSink> {
        &self.metrics
    }

    /// Default-configured detector.
    ///
    /// # Errors
    ///
    /// Same as [`Self::new`].
    pub fn with_defaults() -> RcfResult<Self> {
        Self::new(CusumConfig::default())
    }

    /// Read-only CUSUM config.
    #[must_use]
    pub fn config(&self) -> &CusumConfig {
        &self.config
    }

    /// Running EMA of the score stream.
    #[must_use]
    pub fn stats(&self) -> &EmaStats {
        &self.stats
    }

    /// Upward cumulative sum accumulator.
    #[must_use]
    pub fn s_high(&self) -> f64 {
        self.s_high
    }

    /// Downward cumulative sum accumulator.
    #[must_use]
    pub fn s_low(&self) -> f64 {
        self.s_low
    }

    /// Fold `score` into the detector and emit a verdict.
    ///
    /// Non-finite inputs are silently ignored and return a verdict
    /// flagged as not-ready without mutating the detector.
    pub fn observe(&mut self, score: f64) -> DriftVerdict {
        if !score.is_finite() {
            return DriftVerdict {
                s_high: self.s_high,
                s_low: self.s_low,
                threshold: 0.0,
                mean: self.stats.mean(),
                stddev: self.stats.stddev(),
                ready: false,
                drift: None,
            };
        }

        let prev_mean = self.stats.mean();
        let prev_stddev = self.stats.stddev();
        let prev_observations = self.stats.observations();
        self.stats.update(score);

        let ready = prev_observations >= self.config.min_observations && prev_stddev > 0.0;
        if !ready {
            return DriftVerdict {
                s_high: self.s_high,
                s_low: self.s_low,
                threshold: 0.0,
                mean: prev_mean,
                stddev: prev_stddev,
                ready: false,
                drift: None,
            };
        }

        let k = self.config.allowance_k * prev_stddev;
        let h = self.config.threshold_h * prev_stddev;
        let dev = score - prev_mean;

        self.s_high = (self.s_high + dev - k).max(0.0);
        self.s_low = (self.s_low - dev - k).max(0.0);

        let drift = if self.s_high > h {
            Some(DriftKind::Upward)
        } else if self.s_low > h {
            Some(DriftKind::Downward)
        } else {
            None
        };

        #[cfg(feature = "std")]
        {
            use crate::metrics::names;
            self.metrics
                .observe_histogram(names::DRIFT_S_HIGH, self.s_high);
            self.metrics
                .observe_histogram(names::DRIFT_S_LOW, self.s_low);
            match drift {
                Some(DriftKind::Upward) => {
                    self.metrics.inc_counter(names::DRIFT_FIRES_TOTAL, 1);
                    self.metrics.inc_counter(names::DRIFT_UP_TOTAL, 1);
                }
                Some(DriftKind::Downward) => {
                    self.metrics.inc_counter(names::DRIFT_FIRES_TOTAL, 1);
                    self.metrics.inc_counter(names::DRIFT_DOWN_TOTAL, 1);
                }
                None => {}
            }
        }

        DriftVerdict {
            s_high: self.s_high,
            s_low: self.s_low,
            threshold: h,
            mean: prev_mean,
            stddev: prev_stddev,
            ready: true,
            drift,
        }
    }

    /// Clear the CUSUM accumulators while keeping the EMA reference
    /// intact. Call this after reacting to a drift event — the new
    /// distribution becomes the next reference.
    pub fn reset(&mut self) {
        self.s_high = 0.0;
        self.s_low = 0.0;
    }

    /// Clear everything — accumulators and EMA reference. Returns
    /// the detector to its warmup state.
    pub fn reset_stats(&mut self) {
        self.s_high = 0.0;
        self.s_low = 0.0;
        self.stats.reset();
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert closed-form CUSUM behaviour.
mod tests {
    use super::*;

    fn detector(h: f64) -> MetaDriftDetector {
        MetaDriftDetector::new(CusumConfig {
            allowance_k: 0.5,
            threshold_h: h,
            min_observations: 8,
            decay: 0.1,
        })
        .unwrap()
    }

    #[test]
    fn default_config_validates() {
        CusumConfig::default().validate().unwrap();
    }

    fn cfg(k: f64, h: f64, min_obs: u64, decay: f64) -> CusumConfig {
        CusumConfig {
            allowance_k: k,
            threshold_h: h,
            min_observations: min_obs,
            decay,
        }
    }

    #[test]
    fn validate_rejects_negative_allowance_k() {
        assert!(
            cfg(-0.1, DEFAULT_THRESHOLD_H, 8, DEFAULT_DECAY)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_rejects_zero_threshold_h() {
        assert!(
            cfg(DEFAULT_ALLOWANCE_K, 0.0, 8, DEFAULT_DECAY)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_rejects_decay_outside_range() {
        assert!(
            cfg(DEFAULT_ALLOWANCE_K, DEFAULT_THRESHOLD_H, 8, 0.0)
                .validate()
                .is_err()
        );
        assert!(
            cfg(DEFAULT_ALLOWANCE_K, DEFAULT_THRESHOLD_H, 8, 1.5)
                .validate()
                .is_err()
        );
        assert!(
            cfg(DEFAULT_ALLOWANCE_K, DEFAULT_THRESHOLD_H, 8, f64::NAN)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn warmup_never_fires() {
        let mut d = detector(5.0);
        for _ in 0..8 {
            let v = d.observe(1.0);
            assert!(!v.ready);
            assert!(v.drift.is_none());
        }
    }

    #[test]
    fn constant_stream_does_not_fire() {
        // A perfectly constant stream has stddev → 0; the detector
        // should never leave the warming-up state.
        let mut d = detector(5.0);
        for _ in 0..200 {
            let v = d.observe(1.0);
            assert!(v.drift.is_none());
        }
        // Accumulators stay at 0.
        assert_eq!(d.s_high(), 0.0);
        assert_eq!(d.s_low(), 0.0);
    }

    #[test]
    fn upward_shift_fires_upward() {
        let mut d = detector(3.0);
        // Warmup: noisy baseline.
        for i in 0..64 {
            let noise = if i % 2 == 0 { 0.95 } else { 1.05 };
            d.observe(noise);
        }
        // Sustained shift upward.
        let mut saw_upward = false;
        for _ in 0..100 {
            let v = d.observe(5.0);
            if matches!(v.drift, Some(DriftKind::Upward)) {
                saw_upward = true;
                break;
            }
        }
        assert!(saw_upward, "CUSUM should fire upward on sustained shift");
    }

    #[test]
    fn downward_shift_fires_downward() {
        let mut d = detector(3.0);
        for i in 0..64 {
            let noise = if i % 2 == 0 { 4.95 } else { 5.05 };
            d.observe(noise);
        }
        let mut saw_downward = false;
        for _ in 0..100 {
            let v = d.observe(1.0);
            if matches!(v.drift, Some(DriftKind::Downward)) {
                saw_downward = true;
                break;
            }
        }
        assert!(
            saw_downward,
            "CUSUM should fire downward on sustained shift"
        );
    }

    #[test]
    fn non_finite_input_ignored() {
        let mut d = detector(3.0);
        for _ in 0..16 {
            d.observe(1.0);
        }
        let obs_before = d.stats().observations();
        let v_nan = d.observe(f64::NAN);
        let v_inf = d.observe(f64::INFINITY);
        assert!(v_nan.drift.is_none());
        assert!(v_inf.drift.is_none());
        assert_eq!(d.stats().observations(), obs_before);
    }

    #[test]
    fn reset_clears_accumulators_but_keeps_stats() {
        let mut d = detector(3.0);
        for i in 0..64 {
            let noise = if i % 2 == 0 { 0.95 } else { 1.05 };
            d.observe(noise);
        }
        for _ in 0..50 {
            d.observe(5.0);
        }
        assert!(d.s_high() > 0.0);
        let stats_obs = d.stats().observations();
        d.reset();
        assert_eq!(d.s_high(), 0.0);
        assert_eq!(d.s_low(), 0.0);
        assert_eq!(
            d.stats().observations(),
            stats_obs,
            "reset() must keep the EMA reference"
        );
    }

    #[test]
    fn reset_stats_clears_everything() {
        let mut d = detector(3.0);
        for _ in 0..64 {
            d.observe(1.0);
        }
        d.reset_stats();
        assert_eq!(d.s_high(), 0.0);
        assert_eq!(d.s_low(), 0.0);
        assert_eq!(d.stats().observations(), 0);
    }

    #[test]
    fn verdict_exposes_reference_mean_and_stddev() {
        let mut d = detector(5.0);
        for _ in 0..32 {
            d.observe(2.0);
        }
        let v = d.observe(2.5);
        // After many `2.0` inputs the EMA should be near 2.0; the
        // verdict's mean/stddev are the *pre-update* values so they
        // reflect the reference the CUSUM compared against.
        assert!((v.mean - 2.0).abs() < 0.5);
        assert!(v.stddev >= 0.0);
    }

    #[test]
    fn with_defaults_builds() {
        let d = MetaDriftDetector::with_defaults().unwrap();
        assert_eq!(d.config().allowance_k, DEFAULT_ALLOWANCE_K);
        assert_eq!(d.config().threshold_h, DEFAULT_THRESHOLD_H);
    }
}
