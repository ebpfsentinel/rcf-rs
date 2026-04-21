//! Exponential-moving mean and variance of the anomaly-score stream.
//!
//! [`EmaStats`] tracks the running mean and variance of the per-point
//! scores a [`crate::ThresholdedForest`] observes, using a single decay
//! factor `α ∈ (0, 1]`. The recurrence is
//!
//! ```text
//! delta  = x − mean
//! mean' = mean  + α · delta
//! var'  = (1 − α) · (var + α · delta²)
//! ```
//!
//! which is the standard exponentially-weighted Welford update:
//! `mean'` is the biased EMA and `var'` is the EMA of the squared
//! deviation *about the previous mean* — the correct estimator under
//! exponential weighting (West 1979).
//!
//! The decay factor controls the effective memory window: with
//! `α = 0.01` the statistics "forget" roughly after `1 / α = 100`
//! observations. [`EmaStats`] is also the place where `observations`
//! is counted so callers (threshold derivation, warmup gating) can
//! refuse to emit an anomaly verdict before enough data has been
//! seen.

use alloc::format;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::error::{RcfError, RcfResult};

/// Exponentially-weighted running mean + variance.
///
/// Invariants:
/// - `decay` is finite, in `(0, 1]`.
/// - `observations` is monotonically non-decreasing.
/// - `mean` and `variance` are finite whenever `observations > 0`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmaStats {
    /// EMA of the observed values.
    mean: f64,
    /// EMA of squared deviation about the previous mean.
    variance: f64,
    /// Per-update smoothing factor.
    decay: f64,
    /// Total number of [`EmaStats::update`] calls since construction
    /// or [`EmaStats::reset`].
    observations: u64,
}

impl EmaStats {
    /// Build a fresh tracker with the supplied smoothing factor.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `decay` is non-finite
    /// or falls outside `(0.0, 1.0]`.
    pub fn new(decay: f64) -> RcfResult<Self> {
        if !decay.is_finite() || decay <= 0.0 || decay > 1.0 {
            return Err(RcfError::InvalidConfig(format!(
                "EmaStats decay must be in (0.0, 1.0], got {decay}"
            )));
        }
        Ok(Self {
            mean: 0.0,
            variance: 0.0,
            decay,
            observations: 0,
        })
    }

    /// Fold a new observation into the running statistics.
    ///
    /// Non-finite inputs are silently ignored — callers should reject
    /// or sanitise `NaN`/`±∞` before feeding them in. The observation
    /// counter is still incremented only on accepted inputs so that
    /// [`EmaStats::observations`] reflects the size of the sample the
    /// statistics were actually built from.
    pub fn update(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        let delta = value - self.mean;
        if self.observations == 0 {
            // Bootstrap: first sample is the exact mean, variance stays
            // at zero. Avoids an initial spike that would drag the EMA
            // toward 0 for callers observing scores far from origin.
            self.mean = value;
            self.variance = 0.0;
        } else {
            self.mean += self.decay * delta;
            self.variance = (1.0 - self.decay) * (self.variance + self.decay * delta * delta);
        }
        self.observations = self.observations.saturating_add(1);
    }

    /// Running mean estimate. Zero when no observation has been folded
    /// in yet.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Running variance estimate (always non-negative by construction).
    #[must_use]
    pub fn variance(&self) -> f64 {
        self.variance.max(0.0)
    }

    /// Running standard deviation estimate.
    #[must_use]
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Number of observations folded in since the last [`EmaStats::reset`].
    #[must_use]
    pub fn observations(&self) -> u64 {
        self.observations
    }

    /// Configured smoothing factor.
    #[must_use]
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// Drop every aggregated quantity and restart from the bootstrap
    /// state. Used by the thresholded detector's own `reset` path and
    /// by tests.
    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.variance = 0.0;
        self.observations = 0;
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert exact equality on bootstrap + closed-form expectations.
mod tests {
    use super::*;

    #[test]
    fn new_rejects_non_finite_decay() {
        assert!(EmaStats::new(f64::NAN).is_err());
        assert!(EmaStats::new(f64::INFINITY).is_err());
    }

    #[test]
    fn new_rejects_non_positive_decay() {
        assert!(EmaStats::new(0.0).is_err());
        assert!(EmaStats::new(-0.1).is_err());
    }

    #[test]
    fn new_rejects_decay_above_one() {
        assert!(EmaStats::new(1.001).is_err());
    }

    #[test]
    fn new_accepts_decay_at_one() {
        // decay=1 is legal (full replacement every update).
        EmaStats::new(1.0).unwrap();
    }

    #[test]
    fn first_update_sets_mean_exactly() {
        let mut s = EmaStats::new(0.1).unwrap();
        s.update(7.0);
        assert_eq!(s.mean(), 7.0);
        assert_eq!(s.variance(), 0.0);
        assert_eq!(s.observations(), 1);
    }

    #[test]
    fn non_finite_update_is_ignored() {
        let mut s = EmaStats::new(0.1).unwrap();
        s.update(f64::NAN);
        s.update(f64::INFINITY);
        assert_eq!(s.observations(), 0);
        assert_eq!(s.mean(), 0.0);
    }

    #[test]
    fn mean_tracks_constant_stream_with_zero_variance() {
        let mut s = EmaStats::new(0.1).unwrap();
        for _ in 0..1000 {
            s.update(5.0);
        }
        assert!((s.mean() - 5.0).abs() < 1e-9);
        assert!(s.variance() < 1e-12);
    }

    #[test]
    fn variance_tracks_spread() {
        let mut s = EmaStats::new(0.05).unwrap();
        // Alternating ±1 around 0 → variance should settle near 1.
        for i in 0..5_000 {
            let v = if i % 2 == 0 { 1.0 } else { -1.0 };
            s.update(v);
        }
        // Mean should be near 0, stddev near 1.
        assert!(s.mean().abs() < 0.1);
        assert!(s.stddev() > 0.5);
        assert!(s.stddev() < 1.5);
    }

    #[test]
    fn reset_clears_state() {
        let mut s = EmaStats::new(0.1).unwrap();
        for i in 0..10 {
            s.update(f64::from(i));
        }
        assert!(s.observations() > 0);
        s.reset();
        assert_eq!(s.mean(), 0.0);
        assert_eq!(s.variance(), 0.0);
        assert_eq!(s.observations(), 0);
    }

    #[test]
    fn observations_saturates_at_u64_max() {
        let mut s = EmaStats::new(1.0).unwrap();
        s.observations = u64::MAX;
        s.update(1.0);
        assert_eq!(s.observations(), u64::MAX);
    }

    #[test]
    fn variance_is_never_negative() {
        // A pathological sequence that floats near zero should still
        // keep variance >= 0 when read via the accessor.
        let mut s = EmaStats::new(0.5).unwrap();
        s.update(1e-300);
        s.update(-1e-300);
        assert!(s.variance() >= 0.0);
    }
}
