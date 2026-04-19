//! Output of a [`crate::ThresholdedForest::process`] call.
//!
//! [`AnomalyGrade`] bundles the raw RCF score, the adaptive
//! threshold that was in effect at observation time, and a bounded
//! `[0, 1]` grade that scales linearly in "sigmas above threshold":
//!
//! ```text
//! grade = clamp01( (score − threshold) / (z_factor · stddev) )
//! ```
//!
//! A grade of `0.0` means the score did not exceed the adaptive
//! threshold. A grade of `1.0` means the score sat a full `z_factor`
//! standard deviations *above* the threshold — already `2·z_factor`
//! above the running mean. Callers can route on the boolean
//! [`AnomalyGrade::is_anomaly`] for simple alerting, or read the
//! continuous grade when a severity channel is required.

use crate::domain::AnomalyScore;
use crate::error::{RcfError, RcfResult};

/// Graded anomaly verdict emitted by a thresholded forest.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnomalyGrade {
    /// Raw RCF anomaly score.
    score: AnomalyScore,
    /// Adaptive threshold (`max(min_threshold, mean + z_factor · stddev)`)
    /// in effect when the score was produced.
    threshold: f64,
    /// Severity in `[0, 1]` — linearly scaled between `threshold`
    /// (grade `0`) and `threshold + z_factor · stddev` (grade `1`).
    grade: f64,
    /// Whether the score exceeded the adaptive threshold at observation
    /// time.
    is_anomaly: bool,
    /// Whether the internal running statistics had enough samples to
    /// produce a meaningful threshold (`observations >= min_observations`
    /// and `stddev > 0`). When `false`, [`Self::is_anomaly`] is always
    /// `false` and [`Self::grade`] is `0.0` — downstream consumers
    /// should treat the verdict as "unknown, warming up".
    ready: bool,
}

impl AnomalyGrade {
    /// Build a grade from components, validating the invariants the
    /// [`AnomalyGrade::grade`] contract promises.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::NaNValue`] when `threshold` is non-finite or
    /// when `grade` is non-finite or outside `[0, 1]`.
    pub fn new(
        score: AnomalyScore,
        threshold: f64,
        grade: f64,
        is_anomaly: bool,
        ready: bool,
    ) -> RcfResult<Self> {
        if !threshold.is_finite() {
            return Err(RcfError::NaNValue);
        }
        if !grade.is_finite() || !(0.0..=1.0).contains(&grade) {
            return Err(RcfError::NaNValue);
        }
        Ok(Self {
            score,
            threshold,
            grade,
            is_anomaly,
            ready,
        })
    }

    /// Raw RCF anomaly score.
    #[must_use]
    pub fn score(&self) -> AnomalyScore {
        self.score
    }

    /// Adaptive threshold in effect when the score was emitted.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Severity in `[0, 1]`.
    #[must_use]
    pub fn grade(&self) -> f64 {
        self.grade
    }

    /// Whether this observation crossed the adaptive threshold.
    #[must_use]
    pub fn is_anomaly(&self) -> bool {
        self.is_anomaly
    }

    /// Whether the detector's running statistics had enough samples
    /// to produce a meaningful verdict. During warmup this is
    /// `false` and callers should ignore [`Self::is_anomaly`].
    #[must_use]
    pub fn ready(&self) -> bool {
        self.ready
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert exact equality on closed-form values.
mod tests {
    use super::*;

    fn score(v: f64) -> AnomalyScore {
        AnomalyScore::new(v).unwrap()
    }

    #[test]
    fn new_rejects_non_finite_threshold() {
        assert!(AnomalyGrade::new(score(1.0), f64::NAN, 0.5, true, true).is_err());
        assert!(AnomalyGrade::new(score(1.0), f64::INFINITY, 0.5, true, true).is_err());
    }

    #[test]
    fn new_rejects_grade_outside_unit_interval() {
        assert!(AnomalyGrade::new(score(1.0), 1.0, -0.01, false, true).is_err());
        assert!(AnomalyGrade::new(score(1.0), 1.0, 1.01, true, true).is_err());
    }

    #[test]
    fn new_rejects_non_finite_grade() {
        assert!(AnomalyGrade::new(score(1.0), 1.0, f64::NAN, false, true).is_err());
    }

    #[test]
    fn new_accepts_grade_at_bounds() {
        assert!(AnomalyGrade::new(score(1.0), 1.0, 0.0, false, true).is_ok());
        assert!(AnomalyGrade::new(score(1.0), 1.0, 1.0, true, true).is_ok());
    }

    #[test]
    fn accessors_expose_fields() {
        let g = AnomalyGrade::new(score(2.5), 1.25, 0.5, true, true).unwrap();
        assert_eq!(f64::from(g.score()), 2.5);
        assert_eq!(g.threshold(), 1.25);
        assert_eq!(g.grade(), 0.5);
        assert!(g.is_anomaly());
        assert!(g.ready());
    }

    #[test]
    fn warming_up_grade_is_not_anomaly() {
        let g = AnomalyGrade::new(score(10.0), 0.0, 0.0, false, false).unwrap();
        assert!(!g.ready());
        assert!(!g.is_anomaly());
        assert_eq!(g.grade(), 0.0);
    }
}
