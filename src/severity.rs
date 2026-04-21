//! Ordinal severity bands derived from a raw anomaly score.
//!
//! A thin convenience layer over [`crate::AnomalyScore`] /
//! [`crate::AnomalyGrade`] that maps a score into one of five
//! ordinal bands (`Normal`, `Low`, `Medium`, `High`, `Critical`).
//! Defaults match the eBPFsentinel Enterprise ml-detection
//! thresholds (`2.0 / 3.0 / 4.0 / 5.0`), so downstream alert
//! routing can share the same vocabulary the agent already uses.
//!
//! Lib stays policy-free — the bands are caller-supplied. The
//! `Default` provides a sensible starting point for RCF scores
//! derived from the crate's Guha-2016-style scoring convention.

use alloc::format;

use crate::domain::AnomalyScore;
use crate::error::{RcfError, RcfResult};
use crate::thresholded::AnomalyGrade;

/// Ordinal severity label.
///
/// Ordered by increasing urgency; `PartialOrd` / `Ord` make it
/// trivial to compare (`sev >= Severity::High`) for alert routing
/// policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Severity {
    /// Score below the `low` threshold — no alert.
    Normal,
    /// Score in `[low, medium)`.
    Low,
    /// Score in `[medium, high)`.
    Medium,
    /// Score in `[high, critical)`.
    High,
    /// Score at or above `critical`.
    Critical,
}

impl Severity {
    /// Caller-facing label, matching SOC dashboard conventions.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Severity::Normal => "normal",
            Severity::Low => "low",
            Severity::Medium => "medium",
            Severity::High => "high",
            Severity::Critical => "critical",
        }
    }
}

impl core::fmt::Display for Severity {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.label())
    }
}

/// Ascending thresholds defining the four transition points
/// between the five severity bands.
///
/// Ordering invariant enforced at construction:
/// `0 ≤ low < medium < high < critical`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SeverityBands {
    /// `Normal → Low` transition.
    pub low: f64,
    /// `Low → Medium` transition.
    pub medium: f64,
    /// `Medium → High` transition.
    pub high: f64,
    /// `High → Critical` transition.
    pub critical: f64,
}

impl Default for SeverityBands {
    fn default() -> Self {
        // eBPFsentinel Enterprise ml-detection defaults.
        Self {
            low: 2.0,
            medium: 3.0,
            high: 4.0,
            critical: 5.0,
        }
    }
}

impl SeverityBands {
    /// Build a validated set of bands.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when any threshold is
    /// non-finite, when `low < 0`, or when the four thresholds are
    /// not strictly ascending.
    pub fn new(low: f64, medium: f64, high: f64, critical: f64) -> RcfResult<Self> {
        let bands = Self {
            low,
            medium,
            high,
            critical,
        };
        bands.validate()?;
        Ok(bands)
    }

    /// Validate the ordering invariant.
    ///
    /// # Errors
    ///
    /// Same as [`Self::new`].
    pub fn validate(&self) -> RcfResult<()> {
        for (name, value) in [
            ("low", self.low),
            ("medium", self.medium),
            ("high", self.high),
            ("critical", self.critical),
        ] {
            if !value.is_finite() {
                return Err(RcfError::InvalidConfig(format!(
                    "SeverityBands::{name} must be finite, got {value}"
                )));
            }
        }
        if self.low < 0.0 {
            return Err(RcfError::InvalidConfig(format!(
                "SeverityBands::low must be >= 0, got {}",
                self.low
            )));
        }
        if !(self.low < self.medium && self.medium < self.high && self.high < self.critical) {
            return Err(RcfError::InvalidConfig(format!(
                "SeverityBands must be strictly ascending: low={} medium={} high={} critical={}",
                self.low, self.medium, self.high, self.critical
            )));
        }
        Ok(())
    }

    /// Classify a raw score into its severity band.
    #[must_use]
    pub fn classify(&self, score: f64) -> Severity {
        if !score.is_finite() || score < self.low {
            return Severity::Normal;
        }
        if score < self.medium {
            return Severity::Low;
        }
        if score < self.high {
            return Severity::Medium;
        }
        if score < self.critical {
            return Severity::High;
        }
        Severity::Critical
    }
}

impl AnomalyScore {
    /// Classify this raw score into a [`Severity`] band using the
    /// supplied thresholds.
    #[must_use]
    pub fn severity(&self, bands: &SeverityBands) -> Severity {
        bands.classify(f64::from(*self))
    }
}

impl AnomalyGrade {
    /// Classify the graded verdict into a [`Severity`] band. Uses
    /// the raw score, **not** the bounded `grade ∈ [0, 1]` —
    /// severity bands live in raw-score space so a caller switching
    /// between `ThresholdedForest` and the bare
    /// [`crate::RandomCutForest`] gets consistent labels.
    #[must_use]
    pub fn severity(&self, bands: &SeverityBands) -> Severity {
        self.score().severity(bands)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert exact band boundaries.
mod tests {
    use super::*;

    #[test]
    fn default_matches_ebpfsentinel_ml_detection() {
        let b = SeverityBands::default();
        assert_eq!(b.low, 2.0);
        assert_eq!(b.medium, 3.0);
        assert_eq!(b.high, 4.0);
        assert_eq!(b.critical, 5.0);
    }

    #[test]
    fn classify_routes_every_band() {
        let b = SeverityBands::default();
        assert_eq!(b.classify(0.0), Severity::Normal);
        assert_eq!(b.classify(1.99), Severity::Normal);
        assert_eq!(b.classify(2.0), Severity::Low);
        assert_eq!(b.classify(2.99), Severity::Low);
        assert_eq!(b.classify(3.0), Severity::Medium);
        assert_eq!(b.classify(3.99), Severity::Medium);
        assert_eq!(b.classify(4.0), Severity::High);
        assert_eq!(b.classify(4.99), Severity::High);
        assert_eq!(b.classify(5.0), Severity::Critical);
        assert_eq!(b.classify(1_000.0), Severity::Critical);
    }

    #[test]
    fn classify_handles_non_finite() {
        let b = SeverityBands::default();
        // Every non-finite input maps to `Normal` — safer default
        // than forcing a Critical on NaN poisoning upstream.
        assert_eq!(b.classify(f64::NAN), Severity::Normal);
        assert_eq!(b.classify(f64::NEG_INFINITY), Severity::Normal);
        assert_eq!(b.classify(f64::INFINITY), Severity::Normal);
    }

    #[test]
    fn new_rejects_non_ascending() {
        assert!(SeverityBands::new(3.0, 2.0, 4.0, 5.0).is_err());
        assert!(SeverityBands::new(2.0, 2.0, 4.0, 5.0).is_err());
        assert!(SeverityBands::new(2.0, 3.0, 4.0, 4.0).is_err());
    }

    #[test]
    fn new_rejects_negative_low() {
        assert!(SeverityBands::new(-0.1, 1.0, 2.0, 3.0).is_err());
    }

    #[test]
    fn new_rejects_non_finite() {
        assert!(SeverityBands::new(f64::NAN, 1.0, 2.0, 3.0).is_err());
        assert!(SeverityBands::new(2.0, 3.0, f64::INFINITY, 5.0).is_err());
    }

    #[test]
    fn severity_ordering_is_monotonic() {
        assert!(Severity::Normal < Severity::Low);
        assert!(Severity::Low < Severity::Medium);
        assert!(Severity::Medium < Severity::High);
        assert!(Severity::High < Severity::Critical);
    }

    #[test]
    fn severity_labels_match_soc_vocab() {
        assert_eq!(Severity::Normal.label(), "normal");
        assert_eq!(Severity::Low.label(), "low");
        assert_eq!(Severity::Medium.label(), "medium");
        assert_eq!(Severity::High.label(), "high");
        assert_eq!(Severity::Critical.label(), "critical");
        assert_eq!(format!("{}", Severity::High), "high");
    }

    #[test]
    fn anomaly_score_severity_routes_correctly() {
        let b = SeverityBands::default();
        let s = AnomalyScore::new(3.5).unwrap();
        assert_eq!(s.severity(&b), Severity::Medium);
    }

    #[test]
    fn anomaly_grade_severity_uses_raw_score() {
        let b = SeverityBands::default();
        let grade =
            AnomalyGrade::new(AnomalyScore::new(6.0).unwrap(), 4.5, 1.0, true, true).unwrap();
        assert_eq!(grade.severity(&b), Severity::Critical);
    }

    #[test]
    fn custom_bands_work() {
        let b = SeverityBands::new(0.5, 1.0, 2.0, 3.0).unwrap();
        assert_eq!(b.classify(0.4), Severity::Normal);
        assert_eq!(b.classify(0.5), Severity::Low);
        assert_eq!(b.classify(1.5), Severity::Medium);
        assert_eq!(b.classify(2.5), Severity::High);
        assert_eq!(b.classify(10.0), Severity::Critical);
    }
}
