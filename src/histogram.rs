//! Fixed-bin histogram for score / grade / CUSUM streams.
//!
//! [`ScoreHistogram`] bins a stream of `f64` observations into
//! equal-width buckets on a caller-specified `[min, max)` range.
//! Under- and overflow counters capture values outside the range
//! so the total bin count plus `underflow + overflow` always sums
//! to `total()`.
//!
//! Used for operator-facing dashboards: export the bin counts to
//! Prometheus / Grafana, plot the distribution shape, and diagnose
//! whether the detector's threshold is well-calibrated without
//! re-deriving per-minute aggregates in the monitoring pipeline.
//!
//! The histogram itself is standalone — callers feed it the scores
//! or grades they already collect through
//! [`crate::ThresholdedForest::process`],
//! [`crate::MetaDriftDetector::observe`], etc. Composition over
//! entanglement: no dependency on the forest machinery.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::error::{RcfError, RcfResult};

/// Default number of equal-width bins used when the caller does not
/// override [`HistogramConfig::bin_count`].
pub const DEFAULT_BIN_COUNT: usize = 32;

/// Validated configuration of a [`ScoreHistogram`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HistogramConfig {
    /// Number of equal-width bins covering `[min, max)`.
    pub bin_count: usize,
    /// Inclusive lower bound of the binned range.
    pub min: f64,
    /// Exclusive upper bound of the binned range.
    pub max: f64,
}

impl HistogramConfig {
    /// Build a config with the declared bounds and
    /// [`DEFAULT_BIN_COUNT`] equal-width buckets.
    ///
    /// # Errors
    ///
    /// Same as [`Self::validate`].
    pub fn with_range(min: f64, max: f64) -> RcfResult<Self> {
        let c = Self {
            bin_count: DEFAULT_BIN_COUNT,
            min,
            max,
        };
        c.validate()?;
        Ok(c)
    }

    /// Validate every field.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `bin_count == 0`,
    /// when `min`/`max` are non-finite, or when `min >= max`.
    pub fn validate(&self) -> RcfResult<()> {
        if self.bin_count == 0 {
            return Err(RcfError::InvalidConfig(
                "HistogramConfig::bin_count must be > 0".into(),
            ));
        }
        if !self.min.is_finite() || !self.max.is_finite() {
            return Err(RcfError::InvalidConfig(format!(
                "HistogramConfig bounds must be finite, got min={} max={}",
                self.min, self.max
            )));
        }
        if self.min >= self.max {
            return Err(RcfError::InvalidConfig(format!(
                "HistogramConfig::min ({}) must be strictly less than max ({})",
                self.min, self.max
            )));
        }
        Ok(())
    }

    /// Width of each equal-width bin.
    #[must_use]
    pub fn bin_width(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        {
            (self.max - self.min) / self.bin_count as f64
        }
    }
}

/// Fixed-bin histogram over an `f64` stream.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScoreHistogram {
    /// Validated bin layout.
    config: HistogramConfig,
    /// Per-bin observation counts; `bins[i]` covers
    /// `[min + i · bin_width, min + (i+1) · bin_width)`.
    bins: Vec<u64>,
    /// Observations strictly below `config.min`.
    underflow: u64,
    /// Observations at or above `config.max`.
    overflow: u64,
    /// Non-finite observations (`NaN`, `±∞`) — counted separately so
    /// `total()` can still reason about the sum of bin counts.
    non_finite: u64,
}

impl ScoreHistogram {
    /// Build a fresh histogram with the supplied config.
    ///
    /// # Errors
    ///
    /// Forwards [`HistogramConfig::validate`] errors.
    pub fn new(config: HistogramConfig) -> RcfResult<Self> {
        config.validate()?;
        Ok(Self {
            bins: vec![0; config.bin_count],
            config,
            underflow: 0,
            overflow: 0,
            non_finite: 0,
        })
    }

    /// Short-hand for `new(HistogramConfig::with_range(min, max)?)`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::new`].
    pub fn with_range(min: f64, max: f64) -> RcfResult<Self> {
        Self::new(HistogramConfig::with_range(min, max)?)
    }

    /// Fold `value` into the histogram.
    pub fn record(&mut self, value: f64) {
        if !value.is_finite() {
            self.non_finite = self.non_finite.saturating_add(1);
            return;
        }
        if value < self.config.min {
            self.underflow = self.underflow.saturating_add(1);
            return;
        }
        if value >= self.config.max {
            self.overflow = self.overflow.saturating_add(1);
            return;
        }
        let width = self.config.bin_width();
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let mut idx = ((value - self.config.min) / width) as usize;
        if idx >= self.bins.len() {
            // Defensive: floating drift can nudge a value at the
            // upper edge past the last bin. Cap to the last bin
            // rather than bounce it to overflow.
            idx = self.bins.len() - 1;
        }
        self.bins[idx] = self.bins[idx].saturating_add(1);
    }

    /// Read-only view of the per-bin counts, in ascending-value
    /// order.
    #[must_use]
    pub fn bins(&self) -> &[u64] {
        &self.bins
    }

    /// Inclusive/exclusive edges of every bin — `[(min_0, max_0),
    /// (min_1, max_1), …]`. Useful for Prometheus-style export
    /// where each bucket is named by its upper bound.
    #[must_use]
    pub fn bin_edges(&self) -> Vec<(f64, f64)> {
        let width = self.config.bin_width();
        let mut out = Vec::with_capacity(self.bins.len());
        for i in 0..self.bins.len() {
            #[allow(clippy::cast_precision_loss)]
            let lo = self.config.min + width * i as f64;
            let hi = lo + width;
            out.push((lo, hi));
        }
        out
    }

    /// Configured bin layout.
    #[must_use]
    pub fn config(&self) -> &HistogramConfig {
        &self.config
    }

    /// Observations below [`HistogramConfig::min`].
    #[must_use]
    pub fn underflow(&self) -> u64 {
        self.underflow
    }

    /// Observations at or above [`HistogramConfig::max`].
    #[must_use]
    pub fn overflow(&self) -> u64 {
        self.overflow
    }

    /// Non-finite observations (`NaN`, `±∞`).
    #[must_use]
    pub fn non_finite(&self) -> u64 {
        self.non_finite
    }

    /// Total number of `record` calls — sum of every bin,
    /// underflow, overflow, and non-finite.
    #[must_use]
    pub fn total(&self) -> u64 {
        let sum: u64 = self.bins.iter().copied().sum();
        sum.saturating_add(self.underflow)
            .saturating_add(self.overflow)
            .saturating_add(self.non_finite)
    }

    /// Drop every count, keeping the config.
    pub fn reset(&mut self) {
        for b in &mut self.bins {
            *b = 0;
        }
        self.underflow = 0;
        self.overflow = 0;
        self.non_finite = 0;
    }

    /// Merge `other` into `self`. Both histograms must share the
    /// exact same config.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when the configs differ.
    pub fn merge(&mut self, other: &Self) -> RcfResult<()> {
        if self.config != other.config {
            return Err(RcfError::InvalidConfig(
                "ScoreHistogram::merge requires identical configs".into(),
            ));
        }
        for (a, b) in self.bins.iter_mut().zip(other.bins.iter()) {
            *a = a.saturating_add(*b);
        }
        self.underflow = self.underflow.saturating_add(other.underflow);
        self.overflow = self.overflow.saturating_add(other.overflow);
        self.non_finite = self.non_finite.saturating_add(other.non_finite);
        Ok(())
    }

    /// Linear-interpolated percentile of the recorded values.
    /// Returns `None` when `p` is outside `[0, 1]` or the histogram
    /// has seen no in-range observations. Under-/overflow and
    /// non-finite counts are excluded from the percentile.
    #[must_use]
    pub fn percentile(&self, p: f64) -> Option<f64> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return None;
        }
        let total: u64 = self.bins.iter().copied().sum();
        if total == 0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let target = p * total as f64;
        let width = self.config.bin_width();
        let mut cum: u64 = 0;
        for (i, count) in self.bins.iter().enumerate() {
            let prev = cum;
            cum = cum.saturating_add(*count);
            #[allow(clippy::cast_precision_loss)]
            let prev_f = prev as f64;
            #[allow(clippy::cast_precision_loss)]
            let cum_f = cum as f64;
            if target <= cum_f && *count > 0 {
                let in_bin =
                    (target - prev_f) / f64::from(u32::try_from(*count).unwrap_or(u32::MAX));
                #[allow(clippy::cast_precision_loss)]
                let lo = self.config.min + width * i as f64;
                return Some(lo + width * in_bin.clamp(0.0, 1.0));
            }
        }
        None
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert closed-form histogram behaviour.
mod tests {
    use super::*;

    fn hist() -> ScoreHistogram {
        ScoreHistogram::new(HistogramConfig {
            bin_count: 10,
            min: 0.0,
            max: 10.0,
        })
        .unwrap()
    }

    #[test]
    fn new_rejects_zero_bin_count() {
        assert!(
            ScoreHistogram::new(HistogramConfig {
                bin_count: 0,
                min: 0.0,
                max: 1.0,
            })
            .is_err()
        );
    }

    #[test]
    fn new_rejects_inverted_range() {
        assert!(
            ScoreHistogram::new(HistogramConfig {
                bin_count: 4,
                min: 1.0,
                max: 0.5,
            })
            .is_err()
        );
    }

    #[test]
    fn new_rejects_non_finite_bounds() {
        assert!(
            ScoreHistogram::new(HistogramConfig {
                bin_count: 4,
                min: f64::NAN,
                max: 1.0,
            })
            .is_err()
        );
    }

    #[test]
    fn record_routes_value_to_correct_bin() {
        let mut h = hist();
        h.record(0.5); // bin 0
        h.record(1.5); // bin 1
        h.record(9.9); // bin 9
        assert_eq!(h.bins()[0], 1);
        assert_eq!(h.bins()[1], 1);
        assert_eq!(h.bins()[9], 1);
        assert_eq!(h.total(), 3);
    }

    #[test]
    fn record_under_and_overflow() {
        let mut h = hist();
        h.record(-1.0);
        h.record(10.0); // exclusive max
        h.record(100.0);
        assert_eq!(h.underflow(), 1);
        assert_eq!(h.overflow(), 2);
        assert_eq!(h.total(), 3);
    }

    #[test]
    fn record_non_finite_tallied_separately() {
        let mut h = hist();
        h.record(f64::NAN);
        h.record(f64::INFINITY);
        h.record(f64::NEG_INFINITY);
        assert_eq!(h.non_finite(), 3);
        assert_eq!(h.total(), 3);
        assert!(h.bins().iter().all(|&c| c == 0));
    }

    #[test]
    fn upper_edge_goes_to_last_bin_not_overflow() {
        // `max` is exclusive by contract — a value landing exactly
        // on `max - ε` should fall in the last bin via floating
        // drift fallback.
        let mut h = hist();
        h.record(9.999_999_999);
        assert_eq!(h.bins()[9], 1);
        assert_eq!(h.overflow(), 0);
    }

    #[test]
    fn bin_edges_cover_whole_range() {
        let h = hist();
        let edges = h.bin_edges();
        assert_eq!(edges.len(), 10);
        assert_eq!(edges[0], (0.0, 1.0));
        assert_eq!(edges[9], (9.0, 10.0));
    }

    #[test]
    fn reset_clears_counts_but_keeps_config() {
        let mut h = hist();
        for _ in 0..5 {
            h.record(3.0);
        }
        h.record(-1.0);
        h.reset();
        assert_eq!(h.total(), 0);
        assert_eq!(h.underflow(), 0);
        assert_eq!(h.config().bin_count, 10);
    }

    #[test]
    fn merge_sums_componentwise() {
        let mut a = hist();
        a.record(1.0);
        a.record(5.0);
        let mut b = hist();
        b.record(5.0);
        b.record(20.0);
        a.merge(&b).unwrap();
        assert_eq!(a.bins()[1], 1);
        assert_eq!(a.bins()[5], 2);
        assert_eq!(a.overflow(), 1);
    }

    #[test]
    fn merge_rejects_mismatched_config() {
        let mut a = hist();
        let b = ScoreHistogram::with_range(0.0, 100.0).unwrap();
        assert!(a.merge(&b).is_err());
    }

    #[test]
    fn percentile_handles_empty() {
        let h = hist();
        assert!(h.percentile(0.5).is_none());
    }

    #[test]
    fn percentile_interpolates_within_bin() {
        let mut h = hist();
        for _ in 0..100 {
            h.record(5.0);
        }
        let p50 = h.percentile(0.5).unwrap();
        assert!((5.0..6.0).contains(&p50));
    }

    #[test]
    fn with_range_uses_default_bin_count() {
        let h = ScoreHistogram::with_range(0.0, 1.0).unwrap();
        assert_eq!(h.bins().len(), DEFAULT_BIN_COUNT);
    }
}
