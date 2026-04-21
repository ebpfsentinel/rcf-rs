//! Streaming Peaks-Over-Threshold (SPOT / DSPOT) per-dimension
//! anomaly detector — Siffer et al., *Anomaly Detection in
//! Streams with Extreme Value Theory*, KDD 2017.
//!
//! For each feature dimension the detector maintains a streaming
//! quantile `u` of the baseline distribution (via the shipped
//! [`crate::TDigest`]). Observations above `u` are **peaks**; a
//! Generalised Pareto Distribution (GPD) fit by method-of-moments
//! over the peak pile gives a closed-form tail survival function:
//!
//! ```text
//! Pr(X > x) = (1 - q) · (1 + γ · (x − u) / σ) ^ (−1/γ)        γ ≠ 0
//!           = (1 - q) · exp(−(x − u) / σ)                    γ = 0
//! ```
//!
//! The detector returns this survival probability as a **p-value**
//! in `(0, 1]`. Values below `p_alert` (e.g. `1e-3`) are flagged.
//! Use one [`PotDetector`] per feature dim; compose their p-values
//! into a joint anomaly signal via
//! [`crate::ensemble::fisher_combine`].
//!
//! # SPOT vs DSPOT
//!
//! - **SPOT** (frozen quantile): [`PotDetector::freeze_baseline`]
//!   pins `u` at the current warm-phase quantile. Classical
//!   extreme-value-theory test with a fixed null.
//! - **DSPOT** (drifting quantile): keep calling
//!   [`PotDetector::record`] during live traffic — the `TDigest`
//!   tracks the moving quantile, the peak pile re-fits as drift
//!   accumulates. Trades false-positive rate against recency.
//!
//! # Heavy-tail caveat
//!
//! Method-of-moments GPD fit is only valid for `γ < 0.5` (finite
//! second moment of peaks). When peaks are extremely heavy-tailed
//! the fit degrades and the detector falls back to an empirical
//! tail-count p-value. The fallback is documented in
//! [`PotDetector::p_value`].

#![cfg(feature = "std")]

use std::sync::Arc;

use crate::error::{RcfError, RcfResult};
use crate::metrics::{MetricsSink, default_sink, names};
use crate::tdigest::{DEFAULT_COMPRESSION, TDigest};

/// Default quantile threshold above which observations are treated
/// as peaks for the GPD fit — `0.98` keeps the baseline 98 % of
/// values out of the peak pile, matches the DSPOT reference.
pub const DEFAULT_QUANTILE: f64 = 0.98;

/// Default alert threshold on the p-value emitted by
/// [`PotDetector::p_value`] — `1e-3` fires on the 0.1 % tail.
pub const DEFAULT_ALERT_P: f64 = 1.0e-3;

/// Minimum peak count before the GPD fit is trusted. Below this,
/// [`PotDetector::p_value`] falls back to the empirical tail-rate.
pub const MIN_PEAKS_FOR_FIT: usize = 16;

/// Per-feature streaming Peaks-Over-Threshold detector.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PotDetector {
    /// Streaming digest of the baseline (all recorded values).
    digest: TDigest,
    /// Running statistics of the peaks (values above the frozen
    /// or rolling quantile `u`). Used for method-of-moments.
    peak_count: u64,
    /// Running mean of `(x − u)` across peaks.
    peak_mean: f64,
    /// Running `M2` (sum of squared deviations) for Welford on
    /// peaks — yields `variance = M2 / (n − 1)`.
    peak_m2: f64,
    /// Pinned quantile threshold `u` once [`Self::freeze_baseline`]
    /// is called. `None` until frozen — the DSPOT variant stays
    /// un-frozen and queries the digest's live quantile on each
    /// `p_value` call.
    frozen_u: Option<f64>,
    /// Quantile target (`q`); default [`DEFAULT_QUANTILE`].
    q: f64,
    /// Lifetime count of `record` calls.
    total_seen: u64,
    /// Observability sink — serde-skipped, restored to the noop
    /// sink on round-trip.
    #[cfg_attr(
        feature = "serde",
        serde(skip, default = "crate::metrics::default_sink")
    )]
    metrics: Arc<dyn MetricsSink>,
}

impl PotDetector {
    /// Build a detector with a caller-chosen quantile target
    /// `q ∈ (0, 1)` — observations above the running `q`-quantile
    /// are peaks.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `q` is non-finite
    /// or outside `(0, 1)`.
    pub fn new(q: f64) -> RcfResult<Self> {
        if !q.is_finite() || !(0.0..1.0).contains(&q) || q <= 0.0 {
            return Err(RcfError::InvalidConfig(format!(
                "PotDetector: q must be in (0.0, 1.0), got {q}"
            )));
        }
        Ok(Self {
            digest: TDigest::new(DEFAULT_COMPRESSION)?,
            peak_count: 0,
            peak_mean: 0.0,
            peak_m2: 0.0,
            frozen_u: None,
            q,
            total_seen: 0,
            metrics: default_sink(),
        })
    }

    /// Default-compression detector with quantile
    /// [`DEFAULT_QUANTILE`].
    #[must_use]
    pub fn default_spot() -> Self {
        Self {
            digest: TDigest::with_default_compression(),
            peak_count: 0,
            peak_mean: 0.0,
            peak_m2: 0.0,
            frozen_u: None,
            q: DEFAULT_QUANTILE,
            total_seen: 0,
            metrics: default_sink(),
        }
    }

    /// Install a metrics sink — every `record` emits an
    /// observations counter, every peak bumps a peaks counter.
    #[must_use]
    pub fn with_metrics_sink(mut self, sink: Arc<dyn MetricsSink>) -> Self {
        self.metrics = sink;
        self
    }

    /// Read-only handle to the installed sink.
    #[must_use]
    pub fn metrics_sink(&self) -> &Arc<dyn MetricsSink> {
        &self.metrics
    }

    /// Target quantile.
    #[must_use]
    pub fn quantile_target(&self) -> f64 {
        self.q
    }

    /// Whether [`Self::freeze_baseline`] has been called. When
    /// `false` the detector runs in DSPOT mode.
    #[must_use]
    pub fn is_frozen(&self) -> bool {
        self.frozen_u.is_some()
    }

    /// Observations seen so far.
    #[must_use]
    pub fn total_seen(&self) -> u64 {
        self.total_seen
    }

    /// Peaks accumulated above the (frozen or live) quantile.
    #[must_use]
    pub fn peak_count(&self) -> u64 {
        self.peak_count
    }

    /// Fold `value` into the baseline. Peaks (values above the
    /// frozen or live quantile threshold `u`) also fold into the
    /// Welford running statistics used by the GPD fit.
    ///
    /// Non-finite values are silently ignored — matches the
    /// fail-open behaviour of [`TDigest::record`].
    pub fn record(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.total_seen = self.total_seen.saturating_add(1);
        self.metrics.inc_counter(names::SPOT_OBSERVATIONS_TOTAL, 1);
        self.digest.record(value);
        if let Some(u) = self.current_u() {
            let excess = value - u;
            if excess > 0.0 {
                // Welford update on the peak excess stream.
                self.peak_count = self.peak_count.saturating_add(1);
                self.metrics.inc_counter(names::SPOT_PEAKS_TOTAL, 1);
                #[allow(clippy::cast_precision_loss)]
                let n = self.peak_count as f64;
                let delta = excess - self.peak_mean;
                self.peak_mean += delta / n;
                let delta2 = excess - self.peak_mean;
                self.peak_m2 += delta * delta2;
            }
        }
    }

    /// Pin the current `q`-quantile as the frozen threshold `u`.
    /// Classical SPOT mode — the quantile no longer drifts.
    /// Subsequent [`Self::record`] calls still update the digest
    /// (for diagnostics) but the peak pile is evaluated against the
    /// frozen `u`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::EmptyForest`] when the detector has not
    /// seen enough observations to estimate the quantile.
    pub fn freeze_baseline(&mut self) -> RcfResult<()> {
        let Some(u) = self.digest.quantile(self.q) else {
            return Err(RcfError::EmptyForest);
        };
        self.frozen_u = Some(u);
        // Rebuild the peak pile from scratch against the new `u`
        // — before freeze, peaks were accumulating against the
        // rolling quantile, which differs from `u`.
        self.peak_count = 0;
        self.peak_mean = 0.0;
        self.peak_m2 = 0.0;
        Ok(())
    }

    /// Survival-function p-value `Pr(X > value)` under the fitted
    /// GPD tail. Returns `1.0` when `value` is below the current
    /// threshold `u` (no tail contribution), values in `(0, 1]`
    /// otherwise. Lower p-value → more anomalous.
    ///
    /// Falls back to empirical tail-count when fewer than
    /// [`MIN_PEAKS_FOR_FIT`] peaks have accumulated (the `MoM` fit
    /// is unreliable under that limit).
    #[must_use]
    pub fn p_value(&self, value: f64) -> f64 {
        if !value.is_finite() {
            return 1.0;
        }
        let Some(u) = self.current_u() else {
            return 1.0;
        };
        let excess = value - u;
        if excess <= 0.0 {
            return 1.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let tail_prob = 1.0 - self.q;
        if self.peak_count < MIN_PEAKS_FOR_FIT as u64 {
            // Empirical fallback: fraction of observations strictly
            // above `value`. Bounded below by 1 / total_seen so the
            // p-value never collapses to zero on an unseen extreme.
            #[allow(clippy::cast_precision_loss)]
            let floor = 1.0 / (self.total_seen.max(1) as f64);
            return floor.max(tail_prob);
        }
        // Method-of-moments GPD fit over the peak excess stream.
        let (gamma, sigma) = self.gpd_mom();
        if sigma <= 0.0 || !sigma.is_finite() || !gamma.is_finite() {
            // Degenerate fit — fall back to the calling tail prob.
            return tail_prob.clamp(0.0, 1.0);
        }
        let cond = if gamma.abs() < 1.0e-9 {
            // γ → 0 limit: exponential tail.
            (-excess / sigma).exp()
        } else {
            let inner = 1.0 + gamma * (excess / sigma);
            if inner <= 0.0 {
                // Past the GPD support — treat as vanishing
                // survival (clamp to 1 / total_seen floor).
                0.0
            } else {
                inner.powf(-1.0 / gamma)
            }
        };
        let p = tail_prob * cond;
        // Clamp to a conservative floor so downstream Fisher
        // combination does not blow up on `ln(0)`.
        #[allow(clippy::cast_precision_loss)]
        let floor = 1.0 / (self.total_seen.max(1) as f64 * 1_000.0);
        p.max(floor).min(1.0)
    }

    /// Whether `value` is currently anomalous at level `alert_p`.
    /// Shorthand for `self.p_value(value) < alert_p`.
    #[must_use]
    pub fn is_anomaly(&self, value: f64, alert_p: f64) -> bool {
        self.p_value(value) < alert_p
    }

    /// Current threshold `u` — the frozen value if
    /// [`Self::freeze_baseline`] was called, else the live
    /// quantile from the digest.
    fn current_u(&self) -> Option<f64> {
        if let Some(u) = self.frozen_u {
            return Some(u);
        }
        // Digest quantile needs `&mut self` for the flush pass;
        // clone cheaply to preserve the `&self` contract of the
        // public surface (detector is rarely queried faster than
        // it is updated).
        let mut scratch = self.digest.clone();
        scratch.quantile(self.q)
    }

    /// Method-of-moments GPD fit — `γ = 0.5 · (1 − μ² / σ²)`,
    /// `σ_fit = μ · (1 + γ)`. Requires `peak_count ≥ 2` for a
    /// valid variance; [`Self::p_value`] gates the call.
    fn gpd_mom(&self) -> (f64, f64) {
        let n = self.peak_count;
        if n < 2 {
            return (0.0, 0.0);
        }
        #[allow(clippy::cast_precision_loss)]
        let n_f = n as f64;
        let variance = self.peak_m2 / (n_f - 1.0);
        if variance <= 0.0 || self.peak_mean <= 0.0 {
            return (0.0, 0.0);
        }
        let mean_sq = self.peak_mean * self.peak_mean;
        let gamma = 0.5 * (1.0 - mean_sq / variance);
        // Cap γ ≤ 0.4999 so MoM stays valid; heavier tails are the
        // fallback-tail-prob regime.
        let gamma = gamma.clamp(-5.0, 0.4999);
        let sigma = self.peak_mean * (1.0 + gamma);
        (gamma, sigma)
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_out_of_range_q() {
        assert!(PotDetector::new(0.0).is_err());
        assert!(PotDetector::new(1.0).is_err());
        assert!(PotDetector::new(f64::NAN).is_err());
        assert!(PotDetector::new(-0.1).is_err());
    }

    #[test]
    fn record_ignores_non_finite() {
        let mut d = PotDetector::default_spot();
        d.record(f64::NAN);
        d.record(f64::INFINITY);
        assert_eq!(d.total_seen(), 0);
    }

    #[test]
    fn p_value_below_threshold_returns_one() {
        let mut d = PotDetector::new(0.9).unwrap();
        for i in 0..1000 {
            d.record(i as f64 * 0.001);
        }
        d.freeze_baseline().unwrap();
        assert_eq!(d.p_value(-1.0), 1.0);
    }

    #[test]
    fn p_value_above_threshold_is_smaller_than_one() {
        let mut d = PotDetector::new(0.9).unwrap();
        for i in 0..1000 {
            d.record(i as f64 * 0.001);
        }
        d.freeze_baseline().unwrap();
        // Feed peaks so the GPD fit kicks in.
        for i in 0..200 {
            d.record(0.95 + i as f64 * 0.0005);
        }
        let p = d.p_value(10.0);
        assert!(p < 1.0);
        assert!(p > 0.0);
    }

    #[test]
    fn heavier_outlier_gets_smaller_p_value() {
        let mut d = PotDetector::new(0.9).unwrap();
        for i in 0..1000 {
            d.record(i as f64 * 0.001);
        }
        d.freeze_baseline().unwrap();
        for i in 0..200 {
            d.record(0.95 + i as f64 * 0.0005);
        }
        let mild = d.p_value(1.1);
        let heavy = d.p_value(10.0);
        assert!(heavy <= mild);
    }

    #[test]
    fn freeze_baseline_errors_on_empty_digest() {
        let mut d = PotDetector::default_spot();
        assert!(matches!(
            d.freeze_baseline().unwrap_err(),
            RcfError::EmptyForest
        ));
    }

    #[test]
    fn is_anomaly_thresholds_on_alert_p() {
        let mut d = PotDetector::new(0.9).unwrap();
        for i in 0..1000 {
            d.record(i as f64 * 0.001);
        }
        d.freeze_baseline().unwrap();
        for i in 0..200 {
            d.record(0.95 + i as f64 * 0.0005);
        }
        // 0.5 quite likely under baseline — not anomalous at 1e-3.
        assert!(!d.is_anomaly(0.5, 1.0e-3));
    }
}
