//! `ADWIN` (`ADaptive` `WINdowing`) — streaming change-point
//! detector with automatic window sizing.
//!
//! Bifet & Gavaldà, *Learning from Time-Changing Data with
//! Adaptive Windowing*, SIAM SDM 2007. The window holds recent
//! observations; whenever the mean over two adjacent sub-windows
//! differs by more than a Hoeffding-style bound, the older
//! sub-window is dropped. The remaining window is the longest
//! suffix still consistent with a single distribution.
//!
//! This implementation is a bounded-capacity simplification of
//! Bifet's ADWIN2 (no exponential histograms): a ring buffer of
//! the last `N` items + prefix-sum scans on each `update`. O(N)
//! per update where `N` is the configured window cap. For the
//! use-case in `rcf-rs` (drift on score streams, `N ≤ 4096`) the
//! constant factors dominate over the logarithmic win of
//! exponential histograms.
//!
//! # Drift statistic
//!
//! For every split point `i ∈ [1, N - 1]`:
//!
//! ```text
//! mean_L = (1/n_L) · Σ_{j < i} x_j
//! mean_R = (1/n_R) · Σ_{j ≥ i} x_j
//! m      = 1 / (1/n_L + 1/n_R)                  (harmonic mean of sizes)
//! ε_cut  = sqrt((1 / (2m)) · ln(4 · N / δ)) · (range)
//! ```
//!
//! Drift fires when `|mean_L − mean_R| > ε_cut` for any split.
//! `range` is the caller-declared amplitude of the stream (for
//! bounded-range streams the Hoeffding constant collapses to 1).
//!
//! # Use with rcf-rs
//!
//! `AdwinDetector` is a standalone trigger — feed it the anomaly
//! score stream (or any scalar per-step signal) and route
//! [`AdwinDetector::update`]'s `true` return into
//! [`crate::DriftAwareForest::on_drift`] to spawn a shadow forest.

#![cfg(feature = "std")]

use std::sync::Arc;

use crate::error::{RcfError, RcfResult};
use crate::metrics::{MetricsSink, default_sink, names};

/// Default confidence budget `δ` — lower values = stricter bound,
/// fewer false-positive drift fires. `0.002` matches Bifet's
/// reference `δ = 0.002` experiment setting.
pub const DEFAULT_DELTA: f64 = 0.002;

/// Default window cap (bounded ring buffer). `4096` handles hour-
/// scale streams at 1 observation/s without excess memory.
pub const DEFAULT_WINDOW_CAP: usize = 4096;

/// Minimum sub-window size required for a valid Hoeffding-bound
/// comparison. Below this the detector stays silent — sub-window
/// means are too noisy to distinguish drift from sampling jitter.
pub const MIN_SUBWINDOW_LEN: usize = 16;

/// ADWIN-style streaming change-point detector.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AdwinDetector {
    /// Caller-declared amplitude of the observed stream. Scales
    /// the Hoeffding bound — bounded `[0, 1]` streams pass `1.0`.
    range: f64,
    /// Confidence budget — lower δ => stricter threshold.
    delta: f64,
    /// Maximum items kept in the window.
    window_cap: usize,
    /// Oldest-first ring of recent observations.
    buffer: Vec<f64>,
    /// Cumulative drift fires reported.
    drift_fires: u64,
    /// Observability sink — serde-skipped, restored to the noop
    /// sink on round-trip.
    #[cfg_attr(
        feature = "serde",
        serde(skip, default = "crate::metrics::default_sink")
    )]
    metrics: Arc<dyn MetricsSink>,
}

impl AdwinDetector {
    /// Build a detector with caller-chosen stream amplitude.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] on non-finite / non-
    /// positive `range`, `delta` outside `(0, 1)`, or
    /// `window_cap < 2 · MIN_SUBWINDOW_LEN`.
    pub fn new(range: f64, delta: f64, window_cap: usize) -> RcfResult<Self> {
        if !range.is_finite() || range <= 0.0 {
            return Err(RcfError::InvalidConfig(format!(
                "AdwinDetector: range must be finite and > 0, got {range}"
            )));
        }
        if !delta.is_finite() || !(0.0..1.0).contains(&delta) || delta <= 0.0 {
            return Err(RcfError::InvalidConfig(format!(
                "AdwinDetector: delta must be in (0.0, 1.0), got {delta}"
            )));
        }
        if window_cap < 2 * MIN_SUBWINDOW_LEN {
            return Err(RcfError::InvalidConfig(format!(
                "AdwinDetector: window_cap must be >= {}, got {window_cap}",
                2 * MIN_SUBWINDOW_LEN
            )));
        }
        Ok(Self {
            range,
            delta,
            window_cap,
            buffer: Vec::with_capacity(window_cap),
            drift_fires: 0,
            metrics: default_sink(),
        })
    }

    /// Install a metrics sink — every `update` emits an observed
    /// counter, and every drift fire bumps a drift-fires counter.
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

    /// Convenience: range `1.0`, [`DEFAULT_DELTA`],
    /// [`DEFAULT_WINDOW_CAP`].
    ///
    /// # Panics
    ///
    /// Never — the default values pass `new`'s validation.
    #[must_use]
    pub fn default_bounded() -> Self {
        Self::new(1.0, DEFAULT_DELTA, DEFAULT_WINDOW_CAP).expect("default params valid")
    }

    /// Current window length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// `true` when the window holds no observations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Lifetime drift fires since construction.
    #[must_use]
    pub fn drift_fires(&self) -> u64 {
        self.drift_fires
    }

    /// Running mean of the current window (empty → `0.0`).
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.buffer.is_empty() {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let n = self.buffer.len() as f64;
        self.buffer.iter().sum::<f64>() / n
    }

    /// Fold `value` into the window. Evaluates every split point
    /// against the Hoeffding bound; if any split flags drift, the
    /// older sub-window is dropped and the return is `true`.
    /// Returns `false` otherwise (no drift on this update).
    ///
    /// Non-finite values are silently dropped.
    pub fn update(&mut self, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        self.metrics.inc_counter(names::ADWIN_OBSERVED_TOTAL, 1);
        // Drop the oldest entry if we're at cap. Keeping this an
        // `O(N)` front-removal is fine at N ≤ 4k — benchmarked
        // marginal vs `VecDeque` in the typical rcf-rs use case.
        if self.buffer.len() >= self.window_cap {
            self.buffer.remove(0);
        }
        self.buffer.push(value);
        let fired = self.detect_and_shrink();
        if fired {
            self.metrics.inc_counter(names::ADWIN_DRIFT_FIRES_TOTAL, 1);
        }
        fired
    }

    /// Drop every observation. Counters are preserved.
    pub fn reset_window(&mut self) {
        self.buffer.clear();
    }

    /// Scan every candidate split; drop the older sub-window on
    /// the first drift signal found. Returns `true` iff drift was
    /// detected.
    fn detect_and_shrink(&mut self) -> bool {
        let n = self.buffer.len();
        if n < 2 * MIN_SUBWINDOW_LEN {
            return false;
        }
        // Prefix sums for O(N) split evaluation.
        let mut prefix = Vec::with_capacity(n + 1);
        prefix.push(0.0_f64);
        let mut acc = 0.0_f64;
        for &v in &self.buffer {
            acc += v;
            prefix.push(acc);
        }
        let total = prefix[n];

        for (split, &left_sum) in prefix
            .iter()
            .enumerate()
            .take(n - MIN_SUBWINDOW_LEN + 1)
            .skip(MIN_SUBWINDOW_LEN)
        {
            let n_left = split;
            let n_right = n - split;
            #[allow(clippy::cast_precision_loss)]
            let nl = n_left as f64;
            #[allow(clippy::cast_precision_loss)]
            let nr = n_right as f64;
            let mean_l = left_sum / nl;
            let mean_r = (total - left_sum) / nr;
            let m = 1.0 / (1.0 / nl + 1.0 / nr);
            #[allow(clippy::cast_precision_loss)]
            let log_term = (4.0 * n as f64 / self.delta).ln();
            let eps_cut = self.range * (log_term / (2.0 * m)).sqrt();
            if (mean_l - mean_r).abs() > eps_cut {
                // Drop the older sub-window — keep the right (newer)
                // side which is assumed to reflect the new regime.
                self.buffer.drain(..split);
                self.drift_fires = self.drift_fires.saturating_add(1);
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_invalid_params() {
        assert!(AdwinDetector::new(-1.0, DEFAULT_DELTA, DEFAULT_WINDOW_CAP).is_err());
        assert!(AdwinDetector::new(1.0, 0.0, DEFAULT_WINDOW_CAP).is_err());
        assert!(AdwinDetector::new(1.0, 1.0, DEFAULT_WINDOW_CAP).is_err());
        assert!(AdwinDetector::new(1.0, DEFAULT_DELTA, 5).is_err());
    }

    #[test]
    fn stable_stream_does_not_fire() {
        let mut d = AdwinDetector::default_bounded();
        for _ in 0..512 {
            assert!(!d.update(0.5));
        }
        assert_eq!(d.drift_fires(), 0);
    }

    #[test]
    fn mean_shift_triggers_drift() {
        let mut d = AdwinDetector::default_bounded();
        // Fill with baseline mean 0.1.
        for _ in 0..256 {
            d.update(0.1);
        }
        // Shift to mean 0.9 — should trigger within a handful of
        // post-shift samples.
        let mut fired = false;
        for _ in 0..128 {
            if d.update(0.9) {
                fired = true;
                break;
            }
        }
        assert!(fired, "ADWIN missed mean shift from 0.1 → 0.9");
        assert!(d.drift_fires() >= 1);
    }

    #[test]
    fn non_finite_is_ignored() {
        let mut d = AdwinDetector::default_bounded();
        assert!(!d.update(f64::NAN));
        assert!(!d.update(f64::INFINITY));
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn window_respects_cap() {
        let mut d = AdwinDetector::new(1.0, DEFAULT_DELTA, 64).unwrap();
        for i in 0..200 {
            d.update(f64::from(i % 2));
        }
        assert!(d.len() <= 64);
    }

    #[test]
    fn reset_window_clears_buffer() {
        let mut d = AdwinDetector::default_bounded();
        for _ in 0..100 {
            d.update(0.5);
        }
        d.reset_window();
        assert!(d.is_empty());
    }
}
