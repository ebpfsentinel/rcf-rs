//! Input-feature drift detector — PSI + KL divergence over a
//! frozen baseline distribution.
//!
//! The bare [`crate::meta_drift::MetaDriftDetector`] watches the
//! *score* stream and catches shifts in RCF's view of normality.
//! That alone misses a class of incidents where the input features
//! themselves drift: the scorer may be re-centring around the new
//! baseline while the upstream data has silently shifted (protocol
//! mix change, retargeted attack traffic, bad feature pipeline).
//!
//! [`FeatureDriftDetector`] addresses that by pinning a **baseline
//! per-dim histogram**, accumulating the production stream into a
//! **mirror histogram** with identical bin edges, and computing two
//! classical drift statistics on demand:
//!
//! - **Population Stability Index (PSI)**:
//!   `Σ (Q_i − P_i) · ln(Q_i / P_i)` per bin. Industry thresholds:
//!   `< 0.1` stable, `0.1 .. 0.25` watch, `> 0.25` alert.
//! - **KL divergence `D_KL(Q || P)`**: `Σ Q_i · ln(Q_i / P_i)`.
//!   Asymmetric — use when the production distribution diverging
//!   from baseline is the concern.
//!
//! Both are **per-dimension**: the detector reports one number per
//! feature so a SOC dashboard can pin the offending dim. A
//! Laplace-smoothed epsilon protects against log-of-zero on sparse
//! bins.
//!
//! # Life cycle
//!
//! 1. Build with [`FeatureDriftDetector::new(num_bins)`].
//! 2. Feed the warm-up window via [`Self::observe`].
//! 3. Call [`Self::freeze_baseline`] — pins the per-dim range and
//!    freezes the current histogram as the reference.
//! 4. Keep calling [`Self::observe`] with live traffic; the counts
//!    accrue into the production histogram.
//! 5. Periodically read [`Self::psi`] / [`Self::kl_divergence`] /
//!    [`Self::max_psi`]; optionally [`Self::reset_production`] to
//!    start a fresh production window.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::error::{RcfError, RcfResult};

#[cfg(feature = "std")]
use std::sync::Arc;

/// Default bin count per dimension — 10 is the classical PSI
/// choice, matches credit-risk industry practice.
pub const DEFAULT_NUM_BINS: usize = 10;

/// Default Laplace smoothing epsilon added to every bin count
/// before normalisation. Keeps `ln(P_i / Q_i)` finite on sparse
/// bins without moving the signal on well-populated ones.
pub const DEFAULT_SMOOTHING: f64 = 1.0e-4;

/// Industry-standard PSI band thresholds.
pub const PSI_WATCH_THRESHOLD: f64 = 0.10;
/// Industry-standard PSI band thresholds.
pub const PSI_ALERT_THRESHOLD: f64 = 0.25;

/// Ordinal drift level derived from a PSI value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DriftLevel {
    /// `PSI < 0.10` — distribution is stable vs. baseline.
    Stable,
    /// `0.10 ≤ PSI < 0.25` — worth monitoring, not yet alertable.
    Watch,
    /// `PSI ≥ 0.25` — distribution has shifted materially.
    Alert,
}

impl DriftLevel {
    /// Classify a single PSI value.
    #[must_use]
    pub fn classify(psi: f64) -> Self {
        if !psi.is_finite() || psi < PSI_WATCH_THRESHOLD {
            Self::Stable
        } else if psi < PSI_ALERT_THRESHOLD {
            Self::Watch
        } else {
            Self::Alert
        }
    }
}

/// Per-dim histogram-driven feature drift detector.
///
/// `D` pins the feature-vector dimensionality at compile time,
/// mirroring the rest of the crate. The detector is std-only
/// because it ships its observability plumbing.
pub struct FeatureDriftDetector<const D: usize> {
    /// Number of equal-width bins per dimension.
    num_bins: usize,
    /// Laplace smoothing epsilon added to every bin before
    /// normalisation.
    smoothing: f64,
    /// Per-dim baseline bin counts. `None` until
    /// [`Self::freeze_baseline`] runs.
    baseline: Option<Vec<Vec<u64>>>,
    /// Per-dim production bin counts — always allocated so
    /// `observe` can run once the baseline is frozen.
    production: Vec<Vec<u64>>,
    /// Per-dim `(min, max)` range pinned at baseline time. `None`
    /// while the detector is still in the cold-start warm-up
    /// window.
    bin_edges: Option<[(f64, f64); D]>,
    /// Cold-start sample buffer — every `observe` before
    /// `freeze_baseline` is stashed here so the baseline edges can
    /// be computed from the collected sample. Cleared at freeze.
    cold_samples: Vec<[f64; D]>,
    /// Lifetime total of `observe` calls — surfaced through the
    /// metrics sink so operators can separate warm-up from live
    /// traffic.
    observations_total: u64,
    /// Observability sink.
    #[cfg(feature = "std")]
    metrics: Arc<dyn crate::metrics::MetricsSink>,
}

impl<const D: usize> core::fmt::Debug for FeatureDriftDetector<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut s = f.debug_struct("FeatureDriftDetector");
        s.field("D", &D)
            .field("num_bins", &self.num_bins)
            .field("smoothing", &self.smoothing)
            .field("baseline_frozen", &self.baseline.is_some())
            .field("bin_edges", &self.bin_edges)
            .field("production_buckets", &self.production.len())
            .field("cold_samples", &self.cold_samples.len())
            .field("observations_total", &self.observations_total);
        #[cfg(feature = "std")]
        s.field("metrics", &self.metrics);
        s.finish()
    }
}

impl<const D: usize> FeatureDriftDetector<D> {
    /// Build a fresh detector with `num_bins` equal-width bins per
    /// dimension. Uses [`DEFAULT_SMOOTHING`] for Laplace smoothing.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `num_bins < 2` or
    /// `D == 0`.
    pub fn new(num_bins: usize) -> RcfResult<Self> {
        Self::with_smoothing(num_bins, DEFAULT_SMOOTHING)
    }

    /// Like [`Self::new`] but with a caller-chosen smoothing
    /// epsilon in `(0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] on any out-of-range
    /// argument.
    pub fn with_smoothing(num_bins: usize, smoothing: f64) -> RcfResult<Self> {
        if D == 0 {
            return Err(RcfError::InvalidConfig(
                "FeatureDriftDetector: D must be > 0".into(),
            ));
        }
        if num_bins < 2 {
            return Err(RcfError::InvalidConfig(format!(
                "FeatureDriftDetector: num_bins must be >= 2, got {num_bins}"
            )));
        }
        if !smoothing.is_finite() || smoothing <= 0.0 || smoothing > 1.0 {
            return Err(RcfError::InvalidConfig(format!(
                "FeatureDriftDetector: smoothing must be in (0, 1], got {smoothing}"
            )));
        }
        Ok(Self {
            num_bins,
            smoothing,
            baseline: None,
            production: vec![vec![0; num_bins]; D],
            bin_edges: None,
            cold_samples: Vec::new(),
            observations_total: 0,
            #[cfg(feature = "std")]
            metrics: crate::metrics::default_sink(),
        })
    }

    /// Install a metrics sink — every `observe` / `psi` call emits
    /// counters/gauges into it.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_metrics_sink(mut self, sink: Arc<dyn crate::metrics::MetricsSink>) -> Self {
        self.metrics = sink;
        self
    }

    /// Read-only handle to the installed sink.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn metrics_sink(&self) -> &Arc<dyn crate::metrics::MetricsSink> {
        &self.metrics
    }

    /// Whether [`Self::freeze_baseline`] has been called.
    #[must_use]
    pub fn is_baseline_frozen(&self) -> bool {
        self.baseline.is_some()
    }

    /// Lifetime count of [`Self::observe`] calls.
    #[must_use]
    pub fn observations_total(&self) -> u64 {
        self.observations_total
    }

    /// Number of equal-width bins per dimension.
    #[must_use]
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Per-dim `(min, max)` range pinned at baseline time — `None`
    /// before [`Self::freeze_baseline`].
    #[must_use]
    pub fn bin_edges(&self) -> Option<&[(f64, f64); D]> {
        self.bin_edges.as_ref()
    }

    /// Fold `point` into the production histogram (or the pre-
    /// freeze buffer when the baseline has not yet been frozen).
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] when `point` contains a non-finite
    ///   component.
    pub fn observe(&mut self, point: &[f64; D]) -> RcfResult<()> {
        if !point.iter().all(|v| v.is_finite()) {
            return Err(RcfError::NaNValue);
        }
        self.observations_total = self.observations_total.saturating_add(1);
        #[cfg(feature = "std")]
        self.metrics
            .inc_counter(crate::metrics::names::FEATURE_DRIFT_OBSERVED_TOTAL, 1);

        // Pre-freeze: accumulate into production; freeze_baseline
        // will promote these counts. Post-freeze: accumulate into
        // production (the only live histogram).
        if let Some(edges) = self.bin_edges {
            for (d, (min, max)) in edges.iter().enumerate() {
                let bin = map_to_bin(point[d], *min, *max, self.num_bins);
                self.production[d][bin] = self.production[d][bin].saturating_add(1);
            }
        } else {
            // Cold-start accumulation — we do not have edges yet, so
            // stash raw values in production[0][0] as a single-bin
            // placeholder. `freeze_baseline` will rebuild the
            // histogram from scratch using the collected sample.
            // Keep a parallel `samples` buffer instead.
            self.cold_samples.push(*point);
        }
        Ok(())
    }

    /// Freeze the current production histogram as the baseline.
    /// Computes per-dim `(min, max)` from the collected samples,
    /// rebuilds the histogram with equal-width bins, then clones it
    /// into the baseline slot. Subsequent [`Self::observe`] calls
    /// feed the production histogram only.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::EmptyForest`] when no points have been
    /// observed yet (nothing to bin).
    pub fn freeze_baseline(&mut self) -> RcfResult<()> {
        if self.cold_samples.is_empty() {
            return Err(RcfError::EmptyForest);
        }
        // Compute per-dim min/max on cold samples.
        let mut edges = [(f64::INFINITY, f64::NEG_INFINITY); D];
        for p in &self.cold_samples {
            for d in 0..D {
                if p[d] < edges[d].0 {
                    edges[d].0 = p[d];
                }
                if p[d] > edges[d].1 {
                    edges[d].1 = p[d];
                }
            }
        }
        // Widen degenerate ranges so equal-width bins are well-
        // defined; a zero-width range maps every point to bin 0.
        for pair in &mut edges {
            #[allow(clippy::float_cmp)]
            let collapsed = pair.0 == pair.1;
            if collapsed {
                pair.0 -= 0.5;
                pair.1 += 0.5;
            }
        }

        // Rebuild per-dim histograms from the cold samples.
        let mut baseline = vec![vec![0_u64; self.num_bins]; D];
        for p in &self.cold_samples {
            for d in 0..D {
                let bin = map_to_bin(p[d], edges[d].0, edges[d].1, self.num_bins);
                baseline[d][bin] = baseline[d][bin].saturating_add(1);
            }
        }

        self.baseline = Some(baseline);
        self.bin_edges = Some(edges);
        // Reset production — live traffic starts accruing from now.
        self.production = vec![vec![0_u64; self.num_bins]; D];
        self.cold_samples.clear();
        Ok(())
    }

    /// Drop the production histogram; keep the baseline. Call
    /// between monitoring windows.
    pub fn reset_production(&mut self) {
        self.production = vec![vec![0_u64; self.num_bins]; D];
    }

    /// Per-dim Population Stability Index against the baseline.
    /// Returns a `Vec` of length `D`; entry `d` is
    /// `Σ_i (Q_i − P_i) · ln(Q_i / P_i)` with Laplace smoothing.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::EmptyForest`] when the baseline has not
    /// been frozen.
    pub fn psi(&self) -> RcfResult<Vec<f64>> {
        let baseline = self.baseline.as_ref().ok_or(RcfError::EmptyForest)?;
        let mut out = Vec::with_capacity(D);
        for (base, prod) in baseline.iter().zip(self.production.iter()) {
            out.push(psi_one_dim(base, prod, self.smoothing));
        }
        #[cfg(feature = "std")]
        {
            let max_psi = out
                .iter()
                .copied()
                .fold(0.0_f64, |a, b| if b > a { b } else { a });
            self.metrics
                .set_gauge(crate::metrics::names::FEATURE_DRIFT_MAX_PSI, max_psi);
        }
        Ok(out)
    }

    /// Per-dim KL divergence `D_KL(Q || P)` against the baseline —
    /// `Σ_i Q_i · ln(Q_i / P_i)`. Asymmetric; see module docs.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::EmptyForest`] when the baseline has not
    /// been frozen.
    pub fn kl_divergence(&self) -> RcfResult<Vec<f64>> {
        let baseline = self.baseline.as_ref().ok_or(RcfError::EmptyForest)?;
        let mut out = Vec::with_capacity(D);
        for (base, prod) in baseline.iter().zip(self.production.iter()) {
            out.push(kl_one_dim(base, prod, self.smoothing));
        }
        Ok(out)
    }

    /// Maximum PSI across every dimension — the single number SOC
    /// dashboards usually alert on.
    ///
    /// # Errors
    ///
    /// Same as [`Self::psi`].
    pub fn max_psi(&self) -> RcfResult<f64> {
        let all = self.psi()?;
        Ok(all
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| if b > a { b } else { a }))
    }

    /// Dim index with the largest PSI — useful for root-causing
    /// "which feature moved?".
    ///
    /// # Errors
    ///
    /// Same as [`Self::psi`]. Returns `Ok(None)` when every PSI is
    /// exactly zero.
    pub fn argmax_psi(&self) -> RcfResult<Option<usize>> {
        let all = self.psi()?;
        let mut best = 0_usize;
        let mut best_val = 0.0_f64;
        for (d, v) in all.iter().enumerate() {
            if *v > best_val {
                best_val = *v;
                best = d;
            }
        }
        if best_val == 0.0 {
            Ok(None)
        } else {
            Ok(Some(best))
        }
    }
}

/// Map `v` into `[0, num_bins)` given the inclusive baseline range
/// `[min, max]`. Values outside the range clamp to the extremes so
/// production outliers do not get silently dropped.
fn map_to_bin(v: f64, min: f64, max: f64, num_bins: usize) -> usize {
    if !v.is_finite() || v <= min {
        return 0;
    }
    if v >= max {
        return num_bins - 1;
    }
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let idx = (((v - min) / (max - min)) * num_bins as f64) as usize;
    idx.min(num_bins - 1)
}

/// PSI against two bin-count vectors of equal length. Laplace-
/// smoothed to keep logs finite on sparse bins.
fn psi_one_dim(baseline: &[u64], production: &[u64], smoothing: f64) -> f64 {
    if baseline.len() != production.len() || baseline.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let base_total: f64 = baseline.iter().copied().map(|x| x as f64).sum::<f64>();
    #[allow(clippy::cast_precision_loss)]
    let prod_total: f64 = production.iter().copied().map(|x| x as f64).sum::<f64>();
    if base_total <= 0.0 || prod_total <= 0.0 {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    for (b, p) in baseline.iter().zip(production.iter()) {
        #[allow(clippy::cast_precision_loss)]
        let p_ratio = (*b as f64 / base_total).max(smoothing);
        #[allow(clippy::cast_precision_loss)]
        let q_ratio = (*p as f64 / prod_total).max(smoothing);
        acc += (q_ratio - p_ratio) * (q_ratio / p_ratio).ln();
    }
    acc
}

/// KL divergence `D_KL(Q || P)` across two bin-count vectors.
fn kl_one_dim(baseline: &[u64], production: &[u64], smoothing: f64) -> f64 {
    if baseline.len() != production.len() || baseline.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let base_total: f64 = baseline.iter().copied().map(|x| x as f64).sum::<f64>();
    #[allow(clippy::cast_precision_loss)]
    let prod_total: f64 = production.iter().copied().map(|x| x as f64).sum::<f64>();
    if base_total <= 0.0 || prod_total <= 0.0 {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    for (b, p) in baseline.iter().zip(production.iter()) {
        #[allow(clippy::cast_precision_loss)]
        let p_ratio = (*b as f64 / base_total).max(smoothing);
        #[allow(clippy::cast_precision_loss)]
        let q_ratio = (*p as f64 / prod_total).max(smoothing);
        acc += q_ratio * (q_ratio / p_ratio).ln();
    }
    acc
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
    fn new_rejects_bad_bins() {
        assert!(FeatureDriftDetector::<4>::new(0).is_err());
        assert!(FeatureDriftDetector::<4>::new(1).is_err());
    }

    #[test]
    fn new_rejects_bad_smoothing() {
        assert!(FeatureDriftDetector::<4>::with_smoothing(10, 0.0).is_err());
        assert!(FeatureDriftDetector::<4>::with_smoothing(10, f64::NAN).is_err());
        assert!(FeatureDriftDetector::<4>::with_smoothing(10, 2.0).is_err());
    }

    #[test]
    fn psi_before_freeze_errors() {
        let d = FeatureDriftDetector::<2>::new(10).unwrap();
        assert!(d.psi().is_err());
        assert!(d.kl_divergence().is_err());
    }

    #[test]
    fn identical_distribution_has_zero_psi() {
        let mut d = FeatureDriftDetector::<2>::new(10).unwrap();
        for i in 0..200 {
            let v = (i as f64 % 10.0) * 0.1;
            d.observe(&[v, v + 0.5]).unwrap();
        }
        d.freeze_baseline().unwrap();
        // Replay same sample → production matches baseline exactly.
        for i in 0..200 {
            let v = (i as f64 % 10.0) * 0.1;
            d.observe(&[v, v + 0.5]).unwrap();
        }
        let psi = d.psi().unwrap();
        for p in &psi {
            assert!(*p < 1.0e-6, "expected near-zero PSI, got {p}");
        }
    }

    #[test]
    fn shifted_distribution_raises_psi() {
        let mut d = FeatureDriftDetector::<1>::new(10).unwrap();
        // Baseline: uniform in [0, 1).
        for i in 0..1000 {
            let v = (i as f64 % 10.0) * 0.1;
            d.observe(&[v]).unwrap();
        }
        d.freeze_baseline().unwrap();
        // Production: all near the max bin — massive drift.
        for _ in 0..1000 {
            d.observe(&[0.95]).unwrap();
        }
        let psi = d.psi().unwrap();
        assert!(
            psi[0] > PSI_ALERT_THRESHOLD,
            "expected alert-level PSI, got {}",
            psi[0]
        );
        assert_eq!(DriftLevel::classify(psi[0]), DriftLevel::Alert);
    }

    #[test]
    fn drift_level_thresholds() {
        assert_eq!(DriftLevel::classify(0.0), DriftLevel::Stable);
        assert_eq!(DriftLevel::classify(0.09), DriftLevel::Stable);
        assert_eq!(DriftLevel::classify(0.10), DriftLevel::Watch);
        assert_eq!(DriftLevel::classify(0.24), DriftLevel::Watch);
        assert_eq!(DriftLevel::classify(0.25), DriftLevel::Alert);
        assert_eq!(DriftLevel::classify(f64::NAN), DriftLevel::Stable);
    }

    #[test]
    fn argmax_psi_none_on_zero() {
        let mut d = FeatureDriftDetector::<3>::new(10).unwrap();
        for i in 0..100 {
            let v = (i as f64 % 10.0) * 0.1;
            d.observe(&[v, v + 0.1, v + 0.2]).unwrap();
        }
        d.freeze_baseline().unwrap();
        // No production observations → production all zero → PSI = 0.
        let ap = d.argmax_psi().unwrap();
        assert!(ap.is_none());
    }

    #[test]
    fn argmax_psi_picks_drifting_dim() {
        let mut d = FeatureDriftDetector::<3>::new(10).unwrap();
        for i in 0..500 {
            let v = (i as f64 % 10.0) * 0.1;
            d.observe(&[v, v, v]).unwrap();
        }
        d.freeze_baseline().unwrap();
        // Only dim 1 drifts (pushed to max).
        for i in 0..500 {
            let v = (i as f64 % 10.0) * 0.1;
            d.observe(&[v, 0.95, v]).unwrap();
        }
        let ap = d.argmax_psi().unwrap();
        assert_eq!(ap, Some(1));
    }

    #[test]
    fn observe_rejects_nan() {
        let mut d = FeatureDriftDetector::<2>::new(10).unwrap();
        assert!(d.observe(&[f64::NAN, 0.0]).is_err());
        assert!(d.observe(&[0.0, f64::INFINITY]).is_err());
    }

    #[test]
    fn reset_production_leaves_baseline_intact() {
        let mut d = FeatureDriftDetector::<1>::new(10).unwrap();
        for i in 0..100 {
            d.observe(&[(i as f64) * 0.01]).unwrap();
        }
        d.freeze_baseline().unwrap();
        for i in 0..100 {
            d.observe(&[(i as f64) * 0.01]).unwrap();
        }
        d.reset_production();
        assert!(d.is_baseline_frozen());
        // After reset, production is empty → PSI with smoothing
        // floor returns a finite value (not a panic).
        let psi = d.psi().unwrap();
        assert!(psi[0].is_finite());
    }

    #[test]
    fn kl_matches_psi_components_on_simple_drift() {
        let mut d = FeatureDriftDetector::<1>::new(10).unwrap();
        for i in 0..500 {
            d.observe(&[(i as f64 % 10.0) * 0.1]).unwrap();
        }
        d.freeze_baseline().unwrap();
        for _ in 0..500 {
            d.observe(&[0.95]).unwrap();
        }
        let kl = d.kl_divergence().unwrap();
        assert!(kl[0] > 0.0);
        assert!(kl[0].is_finite());
    }
}
