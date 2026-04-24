//! SOC-feedback ingestion — Das et al., *Incorporating Feedback
//! into Tree-based Anomaly Detection*, `arXiv:1708.09441` /
//! KDD 2016 / ADI.
//!
//! An analyst triaging an alert labels it **benign** (false
//! positive — the score was too high) or **confirmed** (true
//! positive — the score was correctly high). A
//! [`FeedbackStore`] keeps a bounded ring of labelled points and
//! exposes [`FeedbackStore::adjust`] which shifts a raw anomaly
//! score toward the label of its nearest labelled neighbours.
//!
//! # What this is (and is not)
//!
//! **Is**: a lightweight feedback-adjustment layer on top of a
//! live [`anomstream_core::RandomCutForest`]. No retraining, no extra model,
//! no mutation of the forest — the raw RCF score survives
//! intact, the adjustment is an additive bias driven by the
//! Gaussian-kernel-weighted contribution of every live label.
//!
//! **Is not**: the full Das et al. Active Anomaly Discovery
//! (AAD) optimiser with per-leaf weights learned via online
//! convex optimisation. AAD is a heavier commitment — it requires
//! scoring-path instrumentation to expose per-leaf contributions,
//! an online solver, and careful hyper-parameter tuning. The
//! simpler nearest-neighbour-kernel variant here captures the
//! Das paper's core signal ("alerts similar to a known-benign
//! point should score down") with zero forest-side changes and
//! fits the anomstream-core API surface cleanly.
//!
//! Swap in AAD later behind the same [`FeedbackStore::adjust`]
//! signature if / when the AUC gap justifies the complexity.

#![cfg(feature = "std")]

use std::collections::VecDeque;
use std::sync::Arc;

use anomstream_core::error::{RcfError, RcfResult};
use anomstream_core::metrics::{MetricsSink, default_sink, names};

/// Default kernel bandwidth. Smaller `sigma` → only near-duplicate
/// labels influence the probe; larger `sigma` → every stored label
/// contributes a little. Starts at `1.0`; tune per feature scale.
pub const DEFAULT_KERNEL_SIGMA: f64 = 1.0;

/// Default contribution strength — coefficient on the label-
/// weighted kernel sum before it's added to the raw score. `1.0`
/// lets a single nearest-benign point cancel a `+1.0` raw score
/// when the probe sits on the label; halve it for softer
/// adjustment.
pub const DEFAULT_STRENGTH: f64 = 1.0;

/// Default bounded ledger capacity. SOC operators rarely label
/// more than a few hundred alerts per tenant per week; `512`
/// keeps a month of labels at typical rates with
/// bounded-memory guarantees.
pub const DEFAULT_CAPACITY: usize = 512;

/// Analyst verdict attached to a stored point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum FeedbackLabel {
    /// False positive — the raw score was too high; the probe
    /// should score **down** toward baseline.
    Benign,
    /// True positive — the raw score was correctly high; nearby
    /// probes should score **up** to match.
    Confirmed,
}

impl FeedbackLabel {
    /// `-1.0` for [`Self::Benign`], `+1.0` for [`Self::Confirmed`].
    /// Drives the signed contribution in
    /// [`FeedbackStore::adjust`].
    #[must_use]
    pub const fn sign(self) -> f64 {
        match self {
            Self::Benign => -1.0,
            Self::Confirmed => 1.0,
        }
    }
}

/// Bounded ledger of labelled points — oldest-first LRU eviction
/// once [`FeedbackStore::capacity`] is reached.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FeedbackStore<const D: usize> {
    /// Labelled points in arrival order. `VecDeque` so that
    /// oldest-first eviction on capacity pressure is `O(1)` — a
    /// `Vec::remove(0)` would memmove the whole ring per label,
    /// `O(capacity)` per call, at 512-cap that is a measurable
    /// hot-path cost under sustained SOC labelling.
    entries: VecDeque<LabelledPoint<D>>,
    /// Capacity ceiling; older entries get dropped once exceeded.
    capacity: usize,
    /// Gaussian kernel bandwidth on the L2 distance from the probe
    /// to each labelled point.
    sigma: f64,
    /// Coefficient multiplied by the kernel-weighted label sum
    /// before it is added to the raw score.
    strength: f64,
    /// Observability sink — serde-skipped, restored to the noop
    /// sink on round-trip.
    #[cfg_attr(
        feature = "serde",
        serde(skip, default = "anomstream_core::metrics::default_sink")
    )]
    metrics: Arc<dyn MetricsSink>,
}

/// Single labelled record.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct LabelledPoint<const D: usize> {
    /// Feature vector at label time (after any upstream scaling).
    #[cfg_attr(
        feature = "serde",
        serde(with = "anomstream_core::serde_util::fixed_array_f64")
    )]
    point: [f64; D],
    /// SOC verdict.
    label: FeedbackLabel,
}

impl<const D: usize> FeedbackStore<D> {
    /// Build a store with defaults
    /// ([`DEFAULT_CAPACITY`] / [`DEFAULT_KERNEL_SIGMA`] /
    /// [`DEFAULT_STRENGTH`]).
    #[must_use]
    pub fn default_store() -> Self {
        Self {
            entries: VecDeque::with_capacity(DEFAULT_CAPACITY),
            capacity: DEFAULT_CAPACITY,
            sigma: DEFAULT_KERNEL_SIGMA,
            strength: DEFAULT_STRENGTH,
            metrics: default_sink(),
        }
    }

    /// Caller-configured constructor.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] on `capacity == 0`,
    /// non-positive `sigma`, or non-finite `strength`.
    pub fn new(capacity: usize, sigma: f64, strength: f64) -> RcfResult<Self> {
        if capacity == 0 {
            return Err(RcfError::InvalidConfig(
                "FeedbackStore: capacity must be > 0".into(),
            ));
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(RcfError::InvalidConfig(format!(
                "FeedbackStore: sigma must be finite and > 0, got {sigma}"
            )));
        }
        if !strength.is_finite() {
            return Err(RcfError::InvalidConfig(format!(
                "FeedbackStore: strength must be finite, got {strength}"
            )));
        }
        Ok(Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
            sigma,
            strength,
            metrics: default_sink(),
        })
    }

    /// Install a metrics sink — every `label` call emits counters
    /// keyed by verdict (benign vs confirmed).
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

    /// Labels currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when no labels are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Capacity ceiling.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current Gaussian kernel bandwidth.
    #[must_use]
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Current contribution strength.
    #[must_use]
    pub fn strength(&self) -> f64 {
        self.strength
    }

    /// Record an analyst label. On capacity pressure the oldest
    /// entry is evicted (FIFO). Non-finite `point` components are
    /// rejected.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::NaNValue`] on non-finite input.
    pub fn label(&mut self, point: [f64; D], label: FeedbackLabel) -> RcfResult<()> {
        if !point.iter().all(|v| v.is_finite()) {
            return Err(RcfError::NaNValue);
        }
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(LabelledPoint { point, label });
        self.metrics
            .inc_counter(names::FEEDBACK_LABELS_OBSERVED_TOTAL, 1);
        let sub = match label {
            FeedbackLabel::Benign => names::FEEDBACK_LABELS_BENIGN_TOTAL,
            FeedbackLabel::Confirmed => names::FEEDBACK_LABELS_CONFIRMED_TOTAL,
        };
        self.metrics.inc_counter(sub, 1);
        Ok(())
    }

    /// Drop every stored label.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Shift `raw_score` toward the label of its nearest labelled
    /// neighbours via a Gaussian-kernel-weighted sum:
    ///
    /// ```text
    /// adjusted = raw + strength · Σ_i sign(label_i) · exp(−|probe − p_i|² / (2 σ²))
    /// ```
    ///
    /// Benign labels push the score **down**; confirmed labels
    /// push it **up**. The returned value is clamped to `≥ 0` so
    /// the output is a valid anomaly score.
    ///
    /// Returns `raw_score` unchanged when the store is empty or
    /// when `probe` contains a non-finite component.
    #[must_use = "detector output should be checked — dropping it silently usually indicates a logic bug"]
    pub fn adjust(&self, probe: &[f64; D], raw_score: f64) -> f64 {
        if self.entries.is_empty() || !probe.iter().all(|v| v.is_finite()) {
            return raw_score;
        }
        let two_sigma_sq = 2.0 * self.sigma * self.sigma;
        let mut bias = 0.0_f64;
        for entry in &self.entries {
            let mut sq = 0.0_f64;
            for (probe_d, entry_d) in probe.iter().zip(entry.point.iter()) {
                let diff = probe_d - entry_d;
                sq += diff * diff;
            }
            let kernel = (-sq / two_sigma_sq).exp();
            bias += entry.label.sign() * kernel;
        }
        (raw_score + self.strength * bias).max(0.0)
    }

    /// Read-only view of the labelled entries — tests + doctool
    /// usage.
    pub fn entries(&self) -> impl Iterator<Item = (&[f64; D], FeedbackLabel)> {
        self.entries.iter().map(|e| (&e.point, e.label))
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
        assert!(FeedbackStore::<2>::new(0, 1.0, 1.0).is_err());
        assert!(FeedbackStore::<2>::new(10, 0.0, 1.0).is_err());
        assert!(FeedbackStore::<2>::new(10, -0.1, 1.0).is_err());
        assert!(FeedbackStore::<2>::new(10, 1.0, f64::NAN).is_err());
    }

    #[test]
    fn label_rejects_non_finite() {
        let mut s = FeedbackStore::<2>::default_store();
        assert!(matches!(
            s.label([f64::NAN, 0.0], FeedbackLabel::Benign).unwrap_err(),
            RcfError::NaNValue
        ));
        assert!(s.is_empty());
    }

    #[test]
    fn empty_store_leaves_raw_score_untouched() {
        let s = FeedbackStore::<2>::default_store();
        assert_eq!(s.adjust(&[1.0, 2.0], 0.7), 0.7);
    }

    #[test]
    fn benign_label_pulls_nearby_probe_down() {
        let mut s = FeedbackStore::<2>::new(10, 1.0, 1.0).unwrap();
        s.label([1.0, 2.0], FeedbackLabel::Benign).unwrap();
        // Same point as the label → kernel = 1.0, bias = -1.0,
        // adjusted = max(0, 0.7 - 1.0) = 0.0.
        let adjusted = s.adjust(&[1.0, 2.0], 0.7);
        assert!(adjusted < 0.7);
        assert!(adjusted >= 0.0);
    }

    #[test]
    fn confirmed_label_pushes_nearby_probe_up() {
        let mut s = FeedbackStore::<2>::new(10, 1.0, 1.0).unwrap();
        s.label([1.0, 2.0], FeedbackLabel::Confirmed).unwrap();
        let adjusted = s.adjust(&[1.0, 2.0], 0.5);
        // Same point → kernel 1.0, bias +1.0, adjusted = 1.5.
        assert!(adjusted > 0.5);
    }

    #[test]
    fn far_probe_gets_negligible_adjustment() {
        let mut s = FeedbackStore::<2>::new(10, 0.1, 1.0).unwrap();
        s.label([0.0, 0.0], FeedbackLabel::Benign).unwrap();
        // `sigma = 0.1`, probe 10 units away → kernel ≈ 0.
        let raw = 0.7;
        let adjusted = s.adjust(&[10.0, 10.0], raw);
        assert!((adjusted - raw).abs() < 1.0e-9);
    }

    #[test]
    fn capacity_evicts_oldest_entry() {
        let mut s = FeedbackStore::<1>::new(2, 1.0, 1.0).unwrap();
        s.label([0.0], FeedbackLabel::Benign).unwrap();
        s.label([1.0], FeedbackLabel::Confirmed).unwrap();
        s.label([2.0], FeedbackLabel::Benign).unwrap(); // evicts [0.0].
        assert_eq!(s.len(), 2);
        let kept: Vec<_> = s.entries().map(|(p, _)| p[0]).collect();
        assert_eq!(kept, vec![1.0, 2.0]);
    }

    #[test]
    fn clear_drops_all_entries() {
        let mut s = FeedbackStore::<2>::default_store();
        s.label([0.0, 0.0], FeedbackLabel::Benign).unwrap();
        s.clear();
        assert!(s.is_empty());
    }

    #[test]
    fn mixed_labels_net_zero_near_centroid() {
        let mut s = FeedbackStore::<2>::new(10, 1.0, 1.0).unwrap();
        s.label([1.0, 0.0], FeedbackLabel::Confirmed).unwrap();
        s.label([-1.0, 0.0], FeedbackLabel::Benign).unwrap();
        // Midpoint probe sees equal-distance labels with opposite
        // signs → bias ≈ 0, adjusted ≈ raw.
        let raw = 0.6;
        let adjusted = s.adjust(&[0.0, 0.0], raw);
        assert!((adjusted - raw).abs() < 1.0e-9);
    }

    #[test]
    fn label_sign_values() {
        assert_eq!(FeedbackLabel::Benign.sign(), -1.0);
        assert_eq!(FeedbackLabel::Confirmed.sign(), 1.0);
    }
}
