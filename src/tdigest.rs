//! Streaming quantile estimator — Ted Dunning's t-digest
//! (Computing Extremely Accurate Quantiles Using t-Digests, 2019).
//!
//! [`crate::ScoreHistogram`] bins values in a fixed number of
//! equal-width buckets — fine for central percentiles, lossy at the
//! tails where SOC SLOs typically live (p99, p99.9). [`TDigest`]
//! maintains a small set of **centroids** whose weight grows near
//! the distribution tails and stays tight in the centre, giving
//! sub-percent error on tail quantiles for `O(δ)` memory where `δ`
//! is the compression parameter.
//!
//! # Scale function
//!
//! This implementation uses Dunning's **scale function 1**
//! (`k_1(q) = (δ / 2π) · asin(2q − 1)`), which gives near-uniform
//! error across the quantile range. Centroids may grow up to a
//! weight of `total · (q_limit(q) − q)` where
//! `q_limit(q) = (sin(2π · (k_1(q) + 1) / δ) + 1) / 2`, i.e. the
//! quantile a single scale-function step above `q`.
//!
//! # Merging variant
//!
//! `record(x)` pushes into an **unsorted buffer**. When the buffer
//! length exceeds `compression · 10`, or a quantile is queried, the
//! buffer is flushed: sorted, then merged with the existing
//! centroids via one linear pass that respects the scale-function
//! weight bound. This amortises per-record cost to `O(1)` and keeps
//! query latency bounded in `O(δ)`.

use alloc::format;
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::error::{RcfError, RcfResult};

/// Default compression parameter — 100 balances accuracy and
/// memory, matches Dunning's reference implementation.
pub const DEFAULT_COMPRESSION: f64 = 100.0;

/// Buffer-flush trigger — when pending inserts exceed
/// `compression · BUFFER_MULT`, flush and merge.
const BUFFER_MULT: usize = 10;

/// One centroid: mean plus weight. Weight is `f64` because
/// compaction merges centroids; non-integer accumulations are
/// native.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Centroid {
    /// Centroid mean — running average of every value that landed
    /// in this centroid during compaction.
    pub mean: f64,
    /// Centroid weight — number of `record` values this centroid
    /// summarises (sub-1 weights are possible in principle but this
    /// implementation always starts them at 1.0).
    pub weight: f64,
}

/// Streaming quantile estimator with tight-tail accuracy.
///
/// `TDigest` is the streaming analogue of a percentile sketch —
/// `record(x)` is `O(1)` amortised, `quantile(q)` is `O(δ)`, and
/// the maximum centroid count is `~2 · δ`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TDigest {
    /// Compression — larger values → more centroids → more
    /// accurate (especially on tails) at higher memory / merge
    /// cost. Typical range `[20, 1000]`; `100` is a sane default.
    compression: f64,
    /// Sorted centroids (ascending `mean`). Always coherent after
    /// [`Self::flush_buffer`].
    centroids: Vec<Centroid>,
    /// Unsorted insertion buffer. Drained by `flush_buffer`.
    buffer: Vec<f64>,
    /// Cached total weight — `centroids.weight_sum + buffer.len`
    /// once pending inserts are flushed. Surfaced by
    /// [`Self::total_weight`] for diagnostics.
    total_weight: f64,
    /// Running minimum across every `record` call. Lower-tail
    /// queries (`quantile(q)` with very small `q`) extrapolate
    /// between `min` and the first centroid.
    min: f64,
    /// Running maximum. Symmetric role on the upper tail.
    max: f64,
}

impl TDigest {
    /// Build a fresh digest with caller-chosen compression `δ`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `compression` is
    /// non-finite or out of `[2, 10_000]`.
    pub fn new(compression: f64) -> RcfResult<Self> {
        if !compression.is_finite() || !(2.0..=10_000.0).contains(&compression) {
            return Err(RcfError::InvalidConfig(format!(
                "TDigest: compression must be finite in [2, 10000], got {compression}"
            )));
        }
        Ok(Self {
            compression,
            centroids: Vec::new(),
            buffer: Vec::new(),
            total_weight: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        })
    }

    /// Convenience: default compression ([`DEFAULT_COMPRESSION`] =
    /// `100`).
    #[must_use]
    pub fn with_default_compression() -> Self {
        // `DEFAULT_COMPRESSION` is in range by construction.
        Self {
            compression: DEFAULT_COMPRESSION,
            centroids: Vec::new(),
            buffer: Vec::new(),
            total_weight: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Compression parameter `δ`.
    #[must_use]
    pub fn compression(&self) -> f64 {
        self.compression
    }

    /// Total weight across every `record` call (including pending
    /// buffer entries).
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let pending = self.buffer.len() as f64;
        self.total_weight + pending
    }

    /// Number of centroids — bounded by `~2·compression` after a
    /// flush.
    #[must_use]
    pub fn centroid_count(&self) -> usize {
        self.centroids.len()
    }

    /// Observed running minimum. `None` when no values have been
    /// recorded yet.
    #[must_use]
    pub fn min(&self) -> Option<f64> {
        if self.min.is_finite() {
            Some(self.min)
        } else {
            None
        }
    }

    /// Observed running maximum. `None` when no values have been
    /// recorded yet.
    #[must_use]
    pub fn max(&self) -> Option<f64> {
        if self.max.is_finite() {
            Some(self.max)
        } else {
            None
        }
    }

    /// Fold a single observation into the digest. Non-finite
    /// values are silently ignored — the digest has no way to
    /// surface an error per-call and silently dropping matches
    /// [`crate::ScoreHistogram::record`] semantics.
    pub fn record(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        self.buffer.push(value);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let threshold = (self.compression as usize).saturating_mul(BUFFER_MULT);
        if self.buffer.len() >= threshold {
            self.flush_buffer();
        }
    }

    /// Force-flush the pending buffer. Callers normally don't need
    /// this — [`Self::quantile`] flushes transparently — but it
    /// helps bound memory in high-churn scenarios where quantiles
    /// are queried rarely.
    pub fn flush(&mut self) {
        self.flush_buffer();
    }

    /// Quantile `q` in `[0, 1]`. Returns `None` when the digest is
    /// empty; returns `min` at `q = 0` and `max` at `q = 1`.
    #[must_use]
    pub fn quantile(&mut self, q: f64) -> Option<f64> {
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return None;
        }
        self.flush_buffer();
        if self.centroids.is_empty() {
            return None;
        }
        if q <= 0.0 {
            return Some(self.min);
        }
        if q >= 1.0 {
            return Some(self.max);
        }
        let target = q * self.total_weight;

        // Walk centroids, tracking cumulative weight. Interpolate
        // between the two centroids that straddle `target`.
        let mut cum = 0.0_f64;
        let first = &self.centroids[0];
        // Left of the first centroid → interpolate between `min`
        // and first.mean.
        let first_center = first.weight / 2.0;
        if target < first_center {
            if first.weight <= 1.0 || first_center <= 0.0 {
                return Some(first.mean);
            }
            let frac = target / first_center;
            return Some(self.min + frac * (first.mean - self.min));
        }
        cum += first.weight;

        for i in 1..self.centroids.len() {
            let prev = &self.centroids[i - 1];
            let cur = &self.centroids[i];
            let prev_center = cum - prev.weight / 2.0;
            let cur_center = cum + cur.weight / 2.0;
            if target < cur_center {
                let span = cur_center - prev_center;
                if span <= 0.0 {
                    return Some(prev.mean);
                }
                let frac = (target - prev_center) / span;
                return Some(prev.mean + frac * (cur.mean - prev.mean));
            }
            cum += cur.weight;
        }

        // Right of the last centroid → interpolate toward `max`.
        let last = self.centroids.last()?;
        let last_center = self.total_weight - last.weight / 2.0;
        let span = self.total_weight - last_center;
        if span <= 0.0 {
            return Some(last.mean);
        }
        let frac = ((target - last_center) / span).clamp(0.0, 1.0);
        Some(last.mean + frac * (self.max - last.mean))
    }

    /// Percentile — shorthand for `quantile(p / 100.0)`.
    #[must_use]
    pub fn percentile(&mut self, p: f64) -> Option<f64> {
        self.quantile(p / 100.0)
    }

    /// Merge `other` into `self`. Both digests must share the same
    /// compression parameter; the merge preserves distributional
    /// accuracy by re-running the scale-function compression pass.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `other.compression`
    /// does not match `self.compression`.
    pub fn merge(&mut self, other: &Self) -> RcfResult<()> {
        #[allow(clippy::float_cmp)]
        let compat = self.compression == other.compression;
        if !compat {
            return Err(RcfError::InvalidConfig(format!(
                "TDigest merge: compression mismatch ({} vs {})",
                self.compression, other.compression
            )));
        }
        self.flush_buffer();
        // Fold other's centroids + buffer into self's buffer, then
        // flush — simplest path that round-trips through the same
        // scale-function compaction.
        for c in &other.centroids {
            // Expand centroid back to `weight` copies of its mean —
            // close-enough approximation because `mean` is the
            // centroid's summary value.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let n = c.weight.round() as usize;
            for _ in 0..n.max(1) {
                self.buffer.push(c.mean);
            }
        }
        for v in &other.buffer {
            self.buffer.push(*v);
        }
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
        self.flush_buffer();
        Ok(())
    }

    /// Drop every recorded value — digest goes back to its empty
    /// post-construction state. Compression is preserved.
    pub fn reset(&mut self) {
        self.centroids.clear();
        self.buffer.clear();
        self.total_weight = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
    }

    /// Immutable view of the current centroid set — exposed for
    /// diagnostics / persistence helpers. Empty until the first
    /// flush.
    #[must_use]
    pub fn centroids(&self) -> &[Centroid] {
        &self.centroids
    }

    /// Flush the buffer — sort buffer, merge with centroids using
    /// scale-function 1 weight bounds.
    fn flush_buffer(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        // Merge buffer + existing centroids into a sorted list
        // sorted ascending by mean.
        self.buffer
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

        // Combine buffer entries (unit weight) with prior centroids
        // in a single sorted merge pass.
        let mut combined: Vec<Centroid> =
            Vec::with_capacity(self.centroids.len() + self.buffer.len());
        let mut i = 0_usize;
        let mut j = 0_usize;
        while i < self.centroids.len() && j < self.buffer.len() {
            let c = self.centroids[i];
            let v = self.buffer[j];
            if c.mean <= v {
                combined.push(c);
                i += 1;
            } else {
                combined.push(Centroid {
                    mean: v,
                    weight: 1.0,
                });
                j += 1;
            }
        }
        while i < self.centroids.len() {
            combined.push(self.centroids[i]);
            i += 1;
        }
        while j < self.buffer.len() {
            combined.push(Centroid {
                mean: self.buffer[j],
                weight: 1.0,
            });
            j += 1;
        }
        self.buffer.clear();

        // Recompute total weight.
        let total: f64 = combined.iter().map(|c| c.weight).sum();
        self.total_weight = total;
        if total <= 0.0 {
            self.centroids = combined;
            return;
        }

        // Compact via scale function 1.
        let mut out: Vec<Centroid> = Vec::with_capacity(combined.len());
        let mut cum = 0.0_f64;
        let mut current = combined[0];
        cum += current.weight;
        for centroid in &combined[1..] {
            let q0 = (cum - current.weight) / total;
            let q1 = (cum + centroid.weight) / total;
            let q_limit = q_limit_for(q0, self.compression);
            // Merge into the current centroid when the resulting
            // centroid's cumulative-weight upper bound stays within
            // the quantile limit; else seal `current` and start a
            // new one.
            if q1 <= q_limit {
                let new_weight = current.weight + centroid.weight;
                current.mean =
                    (current.mean * current.weight + centroid.mean * centroid.weight) / new_weight;
                current.weight = new_weight;
            } else {
                out.push(current);
                current = *centroid;
            }
            cum += centroid.weight;
        }
        out.push(current);
        self.centroids = out;
    }
}

/// Scale function 1 upper bound:
/// `q_limit(q, δ) = (sin(2π · (k_1(q) + 1) / δ) + 1) / 2`, where
/// `k_1(q) = (δ / 2π) · asin(2q − 1)`. Returns `q + 1/δ` as a
/// fallback when the k-scale saturates.
fn q_limit_for(q: f64, compression: f64) -> f64 {
    use core::f64::consts::PI;
    let clamped = q.clamp(0.0, 1.0);
    let k = (compression / (2.0 * PI)) * (2.0 * clamped - 1.0).asin();
    let next = (2.0 * PI * (k + 1.0) / compression).sin();
    let limit = f64::midpoint(next, 1.0);
    if limit.is_finite() && limit > clamped {
        limit.min(1.0)
    } else {
        (clamped + 1.0 / compression).min(1.0)
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
    fn new_rejects_bad_compression() {
        assert!(TDigest::new(0.0).is_err());
        assert!(TDigest::new(1.0).is_err());
        assert!(TDigest::new(f64::NAN).is_err());
        assert!(TDigest::new(1.0e6).is_err());
    }

    #[test]
    fn empty_quantile_returns_none() {
        let mut d = TDigest::with_default_compression();
        assert!(d.quantile(0.5).is_none());
    }

    #[test]
    fn record_updates_min_max() {
        let mut d = TDigest::with_default_compression();
        d.record(5.0);
        d.record(2.0);
        d.record(8.0);
        assert_eq!(d.min(), Some(2.0));
        assert_eq!(d.max(), Some(8.0));
    }

    #[test]
    fn record_ignores_nan_and_inf() {
        let mut d = TDigest::with_default_compression();
        d.record(f64::NAN);
        d.record(f64::INFINITY);
        d.record(1.0);
        assert_eq!(d.total_weight(), 1.0);
    }

    #[test]
    fn median_of_uniform_stream() {
        let mut d = TDigest::with_default_compression();
        for i in 0..10_000 {
            d.record(i as f64);
        }
        let median = d.quantile(0.5).unwrap();
        // True median is 4999.5; t-digest target is ~1 % error.
        assert!((median - 4999.5).abs() < 150.0, "median = {median}");
    }

    #[test]
    fn tail_quantiles_accurate_on_uniform() {
        let mut d = TDigest::new(200.0).unwrap();
        for i in 0..10_000 {
            d.record(i as f64);
        }
        let p99 = d.quantile(0.99).unwrap();
        let p999 = d.quantile(0.999).unwrap();
        // True p99 = 9899, p99.9 = 9989. Allow < 1% absolute error
        // on the uniform [0, 9999] range.
        assert!((p99 - 9899.0).abs() < 100.0, "p99 = {p99}");
        assert!((p999 - 9989.0).abs() < 100.0, "p99.9 = {p999}");
    }

    #[test]
    fn percentile_is_quantile_over_100() {
        let mut d = TDigest::with_default_compression();
        for i in 0..1000 {
            d.record(i as f64);
        }
        let q50 = d.quantile(0.5).unwrap();
        let p50 = d.percentile(50.0).unwrap();
        assert_eq!(q50, p50);
    }

    #[test]
    fn quantile_0_returns_min_quantile_1_returns_max() {
        let mut d = TDigest::with_default_compression();
        for v in &[1.0, 2.0, 3.0, 100.0] {
            d.record(*v);
        }
        assert_eq!(d.quantile(0.0), Some(1.0));
        assert_eq!(d.quantile(1.0), Some(100.0));
    }

    #[test]
    fn merge_two_digests_preserves_quantiles() {
        let mut a = TDigest::new(200.0).unwrap();
        let mut b = TDigest::new(200.0).unwrap();
        for i in 0..5_000 {
            a.record(i as f64);
        }
        for i in 5_000..10_000 {
            b.record(i as f64);
        }
        a.merge(&b).unwrap();
        let median = a.quantile(0.5).unwrap();
        assert!((median - 4999.5).abs() < 200.0, "median = {median}");
        assert_eq!(a.min(), Some(0.0));
        assert_eq!(a.max(), Some(9999.0));
    }

    #[test]
    fn merge_rejects_compression_mismatch() {
        let mut a = TDigest::new(100.0).unwrap();
        let b = TDigest::new(200.0).unwrap();
        assert!(a.merge(&b).is_err());
    }

    #[test]
    fn reset_drops_state() {
        let mut d = TDigest::with_default_compression();
        for i in 0..100 {
            d.record(i as f64);
        }
        d.reset();
        assert_eq!(d.total_weight(), 0.0);
        assert!(d.min().is_none());
        assert!(d.max().is_none());
        assert!(d.quantile(0.5).is_none());
    }

    #[test]
    fn centroid_count_bounded_by_compression() {
        let mut d = TDigest::new(100.0).unwrap();
        for i in 0..50_000 {
            d.record(i as f64);
        }
        d.flush();
        // Scale-function-1 bound gives <= ~ 2·δ centroids post-
        // compaction. Allow a small slack for implementation
        // rounding.
        assert!(
            d.centroid_count() <= 250,
            "centroids = {}",
            d.centroid_count()
        );
    }

    #[test]
    fn quantile_rejects_out_of_range() {
        let mut d = TDigest::with_default_compression();
        d.record(1.0);
        assert!(d.quantile(-0.1).is_none());
        assert!(d.quantile(1.1).is_none());
        assert!(d.quantile(f64::NAN).is_none());
    }

    #[cfg(all(feature = "serde", feature = "postcard"))]
    #[test]
    fn postcard_roundtrip_preserves_quantiles() {
        let mut d = TDigest::new(200.0).unwrap();
        for i in 0..2_000 {
            d.record(i as f64);
        }
        d.flush();
        let bytes = postcard::to_allocvec(&d).unwrap();
        let mut back: TDigest = postcard::from_bytes(&bytes).unwrap();
        let before = d.quantile(0.9).unwrap();
        let after = back.quantile(0.9).unwrap();
        assert_eq!(before, after);
    }
}
