//! Welford online mean + variance accumulator.
//!
//! Numerically stable single-pass variance (Welford 1962 / Knuth
//! TAOCP vol.2 §4.2.2) — updates `(mean, M2)` in place so the
//! returned sample variance is equivalent to a two-pass algorithm
//! without storing the stream.
//!
//! Shared by the normalizer, per-feature CUSUM / EWMA detectors,
//! and any downstream consumer that needs a cheap streaming
//! `(mean, std_dev)` summary. Keep the shape minimal — callers
//! layer warmup / decay / per-feature arrays on top.
//!
//! # References
//!
//! 1. B. P. Welford, "Note on a Method for Calculating Corrected
//!    Sums of Squares and Products", *Technometrics* 4(3), 1962.
//! 2. D. E. Knuth, *The Art of Computer Programming*, vol. 2,
//!    §4.2.2 "Statistical Calculations", 3rd ed., 1997.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

/// Online accumulator for a univariate stream.
///
/// Tracks sample count, running mean, and Welford `M2` accumulator
/// so both sample variance and standard deviation can be read at
/// `O(1)` without retaining the stream.
///
/// # Examples
///
/// ```
/// use rcf_rs::OnlineStats;
///
/// let mut s = OnlineStats::default();
/// for v in [10.0, 20.0, 30.0] {
///     s.update(v);
/// }
/// assert_eq!(s.count, 3);
/// assert!((s.mean - 20.0).abs() < 1e-12);
/// assert!((s.std_dev() - 10.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(clippy::cast_precision_loss)]
pub struct OnlineStats {
    /// Number of samples observed.
    pub count: u64,
    /// Running sample mean.
    pub mean: f64,
    /// Welford sum-of-squared-deviations accumulator.
    m2: f64,
}

impl OnlineStats {
    /// Construct an empty accumulator. Equivalent to
    /// [`Default::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest a new sample — `O(1)` update to mean and `M2`.
    #[allow(clippy::cast_precision_loss)]
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Sample variance (Bessel-corrected, divides by `count − 1`).
    /// Returns `0.0` when fewer than two samples have been seen.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    /// Sample standard deviation — `sqrt(variance())`.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    #[test]
    fn new_and_default_match() {
        let a = OnlineStats::new();
        let b = OnlineStats::default();
        assert_eq!(a.count, b.count);
        assert_eq!(a.mean, b.mean);
        assert_eq!(a.variance(), b.variance());
    }

    #[test]
    fn empty_has_zero_variance() {
        let s = OnlineStats::default();
        assert_eq!(s.count, 0);
        assert_eq!(s.mean, 0.0);
        assert_eq!(s.variance(), 0.0);
        assert_eq!(s.std_dev(), 0.0);
    }

    #[test]
    fn single_sample_has_zero_variance() {
        let mut s = OnlineStats::default();
        s.update(42.0);
        assert_eq!(s.count, 1);
        assert!((s.mean - 42.0).abs() < f64::EPSILON);
        assert_eq!(s.variance(), 0.0);
    }

    #[test]
    fn three_sample_known_mean_and_std() {
        // Reference: mean([10, 20, 30]) = 20, stddev (Bessel) = 10.
        let mut s = OnlineStats::default();
        for v in [10.0, 20.0, 30.0] {
            s.update(v);
        }
        assert_eq!(s.count, 3);
        assert!((s.mean - 20.0).abs() < 1e-12);
        assert!((s.variance() - 100.0).abs() < 1e-9);
        assert!((s.std_dev() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn matches_two_pass_on_uniform_sequence() {
        use alloc::vec::Vec;

        // Parity check against naive two-pass variance on a
        // 1000-sample deterministic sequence.
        let xs: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.5).collect();
        let mut s = OnlineStats::default();
        for &x in &xs {
            s.update(x);
        }

        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);

        assert!((s.mean - mean).abs() < 1e-9);
        assert!((s.variance() - var).abs() < 1e-6);
    }

    #[test]
    fn clone_preserves_state() {
        let mut s = OnlineStats::default();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            s.update(v);
        }
        let c = s.clone();
        assert_eq!(c.count, s.count);
        assert_eq!(c.mean, s.mean);
        assert_eq!(c.variance(), s.variance());
    }

    #[cfg(all(feature = "serde", feature = "postcard"))]
    #[test]
    fn postcard_roundtrip_preserves_variance() {
        let mut s = OnlineStats::default();
        for v in [10.0, 20.0, 30.0] {
            s.update(v);
        }
        let bytes = postcard::to_allocvec(&s).expect("serde ok");
        let back: OnlineStats = postcard::from_bytes(&bytes).expect("serde ok");
        assert_eq!(back.count, s.count);
        assert!((back.mean - s.mean).abs() < f64::EPSILON);
        assert!((back.variance() - s.variance()).abs() < f64::EPSILON);
    }
}
