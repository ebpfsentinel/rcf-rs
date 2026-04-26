//! Per-feature exponentially-weighted moving average detector.
//!
//! `D` parallel univariate EWMA accumulators track per-dim mean +
//! variance with geometric decay `α`. After a warmup budget,
//! `observe` reports per-feature z-scores and the max across dims
//! — policy-free, so caller maps max-z into alert severity via
//! [`crate::SeverityBands`] or a custom rule.
//!
//! Scoring is applied *before* the accumulator update so the
//! current observation is judged against the prior distribution
//! (textbook EWCD / EWMA-Z convention) — without this, a large
//! step would fold into the mean in the same tick and miss the
//! alert.
//!
//! # EWMA recurrence
//!
//! ```text
//! mean_t     ← α·x_t + (1 − α)·mean_{t−1}
//! variance_t ← α·(x_t − mean_{t−1})² + (1 − α)·variance_{t−1}
//! ```
//!
//! First observation seeds `mean_0 = x_0`, `variance_0 = 0`.
//!
//! # References
//!
//! 1. J. S. Hunter, "The Exponentially Weighted Moving Average",
//!    *Journal of Quality Technology* 18(4), 1986.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

/// One univariate EWMA accumulator — public so callers can read
/// the evolving `(mean, variance, count)` for telemetry.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EwmaAccumulator {
    /// Exponentially-weighted running mean.
    pub mean: f64,
    /// Exponentially-weighted running variance.
    pub variance: f64,
    /// Samples observed by this accumulator.
    pub count: u64,
}

impl EwmaAccumulator {
    /// Fresh accumulator — zeroed.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            mean: 0.0,
            variance: 0.0,
            count: 0,
        }
    }

    /// Ingest `value` under decay factor `alpha`.
    #[inline]
    pub fn update(&mut self, value: f64, alpha: f64) {
        if self.count == 0 {
            self.mean = value;
            self.variance = 0.0;
        } else {
            let diff = value - self.mean;
            self.mean = alpha.mul_add(value, (1.0 - alpha) * self.mean);
            self.variance = alpha.mul_add(diff * diff, (1.0 - alpha) * self.variance);
        }
        self.count += 1;
    }

    /// `|value − mean| / sqrt(variance)`. Returns `f64::MAX`
    /// when variance is effectively zero and the observation
    /// diverges from the mean (perfectly stable baseline broken).
    #[inline]
    #[must_use]
    pub fn z_score(&self, value: f64) -> f64 {
        let diff = (value - self.mean).abs();
        let std_dev = self.variance.sqrt();
        if std_dev < f64::EPSILON {
            if diff < f64::EPSILON { 0.0 } else { f64::MAX }
        } else {
            diff / std_dev
        }
    }

    /// Reset to the zero state.
    #[inline]
    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.variance = 0.0;
        self.count = 0;
    }
}

impl Default for EwmaAccumulator {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Hyper-parameters for [`PerFeatureEwma`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PerFeatureEwmaConfig {
    /// Decay factor; smaller = heavier weight on history. Typical
    /// `[0.01, 0.3]`.
    pub alpha: f64,
    /// Samples required before [`PerFeatureEwma::observe`] starts
    /// returning `Some(_)`. Scoring before warmup is unstable.
    pub warmup_samples: u32,
}

impl Default for PerFeatureEwmaConfig {
    fn default() -> Self {
        Self {
            alpha: 0.01,
            warmup_samples: 100,
        }
    }
}

/// Result of one observation after warmup.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PerFeatureEwmaResult<const D: usize> {
    /// `max(per_feature_z)` — single-number summary so a caller
    /// can classify severity without scanning the full array.
    pub max_z: f64,
    /// Per-dimension z-scores; index matches the input vector.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_util::fixed_array_f64"))]
    pub per_feature_z: [f64; D],
}

/// `D` parallel EWMA accumulators with shared warmup budget.
///
/// # Examples
///
/// ```
/// use anomstream_core::{PerFeatureEwma, PerFeatureEwmaConfig};
///
/// let mut ewma: PerFeatureEwma<3> = PerFeatureEwma::new(PerFeatureEwmaConfig {
///     alpha: 0.1,
///     warmup_samples: 5,
/// });
/// for _ in 0..10 {
///     ewma.observe(&[10.0, 20.0, 30.0]);
/// }
/// let out = ewma.observe(&[10.0, 20.0, 30.0]).expect("warmed");
/// assert!(out.max_z.is_finite());
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PerFeatureEwma<const D: usize> {
    /// Per-dimension accumulator state.
    #[cfg_attr(feature = "serde", serde(with = "serde_accumulators"))]
    accumulators: [EwmaAccumulator; D],
    /// Active configuration.
    config: PerFeatureEwmaConfig,
    /// Observations ingested so far.
    total_samples: u64,
}

#[cfg(feature = "serde")]
mod serde_accumulators {
    //! `serde` adapter for `[EwmaAccumulator; D]` — the derive
    //! macro does not cover arbitrary-`D` arrays, so round-trip
    //! through a length-prefixed slice.
    use super::EwmaAccumulator;
    use alloc::vec::Vec;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Serialize `[EwmaAccumulator; D]` as a length-prefixed slice.
    pub fn serialize<S: Serializer, const D: usize>(
        accs: &[EwmaAccumulator; D],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        accs.as_slice().serialize(s)
    }

    /// Deserialize a length-prefixed slice back into `[EwmaAccumulator; D]`.
    pub fn deserialize<'de, DSer: Deserializer<'de>, const D: usize>(
        d: DSer,
    ) -> Result<[EwmaAccumulator; D], DSer::Error> {
        let v: Vec<EwmaAccumulator> = Vec::deserialize(d)?;
        if v.len() != D {
            return Err(serde::de::Error::invalid_length(
                v.len(),
                &"expected D accumulators",
            ));
        }
        let mut out = [EwmaAccumulator::new(); D];
        for (slot, acc) in out.iter_mut().zip(v) {
            *slot = acc;
        }
        Ok(out)
    }
}

impl<const D: usize> PerFeatureEwma<D> {
    /// Build an empty detector.
    #[inline]
    #[must_use]
    pub const fn new(config: PerFeatureEwmaConfig) -> Self {
        Self {
            accumulators: [EwmaAccumulator::new(); D],
            config,
            total_samples: 0,
        }
    }

    /// Active configuration.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &PerFeatureEwmaConfig {
        &self.config
    }

    /// Observations ingested so far.
    #[inline]
    #[must_use]
    pub const fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// `true` once `total_samples ≥ warmup_samples`.
    #[inline]
    #[must_use]
    pub const fn is_warmed_up(&self) -> bool {
        self.total_samples >= self.config.warmup_samples as u64
    }

    /// Per-dimension accumulator snapshot (useful for status
    /// endpoints and debugging; callers should not mutate in
    /// place — use [`Self::reset`] instead).
    #[inline]
    #[must_use]
    pub const fn accumulators(&self) -> &[EwmaAccumulator; D] {
        &self.accumulators
    }

    /// Ingest `input`, returning the per-feature z-scores *iff*
    /// warmup is complete. Accumulators are always updated so a
    /// disabled caller can still warm the detector.
    #[inline]
    #[must_use = "detector output should be checked — dropping it silently usually indicates a logic bug"]
    pub fn observe(&mut self, input: &[f64; D]) -> Option<PerFeatureEwmaResult<D>> {
        let result = if self.is_warmed_up() {
            let mut per_feature_z = [0.0_f64; D];
            for (i, &value) in input.iter().enumerate() {
                per_feature_z[i] = self.accumulators[i].z_score(value);
            }
            let max_z = per_feature_z.iter().copied().fold(0.0_f64, f64::max);
            Some(PerFeatureEwmaResult {
                max_z,
                per_feature_z,
            })
        } else {
            None
        };

        for (i, &value) in input.iter().enumerate() {
            self.accumulators[i].update(value, self.config.alpha);
        }
        self.total_samples += 1;

        result
    }

    /// Zero every accumulator and the sample counter.
    #[inline]
    pub fn reset(&mut self) {
        for acc in &mut self.accumulators {
            acc.reset();
        }
        self.total_samples = 0;
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn warmup_yields_none() {
        let mut ewma = PerFeatureEwma::<4>::new(PerFeatureEwmaConfig {
            alpha: 0.01,
            warmup_samples: 10,
        });
        for _ in 0..9 {
            assert!(ewma.observe(&[1.0, 2.0, 3.0, 4.0]).is_none());
        }
        assert!(!ewma.is_warmed_up());
    }

    #[test]
    fn after_warmup_returns_some() {
        let mut ewma = PerFeatureEwma::<2>::new(PerFeatureEwmaConfig {
            alpha: 0.1,
            warmup_samples: 5,
        });
        for _ in 0..10 {
            let _ = ewma.observe(&[1.0, 2.0]);
        }
        assert!(ewma.is_warmed_up());
        let out = ewma.observe(&[1.0, 2.0]).expect("warmed");
        assert!(out.max_z.is_finite());
    }

    #[test]
    fn zero_variance_stable_matches_mean() {
        // Same value forever → variance ≈ 0 → re-observing same
        // value yields z=0 (not MAX) because diff < EPSILON.
        let mut ewma = PerFeatureEwma::<1>::new(PerFeatureEwmaConfig {
            alpha: 0.1,
            warmup_samples: 5,
        });
        for _ in 0..20 {
            let _ = ewma.observe(&[7.0]);
        }
        let out = ewma.observe(&[7.0]).expect("warmed");
        assert_eq!(out.per_feature_z[0], 0.0);
    }

    #[test]
    fn zero_variance_spike_yields_max() {
        // Same value forever then a spike → variance ≈ 0, diff ≠ 0
        // → z = f64::MAX.
        let mut ewma = PerFeatureEwma::<1>::new(PerFeatureEwmaConfig {
            alpha: 0.01,
            warmup_samples: 5,
        });
        for _ in 0..50 {
            let _ = ewma.observe(&[7.0]);
        }
        let out = ewma.observe(&[1000.0]).expect("warmed");
        assert_eq!(out.per_feature_z[0], f64::MAX);
        assert_eq!(out.max_z, f64::MAX);
    }

    #[test]
    fn max_z_picks_largest_feature() {
        // Only dim 1 spikes — max_z must equal that dim's z.
        let mut ewma = PerFeatureEwma::<3>::new(PerFeatureEwmaConfig {
            alpha: 0.1,
            warmup_samples: 5,
        });
        // Warmup with non-zero variance on every dim so z is finite.
        for i in 0..30 {
            let v = 10.0 + f64::from(i % 3) - 1.0;
            let _ = ewma.observe(&[v, v, v]);
        }
        let mut probe = [10.0_f64; 3];
        probe[1] = 1000.0;
        let out = ewma.observe(&probe).expect("warmed");
        assert_eq!(out.max_z, out.per_feature_z[1]);
        assert!(out.per_feature_z[1] > out.per_feature_z[0]);
        assert!(out.per_feature_z[1] > out.per_feature_z[2]);
    }

    #[test]
    fn alpha_sensitivity_fast_vs_slow() {
        let mut fast = PerFeatureEwma::<1>::new(PerFeatureEwmaConfig {
            alpha: 0.5,
            warmup_samples: 5,
        });
        let mut slow = PerFeatureEwma::<1>::new(PerFeatureEwmaConfig {
            alpha: 0.01,
            warmup_samples: 5,
        });
        for _ in 0..20 {
            let _ = fast.observe(&[10.0]);
            let _ = slow.observe(&[10.0]);
        }
        for _ in 0..10 {
            let _ = fast.observe(&[20.0]);
            let _ = slow.observe(&[20.0]);
        }
        let fast_out = fast.observe(&[20.0]).expect("warmed");
        let slow_out = slow.observe(&[20.0]).expect("warmed");
        assert!(fast_out.max_z <= slow_out.max_z);
    }

    #[test]
    fn reset_clears_state() {
        let mut ewma = PerFeatureEwma::<2>::new(PerFeatureEwmaConfig {
            alpha: 0.1,
            warmup_samples: 5,
        });
        for _ in 0..20 {
            let _ = ewma.observe(&[10.0, 20.0]);
        }
        assert!(ewma.is_warmed_up());
        ewma.reset();
        assert_eq!(ewma.total_samples(), 0);
        assert!(!ewma.is_warmed_up());
        for acc in ewma.accumulators() {
            assert_eq!(acc.count, 0);
            assert_eq!(acc.mean, 0.0);
            assert_eq!(acc.variance, 0.0);
        }
    }

    #[test]
    fn first_observation_seeds_mean() {
        let mut acc = EwmaAccumulator::new();
        acc.update(42.0, 0.1);
        assert_eq!(acc.mean, 42.0);
        assert_eq!(acc.variance, 0.0);
        assert_eq!(acc.count, 1);
    }

    #[cfg(all(feature = "serde", feature = "postcard"))]
    #[test]
    fn postcard_roundtrip_preserves_state() {
        let mut ewma = PerFeatureEwma::<3>::new(PerFeatureEwmaConfig {
            alpha: 0.1,
            warmup_samples: 5,
        });
        for i in 0..30 {
            let v = 10.0 + f64::from(i % 5);
            let _ = ewma.observe(&[v, v * 2.0, v * 3.0]);
        }
        let bytes = postcard::to_allocvec(&ewma).expect("serde ok");
        let mut back: PerFeatureEwma<3> = postcard::from_bytes(&bytes).expect("serde ok");
        let probe = [12.0, 24.0, 36.0];
        let before = ewma.observe(&probe).expect("warmed");
        let after = back.observe(&probe).expect("warmed");
        assert_eq!(before.max_z, after.max_z);
    }
}
