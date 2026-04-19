//! Imputation-like forensic baseline — answer "what would this
//! dim have looked like if the point were normal?" by aggregating
//! the per-dim distribution of the forest's currently-held sample
//! points.
//!
//! Inspired by AWS's `ImputeVisitor` but repurposed: instead of
//! imputing a `NaN` feature, this helper tells an SOC analyst how
//! far an observed point sits from the forest's current idea of
//! "normal" on every dimension — the *expected value under
//! normality* plus a z-score-style delta.
//!
//! # Semantics
//!
//! - `expected[d]` — mean of dim `d` across every point currently
//!   held in any tree's reservoir (the forest's live baseline).
//! - `stddev[d]` — population standard deviation of the same set.
//! - `observed[d]` — the caller's raw query value.
//! - `delta[d] = observed[d] − expected[d]`.
//! - `zscore[d] = delta[d] / stddev[d]` (clamped to `0` when the
//!   baseline stddev is zero on a dim — constant baseline means
//!   no meaningful z-score).
//! - `live_points` — number of unique points contributing to the
//!   baseline.
//!
//! The baseline is computed in raw-point space: `feature_scales`
//! is applied to the stored points for averaging then inverted so
//! `expected` / `stddev` / `delta` live in the caller's original
//! coordinate system. SOC dashboards don't need to know about the
//! internal scaling.

/// Per-dim forensic baseline comparing an observed point against
/// the forest's current live sample distribution.
///
/// Serialisable under the `serde` feature through the crate's
/// `fixed_array_f64` adapter — callers that persist alert records
/// for NIS2 / SOC2 audit trails can embed this struct directly.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ForensicBaseline<const D: usize> {
    /// Raw query point the baseline was computed against.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_util::fixed_array_f64"))]
    pub observed: [f64; D],
    /// Per-dim mean of the live reservoir points (in raw space).
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_util::fixed_array_f64"))]
    pub expected: [f64; D],
    /// Per-dim population stddev of the live reservoir points.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_util::fixed_array_f64"))]
    pub stddev: [f64; D],
    /// `observed − expected` per dim.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_util::fixed_array_f64"))]
    pub delta: [f64; D],
    /// Per-dim z-score: `delta / stddev`, `0` when stddev is zero.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_util::fixed_array_f64"))]
    pub zscore: [f64; D],
    /// Number of unique live points contributing to the baseline.
    pub live_points: usize,
}

impl<const D: usize> ForensicBaseline<D> {
    /// Index of the dimension with the largest `|zscore|` — the dim
    /// most out-of-family relative to the live baseline. Returns
    /// `None` on an empty forest (no live points) or when every
    /// z-score is exactly zero.
    #[must_use]
    pub fn argmax_abs_zscore(&self) -> Option<usize> {
        if D == 0 || self.live_points == 0 {
            return None;
        }
        let mut best: usize = 0;
        let mut best_val = self.zscore[0].abs();
        for d in 1..D {
            let v = self.zscore[d].abs();
            if v > best_val {
                best = d;
                best_val = v;
            }
        }
        if best_val == 0.0 { None } else { Some(best) }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn argmax_abs_zscore_picks_biggest() {
        let b = ForensicBaseline::<4> {
            observed: [0.0, 0.0, 0.0, 0.0],
            expected: [0.0, 0.0, 0.0, 0.0],
            stddev: [1.0, 1.0, 1.0, 1.0],
            delta: [0.1, -2.0, 0.5, 1.0],
            zscore: [0.1, -2.0, 0.5, 1.0],
            live_points: 16,
        };
        assert_eq!(b.argmax_abs_zscore(), Some(1));
    }

    #[test]
    fn argmax_abs_zscore_empty_when_no_live_points() {
        let b = ForensicBaseline::<2> {
            observed: [0.0, 0.0],
            expected: [0.0, 0.0],
            stddev: [0.0, 0.0],
            delta: [0.0, 0.0],
            zscore: [0.0, 0.0],
            live_points: 0,
        };
        assert!(b.argmax_abs_zscore().is_none());
    }

    #[test]
    fn argmax_abs_zscore_empty_when_all_zero() {
        let b = ForensicBaseline::<2> {
            observed: [0.0, 0.0],
            expected: [0.0, 0.0],
            stddev: [1.0, 1.0],
            delta: [0.0, 0.0],
            zscore: [0.0, 0.0],
            live_points: 4,
        };
        assert!(b.argmax_abs_zscore().is_none());
    }
}
