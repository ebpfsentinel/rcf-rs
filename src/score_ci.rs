//! Anomaly score with confidence interval.
//!
//! The bare [`crate::RandomCutForest::score`] returns a single
//! [`crate::AnomalyScore`] — the mean of per-tree scores. SOC
//! threshold tuning benefits from also knowing how tightly the
//! trees agree: a score of `2.1 ± 0.05` is qualitatively different
//! from `2.1 ± 0.8` even though both produce the same alert under
//! a fixed `> 1.5` threshold. [`ScoreWithConfidence`] packages the
//! mean plus a symmetric Gaussian CI derived from the per-tree
//! standard error (`sqrt(var / n)`).
//!
//! # Default confidence level
//!
//! The out-of-the-box factor is `z = 1.96` — the classical 95 %
//! normal-approximation CI. Callers that want a different level
//! (99 % → `z = 2.576`; 90 % → `z = 1.645`) can call
//! [`ScoreWithConfidence::ci`] with the desired `z`.
//!
//! # Statistical caveat
//!
//! Per-tree scores are IID under the RCF sampling contract, but
//! the Gaussian approximation leans on the CLT — at `num_trees ≤
//! 30` the intervals widen slightly vs. a bootstrap estimate.
//! Good enough for SOC tuning, not for publication-grade error
//! bars.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::domain::AnomalyScore;

/// Default `z` factor for a 95 % normal-approximation CI.
pub const DEFAULT_Z_FACTOR: f64 = 1.96;

/// Scored point plus the per-tree sample statistics behind it.
///
/// Serialisable under the `serde` feature so SIEM / audit sinks
/// can emit the full confidence context alongside the raw score.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScoreWithConfidence {
    /// Mean anomaly score — identical to
    /// [`crate::RandomCutForest::score`]'s output.
    pub score: AnomalyScore,
    /// Number of trees that contributed (= ensemble size minus any
    /// empty trees).
    pub trees_evaluated: usize,
    /// Unbiased sample standard deviation across per-tree scores.
    /// `0.0` when only one tree contributed.
    pub stddev: f64,
    /// Standard error of the mean: `stddev / sqrt(n)`. The width of
    /// the `z = 1.96` confidence interval is `2 · z · stderr`.
    pub stderr: f64,
}

impl ScoreWithConfidence {
    /// Symmetric confidence interval `(lower, upper)` at factor `z`.
    /// The interval is clamped at zero on the lower side — anomaly
    /// scores are non-negative by construction.
    #[must_use]
    pub fn ci(&self, z: f64) -> (f64, f64) {
        let mean = f64::from(self.score);
        let half = z * self.stderr;
        ((mean - half).max(0.0), mean + half)
    }

    /// 95 % CI — convenience wrapper around [`Self::ci`] with
    /// [`DEFAULT_Z_FACTOR`].
    #[must_use]
    pub fn ci95(&self) -> (f64, f64) {
        self.ci(DEFAULT_Z_FACTOR)
    }

    /// Relative stderr — `stderr / max(|mean|, ε)`. Mirrors the
    /// metric used by [`crate::RandomCutForest::score_early_term`]
    /// so callers can compare the two paths.
    #[must_use]
    pub fn relative_stderr(&self) -> f64 {
        let denom = f64::from(self.score).abs().max(f64::EPSILON);
        self.stderr / denom
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::float_cmp)]
mod tests {
    use super::*;

    fn mk(score: f64, n: usize, stddev: f64) -> ScoreWithConfidence {
        let stderr = if n > 0 {
            #[allow(clippy::cast_precision_loss)]
            let inv = (n as f64).sqrt();
            stddev / inv
        } else {
            0.0
        };
        ScoreWithConfidence {
            score: AnomalyScore::new(score).unwrap(),
            trees_evaluated: n,
            stddev,
            stderr,
        }
    }

    #[test]
    fn ci95_width_matches_1_96_stderr() {
        let s = mk(2.0, 100, 0.5);
        let (lo, hi) = s.ci95();
        let width = hi - lo;
        let expected = 2.0 * DEFAULT_Z_FACTOR * s.stderr;
        assert!((width - expected).abs() < 1e-9);
    }

    #[test]
    fn ci_lower_bound_clamps_to_zero() {
        // Mean 0.1, huge stderr → raw lower < 0; clamped to 0.
        let s = mk(0.1, 4, 5.0);
        let (lo, _) = s.ci95();
        assert!(lo >= 0.0);
    }

    #[test]
    fn relative_stderr_equals_stderr_over_mean() {
        let s = mk(2.0, 100, 0.5);
        let rel = s.relative_stderr();
        assert!((rel - s.stderr / 2.0).abs() < 1e-9);
    }

    #[test]
    fn custom_z_factor() {
        let s = mk(2.0, 100, 0.5);
        let (lo99, hi99) = s.ci(2.576); // 99% CI
        let (lo95, hi95) = s.ci95();
        // 99% CI wider than 95% CI.
        assert!(lo99 <= lo95);
        assert!(hi99 >= hi95);
    }
}
