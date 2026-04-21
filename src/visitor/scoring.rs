//! Shared scoring functions used by the production visitors.
//!
//! Mirrors Guha et al. (2016) §3 collusive-displacement scoring:
//!
//! ```text
//! score_seen(depth, mass)   = 1 / (depth + log2(mass))
//! score_unseen(depth, mass) = depth + log2(mass)
//! damp(mass, total_mass)    = 1 / (1 + ln(mass) / ln(total_mass))
//! normalizer(total_mass)    = log2(total_mass)
//! ```
//!
//! Each function clamps the degenerate `mass = 0` and `total_mass ≤ 1`
//! cases to a well-defined finite value (`0.0` for `score_*` and
//! `normalizer`, `1.0` for `damp`) so visitors never observe NaN or
//! infinity.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

/// `score_seen(depth, mass) = 1 / (depth + log2(mass))`, clamped to
/// `0.0` when the denominator is non-positive.
#[must_use]
pub fn score_seen(depth: usize, mass: u64) -> f64 {
    if mass == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let denom = depth as f64 + (mass as f64).log2();
    if denom > 0.0 { 1.0 / denom } else { 0.0 }
}

/// `score_unseen(depth, mass) = depth + log2(mass)`, clamped to
/// `0.0` when `mass == 0`.
#[must_use]
pub fn score_unseen(depth: usize, mass: u64) -> f64 {
    if mass == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let v = depth as f64 + (mass as f64).log2();
    v.max(0.0)
}

/// Damping factor `1 / (1 + ln(mass) / ln(total_mass))`. Returns
/// `1.0` when `total_mass <= 1` (single-leaf tree — no damping).
#[must_use]
pub fn damp(mass: u64, total_mass: u64) -> f64 {
    if total_mass <= 1 || mass == 0 {
        return 1.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let ln_total = (total_mass as f64).ln();
    if ln_total <= 0.0 {
        return 1.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let ratio = (mass as f64).ln() / ln_total;
    1.0 / (1.0 + ratio)
}

/// `normalizer(total_mass) = log2(total_mass)`, clamped to `0.0`
/// when `total_mass <= 1`.
#[must_use]
pub fn normalizer(total_mass: u64) -> f64 {
    if total_mass <= 1 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    {
        (total_mass as f64).log2()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests compare exact closed-form values.
mod tests {
    use super::*;

    #[test]
    fn score_seen_handles_zero_mass() {
        assert_eq!(score_seen(0, 0), 0.0);
        assert_eq!(score_seen(7, 0), 0.0);
    }

    #[test]
    fn score_seen_handles_degenerate_root() {
        // depth=0, mass=1 → log2(1)=0 → denom=0 → clamp.
        assert_eq!(score_seen(0, 1), 0.0);
    }

    #[test]
    fn score_seen_returns_inverse() {
        // depth=2, mass=4 → 2 + log2(4) = 4 → 1/4.
        assert!((score_seen(2, 4) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn score_unseen_handles_zero_mass() {
        assert_eq!(score_unseen(0, 0), 0.0);
    }

    #[test]
    fn score_unseen_returns_sum() {
        // depth=3, mass=8 → 3 + log2(8) = 3 + 3 = 6.
        assert!((score_unseen(3, 8) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn damp_clamps_for_total_one() {
        assert_eq!(damp(1, 1), 1.0);
        assert_eq!(damp(0, 5), 1.0);
        assert_eq!(damp(0, 0), 1.0);
    }

    #[test]
    fn damp_value_at_mass_equal_total() {
        // mass=4, total=4 → ln(4)/ln(4) = 1 → 1/(1+1) = 0.5.
        assert!((damp(4, 4) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn damp_value_for_small_mass() {
        // mass=2, total=4 → ln(2)/ln(4) = 0.5 → 1/1.5 ≈ 0.6667.
        let d = damp(2, 4);
        assert!((d - (1.0 / 1.5)).abs() < 1e-9);
    }

    #[test]
    fn normalizer_zero_for_small_total() {
        assert_eq!(normalizer(0), 0.0);
        assert_eq!(normalizer(1), 0.0);
    }

    #[test]
    fn normalizer_log2_otherwise() {
        assert!((normalizer(4) - 2.0).abs() < 1e-12);
        assert!((normalizer(256) - 8.0).abs() < 1e-12);
    }
}
