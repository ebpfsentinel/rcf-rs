//! `NaN`-safe newtype around `f64` for anomaly scores.
//!
//! [`AnomalyScore`] enforces three invariants at construction time:
//!
//! 1. The wrapped value is finite (no `NaN`, no `±∞`).
//! 2. The wrapped value is non-negative — RCF scores per Guha et al.
//!    (2016) are sums of non-negative terms.
//! 3. The wrapped value can be totally ordered without surprise (the
//!    [`Ord`] impl is sound thanks to the `NaN` rejection).

use core::cmp::Ordering;
use core::fmt;

use crate::error::{RcfError, RcfResult};

/// `NaN`-safe non-negative anomaly score.
#[derive(Debug, Clone, Copy)]
pub struct AnomalyScore(f64);

impl AnomalyScore {
    /// Wrap a raw `f64`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::NaNValue`] when the value is `NaN`, `±∞`,
    /// or strictly negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use rcf_rs::AnomalyScore;
    /// assert!(AnomalyScore::new(0.0).is_ok());
    /// assert!(AnomalyScore::new(2.5).is_ok());
    /// assert!(AnomalyScore::new(-0.1).is_err());
    /// assert!(AnomalyScore::new(f64::NAN).is_err());
    /// ```
    pub fn new(value: f64) -> RcfResult<Self> {
        if !value.is_finite() || value < 0.0 {
            return Err(RcfError::NaNValue);
        }
        Ok(Self(value))
    }

    /// Return the wrapped value.
    #[must_use]
    pub fn into_inner(self) -> f64 {
        self.0
    }
}

impl PartialEq for AnomalyScore {
    fn eq(&self, other: &Self) -> bool {
        // Safe: we forbid NaN at construction time so == is total.
        self.0 == other.0
    }
}

impl Eq for AnomalyScore {}

impl PartialOrd for AnomalyScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AnomalyScore {
    fn cmp(&self, other: &Self) -> Ordering {
        // Safe: NaN rejection at construction time guarantees a total
        // order on the underlying f64.
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

impl fmt::Display for AnomalyScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.0)
    }
}

impl From<AnomalyScore> for f64 {
    fn from(score: AnomalyScore) -> Self {
        score.0
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert exact equality on integer-valued scores.
mod tests {
    use super::*;

    #[test]
    fn accepts_zero() {
        let s = AnomalyScore::new(0.0).unwrap();
        assert_eq!(s.into_inner(), 0.0);
    }

    #[test]
    fn accepts_positive() {
        let s = AnomalyScore::new(1.25).unwrap();
        assert_eq!(s.into_inner(), 1.25);
    }

    #[test]
    fn rejects_negative() {
        assert!(matches!(
            AnomalyScore::new(-0.0001).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn rejects_nan() {
        assert!(AnomalyScore::new(f64::NAN).is_err());
    }

    #[test]
    fn rejects_infinity() {
        assert!(AnomalyScore::new(f64::INFINITY).is_err());
        assert!(AnomalyScore::new(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn ord_total() {
        let a = AnomalyScore::new(0.0).unwrap();
        let b = AnomalyScore::new(1.0).unwrap();
        let c = AnomalyScore::new(1.0).unwrap();
        assert!(a < b);
        assert_eq!(b, c);
        assert!(b >= c);
    }

    #[test]
    fn display_uses_six_decimals() {
        let s = AnomalyScore::new(1.5).unwrap();
        assert_eq!(format!("{s}"), "1.500000");
    }

    #[test]
    fn into_f64_roundtrip() {
        let original = 0.42;
        let s = AnomalyScore::new(original).unwrap();
        let raw: f64 = s.into();
        assert_eq!(raw, original);
    }
}
