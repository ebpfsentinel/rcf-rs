//! Per-dimension attribution vector.
//!
//! [`DiVector`] tracks two `f64` accumulators per dimension: `high[d]`
//! holds the contribution to the score of cuts where the queried point
//! lies *above* the cut, `low[d]` of cuts where the point lies *below*
//! the cut. Summing `high[d] + low[d]` per dimension yields the total
//! per-feature contribution; the largest component identifies the most
//! anomalous dimension. The shape mirrors the AWS RCF reference and
//! Guha et al. (2016) attribution algorithm.

use crate::error::{RcfError, RcfResult};

/// Two-sided per-dimension attribution accumulator.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiVector {
    /// Contribution from cuts where the point lies above the cut.
    high: Vec<f64>,
    /// Contribution from cuts where the point lies below the cut.
    low: Vec<f64>,
}

impl DiVector {
    /// Build a zeroed `DiVector` with `dim` dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rcf_rs::domain::DiVector;
    /// let v = DiVector::zeros(4);
    /// assert_eq!(v.dim(), 4);
    /// assert_eq!(v.total(), 0.0);
    /// ```
    #[must_use]
    pub fn zeros(dim: usize) -> Self {
        Self {
            high: vec![0.0; dim],
            low: vec![0.0; dim],
        }
    }

    /// Dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.high.len()
    }

    /// Read-only view of the upper-side contributions.
    #[must_use]
    pub fn high(&self) -> &[f64] {
        &self.high
    }

    /// Read-only view of the lower-side contributions.
    #[must_use]
    pub fn low(&self) -> &[f64] {
        &self.low
    }

    /// Sum of all `high[d]` and `low[d]` entries.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.high.iter().sum::<f64>() + self.low.iter().sum::<f64>()
    }

    /// Per-dimension total: `high[d] + low[d]`.
    ///
    /// # Panics
    ///
    /// Panics when `d >= self.dim()` — call sites size-check first.
    #[must_use]
    pub fn per_dim_total(&self, d: usize) -> f64 {
        self.high[d] + self.low[d]
    }

    /// Index of the dimension with the largest `high[d] + low[d]`.
    /// Returns `None` for an empty vector.
    #[must_use]
    pub fn argmax(&self) -> Option<usize> {
        if self.dim() == 0 {
            return None;
        }
        let mut best = 0_usize;
        let mut best_val = self.per_dim_total(0);
        for d in 1..self.dim() {
            let v = self.per_dim_total(d);
            if v > best_val {
                best = d;
                best_val = v;
            }
        }
        Some(best)
    }

    /// Add `value` to the upper-side contribution for dimension `d`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::OutOfBounds`] when `d >= self.dim()`.
    pub fn add_high(&mut self, d: usize, value: f64) -> RcfResult<()> {
        if d >= self.high.len() {
            return Err(RcfError::OutOfBounds {
                index: d,
                len: self.high.len(),
            });
        }
        self.high[d] += value;
        Ok(())
    }

    /// Add `value` to the lower-side contribution for dimension `d`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::OutOfBounds`] when `d >= self.dim()`.
    pub fn add_low(&mut self, d: usize, value: f64) -> RcfResult<()> {
        if d >= self.low.len() {
            return Err(RcfError::OutOfBounds {
                index: d,
                len: self.low.len(),
            });
        }
        self.low[d] += value;
        Ok(())
    }

    /// Element-wise add `other` into `self`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::DimensionMismatch`] when dimensions differ.
    pub fn accumulate(&mut self, other: &Self) -> RcfResult<()> {
        if other.dim() != self.dim() {
            return Err(RcfError::DimensionMismatch {
                expected: self.dim(),
                got: other.dim(),
            });
        }
        for d in 0..self.dim() {
            self.high[d] += other.high[d];
            self.low[d] += other.low[d];
        }
        Ok(())
    }

    /// Divide every component by `divisor` in place. Used by the
    /// forest layer to convert a sum of per-tree attributions into a
    /// mean.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `divisor` is zero or
    /// non-finite.
    pub fn scale(&mut self, divisor: f64) -> RcfResult<()> {
        if divisor == 0.0 || !divisor.is_finite() {
            return Err(RcfError::InvalidConfig(format!(
                "DiVector::scale divisor must be non-zero and finite, got {divisor}"
            )));
        }
        for d in 0..self.dim() {
            self.high[d] /= divisor;
            self.low[d] /= divisor;
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert exact equality on integer-valued accumulations.
mod tests {
    use super::*;

    #[test]
    fn zeros_creates_dim_sized_vector() {
        let v = DiVector::zeros(5);
        assert_eq!(v.dim(), 5);
        assert_eq!(v.high(), &[0.0; 5]);
        assert_eq!(v.low(), &[0.0; 5]);
        assert_eq!(v.total(), 0.0);
    }

    #[test]
    fn add_high_and_low_accumulate() {
        let mut v = DiVector::zeros(3);
        v.add_high(0, 1.0).unwrap();
        v.add_high(0, 2.0).unwrap();
        v.add_low(2, 4.0).unwrap();
        assert_eq!(v.high(), &[3.0, 0.0, 0.0]);
        assert_eq!(v.low(), &[0.0, 0.0, 4.0]);
        assert_eq!(v.total(), 7.0);
        assert_eq!(v.per_dim_total(0), 3.0);
        assert_eq!(v.per_dim_total(2), 4.0);
    }

    #[test]
    fn add_high_oob() {
        let mut v = DiVector::zeros(2);
        let err = v.add_high(3, 1.0).unwrap_err();
        assert!(matches!(err, RcfError::OutOfBounds { index: 3, len: 2 }));
    }

    #[test]
    fn add_low_oob() {
        let mut v = DiVector::zeros(2);
        assert!(matches!(
            v.add_low(99, 1.0).unwrap_err(),
            RcfError::OutOfBounds { .. }
        ));
    }

    #[test]
    fn accumulate_sums_componentwise() {
        let mut a = DiVector::zeros(2);
        a.add_high(0, 1.0).unwrap();
        a.add_low(1, 2.0).unwrap();
        let mut b = DiVector::zeros(2);
        b.add_high(0, 4.0).unwrap();
        b.add_low(1, 8.0).unwrap();
        a.accumulate(&b).unwrap();
        assert_eq!(a.high(), &[5.0, 0.0]);
        assert_eq!(a.low(), &[0.0, 10.0]);
    }

    #[test]
    fn accumulate_rejects_dim_mismatch() {
        let mut a = DiVector::zeros(2);
        let b = DiVector::zeros(3);
        assert!(matches!(
            a.accumulate(&b).unwrap_err(),
            RcfError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn scale_divides_componentwise() {
        let mut v = DiVector::zeros(2);
        v.add_high(0, 10.0).unwrap();
        v.add_low(1, 6.0).unwrap();
        v.scale(2.0).unwrap();
        assert_eq!(v.high(), &[5.0, 0.0]);
        assert_eq!(v.low(), &[0.0, 3.0]);
    }

    #[test]
    fn scale_rejects_zero() {
        let mut v = DiVector::zeros(1);
        assert!(matches!(
            v.scale(0.0).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn scale_rejects_nan_infinity() {
        let mut v = DiVector::zeros(1);
        assert!(v.scale(f64::NAN).is_err());
        assert!(v.scale(f64::INFINITY).is_err());
    }

    #[test]
    fn argmax_picks_largest() {
        let mut v = DiVector::zeros(4);
        v.add_high(2, 5.0).unwrap();
        v.add_low(1, 1.0).unwrap();
        assert_eq!(v.argmax(), Some(2));
    }

    #[test]
    fn argmax_zero_dim_returns_none() {
        let v = DiVector::zeros(0);
        assert!(v.argmax().is_none());
    }

    #[test]
    fn argmax_ties_returns_first() {
        let mut v = DiVector::zeros(3);
        v.add_high(0, 5.0).unwrap();
        v.add_high(2, 5.0).unwrap();
        assert_eq!(v.argmax(), Some(0));
    }
}
