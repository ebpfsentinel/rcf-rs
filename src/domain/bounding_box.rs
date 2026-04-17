//! Axis-aligned bounding box for `d`-dimensional points.
//!
//! [`BoundingBox`] is a value object: every mutating operation
//! ([`extend`](BoundingBox::extend), [`merge_with`](BoundingBox::merge_with))
//! takes `&mut self` but the box is otherwise treated as a plain data
//! container with structural equality.
//!
//! The cut probability machinery follows Guha et al. (2016), §3:
//! the probability that a uniform random cut of the box augmented by
//! `point` would isolate `point` from the rest equals
//! `Σ_d Δ_d / total_range_after`, where `Δ_d` is the per-dimension
//! extension caused by including `point`. Both [`probability_of_cut`]
//! and [`per_dim_cut_probabilities`] return a tuple `(total, per_dim)`
//! so callers (e.g. the future `AttributionVisitor`) can reuse the
//! per-dim breakdown without recomputing it.
//!
//! [`probability_of_cut`]: BoundingBox::probability_of_cut
//! [`per_dim_cut_probabilities`]: BoundingBox::per_dim_cut_probabilities

use crate::domain::point::ensure_dim;
use crate::error::{RcfError, RcfResult};

/// Axis-aligned bounding box for `d`-dimensional points.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BoundingBox {
    /// Per-dimension lower corner.
    min: Vec<f64>,
    /// Per-dimension upper corner.
    max: Vec<f64>,
}

impl BoundingBox {
    /// Build a degenerate bounding box from a single point.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::EmptyBoundingBox`] when `point` is empty.
    pub fn from_point(point: &[f64]) -> RcfResult<Self> {
        if point.is_empty() {
            return Err(RcfError::EmptyBoundingBox);
        }
        Ok(Self {
            min: point.to_vec(),
            max: point.to_vec(),
        })
    }

    /// Dimensionality of the box.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.min.len()
    }

    /// Per-dimension lower corner.
    #[must_use]
    pub fn min(&self) -> &[f64] {
        &self.min
    }

    /// Per-dimension upper corner.
    #[must_use]
    pub fn max(&self) -> &[f64] {
        &self.max
    }

    /// Range (`max_d − min_d`) for dimension `d`.
    ///
    /// # Panics
    ///
    /// Panics when `d >= self.dim()` — call sites are internal and
    /// always size-checked.
    #[must_use]
    pub fn range_at(&self, d: usize) -> f64 {
        self.max[d] - self.min[d]
    }

    /// Sum of per-dimension ranges (`Σ_d (max_d − min_d)`).
    ///
    /// This is the denominator used by [`Cut::random_cut`] to pick a
    /// dimension weighted by its range.
    ///
    /// [`Cut::random_cut`]: crate::domain::Cut::random_cut
    #[must_use]
    pub fn range_sum(&self) -> f64 {
        let mut s = 0.0;
        for d in 0..self.dim() {
            s += self.range_at(d);
        }
        s
    }

    /// Extend the box in place to include `point`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::DimensionMismatch`] when `point.len() != self.dim()`.
    pub fn extend(&mut self, point: &[f64]) -> RcfResult<()> {
        ensure_dim(point, self.dim())?;
        for (d, &v) in point.iter().enumerate() {
            if v < self.min[d] {
                self.min[d] = v;
            }
            if v > self.max[d] {
                self.max[d] = v;
            }
        }
        Ok(())
    }

    /// Merge `other` into `self` in place.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::DimensionMismatch`] when dimensions differ.
    pub fn merge_with(&mut self, other: &Self) -> RcfResult<()> {
        if other.dim() != self.dim() {
            return Err(RcfError::DimensionMismatch {
                expected: self.dim(),
                got: other.dim(),
            });
        }
        for d in 0..self.dim() {
            if other.min[d] < self.min[d] {
                self.min[d] = other.min[d];
            }
            if other.max[d] > self.max[d] {
                self.max[d] = other.max[d];
            }
        }
        Ok(())
    }

    /// Return a new box equal to the union of `self` and `other`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::DimensionMismatch`] when dimensions differ.
    pub fn merged(&self, other: &Self) -> RcfResult<Self> {
        let mut out = self.clone();
        out.merge_with(other)?;
        Ok(out)
    }

    /// Per-dimension extension required to accommodate `point` —
    /// `Δ_d = max(0, point_d − max_d) + max(0, min_d − point_d)`.
    ///
    /// When `point` already lies inside the box every `Δ_d` is `0` and
    /// the cut probability is `0`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::DimensionMismatch`] when `point.len() != self.dim()`.
    pub fn extension_per_dim(&self, point: &[f64]) -> RcfResult<Vec<f64>> {
        ensure_dim(point, self.dim())?;
        let mut out = vec![0.0_f64; self.dim()];
        for d in 0..self.dim() {
            let above = point[d] - self.max[d];
            let below = self.min[d] - point[d];
            let mut delta = 0.0;
            if above > 0.0 {
                delta += above;
            }
            if below > 0.0 {
                delta += below;
            }
            out[d] = delta;
        }
        Ok(out)
    }

    /// Probability that a uniform random cut over the augmented box
    /// would isolate `point` from the original box.
    ///
    /// Returns `(total_probability, per_dim_contributions)` where the
    /// per-dim slice sums to `total_probability`. The per-dim slice is
    /// what the future `AttributionVisitor` uses to attribute
    /// per-feature score contributions.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::DimensionMismatch`] when `point.len() != self.dim()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rcf_rs::domain::BoundingBox;
    /// let bbox = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
    /// // [0,0] is already inside the (degenerate) box → no extension, p=0
    /// let (p, _) = bbox.probability_of_cut(&[0.0, 0.0]).unwrap();
    /// assert_eq!(p, 0.0);
    /// // (10, 0) extends only along dim 0 → p > 0, all mass on dim 0
    /// let (p, per_dim) = bbox.probability_of_cut(&[10.0, 0.0]).unwrap();
    /// assert!(p > 0.0);
    /// assert!(per_dim[0] > 0.0);
    /// assert_eq!(per_dim[1], 0.0);
    /// ```
    pub fn probability_of_cut(&self, point: &[f64]) -> RcfResult<(f64, Vec<f64>)> {
        let extension = self.extension_per_dim(point)?;
        let extension_sum: f64 = extension.iter().sum();
        let denom = self.range_sum() + extension_sum;
        if denom == 0.0 {
            // Degenerate box, point coincident — no cut isolates it.
            return Ok((0.0, vec![0.0; self.dim()]));
        }
        let mut per_dim = vec![0.0_f64; self.dim()];
        for d in 0..self.dim() {
            per_dim[d] = extension[d] / denom;
        }
        let total: f64 = per_dim.iter().sum();
        Ok((total, per_dim))
    }

    /// Convenience accessor returning only the per-dim contributions
    /// (ignores the total). Used by visitors that only care about the
    /// per-dim breakdown.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::DimensionMismatch`] when `point.len() != self.dim()`.
    pub fn per_dim_cut_probabilities(&self, point: &[f64]) -> RcfResult<Vec<f64>> {
        Ok(self.probability_of_cut(point)?.1)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert exact equality on bounding-box constants.
mod tests {
    use super::*;

    #[test]
    fn from_point_creates_degenerate_box() {
        let b = BoundingBox::from_point(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(b.dim(), 3);
        assert_eq!(b.min(), &[1.0, 2.0, 3.0]);
        assert_eq!(b.max(), &[1.0, 2.0, 3.0]);
        assert_eq!(b.range_sum(), 0.0);
    }

    #[test]
    fn from_point_rejects_empty() {
        assert!(matches!(
            BoundingBox::from_point(&[]).unwrap_err(),
            RcfError::EmptyBoundingBox
        ));
    }

    #[test]
    fn extend_grows_box() {
        let mut b = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        b.extend(&[3.0, -2.0]).unwrap();
        assert_eq!(b.min(), &[0.0, -2.0]);
        assert_eq!(b.max(), &[3.0, 0.0]);
        assert!((b.range_sum() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn extend_rejects_dim_mismatch() {
        let mut b = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        assert!(matches!(
            b.extend(&[1.0, 2.0, 3.0]).unwrap_err(),
            RcfError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn range_at_per_dim() {
        let mut b = BoundingBox::from_point(&[0.0, 0.0, 0.0]).unwrap();
        b.extend(&[2.0, 4.0, 8.0]).unwrap();
        assert_eq!(b.range_at(0), 2.0);
        assert_eq!(b.range_at(1), 4.0);
        assert_eq!(b.range_at(2), 8.0);
        assert_eq!(b.range_sum(), 14.0);
    }

    #[test]
    fn merge_with_unions_corners() {
        let mut a = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        a.extend(&[2.0, 2.0]).unwrap();
        let mut b = BoundingBox::from_point(&[-1.0, 1.0]).unwrap();
        b.extend(&[1.0, 5.0]).unwrap();
        a.merge_with(&b).unwrap();
        assert_eq!(a.min(), &[-1.0, 0.0]);
        assert_eq!(a.max(), &[2.0, 5.0]);
    }

    #[test]
    fn merged_returns_new_box() {
        let a = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        let b = BoundingBox::from_point(&[5.0, 5.0]).unwrap();
        let union = a.merged(&b).unwrap();
        assert_eq!(union.min(), &[0.0, 0.0]);
        assert_eq!(union.max(), &[5.0, 5.0]);
        // a and b unchanged
        assert_eq!(a.min(), &[0.0, 0.0]);
        assert_eq!(b.max(), &[5.0, 5.0]);
    }

    #[test]
    fn merge_with_rejects_dim_mismatch() {
        let mut a = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        let b = BoundingBox::from_point(&[0.0, 0.0, 0.0]).unwrap();
        assert!(matches!(
            a.merge_with(&b).unwrap_err(),
            RcfError::DimensionMismatch {
                expected: 2,
                got: 3
            }
        ));
    }

    #[test]
    fn extension_zero_when_point_inside() {
        let mut b = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        b.extend(&[10.0, 10.0]).unwrap();
        let ext = b.extension_per_dim(&[5.0, 5.0]).unwrap();
        assert_eq!(ext, vec![0.0, 0.0]);
    }

    #[test]
    fn extension_picks_above_and_below() {
        let mut b = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        b.extend(&[10.0, 10.0]).unwrap();
        // (-3, 15) → extends 3 below on dim 0, 5 above on dim 1
        let ext = b.extension_per_dim(&[-3.0, 15.0]).unwrap();
        assert_eq!(ext, vec![3.0, 5.0]);
    }

    #[test]
    fn probability_of_cut_zero_when_inside() {
        let mut b = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        b.extend(&[10.0, 10.0]).unwrap();
        let (p, per_dim) = b.probability_of_cut(&[5.0, 5.0]).unwrap();
        assert_eq!(p, 0.0);
        assert_eq!(per_dim, vec![0.0, 0.0]);
    }

    #[test]
    fn probability_of_cut_concentrated_on_extending_dim() {
        let mut b = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        b.extend(&[10.0, 10.0]).unwrap();
        // Far outlier on dim 0 only → most of the cut probability mass
        // should land on dim 0.
        let (total, per_dim) = b.probability_of_cut(&[1000.0, 5.0]).unwrap();
        assert!(per_dim[0] > per_dim[1]);
        assert!((per_dim[0] + per_dim[1] - total).abs() < 1e-12);
    }

    #[test]
    fn probability_of_cut_handles_degenerate_box() {
        // Box collapsed to a single point at origin; querying the same
        // point should produce zero (range_sum = 0 + extension = 0).
        let b = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        let (p, per_dim) = b.probability_of_cut(&[0.0, 0.0]).unwrap();
        assert_eq!(p, 0.0);
        assert_eq!(per_dim, vec![0.0, 0.0]);
    }

    #[test]
    fn probability_of_cut_per_dim_sums_to_total() {
        let mut b = BoundingBox::from_point(&[0.0, 0.0, 0.0]).unwrap();
        b.extend(&[1.0, 1.0, 1.0]).unwrap();
        let (total, per_dim) = b.probability_of_cut(&[5.0, -3.0, 0.5]).unwrap();
        let sum: f64 = per_dim.iter().sum();
        assert!((sum - total).abs() < 1e-12);
        // dim 0 (extends by 4) and dim 1 (extends by 3) carry the mass.
        assert!(per_dim[0] > 0.0);
        assert!(per_dim[1] > 0.0);
        assert_eq!(per_dim[2], 0.0);
    }

    #[test]
    fn probability_of_cut_rejects_dim_mismatch() {
        let b = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        assert!(matches!(
            b.probability_of_cut(&[1.0]).unwrap_err(),
            RcfError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn per_dim_cut_probabilities_matches_full_call() {
        let mut b = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        b.extend(&[1.0, 1.0]).unwrap();
        let (_, full) = b.probability_of_cut(&[5.0, -3.0]).unwrap();
        let only_per_dim = b.per_dim_cut_probabilities(&[5.0, -3.0]).unwrap();
        assert_eq!(full, only_per_dim);
    }
}
