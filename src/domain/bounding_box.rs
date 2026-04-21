//! Axis-aligned bounding box for `D`-dimensional points.
//!
//! [`BoundingBox<D>`] is a value object: every mutating operation
//! ([`extend`](BoundingBox::extend), [`merge_with`](BoundingBox::merge_with))
//! takes `&mut self` but the box is otherwise treated as a plain data
//! container with structural equality.
//!
//! Storage is a stack-allocated `[f64; D]` so the compiler can unroll
//! per-dim loops, vectorise via SIMD, and avoid all heap traffic. The
//! AWS-default `feature_dim = 16` is the canonical instantiation.
//!
//! The cut probability machinery follows Guha et al. (2016), §3:
//! the probability that a uniform random cut of the box augmented by
//! `point` would isolate `point` from the rest equals
//! `Σ_d Δ_d / total_range_after`, where `Δ_d` is the per-dimension
//! extension caused by including `point`. Both [`probability_of_cut`]
//! and [`per_dim_cut_probabilities`] return a tuple `(total, per_dim)`
//! so callers (e.g. the future `AttributionVisitor`) can reuse the
//! per-dim breakdown for attribution without recomputing it.
//!
//! [`probability_of_cut`]: BoundingBox::probability_of_cut
//! [`per_dim_cut_probabilities`]: BoundingBox::per_dim_cut_probabilities

use wide::f64x4;

use crate::domain::cut::Cut;
use crate::error::{RcfError, RcfResult};

/// Axis-aligned bounding box for `D`-dimensional points. Storage is
/// stack-allocated `[f64; D]` so the compiler can unroll the
/// per-dim loops, vectorise via SIMD, and avoid any heap traffic.
///
/// # Examples
///
/// ```
/// use rcf_rs::BoundingBox;
///
/// let mut bbox = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
/// bbox.extend(&[3.0, 4.0]).unwrap();
/// assert_eq!(bbox.range_sum(), 7.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BoundingBox<const D: usize> {
    /// Per-dimension lower corner. Serialised through
    /// [`crate::serde_util::fixed_array_f64`] because `serde` does
    /// not yet ship `Deserialize` for `[T; N]` at arbitrary `N`.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_util::fixed_array_f64"))]
    min: [f64; D],
    /// Per-dimension upper corner.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_util::fixed_array_f64"))]
    max: [f64; D],
}

impl<const D: usize> BoundingBox<D> {
    /// Build a degenerate bounding box from a single point.
    ///
    /// # Errors
    ///
    /// - [`RcfError::EmptyBoundingBox`] when `D == 0`.
    /// - [`RcfError::DimensionMismatch`] when `point.len() != D`.
    pub fn from_point(point: &[f64]) -> RcfResult<Self> {
        if D == 0 {
            return Err(RcfError::EmptyBoundingBox);
        }
        if point.len() != D {
            return Err(RcfError::DimensionMismatch {
                expected: D,
                got: point.len(),
            });
        }
        let mut min = [0.0_f64; D];
        let mut max = [0.0_f64; D];
        min.copy_from_slice(point);
        max.copy_from_slice(point);
        Ok(Self { min, max })
    }

    /// Dimensionality of the box (compile-time constant `D`).
    #[must_use]
    #[inline]
    pub const fn dim(&self) -> usize {
        D
    }

    /// Per-dimension lower corner.
    #[must_use]
    #[inline]
    pub fn min(&self) -> &[f64; D] {
        &self.min
    }

    /// Per-dimension upper corner.
    #[must_use]
    #[inline]
    pub fn max(&self) -> &[f64; D] {
        &self.max
    }

    /// Range (`max_d − min_d`) for dimension `d`.
    ///
    /// # Panics
    ///
    /// Panics when `d >= D` — call sites are internal and always
    /// size-checked.
    #[must_use]
    #[inline]
    pub fn range_at(&self, d: usize) -> f64 {
        self.max[d] - self.min[d]
    }

    /// Sum of per-dimension ranges (`Σ_d (max_d − min_d)`).
    ///
    /// This is the denominator used by [`Cut::random_cut`] to pick a
    /// dimension weighted by its range. Vectorised in 4-lane f64
    /// chunks via [`wide::f64x4`] for the AWS-default `D = 16` hot
    /// path; a scalar tail handles dims that are not a multiple of 4.
    ///
    /// [`Cut::random_cut`]: crate::domain::Cut::random_cut
    #[must_use]
    #[inline]
    pub fn range_sum(&self) -> f64 {
        let chunks = D / 4;
        let mut acc_simd = f64x4::splat(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let mn = f64x4::from([
                self.min[off],
                self.min[off + 1],
                self.min[off + 2],
                self.min[off + 3],
            ]);
            let mx = f64x4::from([
                self.max[off],
                self.max[off + 1],
                self.max[off + 2],
                self.max[off + 3],
            ]);
            acc_simd += mx - mn;
        }
        let mut s = acc_simd.reduce_add();
        for d in (chunks * 4)..D {
            s += self.max[d] - self.min[d];
        }
        s
    }

    /// Extend the box in place to include `point`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::DimensionMismatch`] when `point.len() != D`.
    pub fn extend(&mut self, point: &[f64]) -> RcfResult<()> {
        if point.len() != D {
            return Err(RcfError::DimensionMismatch {
                expected: D,
                got: point.len(),
            });
        }
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

    /// Merge `other` into `self` in place. Both boxes have the same
    /// type-level dimensionality so this is infallible.
    pub fn merge_with(&mut self, other: &Self) {
        for d in 0..D {
            if other.min[d] < self.min[d] {
                self.min[d] = other.min[d];
            }
            if other.max[d] > self.max[d] {
                self.max[d] = other.max[d];
            }
        }
    }

    /// Return a new box equal to the union of `self` and `other`.
    #[must_use]
    pub fn merged(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.merge_with(other);
        out
    }

    /// Per-dimension extension required to accommodate `point` —
    /// `Δ_d = max(0, point_d − max_d) + max(0, min_d − point_d)`.
    ///
    /// When `point` already lies inside the box every `Δ_d` is `0` and
    /// the cut probability is `0`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::DimensionMismatch`] when `point.len() != D`.
    pub fn extension_per_dim(&self, point: &[f64]) -> RcfResult<[f64; D]> {
        if point.len() != D {
            return Err(RcfError::DimensionMismatch {
                expected: D,
                got: point.len(),
            });
        }
        let mut out = [0.0_f64; D];
        for d in 0..D {
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
    /// per-dim array sums to `total_probability`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::DimensionMismatch`] when `point.len() != D`.
    pub fn probability_of_cut(&self, point: &[f64]) -> RcfResult<(f64, [f64; D])> {
        let extension = self.extension_per_dim(point)?;
        let extension_sum: f64 = extension.iter().sum();
        let denom = self.range_sum() + extension_sum;
        if denom == 0.0 {
            return Ok((0.0, [0.0; D]));
        }
        let mut per_dim = [0.0_f64; D];
        for d in 0..D {
            per_dim[d] = extension[d] / denom;
        }
        let total: f64 = per_dim.iter().sum();
        Ok((total, per_dim))
    }

    /// Convenience accessor returning only the per-dim contributions
    /// (ignores the total).
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::DimensionMismatch`] when `point.len() != D`.
    pub fn per_dim_cut_probabilities(&self, point: &[f64]) -> RcfResult<[f64; D]> {
        Ok(self.probability_of_cut(point)?.1)
    }

    /// Per-dimension range of the bounding box augmented by `point`
    /// without materialising a fresh [`BoundingBox`].
    ///
    /// # Panics
    ///
    /// Panics in debug builds when `d >= D` or `point.len() != D`.
    #[inline]
    #[must_use]
    pub fn augmented_range_at(&self, d: usize, point: &[f64]) -> f64 {
        let lo = self.min[d].min(point[d]);
        let hi = self.max[d].max(point[d]);
        hi - lo
    }

    /// Sum of [`augmented_range_at`](Self::augmented_range_at) over
    /// every dimension.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when `point.len() != D`.
    #[inline]
    #[must_use]
    pub fn augmented_range_sum(&self, point: &[f64]) -> f64 {
        let chunks = D / 4;
        let mut acc_simd = f64x4::splat(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let p = f64x4::from([point[off], point[off + 1], point[off + 2], point[off + 3]]);
            let mn = f64x4::from([
                self.min[off],
                self.min[off + 1],
                self.min[off + 2],
                self.min[off + 3],
            ]);
            let mx = f64x4::from([
                self.max[off],
                self.max[off + 1],
                self.max[off + 2],
                self.max[off + 3],
            ]);
            let lo = mn.fast_min(p);
            let hi = mx.fast_max(p);
            acc_simd += hi - lo;
        }
        let mut s = acc_simd.reduce_add();
        let tail_start = chunks * 4;
        for ((&p, &mn), &mx) in point[tail_start..D]
            .iter()
            .zip(self.min[tail_start..D].iter())
            .zip(self.max[tail_start..D].iter())
        {
            let lo = mn.min(p);
            let hi = mx.max(p);
            s += hi - lo;
        }
        s
    }

    /// Sample a random cut over the bounding box augmented by
    /// `point` without materialising the augmented box.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::EmptyBoundingBox`] when every per-dim
    /// range of the augmented box is zero.
    #[inline]
    pub fn augmented_random_cut<R: rand::RngCore + ?Sized>(
        &self,
        point: &[f64],
        rng: &mut R,
    ) -> RcfResult<Cut> {
        let total = self.augmented_range_sum(point);
        if total <= 0.0 {
            return Err(RcfError::EmptyBoundingBox);
        }
        let mut target = rand::Rng::random::<f64>(rng) * total;
        let mut chosen = 0_usize;
        for d in 0..D {
            let r = self.augmented_range_at(d, point);
            if target < r {
                chosen = d;
                break;
            }
            target -= r;
            chosen = d;
        }
        let lo = self.min[chosen].min(point[chosen]);
        let hi = self.max[chosen].max(point[chosen]);
        let value = if (hi - lo).abs() < f64::EPSILON {
            lo
        } else {
            lo + rand::Rng::random::<f64>(rng) * (hi - lo)
        };
        Ok(Cut::new(chosen, value))
    }

    /// Total cut probability without allocating the per-dim
    /// breakdown — fast path for [`crate::ScalarScoreVisitor`].
    ///
    /// Fuses the `range_sum` and extension passes into a single SIMD
    /// loop so `self.min` / `self.max` are loaded once per chunk. The
    /// previous split implementation did two passes and ate 2× the L1
    /// bandwidth on deep tree descents where bbox reload dominates.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::DimensionMismatch`] when `point.len() != D`.
    pub fn total_probability_of_cut(&self, point: &[f64]) -> RcfResult<f64> {
        if point.len() != D {
            return Err(RcfError::DimensionMismatch {
                expected: D,
                got: point.len(),
            });
        }
        let chunks = D / 4;
        let zero = f64x4::splat(0.0);
        let mut range_acc = f64x4::splat(0.0);
        let mut ext_acc = f64x4::splat(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let p = f64x4::from([point[off], point[off + 1], point[off + 2], point[off + 3]]);
            let mn = f64x4::from([
                self.min[off],
                self.min[off + 1],
                self.min[off + 2],
                self.min[off + 3],
            ]);
            let mx = f64x4::from([
                self.max[off],
                self.max[off + 1],
                self.max[off + 2],
                self.max[off + 3],
            ]);
            range_acc += mx - mn;
            let above = (p - mx).fast_max(zero);
            let below = (mn - p).fast_max(zero);
            ext_acc += above + below;
        }
        let mut range_sum = range_acc.reduce_add();
        let mut extension_sum = ext_acc.reduce_add();
        let tail_start = chunks * 4;
        for ((&p, &mn), &mx) in point[tail_start..D]
            .iter()
            .zip(self.min[tail_start..D].iter())
            .zip(self.max[tail_start..D].iter())
        {
            range_sum += mx - mn;
            let above = p - mx;
            let below = mn - p;
            if above > 0.0 {
                extension_sum += above;
            }
            if below > 0.0 {
                extension_sum += below;
            }
        }
        let denom = range_sum + extension_sum;
        if denom == 0.0 {
            return Ok(0.0);
        }
        Ok(extension_sum / denom)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn from_point_creates_degenerate_box() {
        let b = BoundingBox::<3>::from_point(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(b.dim(), 3);
        assert_eq!(b.min(), &[1.0, 2.0, 3.0]);
        assert_eq!(b.max(), &[1.0, 2.0, 3.0]);
        assert_eq!(b.range_sum(), 0.0);
    }

    #[test]
    fn from_point_rejects_zero_dim() {
        assert!(matches!(
            BoundingBox::<0>::from_point(&[]).unwrap_err(),
            RcfError::EmptyBoundingBox
        ));
    }

    #[test]
    fn from_point_rejects_dim_mismatch() {
        assert!(matches!(
            BoundingBox::<3>::from_point(&[1.0, 2.0]).unwrap_err(),
            RcfError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn extend_grows_box() {
        let mut b = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
        b.extend(&[3.0, -2.0]).unwrap();
        assert_eq!(b.min(), &[0.0, -2.0]);
        assert_eq!(b.max(), &[3.0, 0.0]);
        assert!((b.range_sum() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn extend_rejects_dim_mismatch() {
        let mut b = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
        assert!(matches!(
            b.extend(&[1.0, 2.0, 3.0]).unwrap_err(),
            RcfError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn range_at_per_dim() {
        let mut b = BoundingBox::<3>::from_point(&[0.0, 0.0, 0.0]).unwrap();
        b.extend(&[2.0, 4.0, 8.0]).unwrap();
        assert_eq!(b.range_at(0), 2.0);
        assert_eq!(b.range_at(1), 4.0);
        assert_eq!(b.range_at(2), 8.0);
        assert_eq!(b.range_sum(), 14.0);
    }

    #[test]
    fn merge_with_unions_corners() {
        let mut a = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
        a.extend(&[2.0, 2.0]).unwrap();
        let mut b = BoundingBox::<2>::from_point(&[-1.0, 1.0]).unwrap();
        b.extend(&[1.0, 5.0]).unwrap();
        a.merge_with(&b);
        assert_eq!(a.min(), &[-1.0, 0.0]);
        assert_eq!(a.max(), &[2.0, 5.0]);
    }

    #[test]
    fn merged_returns_new_box() {
        let a = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
        let b = BoundingBox::<2>::from_point(&[5.0, 5.0]).unwrap();
        let union = a.merged(&b);
        assert_eq!(union.min(), &[0.0, 0.0]);
        assert_eq!(union.max(), &[5.0, 5.0]);
        assert_eq!(a.min(), &[0.0, 0.0]);
        assert_eq!(b.max(), &[5.0, 5.0]);
    }

    #[test]
    fn extension_zero_when_point_inside() {
        let mut b = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
        b.extend(&[10.0, 10.0]).unwrap();
        let ext = b.extension_per_dim(&[5.0, 5.0]).unwrap();
        assert_eq!(ext, [0.0, 0.0]);
    }

    #[test]
    fn extension_picks_above_and_below() {
        let mut b = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
        b.extend(&[10.0, 10.0]).unwrap();
        let ext = b.extension_per_dim(&[-3.0, 15.0]).unwrap();
        assert_eq!(ext, [3.0, 5.0]);
    }

    #[test]
    fn probability_of_cut_zero_when_inside() {
        let mut b = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
        b.extend(&[10.0, 10.0]).unwrap();
        let (p, per_dim) = b.probability_of_cut(&[5.0, 5.0]).unwrap();
        assert_eq!(p, 0.0);
        assert_eq!(per_dim, [0.0, 0.0]);
    }

    #[test]
    fn probability_of_cut_concentrated_on_extending_dim() {
        let mut b = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
        b.extend(&[10.0, 10.0]).unwrap();
        let (total, per_dim) = b.probability_of_cut(&[1000.0, 5.0]).unwrap();
        assert!(per_dim[0] > per_dim[1]);
        assert!((per_dim[0] + per_dim[1] - total).abs() < 1e-12);
    }

    #[test]
    fn probability_of_cut_handles_degenerate_box() {
        let b = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
        let (p, per_dim) = b.probability_of_cut(&[0.0, 0.0]).unwrap();
        assert_eq!(p, 0.0);
        assert_eq!(per_dim, [0.0, 0.0]);
    }

    #[test]
    fn probability_of_cut_per_dim_sums_to_total() {
        let mut b = BoundingBox::<3>::from_point(&[0.0, 0.0, 0.0]).unwrap();
        b.extend(&[1.0, 1.0, 1.0]).unwrap();
        let (total, per_dim) = b.probability_of_cut(&[5.0, -3.0, 0.5]).unwrap();
        let sum: f64 = per_dim.iter().sum();
        assert!((sum - total).abs() < 1e-12);
        assert!(per_dim[0] > 0.0);
        assert!(per_dim[1] > 0.0);
        assert_eq!(per_dim[2], 0.0);
    }

    #[test]
    fn probability_of_cut_rejects_dim_mismatch() {
        let b = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
        assert!(matches!(
            b.probability_of_cut(&[1.0]).unwrap_err(),
            RcfError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn per_dim_cut_probabilities_matches_full_call() {
        let mut b = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
        b.extend(&[1.0, 1.0]).unwrap();
        let (_, full) = b.probability_of_cut(&[5.0, -3.0]).unwrap();
        let only_per_dim = b.per_dim_cut_probabilities(&[5.0, -3.0]).unwrap();
        assert_eq!(full, only_per_dim);
    }
}
