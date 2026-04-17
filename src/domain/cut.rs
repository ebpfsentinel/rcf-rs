//! A single random cut: a hyperplane perpendicular to one axis at a
//! given coordinate.
//!
//! [`Cut::random_cut`] picks the dimension proportionally to the
//! per-dimension range of the [`BoundingBox`] (Guha et al. 2016, §2)
//! then samples the cut value uniformly in `[min_d, max_d]`. Callers
//! pass any [`rand::RngCore`] so reproducibility is purely a function
//! of the seed they own.
//!
//! When the bounding box is fully degenerate (every dimension has
//! zero range) [`Cut::random_cut`] returns [`RcfError::EmptyBoundingBox`]
//! — there is no meaningful cut.

use rand::{Rng, RngCore};

use crate::domain::bounding_box::BoundingBox;
use crate::error::{RcfError, RcfResult};

/// A random cut along one dimension at a given coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cut {
    /// Dimension that the cut is perpendicular to.
    dim: usize,
    /// Cut coordinate in `[bbox.min[dim], bbox.max[dim]]`.
    value: f64,
}

impl Cut {
    /// Build a cut explicitly. Used by tests and the persistence layer.
    #[must_use]
    pub fn new(dim: usize, value: f64) -> Self {
        Self { dim, value }
    }

    /// Cut dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Cut coordinate.
    #[must_use]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Decide which side of the cut a `point` lies on.
    ///
    /// Returns `true` when `point[dim] <= self.value`. The exact-tie
    /// behaviour is consistent with the AWS reference (left subtree
    /// gets the equal-or-below half).
    ///
    /// # Panics
    ///
    /// Panics when `self.dim() >= point.len()` — call sites always
    /// size-check the point first via
    /// [`crate::domain::ensure_dim`].
    #[must_use]
    pub fn left_of(&self, point: &[f64]) -> bool {
        point[self.dim] <= self.value
    }

    /// Sample a random cut over `bbox` with the dimension chosen
    /// proportionally to its range and the value uniform in
    /// `[min_d, max_d]`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::EmptyBoundingBox`] when every per-dimension
    /// range is zero (no cut is meaningful).
    ///
    /// # Examples
    ///
    /// ```
    /// use rand::SeedableRng;
    /// use rand_chacha::ChaCha8Rng;
    /// use rcf_rs::domain::{BoundingBox, Cut};
    ///
    /// let mut rng = ChaCha8Rng::seed_from_u64(42);
    /// let mut bbox = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
    /// bbox.extend(&[1.0, 4.0]).unwrap(); // dim 1 has 4× the range of dim 0
    ///
    /// let cut = Cut::random_cut(&bbox, &mut rng).unwrap();
    /// assert!(cut.dim() < bbox.dim());
    /// let lo = bbox.min()[cut.dim()];
    /// let hi = bbox.max()[cut.dim()];
    /// assert!(cut.value() >= lo && cut.value() <= hi);
    /// ```
    pub fn random_cut<R: RngCore + ?Sized>(bbox: &BoundingBox, rng: &mut R) -> RcfResult<Self> {
        let total = bbox.range_sum();
        if total <= 0.0 {
            return Err(RcfError::EmptyBoundingBox);
        }

        let mut target = rng.random::<f64>() * total;
        let mut chosen = 0_usize;
        for d in 0..bbox.dim() {
            let r = bbox.range_at(d);
            if target < r {
                chosen = d;
                break;
            }
            target -= r;
            chosen = d;
        }

        let lo = bbox.min()[chosen];
        let hi = bbox.max()[chosen];
        let value = if (hi - lo).abs() < f64::EPSILON {
            // Edge case: chosen dim was zero-range (only possible
            // through floating accumulation drift). Fall back to the
            // axis coordinate so the cut is degenerate but well-formed.
            lo
        } else {
            lo + rng.random::<f64>() * (hi - lo)
        };

        Ok(Self { dim: chosen, value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn unit_box(dim: usize) -> BoundingBox {
        let mut b = BoundingBox::from_point(&vec![0.0; dim]).unwrap();
        b.extend(&vec![1.0; dim]).unwrap();
        b
    }

    #[test]
    fn left_of_strictly_below_is_left() {
        let cut = Cut::new(0, 0.5);
        assert!(cut.left_of(&[0.4, 9.9]));
    }

    #[test]
    fn left_of_at_value_is_left() {
        let cut = Cut::new(1, 2.0);
        assert!(cut.left_of(&[1.0, 2.0]));
    }

    #[test]
    fn left_of_strictly_above_is_right() {
        let cut = Cut::new(0, 0.5);
        assert!(!cut.left_of(&[0.6, 9.9]));
    }

    #[test]
    fn random_cut_is_in_range() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let bbox = unit_box(3);
        for _ in 0..100 {
            let cut = Cut::random_cut(&bbox, &mut rng).unwrap();
            assert!(cut.dim() < bbox.dim());
            assert!(cut.value() >= bbox.min()[cut.dim()]);
            assert!(cut.value() <= bbox.max()[cut.dim()]);
        }
    }

    #[test]
    fn random_cut_degenerate_box_fails() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let bbox = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        let err = Cut::random_cut(&bbox, &mut rng).unwrap_err();
        assert!(matches!(err, RcfError::EmptyBoundingBox));
    }

    #[test]
    fn random_cut_dim_distribution_proportional_to_range() {
        // Build a box where dim 1 has 9× the range of dim 0 — over 5000
        // draws dim 1 should receive roughly 90% of the mass.
        let mut bbox = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        bbox.extend(&[1.0, 9.0]).unwrap();

        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut counts = [0_u32; 2];
        let trials = 5000;
        for _ in 0..trials {
            let cut = Cut::random_cut(&bbox, &mut rng).unwrap();
            counts[cut.dim()] += 1;
        }
        let p1 = f64::from(counts[1]) / f64::from(trials);
        // Expected 0.9, allow a generous ±0.03 tolerance for ChaCha8 noise.
        assert!(
            (0.87..=0.93).contains(&p1),
            "dim-1 share = {p1} outside [0.87, 0.93]"
        );
    }

    #[test]
    fn random_cut_deterministic_for_same_seed() {
        let bbox = unit_box(4);
        let mut rng_a = ChaCha8Rng::seed_from_u64(42);
        let mut rng_b = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..20 {
            let a = Cut::random_cut(&bbox, &mut rng_a).unwrap();
            let b = Cut::random_cut(&bbox, &mut rng_b).unwrap();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn cut_constructor_accessors() {
        let c = Cut::new(7, 1.25);
        assert_eq!(c.dim(), 7);
        assert!((c.value() - 1.25).abs() < f64::EPSILON);
    }
}
