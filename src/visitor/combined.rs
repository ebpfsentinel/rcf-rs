//! Single-walk visitor producing both [`AnomalyScore`] and
//! [`DiVector`] attribution.
//!
//! Rationale: the score and attribution paths walk the same root→leaf
//! sequence, touch the same bounding boxes, and compute the same
//! `dampened = blend · damp` per internal node. The only difference
//! is how they spread that scalar — scalar score accumulates it
//! verbatim, attribution splits it across dims via
//! `per_dim_prob[d] / p`. Walking the tree twice costs a second round
//! of cache loads + bounding-box probability SIMD passes.
//!
//! [`ScoreAttributionVisitor`] folds both accumulators in a single
//! traversal. Every `accept_internal` call computes the shared
//! `dampened` once and forwards it to both outputs. At the leaf, the
//! scalar `score_seen` contribution is added to the scalar
//! accumulator only — attribution has no leaf contribution by design
//! (no cut at the leaf to attribute).

use crate::domain::{AnomalyScore, BoundingBox, Cut, DiVector, ensure_finite};
use crate::error::{RcfError, RcfResult};
use crate::visitor::Visitor;
use crate::visitor::scoring::{damp, normalizer, score_seen, score_unseen};

/// Visitor producing the `(score, attribution)` tuple from one
/// root→leaf walk.
///
/// `total_mass` is the per-tree leaf-mass total, read from the tree
/// root before traversal (same convention as
/// [`crate::ScalarScoreVisitor`] / [`crate::AttributionVisitor`]).
///
/// # Examples
///
/// ```
/// use rcf_rs::visitor::combined::ScoreAttributionVisitor;
///
/// let q = [0.5_f64, 0.5];
/// let v = ScoreAttributionVisitor::new(&q, 16).unwrap();
/// assert_eq!(v.total_mass(), 16);
/// ```
#[derive(Debug, Clone)]
pub struct ScoreAttributionVisitor<'a> {
    /// Sum of per-depth dampened contributions — matches
    /// [`crate::ScalarScoreVisitor::accumulated`].
    accumulated: f64,
    /// Per-dimension attribution accumulator.
    di: DiVector,
    /// Queried point — borrowed for the walk.
    point: &'a [f64],
    /// Tree-wide leaf-mass total used for damping + normalisation.
    total_mass: u64,
}

impl<'a> ScoreAttributionVisitor<'a> {
    /// Build a visitor that borrows `point` for the duration of one
    /// traversal.
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] when `point` contains a non-finite component.
    /// - [`RcfError::InvalidConfig`] when `point.is_empty()`.
    pub fn new(point: &'a [f64], total_mass: u64) -> RcfResult<Self> {
        if point.is_empty() {
            return Err(RcfError::InvalidConfig(
                "ScoreAttributionVisitor: point must not be empty".into(),
            ));
        }
        ensure_finite(point)?;
        Ok(Self {
            accumulated: 0.0,
            di: DiVector::zeros(point.len()),
            point,
            total_mass,
        })
    }

    /// Tree-wide leaf-mass total.
    #[must_use]
    pub fn total_mass(&self) -> u64 {
        self.total_mass
    }

    /// Partial scalar accumulator snapshot (pre-normalisation).
    #[must_use]
    pub fn accumulated(&self) -> f64 {
        self.accumulated
    }
}

impl<const D: usize> Visitor<D> for ScoreAttributionVisitor<'_> {
    type Output = (AnomalyScore, DiVector);

    fn accept_internal(
        &mut self,
        depth: usize,
        mass: u64,
        _cut: &Cut,
        bbox: &BoundingBox<D>,
        prob_cut: f64,
        per_dim_prob: &[f64],
    ) {
        let p = prob_cut.clamp(0.0, 1.0);
        let blend = (1.0 - p) * score_seen(depth, mass) + p * score_unseen(depth, mass);
        let dampened = blend * damp(mass, self.total_mass);
        // Scalar accumulator — identical to ScalarScoreVisitor.
        self.accumulated += dampened;

        // Attribution split — only when prob > 0 (else no dim
        // contributed to this cut and the shares would divide by zero).
        if p <= 0.0 {
            return;
        }
        let dim = self.di.dim().min(per_dim_prob.len()).min(bbox.dim());
        for (d, &dim_prob) in per_dim_prob.iter().take(dim).enumerate() {
            if dim_prob <= 0.0 {
                continue;
            }
            let share = dim_prob / p;
            let contribution = dampened * share;
            if self.point[d] > bbox.max()[d] {
                let _ = self.di.add_high(d, contribution);
            } else if self.point[d] < bbox.min()[d] {
                let _ = self.di.add_low(d, contribution);
            }
        }
    }

    fn accept_leaf(&mut self, depth: usize, mass: u64, _point_idx: usize) {
        // Scalar path adds the leaf "seen" contribution; attribution
        // has no leaf split (no cut to attribute).
        self.accumulated += score_seen(depth, mass) * damp(mass, self.total_mass);
    }

    fn needs_per_dim_prob(&self) -> bool {
        true
    }

    fn result(self) -> (AnomalyScore, DiVector) {
        let norm = normalizer(self.total_mass);
        let raw = if norm > 0.0 {
            self.accumulated / norm
        } else {
            self.accumulated
        };
        let clamped = if raw.is_finite() { raw.max(0.0) } else { 0.0 };
        let score = AnomalyScore::new(clamped).expect("clamped score is finite and non-negative");

        let mut di = self.di;
        if norm > 0.0 {
            let _ = di.scale(norm);
        }
        (score, di)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::visitor::{AttributionVisitor, ScalarScoreVisitor};

    fn unit_bbox<const D: usize>() -> BoundingBox<D> {
        let mut b = BoundingBox::<D>::from_point(&vec![0.0; D]).unwrap();
        b.extend(&vec![1.0; D]).unwrap();
        b
    }

    #[test]
    fn new_rejects_empty_point() {
        let err = ScoreAttributionVisitor::new(&[], 4).unwrap_err();
        assert!(matches!(err, RcfError::InvalidConfig(_)));
    }

    #[test]
    fn new_rejects_non_finite() {
        assert!(matches!(
            ScoreAttributionVisitor::new(&[1.0, f64::NAN], 4).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn merged_scalar_matches_scalar_visitor() {
        // Same sequence of accept_internal / accept_leaf calls should
        // produce a scalar component identical to ScalarScoreVisitor
        // when consumed by result().
        let probe = [100.0_f64, 0.5];
        let bbox = unit_bbox::<2>();

        let mut scalar = ScalarScoreVisitor::new(8);
        let mut merged = ScoreAttributionVisitor::new(&probe, 8).unwrap();
        for (depth, mass, p, per_dim) in [
            (1_usize, 4_u64, 0.5_f64, [0.4_f64, 0.1]),
            (2, 2, 0.3, [0.3, 0.0]),
        ] {
            scalar.accept_internal(depth, mass, &Cut::new(0, 0.5), &bbox, p, &per_dim);
            <ScoreAttributionVisitor<'_> as Visitor<2>>::accept_internal(
                &mut merged,
                depth,
                mass,
                &Cut::new(0, 0.5),
                &bbox,
                p,
                &per_dim,
            );
        }
        <ScalarScoreVisitor as Visitor<2>>::accept_leaf(&mut scalar, 3, 1, 0);
        <ScoreAttributionVisitor<'_> as Visitor<2>>::accept_leaf(&mut merged, 3, 1, 0);

        let s_scalar = <ScalarScoreVisitor as Visitor<2>>::result(scalar);
        let (s_merged, _di) = <ScoreAttributionVisitor<'_> as Visitor<2>>::result(merged);
        assert!((f64::from(s_scalar) - f64::from(s_merged)).abs() < 1e-12);
    }

    #[test]
    fn merged_attribution_matches_attribution_visitor() {
        let probe = [100.0_f64, 0.5];
        let bbox = unit_bbox::<2>();

        let mut attr = AttributionVisitor::new(&probe, 8).unwrap();
        let mut merged = ScoreAttributionVisitor::new(&probe, 8).unwrap();
        for (depth, mass, p, per_dim) in [
            (1_usize, 4_u64, 0.5_f64, [0.4_f64, 0.1]),
            (2, 2, 0.3, [0.3, 0.0]),
        ] {
            attr.accept_internal(depth, mass, &Cut::new(0, 0.5), &bbox, p, &per_dim);
            <ScoreAttributionVisitor<'_> as Visitor<2>>::accept_internal(
                &mut merged,
                depth,
                mass,
                &Cut::new(0, 0.5),
                &bbox,
                p,
                &per_dim,
            );
        }
        <AttributionVisitor<'_> as Visitor<2>>::accept_leaf(&mut attr, 3, 1, 0);
        <ScoreAttributionVisitor<'_> as Visitor<2>>::accept_leaf(&mut merged, 3, 1, 0);

        let di_attr = <AttributionVisitor<'_> as Visitor<2>>::result(attr);
        let (_s, di_merged) = <ScoreAttributionVisitor<'_> as Visitor<2>>::result(merged);
        for d in 0..2 {
            assert!((di_attr.high()[d] - di_merged.high()[d]).abs() < 1e-12);
            assert!((di_attr.low()[d] - di_merged.low()[d]).abs() < 1e-12);
        }
    }

    #[test]
    fn needs_per_dim_prob_is_true() {
        let v = ScoreAttributionVisitor::new(&[1.0, 2.0], 4).unwrap();
        assert!(<ScoreAttributionVisitor<'_> as Visitor<2>>::needs_per_dim_prob(&v));
    }
}
