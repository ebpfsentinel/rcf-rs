//! Scalar anomaly score visitor — collusive displacement per Guha
//! et al. (2016) §3.
//!
//! Walks the path from the root to the leaf matching the queried
//! point and accumulates
//!
//! ```text
//! contribution = ((1 − p) · score_seen + p · score_unseen) · damp
//! ```
//!
//! at every internal node (where `p` is the probability that a
//! uniform random cut would isolate the queried point at that
//! depth). At the leaf the contribution collapses to
//! `score_seen · damp`. The final score is `accumulated / log2(total_mass)`.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::domain::{AnomalyScore, BoundingBox, Cut};
use crate::visitor::Visitor;
use crate::visitor::scoring::{damp, normalizer, score_seen, score_unseen};

/// Visitor that produces a non-negative scalar [`AnomalyScore`].
///
/// `total_mass` is the number of points the *whole tree* holds (the
/// root's mass) — the [`crate::RandomCutForest`] reads it from its
/// trees and forwards it to every per-tree visitor.
///
/// # Examples
///
/// ```
/// use rcf_rs::ScalarScoreVisitor;
///
/// let v = ScalarScoreVisitor::new(64);
/// assert_eq!(v.accumulated(), 0.0);
/// assert_eq!(v.total_mass(), 64);
/// ```
#[derive(Debug, Clone)]
pub struct ScalarScoreVisitor {
    /// Sum of per-depth dampened contributions.
    accumulated: f64,
    /// Tree-wide leaf-mass total used for damping and normalisation.
    total_mass: u64,
}

impl ScalarScoreVisitor {
    /// Build a fresh visitor for a tree whose root holds `total_mass`
    /// distinct leaves (counting duplicates via leaf mass).
    #[must_use]
    pub fn new(total_mass: u64) -> Self {
        Self {
            accumulated: 0.0,
            total_mass,
        }
    }

    /// Read-only access to the accumulated raw score before
    /// normalisation. Used by tests and diagnostics.
    #[must_use]
    pub fn accumulated(&self) -> f64 {
        self.accumulated
    }

    /// Tree-wide leaf-mass total this visitor was built with.
    #[must_use]
    pub fn total_mass(&self) -> u64 {
        self.total_mass
    }
}

impl<const D: usize> Visitor<D> for ScalarScoreVisitor {
    type Output = AnomalyScore;

    fn accept_internal(
        &mut self,
        depth: usize,
        mass: u64,
        _cut: &Cut,
        _bbox: &BoundingBox<D>,
        prob_cut: f64,
        _per_dim_prob: &[f64],
    ) {
        let p = prob_cut.clamp(0.0, 1.0);
        let blend = (1.0 - p) * score_seen(depth, mass) + p * score_unseen(depth, mass);
        self.accumulated += blend * damp(mass, self.total_mass);
    }

    fn accept_leaf(&mut self, depth: usize, mass: u64, _point_idx: usize) {
        // The queried point matched this leaf — contribute as a
        // "seen" point (no isolation probability at the leaf).
        self.accumulated += score_seen(depth, mass) * damp(mass, self.total_mass);
    }

    fn result(self) -> AnomalyScore {
        let norm = normalizer(self.total_mass);
        let score = if norm > 0.0 {
            self.accumulated / norm
        } else {
            self.accumulated
        };
        let clamped = if score.is_finite() {
            score.max(0.0)
        } else {
            0.0
        };
        AnomalyScore::new(clamped).expect("clamped score is finite and non-negative")
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests compare exact closed-form constants.
mod tests {
    use super::*;
    use crate::domain::BoundingBox;

    fn unit_bbox<const D: usize>() -> BoundingBox<D> {
        let mut b = BoundingBox::<D>::from_point(&vec![0.0; D]).unwrap();
        b.extend(&vec![1.0; D]).unwrap();
        b
    }

    #[test]
    fn fresh_visitor_starts_at_zero() {
        let v = ScalarScoreVisitor::new(8);
        assert_eq!(v.accumulated(), 0.0);
        assert_eq!(v.total_mass(), 8);
    }

    #[test]
    fn accept_internal_accumulates_blend() {
        let mut v = ScalarScoreVisitor::new(8);
        v.accept_internal(
            1,
            4,
            &Cut::new(0, 0.5),
            &unit_bbox::<2>(),
            0.5,
            &[0.25, 0.25],
        );
        // depth=1, mass=4, p=0.5 → blend = 0.5*(1/(1+log2 4)) + 0.5*(1+log2 4)
        //                              = 0.5 * (1/3) + 0.5 * 3
        //                              = 0.16667 + 1.5 = 1.66667
        // damp(4, 8) = 1 / (1 + ln4/ln8) = 1 / (1 + 0.6667) = 0.6
        let expected = (0.5 * (1.0 / 3.0) + 0.5 * 3.0) * (1.0 / (1.0 + 4f64.ln() / 8f64.ln()));
        assert!((v.accumulated() - expected).abs() < 1e-9);
    }

    #[test]
    fn accept_leaf_adds_seen_contribution() {
        let mut v = ScalarScoreVisitor::new(8);
        <ScalarScoreVisitor as Visitor<2>>::accept_leaf(&mut v, 3, 1, 7);
        // depth=3, mass=1 → score_seen = 1/(3+log2 1) = 1/3
        // damp(1, 8) = 1 / (1 + ln1/ln8) = 1 / (1 + 0) = 1
        let expected = (1.0 / 3.0) * 1.0;
        assert!((v.accumulated() - expected).abs() < 1e-12);
    }

    #[test]
    fn result_normalises_by_log2_total() {
        let mut v = ScalarScoreVisitor::new(4);
        <ScalarScoreVisitor as Visitor<2>>::accept_leaf(&mut v, 2, 1, 0);
        // accumulated = (1/(2+log2 1)) * damp(1,4) = 0.5 * 1 = 0.5
        // normalizer(4) = log2 4 = 2
        let score: f64 = <ScalarScoreVisitor as Visitor<2>>::result(v).into();
        assert!((score - 0.25).abs() < 1e-12);
    }

    #[test]
    fn result_with_total_mass_one_returns_zero() {
        // Single-leaf tree: no anomaly signal.
        let mut v = ScalarScoreVisitor::new(1);
        <ScalarScoreVisitor as Visitor<2>>::accept_leaf(&mut v, 0, 1, 0);
        let score: f64 = <ScalarScoreVisitor as Visitor<2>>::result(v).into();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn result_returns_non_negative() {
        let v = ScalarScoreVisitor::new(8);
        let score: f64 = <ScalarScoreVisitor as Visitor<2>>::result(v).into();
        assert!(score >= 0.0);
    }

    #[test]
    fn prob_cut_outside_range_is_clamped() {
        let mut v = ScalarScoreVisitor::new(8);
        v.accept_internal(
            1,
            4,
            &Cut::new(0, 0.5),
            &unit_bbox::<2>(),
            1.5, // out of [0, 1]
            &[],
        );
        // p clamped to 1.0 → blend = 1.0 * score_unseen(1, 4) = 1 + 2 = 3
        // damp(4, 8) = 0.6
        let expected = 3.0 * (1.0 / (1.0 + 4f64.ln() / 8f64.ln()));
        assert!((v.accumulated() - expected).abs() < 1e-9);
    }

    #[test]
    fn higher_prob_cut_yields_higher_contribution() {
        let mut low = ScalarScoreVisitor::new(64);
        let mut high = ScalarScoreVisitor::new(64);
        low.accept_internal(2, 16, &Cut::new(0, 0.5), &unit_bbox::<2>(), 0.0, &[]);
        high.accept_internal(2, 16, &Cut::new(0, 0.5), &unit_bbox::<2>(), 1.0, &[]);
        // p=0 → contribution = score_seen
        // p=1 → contribution = score_unseen
        // score_unseen(2, 16) = 2 + log2 16 = 6
        // score_seen(2, 16)   = 1 / 6 ≈ 0.167
        // High should accumulate more.
        assert!(high.accumulated() > low.accumulated());
    }
}
