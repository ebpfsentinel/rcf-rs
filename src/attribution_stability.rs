//! Inter-tree dispersion of the per-dim attribution vector.
//!
//! [`crate::RandomCutForest::attribution`] already returns the mean
//! [`DiVector`] across every tree in the forest — but the mean hides
//! how *unanimously* the trees agreed on that answer. An attribution
//! of `high[4] + low[4] = 10` where every tree saw dim 4 as the top
//! contributor is very different from the same mean synthesised from
//! a handful of trees strongly flagging dim 4 and the rest flagging
//! different dims. The second case is a lucky coincidence, not a
//! stable signal.
//!
//! [`AttributionStability`] exposes the mean *and* the per-dim
//! variance / stddev across trees, derives a coefficient of
//! variation and a bounded `confidence ∈ [0, 1]` per dim, and offers
//! two ways to pick the driver dimension:
//!
//! - [`AttributionStability::argmax_mean`] — classic
//!   [`DiVector::argmax`] behaviour, ignores disagreement.
//! - [`AttributionStability::argmax_weighted`] — picks the dim that
//!   maximises `mean × confidence`; downranks dims where the trees
//!   disagree. Safer for SOC-facing alerts.
//!
//! The helper runs one attribution visitor pass per tree, collects
//! the per-tree [`DiVector`]s into a `Vec`, then computes mean and
//! variance in two sweeps. For an AWS-default forest (100 trees, D=16)
//! the extra allocation is ~26 KB.

use alloc::vec;
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::domain::DiVector;
use crate::domain::point::ensure_finite;
use crate::error::{RcfError, RcfResult};
use crate::forest::RandomCutForest;
use crate::thresholded::ThresholdedForest;
use crate::visitor::AttributionVisitor;

/// Inter-tree dispersion of the attribution vector, paired with the
/// mean.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AttributionStability {
    /// Mean per-dim contribution across every tree that produced a
    /// non-trivial attribution. Identical to what
    /// [`crate::RandomCutForest::attribution`] returns.
    mean: DiVector,
    /// Per-dim population variance (not the unbiased estimator —
    /// divides by `tree_count`, not `tree_count − 1`).
    variance: Vec<f64>,
    /// Per-dim standard deviation (`sqrt(variance)`), cached so
    /// callers that query [`Self::confidence`] in a hot loop do not
    /// re-square-root on every call.
    stddev: Vec<f64>,
    /// Number of trees that actually contributed — trees with an
    /// empty reservoir are skipped so `tree_count` may be less than
    /// the forest's configured `num_trees`.
    tree_count: usize,
}

impl AttributionStability {
    /// Mean attribution across trees.
    #[must_use]
    pub fn mean(&self) -> &DiVector {
        &self.mean
    }

    /// Per-dim population variance of the contributions.
    #[must_use]
    pub fn variance(&self) -> &[f64] {
        &self.variance
    }

    /// Per-dim standard deviation of the contributions.
    #[must_use]
    pub fn stddev(&self) -> &[f64] {
        &self.stddev
    }

    /// Number of trees that contributed an attribution.
    #[must_use]
    pub fn tree_count(&self) -> usize {
        self.tree_count
    }

    /// Per-point dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.mean.dim()
    }

    /// Coefficient of variation for dim `d` — `stddev[d] / |mean[d]|`.
    /// Returns `0.0` when `|mean[d]| < f64::EPSILON` (no dispersion is
    /// observable when nothing was attributed).
    ///
    /// # Panics
    ///
    /// Panics when `d >= self.dim()` — callers size-check first.
    #[must_use]
    pub fn coefficient_of_variation(&self, d: usize) -> f64 {
        let mean_abs = self.mean.per_dim_total(d).abs();
        if mean_abs < f64::EPSILON {
            return 0.0;
        }
        self.stddev[d] / mean_abs
    }

    /// Bounded `[0, 1]` confidence that dim `d`'s mean contribution
    /// is a stable signal rather than a handful of trees agreeing by
    /// chance. Derived as `1 / (1 + CV)` — `1.0` for perfect
    /// agreement, falling monotonically as CV rises.
    ///
    /// # Panics
    ///
    /// Panics when `d >= self.dim()` — callers size-check first.
    #[must_use]
    pub fn confidence(&self, d: usize) -> f64 {
        1.0 / (1.0 + self.coefficient_of_variation(d))
    }

    /// Classic [`DiVector::argmax`] — dim with the largest mean
    /// contribution, independent of stability. Returns `None` on an
    /// empty attribution vector.
    #[must_use]
    pub fn argmax_mean(&self) -> Option<usize> {
        self.mean.argmax()
    }

    /// Dim maximising `mean × confidence`. Downranks dims where the
    /// trees disagreed. Returns `None` on an empty attribution vector.
    #[must_use]
    pub fn argmax_weighted(&self) -> Option<usize> {
        if self.dim() == 0 {
            return None;
        }
        let mut best: usize = 0;
        let mut best_val = self.mean.per_dim_total(0) * self.confidence(0);
        for d in 1..self.dim() {
            let v = self.mean.per_dim_total(d) * self.confidence(d);
            if v > best_val {
                best = d;
                best_val = v;
            }
        }
        Some(best)
    }
}

/// Collect the per-tree attribution for `point`. Skips trees whose
/// reservoir is still empty (identical to
/// [`crate::RandomCutForest::attribution`]).
fn collect_per_tree<const D: usize>(
    forest: &RandomCutForest<D>,
    point: &[f64; D],
) -> RcfResult<Vec<DiVector>> {
    let mut out = Vec::with_capacity(forest.num_trees());
    for (tree, _, _) in forest.trees() {
        let Some(root) = tree.root() else {
            continue;
        };
        let mass = tree.store().view(root)?.mass();
        let visitor = AttributionVisitor::new(point, mass)?;
        let di = tree.traverse(point, visitor)?;
        out.push(di);
    }
    Ok(out)
}

/// Compute [`AttributionStability`] from a collected per-tree set.
/// Shared by the forest- and pool-level entry points.
#[allow(clippy::cast_precision_loss)] // Tree counts are bounded by num_trees <= 1000.
fn stability_from_collection<const D: usize>(
    per_tree: &[DiVector],
) -> RcfResult<AttributionStability> {
    if per_tree.is_empty() {
        return Err(RcfError::EmptyForest);
    }
    let tree_count = per_tree.len();
    let divisor = tree_count as f64;

    let mut mean = DiVector::zeros(D);
    for di in per_tree {
        mean.accumulate(di)?;
    }
    mean.scale(divisor)?;

    let mut variance = vec![0.0_f64; D];
    for di in per_tree {
        for (d, var_d) in variance.iter_mut().enumerate().take(D) {
            let delta = di.per_dim_total(d) - mean.per_dim_total(d);
            *var_d += delta * delta;
        }
    }
    for v in &mut variance {
        *v /= divisor;
    }
    let stddev: Vec<f64> = variance.iter().map(|v| v.sqrt()).collect();

    Ok(AttributionStability {
        mean,
        variance,
        stddev,
        tree_count,
    })
}

impl<const D: usize> RandomCutForest<D> {
    /// Inter-tree dispersion of the attribution vector on `point`.
    ///
    /// Returns the mean contribution per dim plus the per-dim
    /// variance and stddev across trees — use
    /// [`AttributionStability::confidence`] or
    /// [`AttributionStability::argmax_weighted`] to pick a driver
    /// dim that downranks tree-level disagreement.
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] when the point contains a non-finite
    ///   component.
    /// - [`RcfError::EmptyForest`] when no tree holds any leaf.
    /// - Any error bubbled up from the per-tree attribution path.
    pub fn attribution_stability(&self, point: &[f64; D]) -> RcfResult<AttributionStability> {
        ensure_finite(point)?;
        // Keep parity with `attribution()` — stored points are in the
        // forest's scaled space, so the caller query must be scaled
        // before walking the tree cuts.
        let scaled = self.scale_point_copy(point);
        let per_tree = collect_per_tree(self, &scaled)?;
        stability_from_collection::<D>(&per_tree)
    }
}

impl<const D: usize> ThresholdedForest<D> {
    /// Inter-tree dispersion of the attribution on `point`. Delegates
    /// to the underlying forest — the threshold layer does not
    /// influence attribution.
    ///
    /// # Errors
    ///
    /// Same as [`RandomCutForest::attribution_stability`].
    pub fn attribution_stability(&self, point: &[f64; D]) -> RcfResult<AttributionStability> {
        self.forest().attribution_stability(point)
    }
}

#[cfg(feature = "std")]
impl<K, const D: usize> crate::pool::TenantForestPool<K, D>
where
    K: core::hash::Hash + Eq + Clone,
{
    /// Per-tenant attribution stability. Lazily instantiates the
    /// tenant (like [`Self::process`]).
    ///
    /// # Errors
    ///
    /// Same as [`ThresholdedForest::attribution_stability`] plus
    /// factory errors.
    ///
    /// # Panics
    ///
    /// Never under normal use — the fall-through branch forces a
    /// slot via [`Self::score_only`] before re-borrowing through
    /// [`Self::get_mut`]; the assertion only fires on an impossible
    /// concurrent eviction through `&mut self`.
    pub fn attribution_stability(
        &mut self,
        key: &K,
        point: &[f64; D],
    ) -> RcfResult<AttributionStability> {
        if !self.contains(key) {
            self.score_only(key, point)?;
        }
        let detector = self
            .get_mut(key)
            .expect("tenant was just forced into the pool");
        detector.attribution_stability(point)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert bounds on closed-form quantities.
mod tests {
    use super::*;
    use crate::ForestBuilder;

    fn trained() -> RandomCutForest<2> {
        let mut f = ForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(32)
            .seed(2026)
            .build()
            .unwrap();
        for i in 0_u32..256 {
            let v = f64::from(i) * 0.01;
            f.update([v, v + 0.5]).unwrap();
        }
        f
    }

    #[test]
    fn empty_forest_errors() {
        let f = ForestBuilder::<2>::new().seed(1).build().unwrap();
        let err = f.attribution_stability(&[0.0, 0.0]).unwrap_err();
        assert!(matches!(err, RcfError::EmptyForest));
    }

    #[test]
    fn non_finite_point_rejected() {
        let f = trained();
        let err = f.attribution_stability(&[f64::NAN, 0.0]).unwrap_err();
        assert!(matches!(err, RcfError::NaNValue));
    }

    #[test]
    fn tree_count_matches_forest_size_on_trained_forest() {
        let f = trained();
        let s = f.attribution_stability(&[5.0, 5.0]).unwrap();
        assert_eq!(s.tree_count(), 50);
        assert_eq!(s.dim(), 2);
    }

    #[test]
    fn mean_matches_plain_attribution() {
        let f = trained();
        let probe = [5.0_f64, 5.0];
        let plain = f.attribution(&probe).unwrap();
        let s = f.attribution_stability(&probe).unwrap();
        // Under the `parallel` feature, `attribution()` uses rayon's
        // reorder-safe fold/reduce, which can differ from this
        // helper's serial sum in the last ULP. 1e-10 is orders of
        // magnitude below any observable signal.
        for d in 0..2 {
            let delta = (plain.per_dim_total(d) - s.mean().per_dim_total(d)).abs();
            assert!(delta < 1e-10, "dim {d} drift {delta}");
        }
    }

    #[test]
    fn variance_is_non_negative_per_dim() {
        let f = trained();
        let s = f.attribution_stability(&[5.0_f64, 5.0]).unwrap();
        for v in s.variance() {
            assert!(*v >= 0.0);
        }
        for sd in s.stddev() {
            assert!(*sd >= 0.0);
        }
    }

    #[test]
    fn stddev_is_sqrt_of_variance() {
        let f = trained();
        let s = f.attribution_stability(&[5.0_f64, 5.0]).unwrap();
        for d in 0..s.dim() {
            assert!((s.stddev()[d] - s.variance()[d].sqrt()).abs() < 1e-12);
        }
    }

    #[test]
    fn confidence_is_one_when_variance_zero() {
        // Every tree attributes exactly the same contribution → CV=0 → conf=1.
        let mut mean = DiVector::zeros(3);
        mean.add_high(0, 1.0).unwrap();
        mean.add_low(1, 2.0).unwrap();
        let s = AttributionStability {
            mean,
            variance: vec![0.0, 0.0, 0.0],
            stddev: vec![0.0, 0.0, 0.0],
            tree_count: 10,
        };
        assert!((s.confidence(0) - 1.0).abs() < f64::EPSILON);
        assert!((s.confidence(1) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn confidence_drops_monotonically_with_cv() {
        let mut mean = DiVector::zeros(2);
        mean.add_high(0, 1.0).unwrap();
        mean.add_high(1, 1.0).unwrap();
        let stable = AttributionStability {
            mean: mean.clone(),
            variance: vec![0.01_f64, 0.25],
            stddev: vec![0.1_f64, 0.5],
            tree_count: 10,
        };
        assert!(stable.confidence(0) > stable.confidence(1));
    }

    #[test]
    fn coefficient_of_variation_is_zero_when_mean_zero() {
        let mean = DiVector::zeros(1);
        let s = AttributionStability {
            mean,
            variance: vec![1.0],
            stddev: vec![1.0],
            tree_count: 4,
        };
        // mean[0] == 0 → CV undefined → clamp to 0.
        assert_eq!(s.coefficient_of_variation(0), 0.0);
        assert!((s.confidence(0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn argmax_weighted_prefers_stable_dim_over_unstable() {
        // mean[0] = 10 but very unstable (stddev=30 → CV=3 → conf ~ 0.25)
        // mean[1] = 5 but stable (stddev=0.1 → CV=0.02 → conf ~ 0.98)
        // weighted: 10 * 0.25 = 2.5  vs  5 * 0.98 = 4.9 → pick 1.
        let mut mean = DiVector::zeros(2);
        mean.add_high(0, 10.0).unwrap();
        mean.add_high(1, 5.0).unwrap();
        let s = AttributionStability {
            mean,
            variance: vec![900.0, 0.01],
            stddev: vec![30.0, 0.1],
            tree_count: 10,
        };
        assert_eq!(s.argmax_mean(), Some(0));
        assert_eq!(s.argmax_weighted(), Some(1));
    }

    #[test]
    fn argmax_weighted_empty_returns_none() {
        let s = AttributionStability {
            mean: DiVector::zeros(0),
            variance: vec![],
            stddev: vec![],
            tree_count: 0,
        };
        assert!(s.argmax_weighted().is_none());
        assert!(s.argmax_mean().is_none());
    }

    #[test]
    fn stability_from_collection_rejects_empty() {
        let err = stability_from_collection::<2>(&[]).unwrap_err();
        assert!(matches!(err, RcfError::EmptyForest));
    }
}
