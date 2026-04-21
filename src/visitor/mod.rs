//! [`Visitor`] trait used by
//! [`crate::tree::RandomCutTree::traverse`] to dispatch per-node
//! callbacks during a root→leaf walk, plus the two production
//! visitors:
//!
//! - [`scalar_score::ScalarScoreVisitor`] — collusive-displacement
//!   anomaly scoring per Guha et al. (2016) §3.
//! - [`attribution::AttributionVisitor`] — per-feature
//!   [`crate::DiVector`] attribution exposing which dimensions drove
//!   the score.

use crate::domain::{BoundingBox, Cut};

pub mod attribution;
pub mod combined;
pub mod scalar_score;
pub mod scoring;

pub use attribution::AttributionVisitor;
pub use combined::ScoreAttributionVisitor;
pub use scalar_score::ScalarScoreVisitor;

/// Trait implemented by anyone observing a root→leaf traversal of a
/// [`crate::tree::RandomCutTree`].
///
/// `D` is the per-point dimensionality of the tree being walked. The
/// trait is generic over `D` so the cached bounding boxes can be
/// passed in as `&BoundingBox<D>` without erasing their compile-time
/// dimensionality.
///
/// The visitor receives one callback per visited internal node and a
/// final callback when the matching leaf is reached. After the walk
/// completes, [`Visitor::result`] consumes the visitor and returns
/// the accumulated output.
///
/// # Contract
///
/// - `accept_internal` is called once per ancestor on the path from
///   the root to the leaf, in root→leaf order.
/// - `accept_leaf` is called exactly once, on the leaf where the walk
///   stops.
/// - `result` is called exactly once, after the traversal finishes.
///
/// # Examples
///
/// ```
/// use rcf_rs::{ScalarScoreVisitor, Visitor};
/// // `ScalarScoreVisitor` implements `Visitor<D>` for any `D`.
/// let v: ScalarScoreVisitor = ScalarScoreVisitor::new(8);
/// assert_eq!(v.total_mass(), 8);
/// ```
pub trait Visitor<const D: usize> {
    /// Output produced after the traversal completes.
    type Output;

    /// Callback invoked for each internal node on the path.
    ///
    /// `depth` is the 0-based depth (root = 0). `mass` is the number
    /// of leaf descendants. `cut` is the hyperplane partitioning the
    /// subtree. `bbox` is the cached union bounding box of the
    /// subtree at this depth (the *pre-augmentation* box, i.e. the
    /// extent of the points currently in the subtree). `prob_cut` is
    /// the total probability that a uniform random cut over the
    /// augmented bounding box would isolate the queried point — its
    /// per-dimension breakdown is supplied via `per_dim_prob`.
    fn accept_internal(
        &mut self,
        depth: usize,
        mass: u64,
        cut: &Cut,
        bbox: &BoundingBox<D>,
        prob_cut: f64,
        per_dim_prob: &[f64],
    );

    /// Callback invoked at the matching leaf.
    fn accept_leaf(&mut self, depth: usize, mass: u64, point_idx: usize);

    /// Consume the visitor and return the accumulated output.
    fn result(self) -> Self::Output;

    /// Whether this visitor consumes the `per_dim_prob` argument
    /// passed to [`accept_internal`](Self::accept_internal). The
    /// tree traversal skips the per-dimension probability vector
    /// allocation when this returns `false` (default).
    ///
    /// Override to `true` for attribution-style visitors so the
    /// traversal computes the full per-dim breakdown.
    #[must_use]
    fn needs_per_dim_prob(&self) -> bool {
        false
    }
}
