//! [`Visitor`] trait used by
//! [`crate::tree::RandomCutTree::traverse`] to dispatch per-node
//! callbacks during a root→leaf walk.
//!
//! Concrete implementations land in story RCF.6
//! (`ScalarScoreVisitor`, `AttributionVisitor`).

use crate::domain::Cut;

/// Trait implemented by anyone observing a root→leaf traversal of a
/// [`crate::tree::RandomCutTree`].
///
/// The visitor receives one callback per visited internal node and a
/// final callback when the matching leaf is reached. After the walk
/// completes, [`Visitor::result`] consumes the visitor and returns the
/// accumulated output.
///
/// # Contract
///
/// - `accept_internal` is called once per ancestor on the path from
///   the root to the leaf, in root→leaf order.
/// - `accept_leaf` is called exactly once, on the leaf where the walk
///   stops.
/// - `result` is called exactly once, after the traversal finishes.
pub trait Visitor {
    /// Output produced after the traversal completes.
    type Output;

    /// Callback invoked for each internal node on the path.
    ///
    /// `depth` is the 0-based depth (root = 0). `mass` is the number
    /// of leaf descendants. `cut` is the hyperplane partitioning the
    /// subtree. `prob_cut` is the total probability that a uniform
    /// random cut over the augmented bounding box would isolate the
    /// queried point — its per-dimension breakdown is supplied via
    /// `per_dim_prob`.
    fn accept_internal(
        &mut self,
        depth: usize,
        mass: u64,
        cut: &Cut,
        prob_cut: f64,
        per_dim_prob: &[f64],
    );

    /// Callback invoked at the matching leaf.
    fn accept_leaf(&mut self, depth: usize, mass: u64, point_idx: usize);

    /// Consume the visitor and return the accumulated output.
    fn result(self) -> Self::Output;
}
