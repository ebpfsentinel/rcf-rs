//! Incrementally-maintained random cut binary partition.
//!
//! [`RandomCutTree`] sits on top of [`crate::tree::NodeStore`] and
//! provides three operations:
//!
//! - [`add`](RandomCutTree::add) — insert a point following Guha
//!   et al. (2016) §2: at every visited node, sample a cut over the
//!   augmented bounding box; if the cut isolates the new point from
//!   the existing subtree, splice a new internal node here, otherwise
//!   descend along the existing cut.
//! - [`delete`](RandomCutTree::delete) — remove a leaf, merge its
//!   sibling up into the parent's slot, and recompute ancestor masses
//!   and bounding boxes.
//! - [`traverse`](RandomCutTree::traverse) — walk root→leaf along the
//!   stored cuts, dispatching per-node callbacks to a
//!   [`crate::visitor::Visitor`].
//!
//! The tree never owns point coordinates — callers (the forest layer
//! in story RCF.7) hand a [`PointAccessor`] in for any operation that
//! needs to know a leaf's location.

use std::borrow::Cow;

use rand::RngCore;

use crate::domain::{BoundingBox, Cut, ensure_dim, ensure_finite};
use crate::error::{RcfError, RcfResult};
use crate::tree::node::{Node, NodeRef};
use crate::tree::node_store::NodeStore;
use crate::visitor::Visitor;

/// Borrow points by index. Implemented by the forest layer and by
/// `Vec<Vec<f64>>` for in-tree tests.
pub trait PointAccessor {
    /// Borrow the point stored at `idx`, or return `None` when it
    /// does not exist.
    fn point(&self, idx: usize) -> Option<&[f64]>;
}

impl PointAccessor for Vec<Vec<f64>> {
    fn point(&self, idx: usize) -> Option<&[f64]> {
        self.get(idx).map(Vec::as_slice)
    }
}

impl PointAccessor for [Vec<f64>] {
    fn point(&self, idx: usize) -> Option<&[f64]> {
        self.get(idx).map(Vec::as_slice)
    }
}

/// Incrementally-maintained random cut tree over up to `capacity`
/// distinct points.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RandomCutTree {
    /// Root reference; `None` when the tree holds no live leaves.
    root: Option<NodeRef>,
    /// Backing storage for all live nodes.
    store: NodeStore,
    /// Dimensionality enforced by `add` / `traverse`.
    dimension: usize,
    /// Reverse index: `point_idx → leaf NodeRef` for `O(1)` deletions.
    /// Backed by a sparse `Vec<Option<NodeRef>>` indexed by `point_idx`
    /// — the forest's `PointStore` reuses freed slots so the indices
    /// stay dense in steady state. Maintained alongside
    /// `distinct_count` so `distinct_point_count` stays `O(1)`.
    leaf_index: Vec<Option<NodeRef>>,
    /// Cached count of `Some` entries in `leaf_index`.
    distinct_count: usize,
}

impl RandomCutTree {
    /// Build an empty tree with room for `capacity` distinct points.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `dimension == 0`,
    /// `capacity == 0`, or `capacity` exceeds the store limit.
    pub fn new(capacity: u32, dimension: usize) -> RcfResult<Self> {
        if dimension == 0 {
            return Err(RcfError::InvalidConfig(
                "RandomCutTree dimension must be > 0".into(),
            ));
        }
        Ok(Self {
            root: None,
            store: NodeStore::new(capacity)?,
            dimension,
            leaf_index: Vec::new(),
            distinct_count: 0,
        })
    }

    /// Insert (or overwrite) `leaf_index[idx] = Some(node)`,
    /// updating [`distinct_count`](Self::distinct_point_count). Grows
    /// the backing `Vec` on demand.
    #[inline]
    fn leaf_index_set(&mut self, idx: usize, node: NodeRef) {
        if idx >= self.leaf_index.len() {
            self.leaf_index.resize(idx + 1, None);
        }
        if self.leaf_index[idx].is_none() {
            self.distinct_count += 1;
        }
        self.leaf_index[idx] = Some(node);
    }

    /// Clear `leaf_index[idx]` if present, decrementing
    /// [`distinct_count`](Self::distinct_point_count). No-op when
    /// `idx` is out of range or already `None`.
    #[inline]
    fn leaf_index_clear(&mut self, idx: usize) {
        if let Some(slot) = self.leaf_index.get_mut(idx)
            && slot.is_some()
        {
            *slot = None;
            self.distinct_count -= 1;
        }
    }

    /// `Some(node_ref)` when `point_idx` is currently mapped.
    #[inline]
    #[must_use]
    fn leaf_index_get(&self, idx: usize) -> Option<NodeRef> {
        self.leaf_index.get(idx).copied().flatten()
    }

    /// Whether `point_idx` is currently mapped — `O(1)` `Vec` index.
    #[inline]
    #[must_use]
    fn leaf_index_contains(&self, idx: usize) -> bool {
        matches!(self.leaf_index.get(idx), Some(Some(_)))
    }

    /// Root node reference, or `None` when the tree is empty.
    #[must_use]
    pub fn root(&self) -> Option<NodeRef> {
        self.root
    }

    /// Configured dimensionality.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of distinct points currently stored (each leaf counts
    /// once regardless of its mass).
    #[must_use]
    #[inline]
    pub fn distinct_point_count(&self) -> usize {
        self.distinct_count
    }

    /// Borrow the underlying node store. Used by tests and
    /// persistence (story RCF.8).
    #[must_use]
    pub fn store(&self) -> &NodeStore {
        &self.store
    }

    /// Whether the tree currently stores `point_idx`.
    #[must_use]
    #[inline]
    pub fn contains(&self, point_idx: usize) -> bool {
        self.leaf_index_contains(point_idx)
    }

    /// Maximum depth from the root to any leaf, or `None` when the
    /// tree is empty. Used by tests and diagnostics to verify the
    /// expected `O(log n)` depth bound under uniform-random inserts.
    #[must_use]
    pub fn max_depth(&self) -> Option<usize> {
        let root = self.root?;
        Some(self.subtree_depth(root, 0).unwrap_or(0))
    }

    /// Recursive helper for [`max_depth`](Self::max_depth).
    fn subtree_depth(&self, n: NodeRef, depth: usize) -> RcfResult<usize> {
        match self.store.node(n)? {
            Node::Leaf { .. } => Ok(depth),
            Node::Internal { left, right, .. } => {
                let l = self.subtree_depth(*left, depth + 1)?;
                let r = self.subtree_depth(*right, depth + 1)?;
                Ok(l.max(r))
            }
        }
    }

    /// Insert `point` (registered under `point_idx` in the caller's
    /// point store) into the tree.
    ///
    /// When an identical point is already present, the existing leaf's
    /// mass is incremented and `point_idx` is mapped to that same
    /// leaf — duplicate-point handling per Guha 2016.
    ///
    /// # Errors
    ///
    /// - [`RcfError::DimensionMismatch`] when `point.len() != self.dimension()`.
    /// - [`RcfError::NaNValue`] when `point` contains a non-finite component.
    /// - [`RcfError::InvalidConfig`] when `point_idx` is already present.
    /// - [`RcfError::InvalidConfig`] when the underlying store runs out
    ///   of internal or leaf slots.
    pub fn add<R, P>(
        &mut self,
        point_idx: usize,
        point: &[f64],
        points: &P,
        rng: &mut R,
    ) -> RcfResult<()>
    where
        R: RngCore + ?Sized,
        P: PointAccessor + ?Sized,
    {
        ensure_dim(point, self.dimension)?;
        ensure_finite(point)?;
        if self.leaf_index_contains(point_idx) {
            return Err(RcfError::InvalidConfig(format!(
                "RandomCutTree::add: point_idx {point_idx} already present"
            )));
        }

        let Some(root) = self.root else {
            let leaf = self.store.add_leaf(point_idx, None, 1)?;
            self.leaf_index_set(point_idx, leaf);
            self.root = Some(leaf);
            return Ok(());
        };

        self.insert_at(root, point_idx, point, points, rng)?;
        Ok(())
    }

    /// Recursive insertion. Returns the (possibly new) `NodeRef` that
    /// occupies the slot previously held by `n`.
    fn insert_at<R, P>(
        &mut self,
        n: NodeRef,
        point_idx: usize,
        point: &[f64],
        points: &P,
        rng: &mut R,
    ) -> RcfResult<NodeRef>
    where
        R: RngCore + ?Sized,
        P: PointAccessor + ?Sized,
    {
        let n_bbox = self.bbox_of(n, points)?;

        // Sample over the *virtual* augmented bbox — no allocation
        // unless we end up materialising the cached bbox for the
        // splice path below.
        if n_bbox.augmented_range_sum(point) <= 0.0 {
            // Coincident with an existing leaf — bump its mass.
            drop(n_bbox);
            return self.absorb_duplicate(n, point_idx);
        }

        let cut = n_bbox.augmented_random_cut(point, rng)?;
        let isolates = isolates_point(&cut, point, &n_bbox);

        if isolates {
            // Materialise the augmented bbox once — it becomes the
            // cached bbox of the new internal node we are about to
            // splice in (so the allocation is unavoidable here).
            let mut augmented: BoundingBox = (*n_bbox).clone();
            drop(n_bbox);
            augmented.extend(point)?;
            self.splice_new_internal(n, point_idx, point, cut, augmented)
        } else {
            drop(n_bbox);
            self.descend_or_split(n, point_idx, point, points, rng)
        }
    }

    /// Absorb a duplicate point into an existing leaf and propagate
    /// the mass increment up to the root.
    fn absorb_duplicate(&mut self, n: NodeRef, point_idx: usize) -> RcfResult<NodeRef> {
        if !n.is_leaf() {
            return Err(RcfError::InvalidConfig(
                "RandomCutTree::absorb_duplicate called on internal node".into(),
            ));
        }
        let mass = self.store.node(n)?.mass();
        self.store.set_mass(n, mass + 1)?;
        self.leaf_index_set(point_idx, n);
        let mut cur = n;
        while let Some(parent) = self.store.parent(cur)? {
            let m = self.store.node(parent)?.mass();
            self.store.set_mass(parent, m + 1)?;
            cur = parent;
        }
        Ok(n)
    }

    /// Splice a new internal node at the position currently held by
    /// `n`, using `cut` as the partition over `bbox`.
    fn splice_new_internal(
        &mut self,
        n: NodeRef,
        point_idx: usize,
        point: &[f64],
        cut: Cut,
        bbox: BoundingBox,
    ) -> RcfResult<NodeRef> {
        let new_leaf = self.store.add_leaf(point_idx, None, 1)?;
        self.leaf_index_set(point_idx, new_leaf);

        let parent_of_n = self.store.parent(n)?;
        let n_mass = self.store.node(n)?.mass();
        let new_mass = n_mass + 1;

        let (left, right) = if cut.left_of(point) {
            (new_leaf, n)
        } else {
            (n, new_leaf)
        };

        let new_internal =
            self.store
                .add_internal(cut, bbox, left, right, parent_of_n, new_mass)?;

        self.store.set_parent(new_leaf, Some(new_internal))?;
        self.store.set_parent(n, Some(new_internal))?;

        self.replace_in_parent(parent_of_n, n, new_internal)?;

        if let Some(parent) = parent_of_n {
            self.update_ancestors_after_insert(parent, point)?;
        }

        Ok(new_internal)
    }

    /// Cut did not isolate — descend into the matching subtree along
    /// the existing internal cut, updating mass and bbox on the way
    /// back up.
    fn descend_or_split<R, P>(
        &mut self,
        n: NodeRef,
        point_idx: usize,
        point: &[f64],
        points: &P,
        rng: &mut R,
    ) -> RcfResult<NodeRef>
    where
        R: RngCore + ?Sized,
        P: PointAccessor + ?Sized,
    {
        let (existing_cut, left, right) = match self.store.node(n)? {
            Node::Internal {
                cut, left, right, ..
            } => (*cut, *left, *right),
            Node::Leaf { .. } => {
                // Cut over a non-degenerate augmented bbox always
                // isolates one of two distinct points; reaching here
                // would indicate a bug elsewhere.
                return Err(RcfError::InvalidConfig(
                    "RandomCutTree::descend_or_split: leaf reached without isolation".into(),
                ));
            }
        };

        let go_left = existing_cut.left_of(point);
        let next = if go_left { left } else { right };
        let new_child = self.insert_at(next, point_idx, point, points, rng)?;

        // Update child pointer in case the descendant call replaced its slot.
        if new_child.raw() != next.raw() {
            if go_left {
                self.store.set_internal_children(n, new_child, right)?;
            } else {
                self.store.set_internal_children(n, left, new_child)?;
            }
            self.store.set_parent(new_child, Some(n))?;
        }

        // Mass + bbox already updated by insert_at via the
        // splice/absorb helpers' own ancestor walks.
        Ok(n)
    }

    /// Update the parent's child pointer after `old` is replaced by
    /// `new`, or update `self.root` when there is no parent.
    fn replace_in_parent(
        &mut self,
        parent_opt: Option<NodeRef>,
        old: NodeRef,
        new: NodeRef,
    ) -> RcfResult<()> {
        let Some(parent) = parent_opt else {
            self.root = Some(new);
            return Ok(());
        };
        let (l, r) = match self.store.node(parent)? {
            Node::Internal { left, right, .. } => (*left, *right),
            Node::Leaf { .. } => {
                return Err(RcfError::InvalidConfig(
                    "RandomCutTree::replace_in_parent: parent is leaf".into(),
                ));
            }
        };
        if l.raw() == old.raw() {
            self.store.set_internal_children(parent, new, r)?;
        } else if r.raw() == old.raw() {
            self.store.set_internal_children(parent, l, new)?;
        } else {
            return Err(RcfError::InvalidConfig(
                "RandomCutTree::replace_in_parent: orphan".into(),
            ));
        }
        Ok(())
    }

    /// Walk from `start` up to the root incrementing mass and
    /// extending each cached internal bounding box by `point` —
    /// in-place extend via [`NodeStore::node_mut`] avoids the
    /// `bbox.clone() + set_internal_bbox` round trip on every level.
    fn update_ancestors_after_insert(&mut self, start: NodeRef, point: &[f64]) -> RcfResult<()> {
        let mut cur = Some(start);
        while let Some(node) = cur {
            let parent = match self.store.node_mut(node)? {
                Node::Internal {
                    bbox, mass, parent, ..
                } => {
                    *mass += 1;
                    bbox.extend(point)?;
                    *parent
                }
                Node::Leaf { mass, parent, .. } => {
                    *mass += 1;
                    *parent
                }
            };
            cur = parent;
        }
        Ok(())
    }

    /// Remove the leaf currently mapped to `point_idx`. When the leaf
    /// has mass `> 1` (duplicate point), only the mass is decremented
    /// and the leaf is preserved.
    ///
    /// # Errors
    ///
    /// - [`RcfError::InvalidConfig`] when `point_idx` is not present.
    /// - [`RcfError::OutOfBounds`] when the underlying store cannot
    ///   look up a referenced point.
    pub fn delete<P>(&mut self, point_idx: usize, points: &P) -> RcfResult<()>
    where
        P: PointAccessor + ?Sized,
    {
        let leaf = self.leaf_index_get(point_idx).ok_or_else(|| {
            RcfError::InvalidConfig(format!(
                "RandomCutTree::delete: point_idx {point_idx} not present"
            ))
        })?;

        let leaf_mass = self.store.node(leaf)?.mass();
        if leaf_mass > 1 {
            self.store.set_mass(leaf, leaf_mass - 1)?;
            let mut cur = leaf;
            while let Some(parent) = self.store.parent(cur)? {
                let m = self.store.node(parent)?.mass();
                self.store.set_mass(parent, m - 1)?;
                cur = parent;
            }
            // Drop this idx from the reverse index — the leaf still
            // represents the other copies of the point under their own
            // point_idx, but `point_idx` itself is gone.
            self.leaf_index_clear(point_idx);
            return Ok(());
        }

        // mass == 1: remove the leaf entirely.
        let parent_of_leaf = self.store.parent(leaf)?;
        self.leaf_index_clear(point_idx);
        self.store.delete(leaf)?;

        let Some(parent) = parent_of_leaf else {
            self.root = None;
            return Ok(());
        };

        let sibling = match self.store.node(parent)? {
            Node::Internal { left, right, .. } => {
                if left.raw() == leaf.raw() {
                    *right
                } else {
                    *left
                }
            }
            Node::Leaf { .. } => {
                return Err(RcfError::InvalidConfig(
                    "RandomCutTree::delete: parent is leaf".into(),
                ));
            }
        };

        let grandparent = self.store.parent(parent)?;
        self.store.set_parent(sibling, grandparent)?;
        self.store.delete(parent)?;

        match grandparent {
            None => {
                self.root = Some(sibling);
            }
            Some(gp) => {
                let (l, r) = match self.store.node(gp)? {
                    Node::Internal { left, right, .. } => (*left, *right),
                    Node::Leaf { .. } => {
                        return Err(RcfError::InvalidConfig(
                            "RandomCutTree::delete: grandparent is leaf".into(),
                        ));
                    }
                };
                if l.raw() == parent.raw() {
                    self.store.set_internal_children(gp, sibling, r)?;
                } else if r.raw() == parent.raw() {
                    self.store.set_internal_children(gp, l, sibling)?;
                } else {
                    return Err(RcfError::InvalidConfig(
                        "RandomCutTree::delete: parent not registered with grandparent".into(),
                    ));
                }
                self.recompute_ancestors(gp, points)?;
            }
        }

        Ok(())
    }

    /// Walk from `start` up to the root decrementing mass and
    /// recomputing each internal bounding box from its (post-merge)
    /// children.
    fn recompute_ancestors<P>(&mut self, start: NodeRef, points: &P) -> RcfResult<()>
    where
        P: PointAccessor + ?Sized,
    {
        let mut cur = Some(start);
        while let Some(node) = cur {
            let m = self.store.node(node)?.mass();
            self.store.set_mass(node, m - 1)?;
            if node.is_internal() {
                let new_bbox = self.compute_internal_bbox(node, points)?;
                self.store.set_internal_bbox(node, new_bbox)?;
            }
            cur = self.store.parent(node)?;
        }
        Ok(())
    }

    /// Compute the bounding box of an internal node from its children.
    fn compute_internal_bbox<P>(&self, n: NodeRef, points: &P) -> RcfResult<BoundingBox>
    where
        P: PointAccessor + ?Sized,
    {
        let (left, right) = match self.store.node(n)? {
            Node::Internal { left, right, .. } => (*left, *right),
            Node::Leaf { .. } => {
                return Err(RcfError::InvalidConfig(
                    "RandomCutTree::compute_internal_bbox called on leaf".into(),
                ));
            }
        };
        let lb = self.bbox_of(left, points)?;
        let rb = self.bbox_of(right, points)?;
        lb.merged(&rb)
    }

    /// Borrow or build the bounding box of any node (internal: cached
    /// — borrowed via [`Cow::Borrowed`] to skip an allocation; leaf:
    /// built on the fly from the point store entry as
    /// [`Cow::Owned`]).
    fn bbox_of<'a, P>(&'a self, n: NodeRef, points: &'a P) -> RcfResult<Cow<'a, BoundingBox>>
    where
        P: PointAccessor + ?Sized,
    {
        match self.store.node(n)? {
            Node::Internal { bbox, .. } => Ok(Cow::Borrowed(bbox)),
            Node::Leaf { point_idx, .. } => {
                let p = points.point(*point_idx).ok_or(RcfError::OutOfBounds {
                    index: *point_idx,
                    len: 0,
                })?;
                Ok(Cow::Owned(BoundingBox::from_point(p)?))
            }
        }
    }

    /// Walk from the root to the leaf matching `point` along the
    /// stored cuts, dispatching callbacks to `visitor`. Returns the
    /// visitor's final output.
    ///
    /// # Errors
    ///
    /// - [`RcfError::EmptyForest`] when the tree is empty.
    /// - [`RcfError::DimensionMismatch`] when `point.len() != self.dimension()`.
    /// - [`RcfError::NaNValue`] when `point` contains a non-finite component.
    pub fn traverse<V: Visitor>(&self, point: &[f64], mut visitor: V) -> RcfResult<V::Output> {
        ensure_dim(point, self.dimension)?;
        ensure_finite(point)?;
        let Some(root) = self.root else {
            return Err(RcfError::EmptyForest);
        };
        self.traverse_inner(root, point, 0, &mut visitor)?;
        Ok(visitor.result())
    }

    /// Recursive helper for [`traverse`](Self::traverse).
    fn traverse_inner<V: Visitor>(
        &self,
        n: NodeRef,
        point: &[f64],
        depth: usize,
        visitor: &mut V,
    ) -> RcfResult<()> {
        match self.store.node(n)? {
            Node::Leaf {
                mass, point_idx, ..
            } => {
                visitor.accept_leaf(depth, *mass, *point_idx);
                Ok(())
            }
            Node::Internal {
                cut,
                bbox,
                mass,
                left,
                right,
                ..
            } => {
                if visitor.needs_per_dim_prob() {
                    let (prob, per_dim) = bbox.probability_of_cut(point)?;
                    visitor.accept_internal(depth, *mass, cut, bbox, prob, &per_dim);
                } else {
                    let prob = bbox.total_probability_of_cut(point)?;
                    visitor.accept_internal(depth, *mass, cut, bbox, prob, &[]);
                }
                let next = if cut.left_of(point) { *left } else { *right };
                self.traverse_inner(next, point, depth + 1, visitor)
            }
        }
    }
}

/// Whether `cut` strictly isolates `point` from `n_bbox` (i.e. `point`
/// ends up alone on one side of the hyperplane).
#[inline]
fn isolates_point(cut: &Cut, point: &[f64], n_bbox: &BoundingBox) -> bool {
    let d = cut.dim();
    let v = cut.value();
    let p_d = point[d];
    let n_min = n_bbox.min()[d];
    let n_max = n_bbox.max()[d];
    // Case 1: point alone on the left (`p_d <= v < n_min`).
    if p_d <= v && v < n_min {
        return true;
    }
    // Case 2: point alone on the right (`n_max <= v < p_d`).
    if n_max <= v && v < p_d {
        return true;
    }
    false
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert exact equality on bounding-box bounds.
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::visitor::Visitor;

    /// Visitor that records the path it observed.
    struct PathRecorder {
        depths: Vec<usize>,
        leaf_idx: Option<usize>,
    }
    impl PathRecorder {
        fn new() -> Self {
            Self {
                depths: Vec::new(),
                leaf_idx: None,
            }
        }
    }
    impl Visitor for PathRecorder {
        type Output = (Vec<usize>, Option<usize>);
        fn accept_internal(
            &mut self,
            depth: usize,
            _mass: u64,
            _cut: &Cut,
            _bbox: &BoundingBox,
            _prob: f64,
            _per_dim: &[f64],
        ) {
            self.depths.push(depth);
        }
        fn accept_leaf(&mut self, depth: usize, _mass: u64, point_idx: usize) {
            self.depths.push(depth);
            self.leaf_idx = Some(point_idx);
        }
        fn result(self) -> Self::Output {
            (self.depths, self.leaf_idx)
        }
    }

    fn fresh_rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    #[test]
    fn new_rejects_zero_dimension() {
        assert!(matches!(
            RandomCutTree::new(8, 0).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn new_rejects_zero_capacity() {
        assert!(matches!(
            RandomCutTree::new(0, 2).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn empty_tree_root_is_none() {
        let t = RandomCutTree::new(8, 2).unwrap();
        assert!(t.root().is_none());
        assert_eq!(t.distinct_point_count(), 0);
        assert_eq!(t.dimension(), 2);
    }

    #[test]
    fn add_first_point_creates_root_leaf() {
        let mut t = RandomCutTree::new(8, 2).unwrap();
        let mut points: Vec<Vec<f64>> = vec![vec![1.0, 2.0]];
        let mut rng = fresh_rng(42);
        t.add(0, &points[0].clone(), &points, &mut rng).unwrap();
        let root = t.root().unwrap();
        assert!(root.is_leaf());
        assert_eq!(t.distinct_point_count(), 1);
        // Cleanup: ensure points contain insertion (mock forest workflow).
        points.push(vec![0.0, 0.0]);
        let _ = points;
    }

    #[test]
    fn add_two_points_creates_internal_root() {
        let mut t = RandomCutTree::new(8, 2).unwrap();
        let p0 = vec![0.0, 0.0];
        let p1 = vec![1.0, 1.0];
        let points = vec![p0.clone(), p1.clone()];
        let mut rng = fresh_rng(7);
        t.add(0, &p0, &points, &mut rng).unwrap();
        t.add(1, &p1, &points, &mut rng).unwrap();
        let root = t.root().unwrap();
        assert!(root.is_internal());
        assert_eq!(t.distinct_point_count(), 2);
        assert_eq!(t.store().node(root).unwrap().mass(), 2);
    }

    #[test]
    fn add_rejects_dimension_mismatch() {
        let mut t = RandomCutTree::new(8, 3).unwrap();
        let points: Vec<Vec<f64>> = vec![];
        let mut rng = fresh_rng(1);
        assert!(matches!(
            t.add(0, &[1.0, 2.0], &points, &mut rng).unwrap_err(),
            RcfError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn add_rejects_non_finite() {
        let mut t = RandomCutTree::new(8, 2).unwrap();
        let points: Vec<Vec<f64>> = vec![];
        let mut rng = fresh_rng(1);
        assert!(matches!(
            t.add(0, &[1.0, f64::NAN], &points, &mut rng).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn add_rejects_duplicate_point_idx() {
        let mut t = RandomCutTree::new(8, 2).unwrap();
        let p = vec![0.0, 0.0];
        let points = vec![p.clone()];
        let mut rng = fresh_rng(1);
        t.add(0, &p, &points, &mut rng).unwrap();
        assert!(matches!(
            t.add(0, &p, &points, &mut rng).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn duplicate_coordinate_increments_leaf_mass() {
        let mut t = RandomCutTree::new(8, 2).unwrap();
        let p = vec![3.0, 4.0];
        let mut points = vec![p.clone()];
        let mut rng = fresh_rng(1);
        t.add(0, &p, &points, &mut rng).unwrap();
        // Insert the same coordinates with a different idx — should
        // collapse into the existing leaf.
        points.push(p.clone());
        t.add(1, &p, &points, &mut rng).unwrap();
        let root = t.root().unwrap();
        assert!(root.is_leaf(), "single-point tree stays a leaf");
        assert_eq!(t.store().node(root).unwrap().mass(), 2);
        assert_eq!(t.distinct_point_count(), 2);
    }

    #[test]
    fn add_many_distinct_points_keeps_mass_invariant() {
        let mut t = RandomCutTree::new(64, 4).unwrap();
        let mut rng = fresh_rng(99);
        let mut points: Vec<Vec<f64>> = Vec::new();
        for i in 0_u32..32 {
            let f = f64::from(i);
            let p = vec![f, f * 2.0, f * 0.5, -f];
            points.push(p.clone());
            t.add(i as usize, &p, &points, &mut rng).unwrap();
        }
        let root = t.root().unwrap();
        assert!(root.is_internal());
        assert_eq!(t.store().node(root).unwrap().mass(), 32);
        assert_eq!(t.distinct_point_count(), 32);
    }

    #[test]
    fn delete_unknown_point_idx_is_err() {
        let mut t = RandomCutTree::new(8, 2).unwrap();
        let points: Vec<Vec<f64>> = vec![];
        assert!(matches!(
            t.delete(99, &points).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn delete_root_leaf_clears_tree() {
        let mut t = RandomCutTree::new(8, 2).unwrap();
        let p = vec![1.0, 2.0];
        let points = vec![p.clone()];
        let mut rng = fresh_rng(1);
        t.add(0, &p, &points, &mut rng).unwrap();
        t.delete(0, &points).unwrap();
        assert!(t.root().is_none());
        assert_eq!(t.distinct_point_count(), 0);
        assert_eq!(t.store().live_count(), 0);
    }

    #[test]
    fn delete_one_of_two_points_leaves_sibling_as_root() {
        let mut t = RandomCutTree::new(8, 2).unwrap();
        let p0 = vec![0.0, 0.0];
        let p1 = vec![1.0, 1.0];
        let points = vec![p0.clone(), p1.clone()];
        let mut rng = fresh_rng(7);
        t.add(0, &p0, &points, &mut rng).unwrap();
        t.add(1, &p1, &points, &mut rng).unwrap();
        t.delete(0, &points).unwrap();
        let root = t.root().unwrap();
        assert!(root.is_leaf());
        match t.store().node(root).unwrap() {
            Node::Leaf {
                point_idx, mass, ..
            } => {
                assert_eq!(*point_idx, 1);
                assert_eq!(*mass, 1);
            }
            Node::Internal { .. } => panic!("expected leaf"),
        }
        assert_eq!(t.store().live_count(), 1);
    }

    #[test]
    fn delete_duplicate_decrements_mass_only() {
        let mut t = RandomCutTree::new(8, 2).unwrap();
        let p = vec![1.0, 1.0];
        let mut points = vec![p.clone(), p.clone()];
        let mut rng = fresh_rng(1);
        t.add(0, &p, &points, &mut rng).unwrap();
        t.add(1, &p, &points, &mut rng).unwrap();
        let root = t.root().unwrap();
        assert_eq!(t.store().node(root).unwrap().mass(), 2);
        t.delete(1, &points).unwrap();
        assert_eq!(t.store().node(root).unwrap().mass(), 1);
        assert!(t.root().unwrap().is_leaf());
        // Only one of the two idx mappings survives — the leaf still
        // holds the original point_idx 0 because we absorbed 1 into
        // its leaf and removed it on delete.
        assert!(!t.contains(1));
        assert!(t.contains(0));
        // Remaining mass and point_count.
        assert_eq!(t.store().node(root).unwrap().mass(), 1);
        // Update points to keep len consistent.
        points.pop();
    }

    #[test]
    fn delete_then_re_add_keeps_capacity_bounded() {
        let mut t = RandomCutTree::new(4, 2).unwrap();
        let mut rng = fresh_rng(11);
        let mut points = Vec::new();
        // Seed initial 4 distinct points.
        let mut live_idxs: Vec<usize> = Vec::new();
        for i in 0_u32..4 {
            let f = f64::from(i);
            let p = vec![f, f + 1.0];
            points.push(p.clone());
            let idx = i as usize;
            t.add(idx, &p, &points, &mut rng).unwrap();
            live_idxs.push(idx);
        }
        // Delete the oldest live idx + re-add a fresh one; capacity must
        // hold across many iterations because freed slots are reused.
        for _ in 0..10 {
            let old = live_idxs.remove(0);
            t.delete(old, &points).unwrap();
            let new_idx = points.len();
            let p = points[old].clone();
            points.push(p.clone());
            t.add(new_idx, &p, &points, &mut rng).unwrap();
            live_idxs.push(new_idx);
        }
        assert_eq!(t.distinct_point_count(), 4);
    }

    #[test]
    fn traverse_empty_tree_is_err() {
        let t = RandomCutTree::new(4, 2).unwrap();
        let v = PathRecorder::new();
        assert!(matches!(
            t.traverse(&[1.0, 2.0], v).unwrap_err(),
            RcfError::EmptyForest
        ));
    }

    #[test]
    fn traverse_single_leaf_visits_only_leaf() {
        let mut t = RandomCutTree::new(4, 2).unwrap();
        let p = vec![1.0, 2.0];
        let points = vec![p.clone()];
        let mut rng = fresh_rng(0);
        t.add(0, &p, &points, &mut rng).unwrap();
        let v = PathRecorder::new();
        let (depths, leaf_idx) = t.traverse(&p, v).unwrap();
        assert_eq!(depths, vec![0]);
        assert_eq!(leaf_idx, Some(0));
    }

    #[test]
    fn traverse_visits_in_root_to_leaf_order() {
        let mut t = RandomCutTree::new(8, 2).unwrap();
        let p0 = vec![0.0, 0.0];
        let p1 = vec![10.0, 10.0];
        let points = vec![p0.clone(), p1.clone()];
        let mut rng = fresh_rng(123);
        t.add(0, &p0, &points, &mut rng).unwrap();
        t.add(1, &p1, &points, &mut rng).unwrap();
        let v = PathRecorder::new();
        let (depths, leaf_idx) = t.traverse(&p1, v).unwrap();
        assert!(depths.windows(2).all(|w| w[0] < w[1]));
        assert!(leaf_idx == Some(0) || leaf_idx == Some(1));
    }

    #[test]
    fn traverse_rejects_dim_mismatch_and_nan() {
        let mut t = RandomCutTree::new(4, 2).unwrap();
        let p = vec![1.0, 2.0];
        let points = vec![p.clone()];
        let mut rng = fresh_rng(0);
        t.add(0, &p, &points, &mut rng).unwrap();
        assert!(matches!(
            t.traverse(&[1.0], PathRecorder::new()).unwrap_err(),
            RcfError::DimensionMismatch { .. }
        ));
        assert!(matches!(
            t.traverse(&[f64::NAN, 0.0], PathRecorder::new())
                .unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn deterministic_tree_under_fixed_seed() {
        fn build(seed: u64) -> RandomCutTree {
            let mut t = RandomCutTree::new(64, 3).unwrap();
            let mut rng = fresh_rng(seed);
            let mut points = Vec::new();
            for i in 0_u32..16 {
                let f = f64::from(i);
                let p = vec![f, f * 2.0, f * 0.25 + 1.0];
                points.push(p.clone());
                t.add(i as usize, &p, &points, &mut rng).unwrap();
            }
            t
        }
        let t1 = build(42);
        let t2 = build(42);
        assert_eq!(t1.store().live_count(), t2.store().live_count());
        assert_eq!(t1.distinct_point_count(), t2.distinct_point_count());
        assert_eq!(
            t1.store().live_internal_count(),
            t2.store().live_internal_count()
        );
        assert_eq!(t1.store().live_leaf_count(), t2.store().live_leaf_count());
    }

    #[test]
    fn isolates_point_above_separated() {
        let mut bbox = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        bbox.extend(&[1.0, 1.0]).unwrap();
        // Point well above on dim 0; cut at value=5 separates them.
        assert!(isolates_point(&Cut::new(0, 5.0), &[10.0, 0.5], &bbox));
        // Cut INSIDE bbox does NOT isolate.
        assert!(!isolates_point(&Cut::new(0, 0.5), &[10.0, 0.5], &bbox));
    }

    #[test]
    fn isolates_point_below_separated() {
        let mut bbox = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        bbox.extend(&[1.0, 1.0]).unwrap();
        assert!(isolates_point(&Cut::new(1, -2.0), &[0.5, -10.0], &bbox));
    }

    #[test]
    fn isolates_point_inside_never() {
        let mut bbox = BoundingBox::from_point(&[0.0, 0.0]).unwrap();
        bbox.extend(&[10.0, 10.0]).unwrap();
        assert!(!isolates_point(&Cut::new(0, 5.0), &[5.0, 5.0], &bbox));
    }

    // Property test: under uniform-random insertions, the tree depth
    // stays within `4 · ⌈log₂ N⌉ + 4` — the "expected `O(log n)`"
    // bound from Guha 2016 §2 with a generous constant to absorb
    // the natural variance of random cuts. Backs RCF.4 AC #5.
    proptest::proptest! {
        #![proptest_config(proptest::test_runner::Config { cases: 32, ..proptest::test_runner::Config::default() })]
        #[test]
        fn depth_bounded_under_uniform_inserts(seed in 0_u64..10_000) {
            const N: usize = 64;
            const D: usize = 4;
            #[allow(clippy::cast_possible_truncation)]
            let mut t = RandomCutTree::new(N as u32, D).unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut points: Vec<Vec<f64>> = Vec::with_capacity(N);

            for i in 0..N {
                let p: Vec<f64> = (0..D)
                    .map(|_| <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng) * 100.0)
                    .collect();
                points.push(p.clone());
                t.add(i, &p, &points, &mut rng).unwrap();
            }

            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let log2_n = (N as f64).log2().ceil() as usize;
            let bound = 4 * log2_n + 4;
            let depth = t.max_depth().expect("non-empty tree");
            proptest::prop_assert!(
                depth <= bound,
                "depth = {} exceeds bound 4·⌈log₂ {}⌉ + 4 = {} (seed={})",
                depth, N, bound, seed,
            );
        }
    }
}
