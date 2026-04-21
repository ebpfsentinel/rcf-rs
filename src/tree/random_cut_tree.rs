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
//! The tree never owns point coordinates — callers (the forest
//! layer) hand a [`PointAccessor`] in for any operation that needs
//! to know a leaf's location.

use alloc::borrow::Cow;
use alloc::format;
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;
use rand::RngCore;

use crate::domain::{BoundingBox, Cut, ensure_dim, ensure_finite};
use crate::error::{RcfError, RcfResult};
use crate::tree::node::{NodeRef, NodeView, NodeViewMut};
use crate::tree::node_store::NodeStore;
use crate::visitor::Visitor;

/// Borrow points by index. Implemented by the forest layer and by
/// `Vec<[f64; D]>` for in-tree tests.
///
/// `D` matches the tree's compile-time dimensionality so the returned
/// reference is a fixed-size array rather than a slice — callers can
/// always pass it to [`BoundingBox::from_point`] without paying for a
/// runtime length check.
///
/// # Examples
///
/// ```
/// use rcf_rs::tree::PointAccessor;
///
/// let v: Vec<[f64; 2]> = vec![[1.0, 2.0]];
/// let p: &[f64; 2] = <Vec<[f64; 2]> as PointAccessor<2>>::point(&v, 0).unwrap();
/// assert_eq!(p, &[1.0, 2.0]);
/// ```
pub trait PointAccessor<const D: usize> {
    /// Borrow the point stored at `idx`, or return `None` when it
    /// does not exist.
    fn point(&self, idx: usize) -> Option<&[f64; D]>;
}

impl<const D: usize> PointAccessor<D> for Vec<[f64; D]> {
    fn point(&self, idx: usize) -> Option<&[f64; D]> {
        self.get(idx)
    }
}

impl<const D: usize> PointAccessor<D> for [[f64; D]] {
    fn point(&self, idx: usize) -> Option<&[f64; D]> {
        self.get(idx)
    }
}

/// Incrementally-maintained random cut tree over up to `capacity`
/// distinct `D`-dimensional points.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use rcf_rs::RandomCutTree;
///
/// let mut tree = RandomCutTree::<2>::new(8).unwrap();
/// let p = [1.0_f64, 2.0];
/// let points = vec![p];
/// let mut rng = ChaCha8Rng::seed_from_u64(42);
/// tree.add(0, &p, &points, &mut rng).unwrap();
/// assert!(tree.root().unwrap().is_leaf());
/// ```
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RandomCutTree<const D: usize> {
    /// Root reference; `None` when the tree holds no live leaves.
    root: Option<NodeRef>,
    /// Backing storage for all live nodes.
    store: NodeStore<D>,
    /// Reverse index: `point_idx → leaf NodeRef` for `O(1)` deletions.
    /// Backed by a sparse `Vec<Option<NodeRef>>` indexed by `point_idx`
    /// — the forest's `PointStore` reuses freed slots so the indices
    /// stay dense in steady state. Maintained alongside
    /// `distinct_count` so `distinct_point_count` stays `O(1)`.
    leaf_index: Vec<Option<NodeRef>>,
    /// Cached count of `Some` entries in `leaf_index`.
    distinct_count: usize,
}

impl<const D: usize> RandomCutTree<D> {
    /// Build an empty tree with room for `capacity` distinct points.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `D == 0`, `capacity ==
    /// 0`, or `capacity` exceeds the store limit.
    pub fn new(capacity: u32) -> RcfResult<Self> {
        if D == 0 {
            return Err(RcfError::InvalidConfig(
                "RandomCutTree dimension must be > 0".into(),
            ));
        }
        Ok(Self {
            root: None,
            store: NodeStore::<D>::new(capacity)?,
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

    /// Configured dimensionality (compile-time `D`).
    #[must_use]
    pub const fn dimension(&self) -> usize {
        D
    }

    /// Number of distinct points currently stored (each leaf counts
    /// once regardless of its mass).
    #[must_use]
    #[inline]
    pub fn distinct_point_count(&self) -> usize {
        self.distinct_count
    }

    /// Borrow the underlying node store. Used by tests and
    /// persistence.
    #[must_use]
    pub fn store(&self) -> &NodeStore<D> {
        &self.store
    }

    /// Whether the tree currently stores `point_idx`.
    #[must_use]
    #[inline]
    pub fn contains(&self, point_idx: usize) -> bool {
        self.leaf_index_contains(point_idx)
    }

    /// `NodeRef` of the leaf currently mapped to `point_idx`, or
    /// `None` when the reservoir has evicted it (or never admitted
    /// it). Exposed so forest-level scoring paths that need to walk
    /// the leaf's ancestor chain (e.g. `codisp`-style probe scoring)
    /// can locate the leaf without re-traversing from the root.
    #[must_use]
    #[inline]
    pub fn leaf_of(&self, point_idx: usize) -> Option<NodeRef> {
        self.leaf_index_get(point_idx)
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
        match self.store.view(n)? {
            NodeView::Leaf(_) => Ok(depth),
            NodeView::Internal(i) => {
                let l = self.subtree_depth(i.left, depth + 1)?;
                let r = self.subtree_depth(i.right, depth + 1)?;
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
        P: PointAccessor<D> + ?Sized,
    {
        ensure_dim(point, D)?;
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
    // `drop(n_bbox)` releases the borrow of `self.store` held by the
    // `Cow<BoundingBox<D>>` before we call `&mut self` helpers below.
    // With `[f64; D]` storage `Cow<BoundingBox<D>>` no longer
    // implements `Drop` directly, so clippy flags the explicit drops
    // as redundant — they are not, they terminate the borrow.
    #[allow(clippy::drop_non_drop)]
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
        P: PointAccessor<D> + ?Sized,
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
            let mut augmented: BoundingBox<D> = (*n_bbox).clone();
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
        let mass = self.store.view(n)?.mass();
        self.store.set_mass(n, mass + 1)?;
        self.leaf_index_set(point_idx, n);
        let mut cur = n;
        while let Some(parent) = self.store.parent(cur)? {
            let m = self.store.view(parent)?.mass();
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
        bbox: BoundingBox<D>,
    ) -> RcfResult<NodeRef> {
        let new_leaf = self.store.add_leaf(point_idx, None, 1)?;
        self.leaf_index_set(point_idx, new_leaf);

        let parent_of_n = self.store.parent(n)?;
        let n_mass = self.store.view(n)?.mass();
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
        P: PointAccessor<D> + ?Sized,
    {
        let (existing_cut, left, right) = match self.store.view(n)? {
            NodeView::Internal(i) => (i.cut, i.left, i.right),
            NodeView::Leaf(_) => {
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
        let (l, r) = match self.store.view(parent)? {
            NodeView::Internal(i) => (i.left, i.right),
            NodeView::Leaf(_) => {
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
    /// in-place extend via [`NodeStore::view_mut`] avoids the
    /// `bbox.clone() + set_internal_bbox` round trip on every level.
    fn update_ancestors_after_insert(&mut self, start: NodeRef, point: &[f64]) -> RcfResult<()> {
        let mut cur = Some(start);
        while let Some(node) = cur {
            let parent = match self.store.view_mut(node)? {
                NodeViewMut::Internal(i) => {
                    i.mass += 1;
                    i.bbox.extend(point)?;
                    i.parent
                }
                NodeViewMut::Leaf(l) => {
                    l.mass += 1;
                    l.parent
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
        P: PointAccessor<D> + ?Sized,
    {
        let leaf = self.leaf_index_get(point_idx).ok_or_else(|| {
            RcfError::InvalidConfig(format!(
                "RandomCutTree::delete: point_idx {point_idx} not present"
            ))
        })?;

        let leaf_mass = self.store.view(leaf)?.mass();
        if leaf_mass > 1 {
            self.store.set_mass(leaf, leaf_mass - 1)?;
            let mut cur = leaf;
            while let Some(parent) = self.store.parent(cur)? {
                let m = self.store.view(parent)?.mass();
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

        let sibling = match self.store.view(parent)? {
            NodeView::Internal(i) => {
                if i.left.raw() == leaf.raw() {
                    i.right
                } else {
                    i.left
                }
            }
            NodeView::Leaf(_) => {
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
                let (l, r) = match self.store.view(gp)? {
                    NodeView::Internal(i) => (i.left, i.right),
                    NodeView::Leaf(_) => {
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
        P: PointAccessor<D> + ?Sized,
    {
        let mut cur = Some(start);
        while let Some(node) = cur {
            let m = self.store.view(node)?.mass();
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
    fn compute_internal_bbox<P>(&self, n: NodeRef, points: &P) -> RcfResult<BoundingBox<D>>
    where
        P: PointAccessor<D> + ?Sized,
    {
        let (left, right) = match self.store.view(n)? {
            NodeView::Internal(i) => (i.left, i.right),
            NodeView::Leaf(_) => {
                return Err(RcfError::InvalidConfig(
                    "RandomCutTree::compute_internal_bbox called on leaf".into(),
                ));
            }
        };
        let lb = self.bbox_of(left, points)?;
        let rb = self.bbox_of(right, points)?;
        Ok(lb.merged(&rb))
    }

    /// Borrow or build the bounding box of any node (internal: cached
    /// — borrowed via [`Cow::Borrowed`] to skip an allocation; leaf:
    /// built on the fly from the point store entry as
    /// [`Cow::Owned`]).
    fn bbox_of<'a, P>(&'a self, n: NodeRef, points: &'a P) -> RcfResult<Cow<'a, BoundingBox<D>>>
    where
        P: PointAccessor<D> + ?Sized,
    {
        match self.store.view(n)? {
            NodeView::Internal(i) => Ok(Cow::Borrowed(&i.bbox)),
            NodeView::Leaf(l) => {
                let p = points.point(l.point_idx).ok_or(RcfError::OutOfBounds {
                    index: l.point_idx,
                    len: 0,
                })?;
                Ok(Cow::Owned(BoundingBox::<D>::from_point(p)?))
            }
        }
    }

    /// Non-mutating codisp estimate — walks root → leaf following
    /// the stored cuts and accumulates the maximum per-depth ratio
    /// `sibling_mass / subtree_mass` across the descent path.
    ///
    /// Matches the shape of the mutating [`crate::RandomCutForest::score_codisp`]
    /// walk but **without** inserting the probe into the reservoir.
    /// The classical `codisp` promises a frozen baseline (AWS /
    /// rrcf); the mutating path's insert+delete cycle leaves
    /// reservoir points evicted permanently, eroding that baseline
    /// across long eval streams (observed on NAB `rogue_hold`:
    /// score drifts from 0.69 to 0.20 after ~5 k probes).
    ///
    /// This path preserves the frozen-baseline promise exactly,
    /// takes `&self` so it parallelises across trees, and costs
    /// `O(depth · D)` per call — typically cheaper than the
    /// mutating walk since there is no reservoir housekeeping.
    ///
    /// # Errors
    ///
    /// - [`RcfError::EmptyForest`] when the tree is empty.
    /// - [`RcfError::DimensionMismatch`] when `point.len() != D`.
    /// - [`RcfError::NaNValue`] when `point` contains a non-finite
    ///   component.
    pub fn codisp_stateless(&self, point: &[f64]) -> RcfResult<f64> {
        ensure_dim(point, D)?;
        ensure_finite(point)?;
        let Some(root) = self.root else {
            return Err(RcfError::EmptyForest);
        };
        let mut cur = root;
        let mut max_disp = 0.0_f64;
        loop {
            match self.store.view(cur)? {
                crate::tree::NodeView::Leaf(_) => return Ok(max_disp),
                crate::tree::NodeView::Internal(i) => {
                    let (next, sibling) = if i.cut.left_of(point) {
                        (i.left, i.right)
                    } else {
                        (i.right, i.left)
                    };
                    let next_mass = self.store.view(next)?.mass();
                    let sibling_mass = self.store.view(sibling)?.mass();
                    if next_mass > 0 {
                        #[allow(clippy::cast_precision_loss)]
                        let disp = sibling_mass as f64 / next_mass as f64;
                        if disp > max_disp {
                            max_disp = disp;
                        }
                    }
                    cur = next;
                }
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
    pub fn traverse<V: Visitor<D>>(&self, point: &[f64], mut visitor: V) -> RcfResult<V::Output> {
        ensure_dim(point, D)?;
        ensure_finite(point)?;
        let Some(root) = self.root else {
            return Err(RcfError::EmptyForest);
        };
        self.traverse_inner(root, point, 0, &mut visitor)?;
        Ok(visitor.result())
    }

    /// Recursive helper for [`traverse`](Self::traverse).
    fn traverse_inner<V: Visitor<D>>(
        &self,
        n: NodeRef,
        point: &[f64],
        depth: usize,
        visitor: &mut V,
    ) -> RcfResult<()> {
        match self.store.view(n)? {
            NodeView::Leaf(l) => {
                visitor.accept_leaf(depth, l.mass, l.point_idx);
                Ok(())
            }
            NodeView::Internal(i) => {
                if visitor.needs_per_dim_prob() {
                    let (prob, per_dim) = i.bbox.probability_of_cut(point)?;
                    visitor.accept_internal(depth, i.mass, &i.cut, &i.bbox, prob, &per_dim);
                } else {
                    let prob = i.bbox.total_probability_of_cut(point)?;
                    visitor.accept_internal(depth, i.mass, &i.cut, &i.bbox, prob, &[]);
                }
                let next = if i.cut.left_of(point) {
                    i.left
                } else {
                    i.right
                };
                self.traverse_inner(next, point, depth + 1, visitor)
            }
        }
    }
}

/// Whether `cut` strictly isolates `point` from `n_bbox` (i.e. `point`
/// ends up alone on one side of the hyperplane).
#[inline]
fn isolates_point<const D: usize>(cut: &Cut, point: &[f64], n_bbox: &BoundingBox<D>) -> bool {
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

    /// Visitor that records the path it observed. Generic over `D` so
    /// the trait impl matches every test tree dimensionality.
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
    impl<const D: usize> Visitor<D> for PathRecorder {
        type Output = (Vec<usize>, Option<usize>);
        fn accept_internal(
            &mut self,
            depth: usize,
            _mass: u64,
            _cut: &Cut,
            _bbox: &BoundingBox<D>,
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
            RandomCutTree::<0>::new(8).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn new_rejects_zero_capacity() {
        assert!(matches!(
            RandomCutTree::<2>::new(0).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn empty_tree_root_is_none() {
        let t = RandomCutTree::<2>::new(8).unwrap();
        assert!(t.root().is_none());
        assert_eq!(t.distinct_point_count(), 0);
        assert_eq!(t.dimension(), 2);
    }

    #[test]
    fn add_first_point_creates_root_leaf() {
        let mut t = RandomCutTree::<2>::new(8).unwrap();
        let points: Vec<[f64; 2]> = vec![[1.0, 2.0]];
        let mut rng = fresh_rng(42);
        t.add(0, &points[0], &points, &mut rng).unwrap();
        let root = t.root().unwrap();
        assert!(root.is_leaf());
        assert_eq!(t.distinct_point_count(), 1);
    }

    #[test]
    fn add_two_points_creates_internal_root() {
        let mut t = RandomCutTree::<2>::new(8).unwrap();
        let p0 = [0.0_f64, 0.0];
        let p1 = [1.0_f64, 1.0];
        let points = vec![p0, p1];
        let mut rng = fresh_rng(7);
        t.add(0, &p0, &points, &mut rng).unwrap();
        t.add(1, &p1, &points, &mut rng).unwrap();
        let root = t.root().unwrap();
        assert!(root.is_internal());
        assert_eq!(t.distinct_point_count(), 2);
        assert_eq!(t.store().view(root).unwrap().mass(), 2);
    }

    #[test]
    fn add_rejects_dimension_mismatch() {
        let mut t = RandomCutTree::<3>::new(8).unwrap();
        let points: Vec<[f64; 3]> = vec![];
        let mut rng = fresh_rng(1);
        assert!(matches!(
            t.add(0, &[1.0, 2.0], &points, &mut rng).unwrap_err(),
            RcfError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn add_rejects_non_finite() {
        let mut t = RandomCutTree::<2>::new(8).unwrap();
        let points: Vec<[f64; 2]> = vec![];
        let mut rng = fresh_rng(1);
        assert!(matches!(
            t.add(0, &[1.0, f64::NAN], &points, &mut rng).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn add_rejects_duplicate_point_idx() {
        let mut t = RandomCutTree::<2>::new(8).unwrap();
        let p = [0.0_f64, 0.0];
        let points = vec![p];
        let mut rng = fresh_rng(1);
        t.add(0, &p, &points, &mut rng).unwrap();
        assert!(matches!(
            t.add(0, &p, &points, &mut rng).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn duplicate_coordinate_increments_leaf_mass() {
        let mut t = RandomCutTree::<2>::new(8).unwrap();
        let p = [3.0_f64, 4.0];
        let mut points = vec![p];
        let mut rng = fresh_rng(1);
        t.add(0, &p, &points, &mut rng).unwrap();
        points.push(p);
        t.add(1, &p, &points, &mut rng).unwrap();
        let root = t.root().unwrap();
        assert!(root.is_leaf(), "single-point tree stays a leaf");
        assert_eq!(t.store().view(root).unwrap().mass(), 2);
        assert_eq!(t.distinct_point_count(), 2);
    }

    #[test]
    fn add_many_distinct_points_keeps_mass_invariant() {
        let mut t = RandomCutTree::<4>::new(64).unwrap();
        let mut rng = fresh_rng(99);
        let mut points: Vec<[f64; 4]> = Vec::new();
        for i in 0_u32..32 {
            let f = f64::from(i);
            let p = [f, f * 2.0, f * 0.5, -f];
            points.push(p);
            t.add(i as usize, &p, &points, &mut rng).unwrap();
        }
        let root = t.root().unwrap();
        assert!(root.is_internal());
        assert_eq!(t.store().view(root).unwrap().mass(), 32);
        assert_eq!(t.distinct_point_count(), 32);
    }

    #[test]
    fn delete_unknown_point_idx_is_err() {
        let mut t = RandomCutTree::<2>::new(8).unwrap();
        let points: Vec<[f64; 2]> = vec![];
        assert!(matches!(
            t.delete(99, &points).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn delete_root_leaf_clears_tree() {
        let mut t = RandomCutTree::<2>::new(8).unwrap();
        let p = [1.0_f64, 2.0];
        let points = vec![p];
        let mut rng = fresh_rng(1);
        t.add(0, &p, &points, &mut rng).unwrap();
        t.delete(0, &points).unwrap();
        assert!(t.root().is_none());
        assert_eq!(t.distinct_point_count(), 0);
        assert_eq!(t.store().live_count(), 0);
    }

    #[test]
    fn delete_one_of_two_points_leaves_sibling_as_root() {
        let mut t = RandomCutTree::<2>::new(8).unwrap();
        let p0 = [0.0_f64, 0.0];
        let p1 = [1.0_f64, 1.0];
        let points = vec![p0, p1];
        let mut rng = fresh_rng(7);
        t.add(0, &p0, &points, &mut rng).unwrap();
        t.add(1, &p1, &points, &mut rng).unwrap();
        t.delete(0, &points).unwrap();
        let root = t.root().unwrap();
        assert!(root.is_leaf());
        let leaf = t.store().leaf(root).unwrap();
        assert_eq!(leaf.point_idx, 1);
        assert_eq!(leaf.mass, 1);
        assert_eq!(t.store().live_count(), 1);
    }

    #[test]
    fn delete_duplicate_decrements_mass_only() {
        let mut t = RandomCutTree::<2>::new(8).unwrap();
        let p = [1.0_f64, 1.0];
        let mut points = vec![p, p];
        let mut rng = fresh_rng(1);
        t.add(0, &p, &points, &mut rng).unwrap();
        t.add(1, &p, &points, &mut rng).unwrap();
        let root = t.root().unwrap();
        assert_eq!(t.store().view(root).unwrap().mass(), 2);
        t.delete(1, &points).unwrap();
        assert_eq!(t.store().view(root).unwrap().mass(), 1);
        assert!(t.root().unwrap().is_leaf());
        assert!(!t.contains(1));
        assert!(t.contains(0));
        assert_eq!(t.store().view(root).unwrap().mass(), 1);
        points.pop();
    }

    #[test]
    fn delete_then_re_add_keeps_capacity_bounded() {
        let mut t = RandomCutTree::<2>::new(4).unwrap();
        let mut rng = fresh_rng(11);
        let mut points: Vec<[f64; 2]> = Vec::new();
        let mut live_idxs: Vec<usize> = Vec::new();
        for i in 0_u32..4 {
            let f = f64::from(i);
            let p = [f, f + 1.0];
            points.push(p);
            let idx = i as usize;
            t.add(idx, &p, &points, &mut rng).unwrap();
            live_idxs.push(idx);
        }
        for _ in 0..10 {
            let old = live_idxs.remove(0);
            t.delete(old, &points).unwrap();
            let new_idx = points.len();
            let p = points[old];
            points.push(p);
            t.add(new_idx, &p, &points, &mut rng).unwrap();
            live_idxs.push(new_idx);
        }
        assert_eq!(t.distinct_point_count(), 4);
    }

    #[test]
    fn traverse_empty_tree_is_err() {
        let t = RandomCutTree::<2>::new(4).unwrap();
        let v = PathRecorder::new();
        assert!(matches!(
            t.traverse(&[1.0, 2.0], v).unwrap_err(),
            RcfError::EmptyForest
        ));
    }

    #[test]
    fn traverse_single_leaf_visits_only_leaf() {
        let mut t = RandomCutTree::<2>::new(4).unwrap();
        let p = [1.0_f64, 2.0];
        let points = vec![p];
        let mut rng = fresh_rng(0);
        t.add(0, &p, &points, &mut rng).unwrap();
        let v = PathRecorder::new();
        let (depths, leaf_idx) = t.traverse(&p, v).unwrap();
        assert_eq!(depths, vec![0]);
        assert_eq!(leaf_idx, Some(0));
    }

    #[test]
    fn traverse_visits_in_root_to_leaf_order() {
        let mut t = RandomCutTree::<2>::new(8).unwrap();
        let p0 = [0.0_f64, 0.0];
        let p1 = [10.0_f64, 10.0];
        let points = vec![p0, p1];
        let mut rng = fresh_rng(123);
        t.add(0, &p0, &points, &mut rng).unwrap();
        t.add(1, &p1, &points, &mut rng).unwrap();
        let v = PathRecorder::new();
        let (depths, leaf_idx) = t.traverse(&p1, v).unwrap();
        assert!(depths.array_windows::<2>().all(|[a, b]| a < b));
        assert!(leaf_idx == Some(0) || leaf_idx == Some(1));
    }

    #[test]
    fn traverse_rejects_dim_mismatch_and_nan() {
        let mut t = RandomCutTree::<2>::new(4).unwrap();
        let p = [1.0_f64, 2.0];
        let points = vec![p];
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
        fn build(seed: u64) -> RandomCutTree<3> {
            let mut t = RandomCutTree::<3>::new(64).unwrap();
            let mut rng = fresh_rng(seed);
            let mut points: Vec<[f64; 3]> = Vec::new();
            for i in 0_u32..16 {
                let f = f64::from(i);
                let p = [f, f * 2.0, f * 0.25 + 1.0];
                points.push(p);
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
        let mut bbox = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
        bbox.extend(&[1.0, 1.0]).unwrap();
        assert!(isolates_point(&Cut::new(0, 5.0), &[10.0, 0.5], &bbox));
        assert!(!isolates_point(&Cut::new(0, 0.5), &[10.0, 0.5], &bbox));
    }

    #[test]
    fn isolates_point_below_separated() {
        let mut bbox = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
        bbox.extend(&[1.0, 1.0]).unwrap();
        assert!(isolates_point(&Cut::new(1, -2.0), &[0.5, -10.0], &bbox));
    }

    #[test]
    fn isolates_point_inside_never() {
        let mut bbox = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
        bbox.extend(&[10.0, 10.0]).unwrap();
        assert!(!isolates_point(&Cut::new(0, 5.0), &[5.0, 5.0], &bbox));
    }

    // Property test: under uniform-random insertions, the tree depth
    // stays within `4 · ⌈log₂ N⌉ + 4` — the "expected `O(log n)`"
    // bound from Guha 2016 §2 with a generous constant to absorb
    // the natural variance of random cuts.
    proptest::proptest! {
        #![proptest_config(proptest::test_runner::Config { cases: 32, ..proptest::test_runner::Config::default() })]
        #[test]
        fn depth_bounded_under_uniform_inserts(seed in 0_u64..10_000) {
            const N: usize = 64;
            const D: usize = 4;
            #[allow(clippy::cast_possible_truncation)]
            let mut t = RandomCutTree::<D>::new(N as u32).unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut points: Vec<[f64; D]> = Vec::with_capacity(N);

            for i in 0..N {
                let mut p = [0.0_f64; D];
                for slot in &mut p {
                    *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng) * 100.0;
                }
                points.push(p);
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
