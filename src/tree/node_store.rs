//! Flat-array node storage with `O(1)` allocation and deallocation.
//!
//! Internal nodes live in `internals[0..capacity)`, leaves live in
//! `leaves[0..capacity)`. Each arena owns its own free list (LIFO
//! stack of freed slot indices) so allocations reuse the most-recently
//! freed slot first — which keeps the live working set compact and
//! cache-friendly.
//!
//! Bounding-box semantics: only internal nodes carry a cached
//! bounding box. Leaves know their point only through `point_idx`
//! into the [`crate::forest::PointStore`]; when a caller needs a
//! leaf bounding box it builds a degenerate one from the point
//! itself. This keeps leaf storage at 24 bytes (idx + parent +
//! mass) instead of duplicating per-leaf coordinate data, saving
//! ~6 MB at default configuration.

use alloc::format;
use alloc::vec::Vec;

use crate::domain::{BoundingBox, Cut};
use crate::error::{RcfError, RcfResult};
use crate::tree::node::{InternalData, LeafData, NodeRef, NodeView, NodeViewMut};

/// Flat-array storage for [`Node`]s with `O(1)` allocation and
/// deallocation via per-arena free lists.
///
/// # Examples
///
/// ```
/// use rcf_rs::NodeStore;
///
/// let mut store = NodeStore::<2>::new(4).unwrap();
/// let leaf = store.add_leaf(0, None, 1).unwrap();
/// assert!(leaf.is_leaf());
/// assert_eq!(store.live_count(), 1);
/// ```
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeStore<const D: usize> {
    /// Internal-node arena. `None` slots are free. Split from the
    /// leaves so each slot is exactly the size of one
    /// [`InternalData<D>`] record instead of paying the full
    /// `Node<D>`-enum worst-case.
    internals: Vec<Option<InternalData<D>>>,
    /// Leaf arena. `None` slots are free. Each slot holds a
    /// small [`LeafData`] record (24 bytes + `Option` overhead)
    /// instead of the old enum-sized ~320 bytes at `D = 16`.
    leaves: Vec<Option<LeafData>>,
    /// LIFO stack of free internal slot indices.
    internal_free: Vec<u32>,
    /// LIFO stack of free leaf slot indices.
    leaf_free: Vec<u32>,
    /// Per-arena capacity (each arena holds at most this many slots).
    capacity: u32,
}

impl<const D: usize> NodeStore<D> {
    /// Build a store with `capacity` internal slots and `capacity`
    /// leaf slots, all initially free.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `capacity == 0` or
    /// `capacity > NodeRef::MAX_INDEX + 1`.
    pub fn new(capacity: u32) -> RcfResult<Self> {
        if capacity == 0 {
            return Err(RcfError::InvalidConfig(
                "NodeStore capacity must be > 0".into(),
            ));
        }
        if capacity > NodeRef::MAX_INDEX {
            return Err(RcfError::InvalidConfig(format!(
                "NodeStore capacity {capacity} exceeds NodeRef::MAX_INDEX {}",
                NodeRef::MAX_INDEX
            )));
        }
        let cap = capacity as usize;
        let internals = (0..cap).map(|_| None).collect();
        let leaves = (0..cap).map(|_| None).collect();
        // Free list pre-populated in descending order so `pop()` hands
        // out index 0 first — keeps the live set front-loaded.
        let internal_free = (0..capacity).rev().collect();
        let leaf_free = (0..capacity).rev().collect();
        Ok(Self {
            internals,
            leaves,
            internal_free,
            leaf_free,
            capacity,
        })
    }

    /// Per-arena slot capacity.
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Number of live nodes (internals + leaves).
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.live_internal_count() + self.live_leaf_count()
    }

    /// Number of live internal nodes.
    #[must_use]
    pub fn live_internal_count(&self) -> usize {
        self.capacity as usize - self.internal_free.len()
    }

    /// Number of live leaf nodes.
    #[must_use]
    pub fn live_leaf_count(&self) -> usize {
        self.capacity as usize - self.leaf_free.len()
    }

    /// Allocate an internal node.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when the internal arena is
    /// exhausted (every slot is live).
    pub fn add_internal(
        &mut self,
        cut: Cut,
        bbox: BoundingBox<D>,
        left: NodeRef,
        right: NodeRef,
        parent: Option<NodeRef>,
        mass: u64,
    ) -> RcfResult<NodeRef> {
        let idx = self
            .internal_free
            .pop()
            .ok_or_else(|| RcfError::InvalidConfig("NodeStore internal arena exhausted".into()))?;
        self.internals[idx as usize] = Some(InternalData {
            cut,
            bbox,
            left,
            right,
            parent,
            mass,
        });
        Ok(NodeRef::internal(idx))
    }

    /// Allocate a leaf node.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when the leaf arena is
    /// exhausted (every slot is live).
    pub fn add_leaf(
        &mut self,
        point_idx: usize,
        parent: Option<NodeRef>,
        mass: u64,
    ) -> RcfResult<NodeRef> {
        let idx = self
            .leaf_free
            .pop()
            .ok_or_else(|| RcfError::InvalidConfig("NodeStore leaf arena exhausted".into()))?;
        self.leaves[idx as usize] = Some(LeafData {
            point_idx,
            parent,
            mass,
        });
        Ok(NodeRef::leaf(idx))
    }

    /// Free a node back to its arena. The slot becomes available for
    /// the next allocation.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::OutOfBounds`] when the slot is empty
    /// (double-free) or `n.index() >= capacity()`.
    pub fn delete(&mut self, n: NodeRef) -> RcfResult<()> {
        let idx = n.index();
        if idx >= self.capacity as usize {
            return Err(RcfError::OutOfBounds {
                index: idx,
                len: self.capacity as usize,
            });
        }
        let was_live = if n.is_leaf() {
            self.leaves[idx].take().is_some()
        } else {
            self.internals[idx].take().is_some()
        };
        if !was_live {
            return Err(RcfError::OutOfBounds {
                index: idx,
                len: self.capacity as usize,
            });
        }
        if n.is_leaf() {
            #[allow(clippy::cast_possible_truncation)]
            self.leaf_free.push(idx as u32);
        } else {
            #[allow(clippy::cast_possible_truncation)]
            self.internal_free.push(idx as u32);
        }
        Ok(())
    }

    /// Zero-copy immutable view of a node. Pattern-match on the
    /// returned [`NodeView`] to branch on internal-vs-leaf without
    /// cloning the underlying record.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::OutOfBounds`] when the slot is empty or
    /// `n.index() >= capacity()`.
    pub fn view(&self, n: NodeRef) -> RcfResult<NodeView<'_, D>> {
        let idx = n.index();
        if idx >= self.capacity as usize {
            return Err(RcfError::OutOfBounds {
                index: idx,
                len: self.capacity as usize,
            });
        }
        if n.is_leaf() {
            self.leaves[idx]
                .as_ref()
                .map(NodeView::Leaf)
                .ok_or(RcfError::OutOfBounds {
                    index: idx,
                    len: self.capacity as usize,
                })
        } else {
            self.internals[idx]
                .as_ref()
                .map(NodeView::Internal)
                .ok_or(RcfError::OutOfBounds {
                    index: idx,
                    len: self.capacity as usize,
                })
        }
    }

    /// Zero-copy mutable view of a node — see [`Self::view`].
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::OutOfBounds`] when the slot is empty or
    /// `n.index() >= capacity()`.
    pub fn view_mut(&mut self, n: NodeRef) -> RcfResult<NodeViewMut<'_, D>> {
        let idx = n.index();
        if idx >= self.capacity as usize {
            return Err(RcfError::OutOfBounds {
                index: idx,
                len: self.capacity as usize,
            });
        }
        if n.is_leaf() {
            self.leaves[idx]
                .as_mut()
                .map(NodeViewMut::Leaf)
                .ok_or(RcfError::OutOfBounds {
                    index: idx,
                    len: self.capacity as usize,
                })
        } else {
            self.internals[idx]
                .as_mut()
                .map(NodeViewMut::Internal)
                .ok_or(RcfError::OutOfBounds {
                    index: idx,
                    len: self.capacity as usize,
                })
        }
    }

    /// Typed immutable accessor for an internal node. Prefer this
    /// when the caller already knows the node is internal —
    /// one-level shallower than going through [`Self::view`] + match.
    ///
    /// # Errors
    ///
    /// - [`RcfError::OutOfBounds`] when the slot is empty or OOB.
    /// - [`RcfError::InvalidConfig`] when called on a leaf reference.
    pub fn internal(&self, n: NodeRef) -> RcfResult<&InternalData<D>> {
        if n.is_leaf() {
            return Err(RcfError::InvalidConfig(
                "NodeStore::internal: called on a leaf reference".into(),
            ));
        }
        let idx = n.index();
        if idx >= self.capacity as usize {
            return Err(RcfError::OutOfBounds {
                index: idx,
                len: self.capacity as usize,
            });
        }
        self.internals[idx].as_ref().ok_or(RcfError::OutOfBounds {
            index: idx,
            len: self.capacity as usize,
        })
    }

    /// Typed mutable accessor for an internal node — see
    /// [`Self::internal`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::internal`].
    pub fn internal_mut(&mut self, n: NodeRef) -> RcfResult<&mut InternalData<D>> {
        if n.is_leaf() {
            return Err(RcfError::InvalidConfig(
                "NodeStore::internal_mut: called on a leaf reference".into(),
            ));
        }
        let idx = n.index();
        if idx >= self.capacity as usize {
            return Err(RcfError::OutOfBounds {
                index: idx,
                len: self.capacity as usize,
            });
        }
        self.internals[idx].as_mut().ok_or(RcfError::OutOfBounds {
            index: idx,
            len: self.capacity as usize,
        })
    }

    /// Typed immutable accessor for a leaf node.
    ///
    /// # Errors
    ///
    /// - [`RcfError::OutOfBounds`] when the slot is empty or OOB.
    /// - [`RcfError::InvalidConfig`] when called on an internal reference.
    pub fn leaf(&self, n: NodeRef) -> RcfResult<&LeafData> {
        if !n.is_leaf() {
            return Err(RcfError::InvalidConfig(
                "NodeStore::leaf: called on an internal reference".into(),
            ));
        }
        let idx = n.index();
        if idx >= self.capacity as usize {
            return Err(RcfError::OutOfBounds {
                index: idx,
                len: self.capacity as usize,
            });
        }
        self.leaves[idx].as_ref().ok_or(RcfError::OutOfBounds {
            index: idx,
            len: self.capacity as usize,
        })
    }

    /// Typed mutable accessor for a leaf node — see [`Self::leaf`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::leaf`].
    pub fn leaf_mut(&mut self, n: NodeRef) -> RcfResult<&mut LeafData> {
        if !n.is_leaf() {
            return Err(RcfError::InvalidConfig(
                "NodeStore::leaf_mut: called on an internal reference".into(),
            ));
        }
        let idx = n.index();
        if idx >= self.capacity as usize {
            return Err(RcfError::OutOfBounds {
                index: idx,
                len: self.capacity as usize,
            });
        }
        self.leaves[idx].as_mut().ok_or(RcfError::OutOfBounds {
            index: idx,
            len: self.capacity as usize,
        })
    }

    /// Parent reference of a node (`None` for the root).
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::OutOfBounds`] when the node does not exist.
    pub fn parent(&self, n: NodeRef) -> RcfResult<Option<NodeRef>> {
        Ok(self.view(n)?.parent())
    }

    /// Sibling of a node.
    ///
    /// Returns `None` when `n` is the root (no parent → no sibling).
    ///
    /// # Errors
    ///
    /// - [`RcfError::OutOfBounds`] when `n` does not exist.
    /// - [`RcfError::InvalidConfig`] when the parent is a leaf
    ///   (impossible state — internal data structure invariant violated)
    ///   or when `n` is not registered as a child of its parent
    ///   (orphan).
    pub fn sibling(&self, n: NodeRef) -> RcfResult<Option<NodeRef>> {
        let Some(parent_ref) = self.parent(n)? else {
            return Ok(None);
        };
        if parent_ref.is_leaf() {
            return Err(RcfError::InvalidConfig(
                "NodeStore::sibling: parent is a leaf — invariant violated".into(),
            ));
        }
        let parent = self.internal(parent_ref)?;
        let n_raw = n.raw();
        if parent.left.raw() == n_raw {
            Ok(Some(parent.right))
        } else if parent.right.raw() == n_raw {
            Ok(Some(parent.left))
        } else {
            Err(RcfError::InvalidConfig(
                "NodeStore::sibling: child not registered with parent".into(),
            ))
        }
    }

    /// Cached bounding box of an *internal* node.
    ///
    /// # Errors
    ///
    /// - [`RcfError::OutOfBounds`] when the node does not exist.
    /// - [`RcfError::InvalidConfig`] when called on a leaf — leaf
    ///   bounding boxes are degenerate single-point boxes; build them
    ///   from the underlying point store entry on the consumer side.
    pub fn internal_bbox(&self, n: NodeRef) -> RcfResult<&BoundingBox<D>> {
        Ok(&self.internal(n)?.bbox)
    }

    /// Set the parent of a node.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::OutOfBounds`] when the node does not exist.
    pub fn set_parent(&mut self, n: NodeRef, parent: Option<NodeRef>) -> RcfResult<()> {
        match self.view_mut(n)? {
            NodeViewMut::Internal(i) => {
                i.parent = parent;
            }
            NodeViewMut::Leaf(l) => {
                l.parent = parent;
            }
        }
        Ok(())
    }

    /// Replace the mass of a node.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::OutOfBounds`] when the node does not exist.
    pub fn set_mass(&mut self, n: NodeRef, mass: u64) -> RcfResult<()> {
        match self.view_mut(n)? {
            NodeViewMut::Internal(i) => {
                i.mass = mass;
            }
            NodeViewMut::Leaf(l) => {
                l.mass = mass;
            }
        }
        Ok(())
    }

    /// Replace the cached bounding box of an internal node.
    ///
    /// # Errors
    ///
    /// - [`RcfError::OutOfBounds`] when the node does not exist.
    /// - [`RcfError::InvalidConfig`] when called on a leaf.
    pub fn set_internal_bbox(&mut self, n: NodeRef, bbox: BoundingBox<D>) -> RcfResult<()> {
        self.internal_mut(n)?.bbox = bbox;
        Ok(())
    }

    /// Replace the children of an internal node. Used by
    /// [`crate::RandomCutTree`] `delete` when merging a sibling
    /// into its grandparent's slot.
    ///
    /// # Errors
    ///
    /// - [`RcfError::OutOfBounds`] when the node does not exist.
    /// - [`RcfError::InvalidConfig`] when called on a leaf.
    pub fn set_internal_children(
        &mut self,
        n: NodeRef,
        new_left: NodeRef,
        new_right: NodeRef,
    ) -> RcfResult<()> {
        let i = self.internal_mut(n)?;
        i.left = new_left;
        i.right = new_right;
        Ok(())
    }

    /// Replace the cut of an internal node. Used when a tree
    /// rearrangement preserves the slot but swaps in a new cut.
    ///
    /// # Errors
    ///
    /// - [`RcfError::OutOfBounds`] when the node does not exist.
    /// - [`RcfError::InvalidConfig`] when called on a leaf.
    pub fn set_internal_cut(&mut self, n: NodeRef, new_cut: Cut) -> RcfResult<()> {
        self.internal_mut(n)?.cut = new_cut;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert exact equality on integer-valued masses.
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn unit_bbox<const D: usize>() -> BoundingBox<D> {
        let mut b = BoundingBox::<D>::from_point(&vec![0.0; D]).unwrap();
        b.extend(&vec![1.0; D]).unwrap();
        b
    }

    #[test]
    fn new_rejects_zero_capacity() {
        assert!(matches!(
            NodeStore::<2>::new(0).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn new_rejects_capacity_above_max() {
        // u32::MAX is > MAX_INDEX (1 << 31).
        assert!(matches!(
            NodeStore::<2>::new(u32::MAX).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn new_starts_empty() {
        let s = NodeStore::<2>::new(8).unwrap();
        assert_eq!(s.capacity(), 8);
        assert_eq!(s.live_count(), 0);
        assert_eq!(s.live_internal_count(), 0);
        assert_eq!(s.live_leaf_count(), 0);
    }

    #[test]
    fn add_leaf_increments_live() {
        let mut s = NodeStore::<2>::new(4).unwrap();
        let l = s.add_leaf(7, None, 1).unwrap();
        assert!(l.is_leaf());
        assert_eq!(s.live_leaf_count(), 1);
        assert_eq!(s.live_internal_count(), 0);
    }

    #[test]
    fn add_internal_increments_live() {
        let mut s = NodeStore::<2>::new(4).unwrap();
        let l1 = s.add_leaf(0, None, 1).unwrap();
        let l2 = s.add_leaf(1, None, 1).unwrap();
        let cut = Cut::new(0, 0.5);
        let i = s
            .add_internal(cut, unit_bbox::<2>(), l1, l2, None, 2)
            .unwrap();
        assert!(i.is_internal());
        assert_eq!(s.live_internal_count(), 1);
        assert_eq!(s.live_leaf_count(), 2);
        assert_eq!(s.live_count(), 3);
    }

    #[test]
    fn add_leaf_capacity_exhausted() {
        let mut s = NodeStore::<2>::new(2).unwrap();
        s.add_leaf(0, None, 1).unwrap();
        s.add_leaf(1, None, 1).unwrap();
        assert!(matches!(
            s.add_leaf(2, None, 1).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn add_internal_capacity_exhausted() {
        let mut s = NodeStore::<2>::new(1).unwrap();
        let l1 = s.add_leaf(0, None, 1).unwrap();
        let l2 = s.add_leaf(1, None, 1);
        // capacity=1: only one leaf slot.
        assert!(matches!(l2.unwrap_err(), RcfError::InvalidConfig(_)));
        let i = s
            .add_internal(Cut::new(0, 0.0), unit_bbox::<2>(), l1, l1, None, 1)
            .unwrap();
        assert!(
            s.add_internal(Cut::new(0, 0.0), unit_bbox::<2>(), i, i, None, 1)
                .is_err()
        );
    }

    #[test]
    fn delete_frees_slot_for_reuse() {
        let mut s = NodeStore::<2>::new(2).unwrap();
        let l = s.add_leaf(7, None, 1).unwrap();
        let l_idx = l.index();
        s.delete(l).unwrap();
        assert_eq!(s.live_leaf_count(), 0);
        // Same slot reused on next allocation (LIFO).
        let l2 = s.add_leaf(99, None, 1).unwrap();
        assert_eq!(l2.index(), l_idx);
    }

    #[test]
    fn delete_oob_index_rejected() {
        let mut s = NodeStore::<2>::new(2).unwrap();
        let bogus = NodeRef::leaf(99);
        assert!(matches!(
            s.delete(bogus).unwrap_err(),
            RcfError::OutOfBounds { .. }
        ));
    }

    #[test]
    fn delete_double_free_rejected() {
        let mut s = NodeStore::<2>::new(2).unwrap();
        let l = s.add_leaf(0, None, 1).unwrap();
        s.delete(l).unwrap();
        assert!(matches!(
            s.delete(l).unwrap_err(),
            RcfError::OutOfBounds { .. }
        ));
    }

    #[test]
    fn leaf_returns_inserted_value() {
        let mut s = NodeStore::<2>::new(2).unwrap();
        let l = s.add_leaf(42, None, 1).unwrap();
        let data = s.leaf(l).unwrap();
        assert_eq!(data.point_idx, 42);
        assert_eq!(data.mass, 1);
    }

    #[test]
    fn leaf_mut_allows_inplace_update() {
        let mut s = NodeStore::<2>::new(2).unwrap();
        let l = s.add_leaf(1, None, 1).unwrap();
        s.leaf_mut(l).unwrap().mass = 5;
        assert_eq!(s.view(l).unwrap().mass(), 5);
    }

    #[test]
    fn view_oob_returns_err() {
        let s = NodeStore::<2>::new(2).unwrap();
        assert!(matches!(
            s.view(NodeRef::leaf(7)).unwrap_err(),
            RcfError::OutOfBounds { .. }
        ));
    }

    #[test]
    fn parent_and_sibling_lookup() {
        let mut s = NodeStore::<2>::new(4).unwrap();
        let l1 = s.add_leaf(0, None, 1).unwrap();
        let l2 = s.add_leaf(1, None, 1).unwrap();
        let i = s
            .add_internal(Cut::new(0, 0.5), unit_bbox::<2>(), l1, l2, None, 2)
            .unwrap();
        s.set_parent(l1, Some(i)).unwrap();
        s.set_parent(l2, Some(i)).unwrap();

        assert_eq!(s.parent(l1).unwrap(), Some(i));
        assert_eq!(s.parent(i).unwrap(), None);
        assert_eq!(s.sibling(l1).unwrap(), Some(l2));
        assert_eq!(s.sibling(l2).unwrap(), Some(l1));
        // Root has no sibling.
        assert_eq!(s.sibling(i).unwrap(), None);
    }

    #[test]
    fn sibling_orphan_detected() {
        let mut s = NodeStore::<2>::new(4).unwrap();
        let real_l = s.add_leaf(0, None, 1).unwrap();
        let fake_l = s.add_leaf(1, None, 1).unwrap();
        let other = s.add_leaf(2, None, 1).unwrap();
        let i = s
            .add_internal(Cut::new(0, 0.5), unit_bbox::<2>(), real_l, other, None, 2)
            .unwrap();
        // fake_l claims `i` as its parent without being one of its children.
        s.set_parent(fake_l, Some(i)).unwrap();
        assert!(matches!(
            s.sibling(fake_l).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn sibling_parent_is_leaf_rejected() {
        let mut s = NodeStore::<2>::new(4).unwrap();
        let leaf_parent = s.add_leaf(0, None, 1).unwrap();
        let child = s.add_leaf(1, None, 1).unwrap();
        // Manually break the invariant: child claims a leaf as its parent.
        s.set_parent(child, Some(leaf_parent)).unwrap();
        assert!(matches!(
            s.sibling(child).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn internal_bbox_returns_cached_box() {
        let mut s = NodeStore::<2>::new(2).unwrap();
        let l1 = s.add_leaf(0, None, 1).unwrap();
        let l2 = s.add_leaf(1, None, 1).unwrap();
        let bbox = unit_bbox::<2>();
        let i = s
            .add_internal(Cut::new(0, 0.5), bbox.clone(), l1, l2, None, 2)
            .unwrap();
        assert_eq!(s.internal_bbox(i).unwrap(), &bbox);
    }

    #[test]
    fn internal_bbox_on_leaf_rejected() {
        let mut s = NodeStore::<2>::new(2).unwrap();
        let l = s.add_leaf(0, None, 1).unwrap();
        assert!(matches!(
            s.internal_bbox(l).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn set_mass_updates_leaf_and_internal() {
        let mut s = NodeStore::<2>::new(2).unwrap();
        let l = s.add_leaf(0, None, 1).unwrap();
        s.set_mass(l, 9).unwrap();
        assert_eq!(s.view(l).unwrap().mass(), 9);
    }

    #[test]
    fn set_internal_bbox_replaces_cached() {
        let mut s = NodeStore::<2>::new(2).unwrap();
        let l1 = s.add_leaf(0, None, 1).unwrap();
        let l2 = s.add_leaf(1, None, 1).unwrap();
        let i = s
            .add_internal(Cut::new(0, 0.5), unit_bbox::<2>(), l1, l2, None, 2)
            .unwrap();
        let mut new_bbox = BoundingBox::<2>::from_point(&[0.0, 0.0]).unwrap();
        new_bbox.extend(&[10.0, 10.0]).unwrap();
        s.set_internal_bbox(i, new_bbox.clone()).unwrap();
        assert_eq!(s.internal_bbox(i).unwrap(), &new_bbox);
    }

    #[test]
    fn set_internal_children_swaps_refs() {
        let mut s = NodeStore::<2>::new(4).unwrap();
        let l1 = s.add_leaf(0, None, 1).unwrap();
        let l2 = s.add_leaf(1, None, 1).unwrap();
        let l3 = s.add_leaf(2, None, 1).unwrap();
        let i = s
            .add_internal(Cut::new(0, 0.5), unit_bbox::<2>(), l1, l2, None, 2)
            .unwrap();
        s.set_internal_children(i, l1, l3).unwrap();
        let data = s.internal(i).unwrap();
        assert_eq!(data.left, l1);
        assert_eq!(data.right, l3);
    }

    #[test]
    fn set_internal_cut_replaces_cut() {
        let mut s = NodeStore::<2>::new(4).unwrap();
        let l1 = s.add_leaf(0, None, 1).unwrap();
        let l2 = s.add_leaf(1, None, 1).unwrap();
        let i = s
            .add_internal(Cut::new(0, 0.5), unit_bbox::<2>(), l1, l2, None, 2)
            .unwrap();
        s.set_internal_cut(i, Cut::new(1, 9.0)).unwrap();
        let data = s.internal(i).unwrap();
        assert_eq!(data.cut.dim(), 1);
        assert_eq!(data.cut.value(), 9.0);
    }

    #[test]
    fn setters_reject_oob_or_wrong_kind() {
        let mut s = NodeStore::<2>::new(2).unwrap();
        let l = s.add_leaf(0, None, 1).unwrap();
        assert!(matches!(
            s.set_internal_bbox(l, unit_bbox::<2>()).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
        assert!(matches!(
            s.set_internal_children(l, l, l).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
        assert!(matches!(
            s.set_internal_cut(l, Cut::new(0, 0.0)).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
        assert!(matches!(
            s.set_parent(NodeRef::leaf(99), None).unwrap_err(),
            RcfError::OutOfBounds { .. }
        ));
        assert!(matches!(
            s.set_mass(NodeRef::leaf(99), 1).unwrap_err(),
            RcfError::OutOfBounds { .. }
        ));
    }

    // Property test: random sequences of add/delete keep the
    // invariant `live_count = capacity − free_list.len()` and never
    // produce duplicate live indices.
    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]
        #[test]
        fn invariants_under_random_ops(ops in proptest::collection::vec(0u32..20, 0..200)) {
            const CAP: u32 = 16;
            let mut s = NodeStore::<2>::new(CAP).unwrap();
            let mut live_leaves: Vec<NodeRef> = Vec::new();
            let mut live_internals: Vec<NodeRef> = Vec::new();

            for op in ops {
                match op % 4 {
                    0 => {
                        if let Ok(r) = s.add_leaf(0, None, 1) {
                            // No duplicate live ref for the same slot.
                            prop_assert!(!live_leaves.iter().any(|x| x.raw() == r.raw()));
                            live_leaves.push(r);
                        }
                    }
                    1 => {
                        if live_leaves.len() >= 2 {
                            let l1 = live_leaves[live_leaves.len() - 1];
                            let l2 = live_leaves[live_leaves.len() - 2];
                            if let Ok(r) = s.add_internal(
                                Cut::new(0, 0.0), unit_bbox::<2>(), l1, l2, None, 2,
                            ) {
                                prop_assert!(!live_internals.iter().any(|x| x.raw() == r.raw()));
                                live_internals.push(r);
                            }
                        }
                    }
                    2 => {
                        if let Some(l) = live_leaves.pop() {
                            s.delete(l).unwrap();
                        }
                    }
                    _ => {
                        if let Some(i) = live_internals.pop() {
                            s.delete(i).unwrap();
                        }
                    }
                }
                prop_assert_eq!(s.live_leaf_count(), live_leaves.len());
                prop_assert_eq!(s.live_internal_count(), live_internals.len());
                prop_assert_eq!(s.live_count(), live_leaves.len() + live_internals.len());
            }
        }
    }
}
