//! Tree node algebraic type and packed `NodeRef` reference.
//!
//! [`NodeRef`] packs an internal-vs-leaf discriminator and a slot
//! index into a single `u32`. The high bit (`1 << 31`) marks leaves;
//! the low 31 bits hold the slot index in the corresponding storage
//! arena owned by [`NodeStore`].
//!
//! [`NodeStore`]: crate::tree::NodeStore

use crate::domain::{BoundingBox, Cut};

/// Packed reference to a tree node.
///
/// The high bit discriminates internal (`0`) from leaf (`1`) so node
/// identity is a single 4-byte field — useful for cache-friendly
/// storage in [`NodeStore`].
///
/// [`NodeStore`]: crate::tree::NodeStore
///
/// # Examples
///
/// ```
/// use rcf_rs::ForestBuilder;
///
/// let forest = ForestBuilder::<2>::new()
///     .num_trees(50)
///     .sample_size(8)
///     .seed(1)
///     .build()
///     .unwrap();
/// for (tree, _, _) in forest.trees() {
///     assert!(tree.root().is_none()); // empty forest has no root yet
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeRef(u32);

impl NodeRef {
    /// Bit set on a leaf reference.
    pub(crate) const LEAF_BIT: u32 = 1 << 31;
    /// Mask covering the slot-index bits.
    pub(crate) const INDEX_MASK: u32 = !Self::LEAF_BIT;
    /// Largest representable slot index (`(1 << 31) − 1`).
    pub const MAX_INDEX: u32 = Self::INDEX_MASK;

    /// Build an internal reference from a slot index.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when `idx > MAX_INDEX`. Internal callers
    /// always size-check first via the store's capacity.
    #[must_use]
    pub(crate) fn internal(idx: u32) -> Self {
        debug_assert!(idx <= Self::MAX_INDEX, "internal index overflow");
        Self(idx)
    }

    /// Build a leaf reference from a slot index.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when `idx > MAX_INDEX`. Internal callers
    /// always size-check first via the store's capacity.
    #[must_use]
    pub(crate) fn leaf(idx: u32) -> Self {
        debug_assert!(idx <= Self::MAX_INDEX, "leaf index overflow");
        Self(idx | Self::LEAF_BIT)
    }

    /// Whether this reference points to a leaf.
    #[must_use]
    #[inline]
    pub fn is_leaf(self) -> bool {
        self.0 & Self::LEAF_BIT != 0
    }

    /// Whether this reference points to an internal node.
    #[must_use]
    #[inline]
    pub fn is_internal(self) -> bool {
        !self.is_leaf()
    }

    /// Slot index in the corresponding storage arena.
    #[must_use]
    #[inline]
    pub fn index(self) -> usize {
        (self.0 & Self::INDEX_MASK) as usize
    }

    /// Raw packed `u32` representation. Used by [`NodeStore`] for
    /// child-equality comparisons during sibling lookup.
    ///
    /// [`NodeStore`]: crate::tree::NodeStore
    #[must_use]
    #[inline]
    pub(crate) fn raw(self) -> u32 {
        self.0
    }
}

/// Raw internal-node record. Lives inline in the
/// [`crate::NodeStore`] internal arena — one entry per live
/// internal node. The bounding box is embedded inline so tree
/// traversal stays cache-resident.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InternalData<const D: usize> {
    /// The hyperplane partitioning the subtree.
    pub cut: Cut,
    /// Cached union bounding box of the subtree.
    pub bbox: BoundingBox<D>,
    /// Left child (`point[cut.dim] <= cut.value`).
    pub left: NodeRef,
    /// Right child (`point[cut.dim] > cut.value`).
    pub right: NodeRef,
    /// Parent reference (`None` only at the root).
    pub parent: Option<NodeRef>,
    /// Number of leaf descendants.
    pub mass: u64,
}

/// Raw leaf-node record. Lives inline in the [`crate::NodeStore`]
/// leaf arena — one entry per live leaf. Kept small (no bounding
/// box, no cut) so the leaf arena fits many entries per cache line.
/// Before the v4 split the leaf arena stored the full `Node<D>`
/// enum with its internal-variant shape, wasting ~300 bytes per
/// leaf at `D = 16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LeafData {
    /// Index into the forest point store.
    pub point_idx: usize,
    /// Parent reference (`None` only when the tree contains a
    /// single leaf at the root).
    pub parent: Option<NodeRef>,
    /// Number of stored copies of this point. Always `>= 1`.
    pub mass: u64,
}

/// Zero-copy immutable view of a tree node. Returned by
/// [`crate::NodeStore::view`] — pattern-match to branch on
/// internal-vs-leaf without cloning the underlying record.
#[derive(Debug)]
pub enum NodeView<'a, const D: usize> {
    /// Reference to an internal node's record.
    Internal(&'a InternalData<D>),
    /// Reference to a leaf node's record.
    Leaf(&'a LeafData),
}

impl<const D: usize> NodeView<'_, D> {
    /// Mass of the node (leaf count for internals, copy count for
    /// leaves).
    #[must_use]
    #[inline]
    pub fn mass(&self) -> u64 {
        match self {
            Self::Internal(i) => i.mass,
            Self::Leaf(l) => l.mass,
        }
    }

    /// Parent reference (`None` only for the root).
    #[must_use]
    #[inline]
    pub fn parent(&self) -> Option<NodeRef> {
        match self {
            Self::Internal(i) => i.parent,
            Self::Leaf(l) => l.parent,
        }
    }

    /// Whether this is an internal node.
    #[must_use]
    #[inline]
    pub fn is_internal(&self) -> bool {
        matches!(self, Self::Internal(_))
    }

    /// Whether this is a leaf.
    #[must_use]
    #[inline]
    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf(_))
    }
}

/// Zero-copy mutable view of a tree node. Mirrors [`NodeView`] but
/// hands out `&mut` references for in-place field updates.
#[derive(Debug)]
pub enum NodeViewMut<'a, const D: usize> {
    /// Mutable reference to an internal node's record.
    Internal(&'a mut InternalData<D>),
    /// Mutable reference to a leaf node's record.
    Leaf(&'a mut LeafData),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ref_internal_round_trip() {
        let r = NodeRef::internal(42);
        assert!(r.is_internal());
        assert!(!r.is_leaf());
        assert_eq!(r.index(), 42);
    }

    #[test]
    fn ref_leaf_round_trip() {
        let r = NodeRef::leaf(7);
        assert!(r.is_leaf());
        assert!(!r.is_internal());
        assert_eq!(r.index(), 7);
    }

    #[test]
    fn ref_internal_and_leaf_with_same_index_differ() {
        let i = NodeRef::internal(0);
        let l = NodeRef::leaf(0);
        assert_ne!(i, l);
        assert_eq!(i.index(), l.index());
    }

    #[test]
    fn ref_max_index_round_trips() {
        let r = NodeRef::internal(NodeRef::MAX_INDEX);
        assert_eq!(r.index(), NodeRef::MAX_INDEX as usize);
        assert!(r.is_internal());
    }

    #[test]
    fn leaf_view_mass_and_parent() {
        let data = LeafData {
            point_idx: 0,
            parent: Some(NodeRef::internal(5)),
            mass: 3,
        };
        let view: NodeView<'_, 2> = NodeView::Leaf(&data);
        assert_eq!(view.mass(), 3);
        assert!(view.is_leaf());
        assert!(!view.is_internal());
        assert_eq!(view.parent(), Some(NodeRef::internal(5)));
    }
}
