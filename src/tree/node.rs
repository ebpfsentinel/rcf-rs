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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    pub fn is_leaf(self) -> bool {
        self.0 & Self::LEAF_BIT != 0
    }

    /// Whether this reference points to an internal node.
    #[must_use]
    pub fn is_internal(self) -> bool {
        !self.is_leaf()
    }

    /// Slot index in the corresponding storage arena.
    #[must_use]
    pub fn index(self) -> usize {
        (self.0 & Self::INDEX_MASK) as usize
    }

    /// Raw packed `u32` representation. Used by [`NodeStore`] for
    /// child-equality comparisons during sibling lookup.
    ///
    /// [`NodeStore`]: crate::tree::NodeStore
    #[must_use]
    pub(crate) fn raw(self) -> u32 {
        self.0
    }
}

/// A live tree node — either an internal node holding a cut, bounding
/// box and two children, or a leaf pointing into the future point
/// store (story RCF.7).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Node {
    /// Internal node: a cut hyperplane plus the union bounding box of
    /// the subtree, two children, an optional parent and the mass
    /// (number of leaf descendants).
    Internal {
        /// The hyperplane partitioning the subtree.
        cut: Cut,
        /// Cached union bounding box of the subtree.
        bbox: BoundingBox,
        /// Left child (`point[cut.dim] <= cut.value`).
        left: NodeRef,
        /// Right child (`point[cut.dim] > cut.value`).
        right: NodeRef,
        /// Parent reference (`None` only at the root).
        parent: Option<NodeRef>,
        /// Number of leaf descendants.
        mass: u64,
    },
    /// Leaf node: an index into the forest point store, the parent,
    /// and a mass (always `1` for distinct points; `> 1` only when
    /// the same point is inserted multiple times — RCF.4 collapses
    /// duplicates).
    Leaf {
        /// Index into the forest point store.
        point_idx: usize,
        /// Parent reference (`None` only when the tree contains a
        /// single leaf at the root).
        parent: Option<NodeRef>,
        /// Number of stored copies of this point. Always `>= 1`.
        mass: u64,
    },
}

impl Node {
    /// Mass of the node (leaf count for internals, copy count for
    /// leaves).
    #[must_use]
    pub fn mass(&self) -> u64 {
        match self {
            Self::Internal { mass, .. } | Self::Leaf { mass, .. } => *mass,
        }
    }

    /// Parent reference (`None` only for the root).
    #[must_use]
    pub fn parent(&self) -> Option<NodeRef> {
        match self {
            Self::Internal { parent, .. } | Self::Leaf { parent, .. } => *parent,
        }
    }

    /// Whether this is an internal node.
    #[must_use]
    pub fn is_internal(&self) -> bool {
        matches!(self, Self::Internal { .. })
    }

    /// Whether this is a leaf.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf { .. })
    }
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
    fn node_mass_returns_inner_mass() {
        let leaf = Node::Leaf {
            point_idx: 0,
            parent: None,
            mass: 3,
        };
        assert_eq!(leaf.mass(), 3);
        assert!(leaf.is_leaf());
        assert!(!leaf.is_internal());
    }

    #[test]
    fn node_parent_returns_inner_parent() {
        let leaf = Node::Leaf {
            point_idx: 0,
            parent: Some(NodeRef::internal(5)),
            mass: 1,
        };
        assert_eq!(leaf.parent(), Some(NodeRef::internal(5)));
    }
}
