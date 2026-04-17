//! Tree algorithm primitives.
//!
//! - [`node::Node`] — internal-vs-leaf algebraic data type
//! - [`node::NodeRef`] — `u32` packed reference (high bit discriminates
//!   internal from leaf, low bits hold the slot index)
//! - [`node_store::NodeStore`] — flat-array backing store with
//!   pre-allocated internal and leaf slot arenas plus free lists for
//!   `O(1)` allocation and deallocation
//!
//! `RandomCutTree` (story RCF.4, not yet shipped) will sit on top of
//! [`node_store::NodeStore`] and provide `add` / `delete` / `traverse`
//! for the actual cut tree.

pub mod node;
pub mod node_store;

pub use node::{Node, NodeRef};
pub use node_store::NodeStore;
