//! Forest aggregate root.
//!
//! - [`point_store::PointStore`] — a refcounted ring buffer that
//!   holds the canonical copy of every point currently referenced by
//!   any tree. Trees see it through the
//!   [`crate::tree::PointAccessor`] trait.
//! - [`random_cut_forest::RandomCutForest`] — orchestrates `N`
//!   `(RandomCutTree, ReservoirSampler)` pairs sharing the
//!   [`point_store::PointStore`].

pub mod point_store;
pub mod random_cut_forest;

pub use point_store::PointStore;
pub use random_cut_forest::RandomCutForest;
