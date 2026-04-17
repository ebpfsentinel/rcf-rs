//! Domain primitives: pure value objects + dimension helpers.
//!
//! This module is dependency-free (apart from `rand` for [`cut`]) and
//! holds the geometric building blocks that the tree, sampler, visitor
//! and forest layers compose:
//!
//! - [`point`] — `Point` type alias and dimensionality helpers
//! - [`bounding_box::BoundingBox`] — axis-aligned bounding boxes with
//!   `range_sum` and `probability_of_cut` (Guha 2016 §3)
//! - [`cut::Cut`] — a single random cut (dimension + value) sampled
//!   weighted by the bounding box's per-dimension range
//! - [`divector::DiVector`] — per-feature attribution vector for
//!   `AttributionVisitor` (story RCF.6, not yet shipped)
//! - [`score::AnomalyScore`] — `NaN`-safe newtype around `f64`

pub mod bounding_box;
pub mod cut;
pub mod divector;
pub mod point;
pub mod score;

pub use bounding_box::BoundingBox;
pub use cut::Cut;
pub use divector::DiVector;
pub use point::{Point, ensure_dim, ensure_finite};
pub use score::AnomalyScore;
