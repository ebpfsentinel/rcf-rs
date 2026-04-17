//! Refcounted point store shared across every tree of a
//! [`crate::RandomCutForest`].
//!
//! Avoids the `(num_trees − 1) × sample_size × dimension × 8` byte
//! cost of per-tree point storage by holding the canonical point
//! once and tracking how many trees currently reference it. When the
//! refcount drops to zero the slot returns to a free list and gets
//! reused by the next [`add`](PointStore::add).
//!
//! The store implements [`PointAccessor`] so trees borrow leaf
//! points through it during traversal and bbox recomputation.

use crate::domain::point::{ensure_dim, ensure_finite};
use crate::error::{RcfError, RcfResult};
use crate::tree::PointAccessor;

/// Refcounted ring buffer of points indexed by `point_idx`.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointStore {
    /// Per-slot point payload. `None` slots are free.
    points: Vec<Option<Vec<f64>>>,
    /// Per-slot reference count (number of trees currently holding
    /// the point at that slot).
    ref_counts: Vec<u32>,
    /// LIFO stack of free slot indices.
    free_list: Vec<usize>,
    /// Per-point dimensionality enforced at insertion.
    dimension: usize,
}

impl PointStore {
    /// Build an empty store accepting `dimension`-dimensional points.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `dimension == 0`.
    pub fn new(dimension: usize) -> RcfResult<Self> {
        if dimension == 0 {
            return Err(RcfError::InvalidConfig(
                "PointStore dimension must be > 0".into(),
            ));
        }
        Ok(Self {
            points: Vec::new(),
            ref_counts: Vec::new(),
            free_list: Vec::new(),
            dimension,
        })
    }

    /// Configured dimensionality.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Total slot capacity (live + free).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.points.len()
    }

    /// Number of live (refcount > 0) entries.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.points.iter().filter(|slot| slot.is_some()).count()
    }

    /// Insert `point` into the next free slot (or grow). The new
    /// entry starts with a reference count of zero — callers must
    /// invoke [`incr_ref`](Self::incr_ref) for every tree that
    /// adopts the point so [`decr_ref`](Self::decr_ref) eventually
    /// frees the slot.
    ///
    /// # Errors
    ///
    /// - [`RcfError::DimensionMismatch`] when `point.len() != self.dimension()`.
    /// - [`RcfError::NaNValue`] when `point` contains a non-finite component.
    pub fn add(&mut self, point: Vec<f64>) -> RcfResult<usize> {
        ensure_dim(&point, self.dimension)?;
        ensure_finite(&point)?;
        if let Some(idx) = self.free_list.pop() {
            self.points[idx] = Some(point);
            self.ref_counts[idx] = 0;
            return Ok(idx);
        }
        let idx = self.points.len();
        self.points.push(Some(point));
        self.ref_counts.push(0);
        Ok(idx)
    }

    /// Drop a freshly-added point that no tree adopted (every tree
    /// rejected its `accept`). Callers should invoke this when
    /// [`add`](Self::add)'s return value still has zero refcount
    /// after every tree had a chance to incr it.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::OutOfBounds`] when `idx` is invalid or
    /// already free, and [`RcfError::InvalidConfig`] when the slot
    /// has any live references.
    pub fn drop_unreferenced(&mut self, idx: usize) -> RcfResult<()> {
        if idx >= self.points.len() || self.points[idx].is_none() {
            return Err(RcfError::OutOfBounds {
                index: idx,
                len: self.points.len(),
            });
        }
        if self.ref_counts[idx] != 0 {
            return Err(RcfError::InvalidConfig(format!(
                "PointStore::drop_unreferenced: slot {idx} still has refcount {}",
                self.ref_counts[idx]
            )));
        }
        self.points[idx] = None;
        self.free_list.push(idx);
        Ok(())
    }

    /// Increment the reference count for slot `idx`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::OutOfBounds`] when `idx` is invalid or
    /// the slot is free.
    pub fn incr_ref(&mut self, idx: usize) -> RcfResult<()> {
        if idx >= self.points.len() || self.points[idx].is_none() {
            return Err(RcfError::OutOfBounds {
                index: idx,
                len: self.points.len(),
            });
        }
        self.ref_counts[idx] = self.ref_counts[idx].saturating_add(1);
        Ok(())
    }

    /// Decrement the reference count for slot `idx`. When the count
    /// reaches zero the slot is freed and reused on the next
    /// [`add`](Self::add).
    ///
    /// # Errors
    ///
    /// - [`RcfError::OutOfBounds`] when `idx` is invalid or the
    ///   slot is free.
    /// - [`RcfError::InvalidConfig`] when the count is already zero
    ///   (under-decrement).
    pub fn decr_ref(&mut self, idx: usize) -> RcfResult<()> {
        if idx >= self.points.len() || self.points[idx].is_none() {
            return Err(RcfError::OutOfBounds {
                index: idx,
                len: self.points.len(),
            });
        }
        if self.ref_counts[idx] == 0 {
            return Err(RcfError::InvalidConfig(format!(
                "PointStore::decr_ref: slot {idx} already at zero refcount"
            )));
        }
        self.ref_counts[idx] -= 1;
        if self.ref_counts[idx] == 0 {
            self.points[idx] = None;
            self.free_list.push(idx);
        }
        Ok(())
    }

    /// Current reference count for slot `idx`. Returns `0` for free
    /// or invalid slots.
    #[must_use]
    pub fn ref_count(&self, idx: usize) -> u32 {
        if idx >= self.ref_counts.len() {
            return 0;
        }
        self.ref_counts[idx]
    }

    /// Pessimistic upper bound (in bytes) of the store's payload.
    #[must_use]
    pub fn memory_estimate(&self) -> usize {
        self.points.len() * (self.dimension * core::mem::size_of::<f64>())
            + self.ref_counts.len() * core::mem::size_of::<u32>()
            + self.free_list.len() * core::mem::size_of::<usize>()
    }
}

impl PointAccessor for PointStore {
    fn point(&self, idx: usize) -> Option<&[f64]> {
        self.points.get(idx).and_then(|slot| slot.as_deref())
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert exact equality on stored point payloads.
mod tests {
    use super::*;

    #[test]
    fn new_rejects_zero_dimension() {
        assert!(matches!(
            PointStore::new(0).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn add_validates_dim() {
        let mut s = PointStore::new(3).unwrap();
        assert!(matches!(
            s.add(vec![1.0, 2.0]).unwrap_err(),
            RcfError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn add_validates_finite() {
        let mut s = PointStore::new(2).unwrap();
        assert!(matches!(
            s.add(vec![1.0, f64::NAN]).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn add_returns_increasing_indices_initially() {
        let mut s = PointStore::new(2).unwrap();
        assert_eq!(s.add(vec![0.0, 0.0]).unwrap(), 0);
        assert_eq!(s.add(vec![1.0, 1.0]).unwrap(), 1);
        assert_eq!(s.capacity(), 2);
    }

    #[test]
    fn point_returns_inserted_value() {
        let mut s = PointStore::new(2).unwrap();
        let idx = s.add(vec![1.5, 2.5]).unwrap();
        assert_eq!(s.point(idx), Some(&[1.5, 2.5][..]));
    }

    #[test]
    fn point_returns_none_for_free_or_oob() {
        let mut s = PointStore::new(2).unwrap();
        assert_eq!(s.point(99), None);
        let idx = s.add(vec![0.0, 0.0]).unwrap();
        s.incr_ref(idx).unwrap();
        s.decr_ref(idx).unwrap();
        assert_eq!(s.point(idx), None);
    }

    #[test]
    fn ref_count_starts_at_zero() {
        let mut s = PointStore::new(2).unwrap();
        let idx = s.add(vec![0.0, 0.0]).unwrap();
        assert_eq!(s.ref_count(idx), 0);
    }

    #[test]
    fn incr_decr_cycle_frees_slot() {
        let mut s = PointStore::new(2).unwrap();
        let a = s.add(vec![1.0, 1.0]).unwrap();
        s.incr_ref(a).unwrap();
        s.incr_ref(a).unwrap();
        assert_eq!(s.ref_count(a), 2);
        s.decr_ref(a).unwrap();
        assert_eq!(s.ref_count(a), 1);
        s.decr_ref(a).unwrap();
        assert_eq!(s.ref_count(a), 0);
        // Slot freed and reused.
        let b = s.add(vec![2.0, 2.0]).unwrap();
        assert_eq!(b, a);
    }

    #[test]
    fn incr_oob_or_free_returns_err() {
        let mut s = PointStore::new(2).unwrap();
        assert!(matches!(
            s.incr_ref(99).unwrap_err(),
            RcfError::OutOfBounds { .. }
        ));
        let idx = s.add(vec![0.0, 0.0]).unwrap();
        s.incr_ref(idx).unwrap();
        s.decr_ref(idx).unwrap();
        assert!(matches!(
            s.incr_ref(idx).unwrap_err(),
            RcfError::OutOfBounds { .. }
        ));
    }

    #[test]
    fn decr_at_zero_is_invalid_config() {
        let mut s = PointStore::new(2).unwrap();
        let idx = s.add(vec![0.0, 0.0]).unwrap();
        assert!(matches!(
            s.decr_ref(idx).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn drop_unreferenced_frees_zero_refcount_slot() {
        let mut s = PointStore::new(2).unwrap();
        let idx = s.add(vec![0.0, 0.0]).unwrap();
        s.drop_unreferenced(idx).unwrap();
        assert_eq!(s.point(idx), None);
        assert_eq!(s.live_count(), 0);
        // Slot returned to free list.
        let new_idx = s.add(vec![1.0, 1.0]).unwrap();
        assert_eq!(new_idx, idx);
    }

    #[test]
    fn drop_unreferenced_rejects_live_slot() {
        let mut s = PointStore::new(2).unwrap();
        let idx = s.add(vec![0.0, 0.0]).unwrap();
        s.incr_ref(idx).unwrap();
        assert!(matches!(
            s.drop_unreferenced(idx).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn live_count_tracks_active_slots() {
        let mut s = PointStore::new(2).unwrap();
        let a = s.add(vec![0.0, 0.0]).unwrap();
        let b = s.add(vec![1.0, 1.0]).unwrap();
        assert_eq!(s.live_count(), 2);
        s.incr_ref(a).unwrap();
        s.decr_ref(a).unwrap();
        assert_eq!(s.live_count(), 1);
        // Bump and drop b to confirm decr brings live_count back to 0.
        s.incr_ref(b).unwrap();
        s.decr_ref(b).unwrap();
        assert_eq!(s.live_count(), 0);
    }

    #[test]
    fn point_accessor_impl_works() {
        let mut s = PointStore::new(3).unwrap();
        let idx = s.add(vec![7.0, 8.0, 9.0]).unwrap();
        let acc: &dyn PointAccessor = &s;
        assert_eq!(acc.point(idx), Some(&[7.0, 8.0, 9.0][..]));
    }

    #[test]
    fn memory_estimate_grows_with_capacity() {
        let mut s = PointStore::new(4).unwrap();
        let before = s.memory_estimate();
        s.add(vec![1.0; 4]).unwrap();
        let after = s.memory_estimate();
        assert!(after > before);
    }
}
