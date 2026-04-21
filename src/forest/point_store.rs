//! Refcounted point store shared across every tree of a
//! [`crate::RandomCutForest`].
//!
//! Avoids the `(num_trees − 1) × sample_size × D × 8` byte cost of
//! per-tree point storage by holding the canonical point once and
//! tracking how many trees currently reference it. When the refcount
//! drops to zero the slot returns to a free list and gets reused by
//! the next [`add`](PointStore::add).
//!
//! Points are stored as stack-allocated `[f64; D]` arrays so reads
//! through [`PointAccessor`] never pay for a dynamic-length check.
//!
//! The store implements [`PointAccessor`] so trees borrow leaf
//! points through it during traversal and bbox recomputation.

use alloc::format;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::domain::point::ensure_finite;
use crate::error::{RcfError, RcfResult};
use crate::tree::PointAccessor;

/// Refcounted ring buffer of `D`-dimensional points indexed by
/// `point_idx`.
///
/// Reference counts are stored as [`AtomicU32`] so the per-tree work
/// inside [`crate::RandomCutForest::update`] can call
/// [`incr_ref`](Self::incr_ref) / [`decr_ref`](Self::decr_ref) in
/// parallel without coarse locking. Slot allocation
/// ([`add`](Self::add)) and final freeing
/// ([`set_free`](Self::set_free) /
/// [`drop_unreferenced`](Self::drop_unreferenced)) stay
/// single-threaded — the forest serialises them before/after the
/// parallel block.
///
/// # Examples
///
/// ```
/// use rcf_rs::PointStore;
///
/// let mut store = PointStore::<2>::new().unwrap();
/// let idx = store.add([1.0, 2.0]).unwrap();
/// assert_eq!(store.live_count(), 1);
/// store.drop_unreferenced(idx).unwrap();
/// assert_eq!(store.live_count(), 0);
/// ```
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointStore<const D: usize> {
    /// Per-slot point payload. `None` slots are free. Serialised
    /// through [`point_slots_serde`] because `serde` does not yet
    /// implement [`Deserialize`] for `[T; N]` at arbitrary `N`.
    #[cfg_attr(feature = "serde", serde(with = "point_slots_serde"))]
    points: Vec<Option<[f64; D]>>,
    /// Per-slot reference count (number of trees currently holding
    /// the point at that slot). Atomic so parallel update workers
    /// can adjust it without locking.
    #[cfg_attr(feature = "serde", serde(with = "atomic_u32_vec_serde"))]
    ref_counts: Vec<AtomicU32>,
    /// LIFO stack of free slot indices.
    free_list: Vec<usize>,
}

#[cfg(feature = "serde")]
mod point_slots_serde {
    //! Serde adapter that snapshots `Vec<Option<[f64; D]>>` through a
    //! `Vec<Option<Vec<f64>>>` payload. Needed because `serde` does
    //! not yet ship `Deserialize` for `[T; N]` at arbitrary `N` —
    //! only fixed sizes up to 32. `D` is fixed at the type level so
    //! deserialisation rejects payloads whose array length differs.
    use alloc::format;
    use alloc::vec::Vec;

    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

    /// Snapshot every `Option<[f64; D]>` slot to an
    /// `Option<Vec<f64>>` and serialise the resulting vector.
    pub fn serialize<S, const D: usize>(
        slots: &[Option<[f64; D]>],
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let snapshot: Vec<Option<Vec<f64>>> = slots
            .iter()
            .map(|slot| slot.as_ref().map(|arr| arr.to_vec()))
            .collect();
        snapshot.serialize(serializer)
    }

    /// Reconstitute a `Vec<Option<[f64; D]>>` from the
    /// `Vec<Option<Vec<f64>>>` payload, rejecting any inner vector
    /// whose length does not match the type-level dimension `D`.
    pub fn deserialize<'de, D2, const D: usize>(
        deserializer: D2,
    ) -> Result<Vec<Option<[f64; D]>>, D2::Error>
    where
        D2: Deserializer<'de>,
    {
        let raw: Vec<Option<Vec<f64>>> = Vec::deserialize(deserializer)?;
        raw.into_iter()
            .map(|slot| match slot {
                None => Ok(None),
                Some(v) => {
                    let arr: [f64; D] = v.try_into().map_err(|_v: Vec<f64>| {
                        D2::Error::custom(format!("PointStore slot length mismatch: expected {D}"))
                    })?;
                    Ok(Some(arr))
                }
            })
            .collect()
    }
}

#[cfg(feature = "serde")]
mod atomic_u32_vec_serde {
    //! Serde adapter that snapshots a `Vec<AtomicU32>` to / from a
    //! `Vec<u32>` payload using `Ordering::Relaxed` (consistent with
    //! the runtime's intended ordering for refcount loads).
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicU32, Ordering};

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Snapshot a `&[AtomicU32]` to a `Vec<u32>` payload.
    pub fn serialize<S: Serializer>(v: &[AtomicU32], serializer: S) -> Result<S::Ok, S::Error> {
        let snapshot: Vec<u32> = v.iter().map(|a| a.load(Ordering::Relaxed)).collect();
        snapshot.serialize(serializer)
    }

    /// Reconstitute a `Vec<AtomicU32>` from a `Vec<u32>` payload.
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<AtomicU32>, D::Error> {
        let raw = Vec::<u32>::deserialize(deserializer)?;
        Ok(raw.into_iter().map(AtomicU32::new).collect())
    }
}

impl<const D: usize> PointStore<D> {
    /// Build an empty store accepting `D`-dimensional points.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `D == 0`.
    pub fn new() -> RcfResult<Self> {
        if D == 0 {
            return Err(RcfError::InvalidConfig(
                "PointStore dimension must be > 0".into(),
            ));
        }
        Ok(Self {
            points: Vec::new(),
            ref_counts: Vec::new(),
            free_list: Vec::new(),
        })
    }

    /// Configured dimensionality (compile-time constant `D`).
    #[must_use]
    #[inline]
    pub const fn dimension(&self) -> usize {
        D
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
    /// [`RcfError::NaNValue`] when `point` contains a non-finite
    /// component. Dimension is enforced at the type level.
    pub fn add(&mut self, point: [f64; D]) -> RcfResult<usize> {
        ensure_finite(&point)?;
        if let Some(idx) = self.free_list.pop() {
            self.points[idx] = Some(point);
            self.ref_counts[idx].store(0, Ordering::Relaxed);
            return Ok(idx);
        }
        let idx = self.points.len();
        self.points.push(Some(point));
        self.ref_counts.push(AtomicU32::new(0));
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
        let rc = self.ref_counts[idx].load(Ordering::Acquire);
        if rc != 0 {
            return Err(RcfError::InvalidConfig(format!(
                "PointStore::drop_unreferenced: slot {idx} still has refcount {rc}"
            )));
        }
        self.points[idx] = None;
        self.free_list.push(idx);
        Ok(())
    }

    /// Increment the reference count for slot `idx`. Lock-free
    /// atomic op — safe to call from parallel workers via `&self`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::OutOfBounds`] when `idx` is invalid or
    /// the slot is free.
    pub fn incr_ref(&self, idx: usize) -> RcfResult<()> {
        if idx >= self.points.len() || self.points[idx].is_none() {
            return Err(RcfError::OutOfBounds {
                index: idx,
                len: self.points.len(),
            });
        }
        self.ref_counts[idx].fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// Decrement the reference count for slot `idx`. Returns `true`
    /// when the count just hit zero — the caller (forest layer) must
    /// then invoke [`set_free`](Self::set_free) **single-threaded**
    /// to actually mark the slot as reclaimable. Lock-free atomic op
    /// — safe from parallel workers via `&self`.
    ///
    /// # Errors
    ///
    /// - [`RcfError::OutOfBounds`] when `idx` is invalid or the
    ///   slot is free.
    /// - [`RcfError::InvalidConfig`] when the count is already zero
    ///   (under-decrement detected via the atomic CAS loop).
    pub fn decr_ref(&self, idx: usize) -> RcfResult<bool> {
        if idx >= self.points.len() || self.points[idx].is_none() {
            return Err(RcfError::OutOfBounds {
                index: idx,
                len: self.points.len(),
            });
        }
        loop {
            let current = self.ref_counts[idx].load(Ordering::Acquire);
            if current == 0 {
                return Err(RcfError::InvalidConfig(format!(
                    "PointStore::decr_ref: slot {idx} already at zero refcount"
                )));
            }
            let next = current - 1;
            if self.ref_counts[idx]
                .compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(next == 0);
            }
        }
    }

    /// Mark `idx` as free after [`decr_ref`](Self::decr_ref) returned
    /// `true`. Single-threaded — called from the forest layer outside
    /// the parallel block.
    ///
    /// # Errors
    ///
    /// - [`RcfError::OutOfBounds`] when `idx` is invalid or the slot
    ///   is already free.
    /// - [`RcfError::InvalidConfig`] when the slot's refcount is
    ///   non-zero (something incr'd between the decr and the free).
    pub fn set_free(&mut self, idx: usize) -> RcfResult<()> {
        if idx >= self.points.len() || self.points[idx].is_none() {
            return Err(RcfError::OutOfBounds {
                index: idx,
                len: self.points.len(),
            });
        }
        let rc = self.ref_counts[idx].load(Ordering::Acquire);
        if rc != 0 {
            return Err(RcfError::InvalidConfig(format!(
                "PointStore::set_free: slot {idx} has refcount {rc}, not zero"
            )));
        }
        self.points[idx] = None;
        self.free_list.push(idx);
        Ok(())
    }

    /// Current reference count for slot `idx`. Returns `0` for free
    /// or invalid slots. Loads the atomic with `Ordering::Acquire`.
    #[must_use]
    pub fn ref_count(&self, idx: usize) -> u32 {
        if idx >= self.ref_counts.len() {
            return 0;
        }
        self.ref_counts[idx].load(Ordering::Acquire)
    }

    /// Pessimistic upper bound (in bytes) of the store's payload.
    #[must_use]
    pub fn memory_estimate(&self) -> usize {
        self.points.len() * (D * core::mem::size_of::<f64>())
            + self.ref_counts.len() * core::mem::size_of::<u32>()
            + self.free_list.len() * core::mem::size_of::<usize>()
    }
}

impl<const D: usize> PointAccessor<D> for PointStore<D> {
    fn point(&self, idx: usize) -> Option<&[f64; D]> {
        self.points.get(idx).and_then(|slot| slot.as_ref())
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert exact equality on stored point payloads.
mod tests {
    use super::*;

    #[test]
    fn new_rejects_zero_dimension() {
        assert!(matches!(
            PointStore::<0>::new().unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn add_validates_finite() {
        let mut s = PointStore::<2>::new().unwrap();
        assert!(matches!(
            s.add([1.0, f64::NAN]).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn add_returns_increasing_indices_initially() {
        let mut s = PointStore::<2>::new().unwrap();
        assert_eq!(s.add([0.0, 0.0]).unwrap(), 0);
        assert_eq!(s.add([1.0, 1.0]).unwrap(), 1);
        assert_eq!(s.capacity(), 2);
    }

    #[test]
    fn point_returns_inserted_value() {
        let mut s = PointStore::<2>::new().unwrap();
        let idx = s.add([1.5, 2.5]).unwrap();
        assert_eq!(s.point(idx), Some(&[1.5, 2.5]));
    }

    #[test]
    fn point_returns_none_for_free_or_oob() {
        let mut s = PointStore::<2>::new().unwrap();
        assert_eq!(s.point(99), None);
        let idx = s.add([0.0, 0.0]).unwrap();
        s.incr_ref(idx).unwrap();
        let hit_zero = s.decr_ref(idx).unwrap();
        assert!(hit_zero);
        s.set_free(idx).unwrap();
        assert_eq!(s.point(idx), None);
    }

    #[test]
    fn ref_count_starts_at_zero() {
        let mut s = PointStore::<2>::new().unwrap();
        let idx = s.add([0.0, 0.0]).unwrap();
        assert_eq!(s.ref_count(idx), 0);
    }

    #[test]
    fn incr_decr_cycle_frees_slot() {
        let mut s = PointStore::<2>::new().unwrap();
        let a = s.add([1.0, 1.0]).unwrap();
        s.incr_ref(a).unwrap();
        s.incr_ref(a).unwrap();
        assert_eq!(s.ref_count(a), 2);
        let hit_zero = s.decr_ref(a).unwrap();
        assert!(!hit_zero);
        assert_eq!(s.ref_count(a), 1);
        let hit_zero = s.decr_ref(a).unwrap();
        assert!(hit_zero);
        assert_eq!(s.ref_count(a), 0);
        s.set_free(a).unwrap();
        let b = s.add([2.0, 2.0]).unwrap();
        assert_eq!(b, a);
    }

    #[test]
    fn incr_oob_or_free_returns_err() {
        let mut s = PointStore::<2>::new().unwrap();
        assert!(matches!(
            s.incr_ref(99).unwrap_err(),
            RcfError::OutOfBounds { .. }
        ));
        let idx = s.add([0.0, 0.0]).unwrap();
        s.incr_ref(idx).unwrap();
        assert!(s.decr_ref(idx).unwrap());
        s.set_free(idx).unwrap();
        assert!(matches!(
            s.incr_ref(idx).unwrap_err(),
            RcfError::OutOfBounds { .. }
        ));
    }

    #[test]
    fn decr_at_zero_is_invalid_config() {
        let mut s = PointStore::<2>::new().unwrap();
        let idx = s.add([0.0, 0.0]).unwrap();
        assert!(matches!(
            s.decr_ref(idx).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn drop_unreferenced_frees_zero_refcount_slot() {
        let mut s = PointStore::<2>::new().unwrap();
        let idx = s.add([0.0, 0.0]).unwrap();
        s.drop_unreferenced(idx).unwrap();
        assert_eq!(s.point(idx), None);
        assert_eq!(s.live_count(), 0);
        let new_idx = s.add([1.0, 1.0]).unwrap();
        assert_eq!(new_idx, idx);
    }

    #[test]
    fn drop_unreferenced_rejects_live_slot() {
        let mut s = PointStore::<2>::new().unwrap();
        let idx = s.add([0.0, 0.0]).unwrap();
        s.incr_ref(idx).unwrap();
        assert!(matches!(
            s.drop_unreferenced(idx).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn live_count_tracks_active_slots() {
        let mut s = PointStore::<2>::new().unwrap();
        let a = s.add([0.0, 0.0]).unwrap();
        let b = s.add([1.0, 1.0]).unwrap();
        assert_eq!(s.live_count(), 2);
        s.incr_ref(a).unwrap();
        assert!(s.decr_ref(a).unwrap());
        s.set_free(a).unwrap();
        assert_eq!(s.live_count(), 1);
        s.incr_ref(b).unwrap();
        assert!(s.decr_ref(b).unwrap());
        s.set_free(b).unwrap();
        assert_eq!(s.live_count(), 0);
    }

    #[test]
    fn point_accessor_impl_works() {
        let mut s = PointStore::<3>::new().unwrap();
        let idx = s.add([7.0, 8.0, 9.0]).unwrap();
        let acc: &dyn PointAccessor<3> = &s;
        assert_eq!(acc.point(idx), Some(&[7.0, 8.0, 9.0]));
    }

    #[test]
    fn memory_estimate_grows_with_capacity() {
        let mut s = PointStore::<4>::new().unwrap();
        let before = s.memory_estimate();
        s.add([1.0; 4]).unwrap();
        let after = s.memory_estimate();
        assert!(after > before);
    }
}
