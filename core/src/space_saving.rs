//! Space-Saving — deterministic top-K heavy hitters in `O(K)`
//! memory.
//!
//! Maintains at most `K` monitored keys with `(estimate, error)`
//! pairs. On each observation `x`:
//!
//! - `x` already tracked → increment its estimate.
//! - `x` new and the table has fewer than `K` entries → insert
//!   with `estimate = 1`, `error = 0`.
//! - `x` new and the table is full → evict the current minimum
//!   entry `m` and insert `x` with `estimate = m.estimate + 1`,
//!   `error = m.estimate`. The error tracks the worst-case
//!   overestimate so callers can bound uncertainty.
//!
//! Guarantees (Metwally et al. 2005):
//!
//! - Every key with true frequency `> N/K` is retained.
//! - `estimate(x) − error(x) ≤ true_count(x) ≤ estimate(x)`.
//!
//! Memory is `O(K)` — typical `K = 128` costs ≈ 4 KiB for
//! 16-byte keys (IPv6 addresses / flow-hash tuples). Complements
//! [`crate::CountMinSketch`] — where CMS is probabilistic per-key
//! frequency, Space-Saving is deterministic top-K under a fixed
//! memory cap.
//!
//! Per-observe cost: `O(1)` on the tracked-key path, `O(K)` on
//! the evict path (linear scan for the current minimum). For
//! typical SOC heavy-hitter tables (`K` in the low hundreds) this
//! is sub-microsecond. If per-packet observe is required at
//! 10 Gbps ingress, prefer a two-stage pipeline: pre-filter via
//! [`crate::CountMinSketch`] (saturating inserts at fixed cost),
//! refresh Space-Saving on an aggregation cadence.
//!
//! Gated behind the `std` feature because the inner store uses
//! [`std::collections::HashMap`].
//!
//! # Reference
//!
//! A. Metwally, D. Agrawal, A. El Abbadi, "Efficient Computation
//! of Frequent and Top-k Elements in Data Streams", ICDT 2005.

use alloc::vec::Vec;
use core::hash::Hash;
use std::collections::HashMap;

use crate::error::{RcfError, RcfResult};

/// Default capacity — tracks 128 heavy hitters, ~4 KiB for
/// 16-byte keys.
pub const DEFAULT_CAPACITY: usize = 128;

/// One entry in the monitored table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HeavyHitterEntry {
    /// Caller-facing frequency estimate. Overestimate bounded by
    /// `error`.
    pub estimate: u64,
    /// Worst-case overestimate: `true_count = estimate − error`.
    /// Zero for keys inserted before capacity was reached.
    pub error: u64,
}

/// Scored entry returned by [`SpaceSaving::top_k`]. `K` carries
/// the caller's key type so the report keeps the original
/// identifier (no hash collision on IP addresses, flow tuples,
/// etc.).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HeavyHitter<K> {
    /// Rank in the top-K (0 = highest estimate).
    pub rank: u32,
    /// Caller-supplied key.
    pub key: K,
    /// Frequency estimate (always `≥ true_count`).
    pub estimate: u64,
    /// Worst-case overestimate (`estimate − true_count`
    /// upper-bound).
    pub error: u64,
}

/// Streaming top-K tracker with `O(K)` memory.
///
/// # Examples
///
/// ```
/// use anomstream_core::SpaceSaving;
///
/// let mut ss: SpaceSaving<u32> = SpaceSaving::with_default_capacity().unwrap();
/// // 1 000 observations of "10" versus 5 of "99".
/// for _ in 0..1_000 {
///     ss.observe(10_u32);
/// }
/// for _ in 0..5 {
///     ss.observe(99_u32);
/// }
/// let top = ss.top_k(1);
/// assert_eq!(top[0].key, 10);
/// assert_eq!(top[0].estimate, 1_000);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpaceSaving<K>
where
    K: Hash + Eq + Clone,
{
    /// Bounded table — at most `capacity` entries.
    counts: HashMap<K, HeavyHitterEntry>,
    /// Maximum `counts` length.
    capacity: usize,
    /// Total observations — ops signal, also the divisor of the
    /// `N/K` frequency guarantee.
    total: u64,
}

impl<K> SpaceSaving<K>
where
    K: Hash + Eq + Clone,
{
    /// Build a tracker with caller-chosen `capacity`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] on `capacity == 0`.
    pub fn new(capacity: usize) -> RcfResult<Self> {
        if capacity == 0 {
            return Err(RcfError::InvalidConfig(alloc::string::ToString::to_string(
                "SpaceSaving: capacity must be > 0",
            )));
        }
        Ok(Self {
            counts: HashMap::with_capacity(capacity),
            capacity,
            total: 0,
        })
    }

    /// Default tracker — `capacity = 128`.
    ///
    /// # Errors
    ///
    /// Never in practice — [`DEFAULT_CAPACITY`] is a positive
    /// compile-time constant.
    pub fn with_default_capacity() -> RcfResult<Self> {
        Self::new(DEFAULT_CAPACITY)
    }

    /// Configured capacity (max tracked keys).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current number of tracked keys (`≤ capacity`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// `true` when the table holds no keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Total observations — sum of every [`Self::observe`] weight.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Worst-case per-estimate error bound (`N/K`). Keys whose
    /// true frequency exceeds this are guaranteed to be tracked.
    #[must_use]
    pub fn error_bound(&self) -> u64 {
        if self.capacity == 0 {
            return 0;
        }
        self.total / (self.capacity as u64)
    }

    /// Ingest one occurrence of `key` with unit weight.
    pub fn observe(&mut self, key: K) {
        self.observe_weighted(key, 1);
    }

    /// Ingest `key` with caller-supplied `weight` — byte-count
    /// heavy hitters in NDR workloads (per-packet bytes, not
    /// just packet counts).
    pub fn observe_weighted(&mut self, key: K, weight: u64) {
        if weight == 0 {
            return;
        }
        self.total = self.total.saturating_add(weight);

        if let Some(entry) = self.counts.get_mut(&key) {
            entry.estimate = entry.estimate.saturating_add(weight);
            return;
        }

        if self.counts.len() < self.capacity {
            self.counts.insert(
                key,
                HeavyHitterEntry {
                    estimate: weight,
                    error: 0,
                },
            );
            return;
        }

        // Table full — evict current minimum, reinsert `key` with
        // `estimate = min.estimate + weight` and `error = min.estimate`.
        if let Some((min_key, min_entry)) = self.find_min() {
            self.counts.remove(&min_key);
            let boosted = HeavyHitterEntry {
                estimate: min_entry.estimate.saturating_add(weight),
                error: min_entry.estimate,
            };
            self.counts.insert(key, boosted);
        }
    }

    /// Frequency estimate for `key`. Returns `None` when the key
    /// is not tracked (its true count may still be up to
    /// [`Self::error_bound`]).
    #[must_use]
    pub fn estimate(&self, key: &K) -> Option<HeavyHitterEntry> {
        self.counts.get(key).copied()
    }

    /// Ranked top-`n` snapshot — sorted by descending estimate.
    /// `n` is clamped to [`Self::len`].
    #[must_use]
    pub fn top_k(&self, n: usize) -> Vec<HeavyHitter<K>> {
        let mut entries: Vec<(K, HeavyHitterEntry)> =
            self.counts.iter().map(|(k, e)| (k.clone(), *e)).collect();
        entries.sort_by_key(|(_, e)| core::cmp::Reverse(e.estimate));
        entries.truncate(n);
        entries
            .into_iter()
            .enumerate()
            .map(|(idx, (k, e))| HeavyHitter {
                rank: u32::try_from(idx).unwrap_or(u32::MAX),
                key: k,
                estimate: e.estimate,
                error: e.error,
            })
            .collect()
    }

    /// Iterate every tracked `(&K, HeavyHitterEntry)` in
    /// insertion order (not ranked). Useful for ad-hoc scans /
    /// serialisation sinks.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &HeavyHitterEntry)> {
        self.counts.iter()
    }

    /// Drop every tracked key. Allocation is preserved.
    pub fn reset(&mut self) {
        self.counts.clear();
        self.total = 0;
    }

    /// `O(K)` linear scan for the minimum-estimate entry. Returns
    /// `None` when the table is empty — callers on the evict path
    /// have already confirmed non-emptiness, so `None` is treated
    /// as a no-op insert.
    fn find_min(&self) -> Option<(K, HeavyHitterEntry)> {
        self.counts
            .iter()
            .min_by_key(|(_, e)| e.estimate)
            .map(|(k, e)| (k.clone(), *e))
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_zero_capacity() {
        assert!(SpaceSaving::<u32>::new(0).is_err());
    }

    #[test]
    fn exact_counts_within_capacity() {
        let mut ss: SpaceSaving<u32> = SpaceSaving::new(8).unwrap();
        for i in 0..5_u32 {
            for _ in 0..=u64::from(i) {
                ss.observe(i);
            }
        }
        // 5 distinct keys, capacity 8 → every count is exact,
        // zero error.
        let top = ss.top_k(5);
        assert_eq!(top.len(), 5);
        for hh in &top {
            assert_eq!(hh.error, 0);
        }
        assert_eq!(top[0].key, 4);
        assert_eq!(top[0].estimate, 5);
    }

    #[test]
    fn heavy_hitter_always_retained() {
        // Heavy hitter with frequency > N/K must never be evicted.
        let mut ss: SpaceSaving<u32> = SpaceSaving::new(8).unwrap();
        // 1 000 observations of key 0 → 500× larger than the
        // N/K = 125 bound for the remaining 992 observations.
        for _ in 0..1_000 {
            ss.observe(0_u32);
        }
        // Flood with 2 000 unique noise keys.
        for i in 1..2_001_u32 {
            ss.observe(i);
        }
        let h = ss
            .top_k(8)
            .into_iter()
            .find(|hh| hh.key == 0)
            .expect("heavy hitter retained");
        // Estimate ≥ true count.
        assert!(h.estimate >= 1_000);
    }

    #[test]
    fn error_bound_sandwiches_true_count() {
        let mut ss: SpaceSaving<u32> = SpaceSaving::new(16).unwrap();
        for i in 0..100_u32 {
            for _ in 0..10 {
                ss.observe(i);
            }
        }
        for hh in ss.top_k(16) {
            // estimate − error ≤ true_count ≤ estimate
            assert!(hh.estimate >= hh.error);
            let lower = hh.estimate - hh.error;
            assert!(lower <= 10, "lower={lower}");
            assert!(hh.estimate >= 10 || hh.error > 0);
        }
    }

    #[test]
    fn estimate_returns_none_for_untracked() {
        let mut ss: SpaceSaving<u32> = SpaceSaving::new(2).unwrap();
        ss.observe(1);
        ss.observe(2);
        for _ in 0..5 {
            ss.observe(3);
        }
        // Capacity 2, 3 becomes tracked via eviction of min.
        assert!(ss.estimate(&3).is_some());
        // 100 is never observed.
        assert!(ss.estimate(&100).is_none());
    }

    #[test]
    fn weighted_observe_accumulates() {
        let mut ss: SpaceSaving<u32> = SpaceSaving::new(4).unwrap();
        ss.observe_weighted(7, 1_000);
        ss.observe_weighted(7, 500);
        let h = ss.estimate(&7).expect("tracked");
        assert_eq!(h.estimate, 1_500);
        assert_eq!(ss.total(), 1_500);
    }

    #[test]
    fn zero_weight_is_noop() {
        let mut ss: SpaceSaving<u32> = SpaceSaving::new(4).unwrap();
        ss.observe_weighted(1, 0);
        assert!(ss.is_empty());
        assert_eq!(ss.total(), 0);
    }

    #[test]
    fn error_bound_grows_linearly() {
        let mut ss: SpaceSaving<u32> = SpaceSaving::new(10).unwrap();
        for i in 0..1_000_u32 {
            ss.observe(i);
        }
        // N/K = 1000/10 = 100.
        assert_eq!(ss.error_bound(), 100);
    }

    #[test]
    fn top_k_ranks_descending() {
        let mut ss: SpaceSaving<u32> = SpaceSaving::new(8).unwrap();
        for (key, count) in [(1_u32, 100_u64), (2, 50), (3, 25), (4, 10)] {
            for _ in 0..count {
                ss.observe(key);
            }
        }
        let top = ss.top_k(4);
        assert_eq!(top[0].key, 1);
        assert_eq!(top[1].key, 2);
        assert_eq!(top[2].key, 3);
        assert_eq!(top[3].key, 4);
        assert_eq!(top[0].rank, 0);
        assert_eq!(top[3].rank, 3);
    }

    #[test]
    fn top_k_clamps_to_len() {
        let mut ss: SpaceSaving<u32> = SpaceSaving::new(8).unwrap();
        ss.observe(1);
        assert_eq!(ss.top_k(10).len(), 1);
        assert_eq!(ss.top_k(0).len(), 0);
    }

    #[test]
    fn reset_clears_everything() {
        let mut ss: SpaceSaving<u32> = SpaceSaving::new(4).unwrap();
        for i in 0..100_u32 {
            ss.observe(i);
        }
        ss.reset();
        assert!(ss.is_empty());
        assert_eq!(ss.total(), 0);
        assert_eq!(ss.top_k(4).len(), 0);
    }

    #[test]
    fn byte_key_roundtrip() {
        let mut ss: SpaceSaving<[u8; 16]> = SpaceSaving::new(4).unwrap();
        let k = [0x01_u8; 16];
        for _ in 0..10 {
            ss.observe(k);
        }
        assert_eq!(ss.estimate(&k).unwrap().estimate, 10);
    }

    #[cfg(all(feature = "serde", feature = "postcard"))]
    #[test]
    fn postcard_roundtrip_preserves_top_k() {
        let mut ss: SpaceSaving<u32> = SpaceSaving::new(8).unwrap();
        for i in 0..20_u32 {
            for _ in 0..=u64::from(i) {
                ss.observe(i);
            }
        }
        let bytes = postcard::to_allocvec(&ss).expect("serde ok");
        let back: SpaceSaving<u32> = postcard::from_bytes(&bytes).expect("serde ok");
        let a = ss.top_k(8);
        let b = back.top_k(8);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.key, y.key);
            assert_eq!(x.estimate, y.estimate);
            assert_eq!(x.error, y.error);
        }
    }
}
