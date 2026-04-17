//! Weighted reservoir sampler without replacement.
//!
//! Implements the Efraimidis-Spirakis (2006) weighted reservoir with
//! time-decayed weights as described in Guha et al. (2016) §4. The
//! AWS `SageMaker` reference explicitly cites Park et al. (2004) for
//! sampling **without replacement**, so [`ReservoirSampler`] never
//! lets the same point index live in the reservoir twice.
//!
//! For each candidate the sampler draws `u ~ Uniform(0, 1)` and
//! computes
//!
//! ```text
//! weight = ln(-ln(u)) − entries_seen · time_decay
//! ```
//!
//! The reservoir keeps the `capacity` **smallest** weights via a
//! max-heap; when a new candidate's weight is below the current heap
//! maximum the old entry is evicted. With `time_decay = 0` the
//! sampler degenerates to standard uniform-random reservoir sampling
//! (keeping the smallest values of `ln(-ln(u))` is equivalent to the
//! Efraimidis-Spirakis reservoir on weight 1). With `time_decay > 0`
//! the offset shifts **older** items toward larger weights — so they
//! get evicted first and the reservoir biases toward recent items.
//!
//! Membership is the caller's contract: every `point_idx` passed to
//! [`ReservoirSampler::accept`] must be unique. The forest layer
//! (story RCF.7) guarantees this through its
//! [`crate::forest`-stored point indices](crate::tree::PointAccessor).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rand::{Rng, RngCore};

use crate::error::{RcfError, RcfResult};

/// Outcome of a single [`ReservoirSampler::accept`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerOp {
    /// The candidate was inserted into a free slot of the reservoir.
    Inserted,
    /// The candidate displaced an older entry. The evicted point
    /// index is returned so the caller (forest / tree) can remove it
    /// from the corresponding tree.
    Replaced(usize),
    /// The candidate had a smaller weight than every reservoir
    /// member; the reservoir was unchanged.
    Rejected,
}

/// Weighted reservoir entry; ordered by weight for the heap.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct WeightedEntry {
    /// Sampling weight: `ln(-ln(u)) − entries_seen · time_decay`.
    weight: f64,
    /// Point index in the caller's point store.
    point_idx: usize,
}

impl PartialEq for WeightedEntry {
    fn eq(&self, other: &Self) -> bool {
        // NaN cannot occur — `accept` rejects non-finite u and the
        // formula is otherwise total. Use bitwise equality so two
        // entries with the same weight but different idx still
        // compare equal *for ordering purposes only*.
        self.weight == other.weight
    }
}

impl Eq for WeightedEntry {}

impl PartialOrd for WeightedEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WeightedEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Weights are finite by construction (`accept` clamps `u`
        // away from zero so `ln(-ln(u))` is finite). `unwrap_or` is
        // a defensive fallback on the off chance a hostile RNG
        // returns a corrupted bit pattern.
        self.weight
            .partial_cmp(&other.weight)
            .unwrap_or(Ordering::Equal)
    }
}

/// Max-heap-backed weighted reservoir sampler without replacement.
///
/// The heap holds the `capacity` **smallest** weights seen so far;
/// the heap top is the largest among those, the candidate to evict
/// when a smaller-weight item arrives.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReservoirSampler {
    /// Max-heap on weight: `peek()` returns the largest weight
    /// currently held, which is the next eviction candidate.
    heap: BinaryHeap<WeightedEntry>,
    /// Maximum number of indices the reservoir holds at once.
    capacity: usize,
    /// Total number of [`accept`] calls observed since construction
    /// or [`reset`](Self::reset).
    ///
    /// [`accept`]: Self::accept
    entries_seen: u64,
    /// Per-call decay applied to the weight: bigger value = stronger
    /// recency bias.
    time_decay: f64,
}

impl ReservoirSampler {
    /// Build a fresh sampler with the requested `capacity` and
    /// `time_decay`.
    ///
    /// # Errors
    ///
    /// - [`RcfError::InvalidConfig`] when `capacity == 0`.
    /// - [`RcfError::InvalidConfig`] when `time_decay` is negative or
    ///   non-finite (NaN, ±∞).
    pub fn new(capacity: usize, time_decay: f64) -> RcfResult<Self> {
        if capacity == 0 {
            return Err(RcfError::InvalidConfig(
                "ReservoirSampler capacity must be > 0".into(),
            ));
        }
        if !time_decay.is_finite() || time_decay < 0.0 {
            return Err(RcfError::InvalidConfig(format!(
                "ReservoirSampler time_decay must be finite and >= 0, got {time_decay}"
            )));
        }
        Ok(Self {
            heap: BinaryHeap::with_capacity(capacity),
            capacity,
            entries_seen: 0,
            time_decay,
        })
    }

    /// Reservoir capacity (the maximum number of indices held at any
    /// one time).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of indices currently in the reservoir.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether the reservoir holds any indices.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Total number of [`accept`](Self::accept) calls since the last
    /// [`reset`](Self::reset).
    #[must_use]
    pub fn entries_seen(&self) -> u64 {
        self.entries_seen
    }

    /// Configured time-decay factor.
    #[must_use]
    pub fn time_decay(&self) -> f64 {
        self.time_decay
    }

    /// Drop every reservoir member and reset the entries counter.
    /// Used by tests and by hot-reload paths in the forest layer
    /// that rebuild the sampler in place.
    pub fn reset(&mut self) {
        self.heap.clear();
        self.entries_seen = 0;
    }

    /// Iterate over every point index currently in the reservoir, in
    /// no particular order. Used by tests, by the forest layer to
    /// detect duplicate insertion attempts, and by the persistence
    /// layer (story RCF.8).
    pub fn iter_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.heap.iter().map(|entry| entry.point_idx)
    }

    /// Whether `point_idx` is currently held in the reservoir.
    /// O(`capacity`) — used by tests; the forest layer enforces
    /// uniqueness through its own point store.
    #[must_use]
    pub fn contains(&self, point_idx: usize) -> bool {
        self.iter_indices().any(|idx| idx == point_idx)
    }

    /// Offer `point_idx` to the sampler. Returns the resulting
    /// [`SamplerOp`].
    ///
    /// Caller invariant: every `point_idx` must be unique within the
    /// lifetime of the sampler (until [`reset`](Self::reset)). Passing
    /// the same index twice violates the without-replacement contract.
    ///
    /// # Panics
    ///
    /// Never. The post-capacity branch peeks the heap whose length is
    /// guaranteed to equal `self.capacity > 0` by the prior `if`, so
    /// the `expect`s on `peek`/`pop` are unreachable.
    pub fn accept<R: RngCore + ?Sized>(&mut self, point_idx: usize, rng: &mut R) -> SamplerOp {
        self.entries_seen = self.entries_seen.saturating_add(1);

        // u ∈ (0, 1) — the standard `random::<f64>()` returns [0, 1),
        // so clamp the lower bound away from zero to keep `ln` finite.
        let mut u: f64 = rng.random();
        if u <= 0.0 {
            u = f64::MIN_POSITIVE;
        }

        #[allow(clippy::cast_precision_loss)]
        let decay = self.entries_seen as f64 * self.time_decay;
        let weight = (-u.ln()).ln() - decay;

        if self.heap.len() < self.capacity {
            self.heap.push(WeightedEntry { weight, point_idx });
            return SamplerOp::Inserted;
        }

        // Max-heap: peek returns the largest weight currently held —
        // the next eviction candidate. A smaller-weight new entry
        // bumps it out so the reservoir keeps the K smallest weights.
        let max_weight = self.heap.peek().expect("heap is non-empty").weight;
        if weight < max_weight {
            let evicted = self.heap.pop().expect("heap is non-empty").point_idx;
            self.heap.push(WeightedEntry { weight, point_idx });
            SamplerOp::Replaced(evicted)
        } else {
            SamplerOp::Rejected
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)] // Tests assert exact equality on configured constants and intentionally cast small bounded values.
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashSet;

    fn fresh_rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    #[test]
    fn new_rejects_zero_capacity() {
        assert!(matches!(
            ReservoirSampler::new(0, 0.0).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn new_rejects_negative_time_decay() {
        assert!(matches!(
            ReservoirSampler::new(8, -0.001).unwrap_err(),
            RcfError::InvalidConfig(_)
        ));
    }

    #[test]
    fn new_rejects_non_finite_time_decay() {
        assert!(ReservoirSampler::new(8, f64::NAN).is_err());
        assert!(ReservoirSampler::new(8, f64::INFINITY).is_err());
    }

    #[test]
    fn new_initial_state() {
        let s = ReservoirSampler::new(4, 0.05).unwrap();
        assert_eq!(s.capacity(), 4);
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert_eq!(s.entries_seen(), 0);
        assert_eq!(s.time_decay(), 0.05);
    }

    #[test]
    fn accept_fills_capacity_with_inserts() {
        let mut s = ReservoirSampler::new(3, 0.0).unwrap();
        let mut rng = fresh_rng(1);
        assert_eq!(s.accept(10, &mut rng), SamplerOp::Inserted);
        assert_eq!(s.accept(11, &mut rng), SamplerOp::Inserted);
        assert_eq!(s.accept(12, &mut rng), SamplerOp::Inserted);
        assert_eq!(s.len(), 3);
        assert_eq!(s.entries_seen(), 3);
    }

    #[test]
    fn accept_after_capacity_returns_replaced_or_rejected() {
        let mut s = ReservoirSampler::new(2, 0.0).unwrap();
        let mut rng = fresh_rng(7);
        s.accept(10, &mut rng);
        s.accept(11, &mut rng);
        for i in 12..200 {
            let op = s.accept(i, &mut rng);
            assert!(
                matches!(op, SamplerOp::Replaced(_) | SamplerOp::Rejected),
                "post-capacity op should be Replaced or Rejected"
            );
            assert_eq!(s.len(), 2, "capacity invariant violated");
        }
    }

    #[test]
    fn replaced_evicts_existing_index() {
        let mut s = ReservoirSampler::new(2, 0.0).unwrap();
        let mut rng = fresh_rng(13);
        s.accept(10, &mut rng);
        s.accept(11, &mut rng);
        let mut evicted_set: HashSet<usize> = HashSet::new();
        for i in 12..200 {
            if let SamplerOp::Replaced(evicted) = s.accept(i, &mut rng) {
                evicted_set.insert(evicted);
                // Evicted index must NOT still be in the reservoir.
                assert!(!s.contains(evicted));
            }
        }
        assert!(!evicted_set.is_empty(), "expected at least one Replaced");
    }

    #[test]
    fn no_duplicate_indices_in_reservoir() {
        let mut s = ReservoirSampler::new(50, 0.0).unwrap();
        let mut rng = fresh_rng(2026);
        for i in 0..10_000 {
            s.accept(i, &mut rng);
        }
        let indices: Vec<usize> = s.iter_indices().collect();
        let unique: HashSet<usize> = indices.iter().copied().collect();
        assert_eq!(indices.len(), unique.len());
        assert!(indices.len() <= s.capacity());
    }

    #[test]
    fn reset_clears_state() {
        let mut s = ReservoirSampler::new(4, 0.0).unwrap();
        let mut rng = fresh_rng(0);
        for i in 0..10 {
            s.accept(i, &mut rng);
        }
        assert!(s.entries_seen() > 0);
        s.reset();
        assert_eq!(s.entries_seen(), 0);
        assert!(s.is_empty());
    }

    #[test]
    fn deterministic_under_fixed_seed() {
        fn run(seed: u64) -> Vec<usize> {
            let mut s = ReservoirSampler::new(8, 0.0).unwrap();
            let mut rng = fresh_rng(seed);
            for i in 0..100 {
                s.accept(i, &mut rng);
            }
            let mut idxs: Vec<usize> = s.iter_indices().collect();
            idxs.sort_unstable();
            idxs
        }
        assert_eq!(run(2026), run(2026));
        assert_ne!(run(2026), run(7));
    }

    /// Chi-square style uniformity test. With λ=0 every input position
    /// should land in the final reservoir with probability ≈ K/N.
    #[test]
    fn uniform_distribution_with_zero_decay() {
        const CAP: usize = 32;
        const N: usize = 1024;
        const TRIALS: usize = 256;

        let mut counts = vec![0_u32; N];
        for trial in 0..TRIALS {
            #[allow(clippy::cast_possible_truncation)]
            let mut s = ReservoirSampler::new(CAP, 0.0).unwrap();
            let mut rng = fresh_rng(trial as u64 + 1);
            for i in 0..N {
                s.accept(i, &mut rng);
            }
            for idx in s.iter_indices() {
                counts[idx] += 1;
            }
        }

        let expected = (TRIALS * CAP) as f64 / N as f64;
        // Average per-bucket count and check tightness around expectation.
        let total: u32 = counts.iter().sum();
        assert_eq!(total as usize, TRIALS * CAP);

        // No bucket should be more than 4× the expected count
        // (very loose chi-square style sanity bound).
        let max_count = *counts.iter().max().unwrap();
        assert!(
            f64::from(max_count) <= expected * 4.0,
            "uniform sample biased: max_count={max_count} expected={expected}"
        );
        // Also: at least 80% of buckets received at least one hit.
        let nonzero = counts.iter().filter(|&&c| c > 0).count();
        assert!(
            nonzero as f64 >= 0.80 * N as f64,
            "uniform sample too sparse: {nonzero}/{N} buckets non-zero"
        );
    }

    /// With λ > 0 the reservoir should favour recent input positions.
    #[test]
    fn recency_bias_with_positive_decay() {
        const CAP: usize = 32;
        const N: usize = 2048;
        const TRIALS: usize = 256;
        const LAMBDA: f64 = 0.01;

        let mut recent_count = 0_u32;
        for trial in 0..TRIALS {
            #[allow(clippy::cast_possible_truncation)]
            let mut s = ReservoirSampler::new(CAP, LAMBDA).unwrap();
            let mut rng = fresh_rng(trial as u64 + 100);
            for i in 0..N {
                s.accept(i, &mut rng);
            }
            for idx in s.iter_indices() {
                if idx >= (N - N / 10) {
                    recent_count += 1;
                }
            }
        }

        // With λ=0, expect ~10% of reservoir from last 10% of inputs.
        // With λ > 0, recent items should be over-represented — assert
        // > 25% (well above the uniform baseline).
        let total: u32 = (TRIALS * CAP) as u32;
        let recent_share = f64::from(recent_count) / f64::from(total);
        assert!(
            recent_share > 0.25,
            "expected recency bias, got share={recent_share}"
        );
    }

    /// Without decay, indices match the uniform baseline closely:
    /// the share of "last 10%" stays near 10%.
    #[test]
    fn uniform_baseline_without_decay() {
        const CAP: usize = 32;
        const N: usize = 2048;
        const TRIALS: usize = 256;

        let mut recent_count = 0_u32;
        for trial in 0..TRIALS {
            #[allow(clippy::cast_possible_truncation)]
            let mut s = ReservoirSampler::new(CAP, 0.0).unwrap();
            let mut rng = fresh_rng(trial as u64 + 500);
            for i in 0..N {
                s.accept(i, &mut rng);
            }
            for idx in s.iter_indices() {
                if idx >= (N - N / 10) {
                    recent_count += 1;
                }
            }
        }

        let total = (TRIALS * CAP) as u32;
        let share = f64::from(recent_count) / f64::from(total);
        assert!(
            (0.06..0.15).contains(&share),
            "uniform baseline drifted: share={share}"
        );
    }

    #[test]
    fn iter_indices_matches_len() {
        let mut s = ReservoirSampler::new(5, 0.0).unwrap();
        let mut rng = fresh_rng(1);
        for i in 0..100 {
            s.accept(i, &mut rng);
        }
        assert_eq!(s.iter_indices().count(), s.len());
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn weighted_entry_ord_is_total_on_finite_weights() {
        let a = WeightedEntry {
            weight: 1.0,
            point_idx: 0,
        };
        let b = WeightedEntry {
            weight: 2.0,
            point_idx: 1,
        };
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a.cmp(&a), Ordering::Equal);
    }
}
