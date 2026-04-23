//! Bloom filter — probabilistic set membership with a bounded
//! false-positive rate (`fpr`) and zero false negatives.
//!
//! Sized from the pair `(n, p)` where `n` is the expected insert
//! capacity and `p` the target `fpr`. Optimal parameters (Mitzenmacher,
//! 2002):
//!
//! - `m = ⌈−n · ln(p) / (ln 2)²⌉` — bit count
//! - `k = round((m / n) · ln 2)` — hash count
//!
//! At the design load (`n` inserts) the filter reaches its target
//! `p`; beyond that the false-positive rate grows geometrically.
//! IOC-membership workloads typically sit near `p = 0.01` with `n`
//! sized to the feed — e.g. 1 M IPs at `p = 0.01` costs ≈ 1.2 MiB
//! of bit state for 7 hashes per lookup.
//!
//! Hashing uses a single `SipHash` call per key, split into two
//! 32-bit lanes and combined via the Kirsch-Mitzenmacher
//! double-hashing trick: `h_i(x) = h1(x) + i · h2(x)`. The two
//! hashes are indistinguishable from independent hashes for any
//! filter parameters that matter in practice (Kirsch & Mitzenmacher
//! 2008 — *Less Hashing, Same Performance*).
//!
//! Gated behind `std` (uses [`std::hash::DefaultHasher`]).
//!
//! # References
//!
//! 1. B. Bloom, "Space/Time Trade-offs in Hash Coding with
//!    Allowable Errors", CACM 13(7), 1970.
//! 2. A. Kirsch, M. Mitzenmacher, "Less Hashing, Same Performance:
//!    Building a Better Bloom Filter", ESA 2006 / RSA 2008.

use alloc::vec;
use alloc::vec::Vec;
use core::hash::{Hash, Hasher};
use std::hash::DefaultHasher;

use crate::error::{RcfError, RcfResult};

/// Default target false-positive rate — 1 %.
pub const DEFAULT_FALSE_POSITIVE_RATE: f64 = 0.01;

/// Maximum hash functions per query. Guards against pathological
/// `(n, p)` pairs that would otherwise degenerate into `k > 64`
/// hashes — the 64 cap is well past the useful regime
/// (`p ≈ 2⁻⁶⁴`).
pub const MAX_HASHES: u32 = 64;

/// Probabilistic set-membership sketch. `insert` is O(k);
/// `contains` is O(k) with zero false negatives and a tunable
/// false-positive rate.
///
/// # Examples
///
/// ```
/// use anomstream_core::BloomFilter;
///
/// // Size for 10 000 IOCs at 1 % false-positive rate.
/// let mut bf = BloomFilter::new(10_000, 0.01).expect("params");
/// bf.insert_bytes(b"malicious.example");
/// assert!(bf.contains_bytes(b"malicious.example"));
/// assert!(!bf.contains_bytes(b"benign.example"));
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BloomFilter {
    /// Bit bank — `ceil(num_bits / 64)` words.
    bits: Vec<u64>,
    /// Configured bit count `m`.
    num_bits: usize,
    /// Hash function count `k`.
    num_hashes: u32,
    /// Total `insert` calls — ops signal, also feeds the saturation
    /// check in [`Self::effective_fpr`].
    total_added: u64,
}

impl BloomFilter {
    /// Build a filter sized for `capacity` expected inserts at the
    /// given false-positive rate `fpr`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `capacity == 0`,
    /// `fpr` is outside `(0, 1)` or not finite, or the derived
    /// hash count exceeds [`MAX_HASHES`].
    pub fn new(capacity: usize, fpr: f64) -> RcfResult<Self> {
        if capacity == 0 {
            return Err(RcfError::InvalidConfig(alloc::string::ToString::to_string(
                "BloomFilter: capacity must be > 0",
            )));
        }
        if !fpr.is_finite() || fpr <= 0.0 || fpr >= 1.0 {
            return Err(RcfError::InvalidConfig(alloc::format!(
                "BloomFilter: fpr {fpr} must be in (0, 1)"
            )));
        }
        let ln2 = core::f64::consts::LN_2;
        #[allow(clippy::cast_precision_loss)]
        let n_f = capacity as f64;
        let m_f = (-n_f * fpr.ln() / (ln2 * ln2)).ceil().max(1.0);
        let k_f = ((m_f / n_f) * ln2).round().max(1.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let num_bits = m_f as usize;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let num_hashes = k_f as u32;
        Self::with_params(num_bits, num_hashes)
    }

    /// Default filter sized at `fpr = 0.01` for `capacity` inserts.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] on `capacity == 0`.
    pub fn with_capacity(capacity: usize) -> RcfResult<Self> {
        Self::new(capacity, DEFAULT_FALSE_POSITIVE_RATE)
    }

    /// Build a filter with explicit `num_bits` and `num_hashes`.
    /// Escape hatch for callers with their own sizing policy (e.g.
    /// page-aligned bit banks, fixed `k` for hardware parity).
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] on `num_bits == 0`,
    /// `num_hashes == 0`, or `num_hashes > MAX_HASHES`.
    pub fn with_params(num_bits: usize, num_hashes: u32) -> RcfResult<Self> {
        if num_bits == 0 {
            return Err(RcfError::InvalidConfig(alloc::string::ToString::to_string(
                "BloomFilter: num_bits must be > 0",
            )));
        }
        if num_hashes == 0 || num_hashes > MAX_HASHES {
            return Err(RcfError::InvalidConfig(alloc::format!(
                "BloomFilter: num_hashes {num_hashes} out of (0, {MAX_HASHES}]"
            )));
        }
        let words = num_bits.div_ceil(64);
        Ok(Self {
            bits: vec![0_u64; words],
            num_bits,
            num_hashes,
            total_added: 0,
        })
    }

    /// Configured bit count `m`.
    #[must_use]
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Configured hash count `k`.
    #[must_use]
    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    /// Total `insert` calls — ops signal.
    #[must_use]
    pub fn total_added(&self) -> u64 {
        self.total_added
    }

    /// Memory footprint of the bit bank in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.bits.len() * core::mem::size_of::<u64>()
    }

    /// `true` when no insertions have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_added == 0
    }

    /// Current empirical false-positive rate based on
    /// [`Self::total_added`]. Upper-bounded by
    /// `(1 − e^(−k·n/m))^k`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn effective_fpr(&self) -> f64 {
        let n = self.total_added as f64;
        let m = self.num_bits as f64;
        let k = f64::from(self.num_hashes);
        (1.0 - (-k * n / m).exp()).powf(k)
    }

    /// Insert a `Hash`-able value.
    pub fn insert<T: Hash + ?Sized>(&mut self, value: &T) {
        let (h1, h2) = double_hash(value);
        self.insert_hash(h1, h2);
    }

    /// Insert a raw byte key — skips generic `Hash` dispatch.
    pub fn insert_bytes(&mut self, key: &[u8]) {
        let (h1, h2) = double_hash(key);
        self.insert_hash(h1, h2);
    }

    /// Insert the caller-supplied `(h1, h2)` pair. Escape hatch for
    /// callers with a stronger hasher — accuracy depends on the
    /// pair being uniform modulo `num_bits`.
    pub fn insert_hash(&mut self, h1: u64, h2: u64) {
        self.total_added = self.total_added.saturating_add(1);
        for i in 0..self.num_hashes {
            let idx = self.combined_index(h1, h2, i);
            self.set_bit(idx);
        }
    }

    /// Query a `Hash`-able value. Returns `true` when every probed
    /// bit is set — may be a false positive, never a false negative.
    #[must_use]
    pub fn contains<T: Hash + ?Sized>(&self, value: &T) -> bool {
        let (h1, h2) = double_hash(value);
        self.contains_hash(h1, h2)
    }

    /// Query a raw byte key.
    #[must_use]
    pub fn contains_bytes(&self, key: &[u8]) -> bool {
        let (h1, h2) = double_hash(key);
        self.contains_hash(h1, h2)
    }

    /// Query with a caller-supplied `(h1, h2)` pair.
    #[must_use]
    pub fn contains_hash(&self, h1: u64, h2: u64) -> bool {
        for i in 0..self.num_hashes {
            let idx = self.combined_index(h1, h2, i);
            if !self.get_bit(idx) {
                return false;
            }
        }
        true
    }

    /// Merge `other` into `self` via bitwise OR. Equivalent to the
    /// union of the underlying sets; keeps zero false negatives.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when the two filters
    /// disagree on `num_bits` or `num_hashes`.
    pub fn union(&mut self, other: &Self) -> RcfResult<()> {
        if self.num_bits != other.num_bits || self.num_hashes != other.num_hashes {
            return Err(RcfError::InvalidConfig(alloc::format!(
                "BloomFilter::union: shape mismatch ({}/{} vs {}/{})",
                self.num_bits,
                self.num_hashes,
                other.num_bits,
                other.num_hashes,
            )));
        }
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a |= *b;
        }
        self.total_added = self.total_added.saturating_add(other.total_added);
        Ok(())
    }

    /// Clear every bit. Capacity is preserved.
    pub fn reset(&mut self) {
        for w in &mut self.bits {
            *w = 0;
        }
        self.total_added = 0;
    }

    /// Kirsch-Mitzenmacher combined hash: `g_i(x) = h1 + i·h2
    /// (mod m)`.
    fn combined_index(&self, h1: u64, h2: u64, i: u32) -> usize {
        let combined = h1.wrapping_add(u64::from(i).wrapping_mul(h2));
        #[allow(clippy::cast_possible_truncation)]
        let modded = (combined % (self.num_bits as u64)) as usize;
        modded
    }

    /// Set bit `idx` in the word-packed bit bank.
    fn set_bit(&mut self, idx: usize) {
        let (w, b) = (idx >> 6, idx & 63);
        self.bits[w] |= 1_u64 << b;
    }

    /// Read bit `idx` from the word-packed bit bank.
    fn get_bit(&self, idx: usize) -> bool {
        let (w, b) = (idx >> 6, idx & 63);
        (self.bits[w] >> b) & 1 == 1
    }
}

/// Derive the `(h1, h2)` pair from a single `SipHash` pass — splits
/// the 64-bit digest into `(low32, high32)` then expands each half
/// back to 64 bits with a prime mix. Cheaper than two independent
/// hasher passes; the double-hashing trick covers any residual
/// correlation.
fn double_hash<T: Hash + ?Sized>(value: &T) -> (u64, u64) {
    let mut h = DefaultHasher::new();
    value.hash(&mut h);
    let full = h.finish();
    let h1 = full;
    // Mix the upper half with a 64-bit odd constant (same family
    // Rust's standard hashmap uses as a perturbation) to decorrelate
    // `h2` from `h1` below the low bits where the bit index lives.
    let h2 = full.rotate_left(32).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h1, h2)
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_invalid_params() {
        assert!(BloomFilter::new(0, 0.01).is_err());
        assert!(BloomFilter::new(1_000, 0.0).is_err());
        assert!(BloomFilter::new(1_000, 1.0).is_err());
        assert!(BloomFilter::new(1_000, f64::NAN).is_err());
        assert!(BloomFilter::new(1_000, -0.1).is_err());
    }

    #[test]
    fn with_params_rejects_zero_and_oversized_k() {
        assert!(BloomFilter::with_params(0, 4).is_err());
        assert!(BloomFilter::with_params(1_024, 0).is_err());
        assert!(BloomFilter::with_params(1_024, MAX_HASHES + 1).is_err());
    }

    #[test]
    fn sizing_matches_optimal_formulas() {
        // n = 10 000, p = 0.01 → m ≈ 95 851, k ≈ 7.
        let bf = BloomFilter::new(10_000, 0.01).unwrap();
        assert!((95_000..=96_000).contains(&bf.num_bits()));
        assert_eq!(bf.num_hashes(), 7);
    }

    #[test]
    fn no_false_negatives_on_inserted_keys() {
        let mut bf = BloomFilter::new(1_000, 0.01).unwrap();
        for i in 0..1_000_u32 {
            bf.insert_bytes(&i.to_le_bytes());
        }
        for i in 0..1_000_u32 {
            assert!(bf.contains_bytes(&i.to_le_bytes()));
        }
    }

    #[test]
    fn contains_before_insert_is_false() {
        let bf = BloomFilter::new(1_000, 0.01).unwrap();
        assert!(!bf.contains_bytes(b"never-inserted"));
    }

    #[test]
    fn false_positive_rate_within_budget() {
        // Insert n keys, query n fresh keys — fraction of hits is
        // the empirical FPR.  Tolerance 3× target to absorb noise.
        let target = 0.01_f64;
        let mut bf = BloomFilter::new(10_000, target).unwrap();
        for i in 0..10_000_u32 {
            bf.insert_bytes(&i.to_le_bytes());
        }
        let mut hits = 0_u32;
        for i in 10_000_u32..20_000 {
            if bf.contains_bytes(&i.to_le_bytes()) {
                hits += 1;
            }
        }
        let fpr = f64::from(hits) / 10_000.0;
        assert!(fpr < target * 3.0, "fpr={fpr}");
    }

    #[test]
    fn union_matches_either_insert() {
        let mut a = BloomFilter::new(1_000, 0.01).unwrap();
        let mut b = BloomFilter::new(1_000, 0.01).unwrap();
        a.insert_bytes(b"alpha");
        b.insert_bytes(b"beta");
        a.union(&b).unwrap();
        assert!(a.contains_bytes(b"alpha"));
        assert!(a.contains_bytes(b"beta"));
    }

    #[test]
    fn union_rejects_shape_mismatch() {
        let mut a = BloomFilter::new(1_000, 0.01).unwrap();
        let b = BloomFilter::new(2_000, 0.01).unwrap();
        assert!(a.union(&b).is_err());
    }

    #[test]
    fn reset_clears_bits_but_keeps_capacity() {
        let mut bf = BloomFilter::new(1_000, 0.01).unwrap();
        for i in 0..100_u32 {
            bf.insert_bytes(&i.to_le_bytes());
        }
        bf.reset();
        assert!(bf.is_empty());
        assert!(!bf.contains_bytes(&0_u32.to_le_bytes()));
        // Post-reset inserts still work.
        bf.insert_bytes(b"fresh");
        assert!(bf.contains_bytes(b"fresh"));
    }

    #[test]
    fn generic_hash_and_byte_paths_agree() {
        let mut a = BloomFilter::new(1_000, 0.01).unwrap();
        let mut b = BloomFilter::new(1_000, 0.01).unwrap();
        let key = b"same-key";
        a.insert(&key.as_slice());
        b.insert_bytes(key);
        // Both paths go through `double_hash(<[u8]>)` so the bits
        // set must be identical.
        assert_eq!(a.bits, b.bits);
    }

    #[test]
    fn effective_fpr_grows_with_load() {
        let mut bf = BloomFilter::new(1_000, 0.01).unwrap();
        let empty = bf.effective_fpr();
        for i in 0..500_u32 {
            bf.insert_bytes(&i.to_le_bytes());
        }
        let half = bf.effective_fpr();
        for i in 500..1_000_u32 {
            bf.insert_bytes(&i.to_le_bytes());
        }
        let full = bf.effective_fpr();
        assert!(empty < half && half < full);
        assert!(full < 0.015); // near design target 1 %.
    }

    #[cfg(all(feature = "serde", feature = "postcard"))]
    #[test]
    fn postcard_roundtrip_preserves_membership() {
        let mut bf = BloomFilter::new(1_000, 0.01).unwrap();
        for i in 0..500_u32 {
            bf.insert_bytes(&i.to_le_bytes());
        }
        let bytes = postcard::to_allocvec(&bf).expect("serde ok");
        let back: BloomFilter = postcard::from_bytes(&bytes).expect("serde ok");
        for i in 0..500_u32 {
            assert!(back.contains_bytes(&i.to_le_bytes()));
        }
        assert_eq!(bf.num_bits(), back.num_bits());
        assert_eq!(bf.num_hashes(), back.num_hashes());
    }
}
