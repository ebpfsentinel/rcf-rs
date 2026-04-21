//! Hot-path integration primitives for `rcf-rs` callers that run on
//! latency-critical ingress paths (eBPF TC action, XDP, per-packet
//! anomaly classifier).
//!
//! The bare [`crate::RandomCutForest::update`] takes ~23 µs at the
//! AWS-default `(100 trees, 256 samples, D = 16)` config — an order
//! of magnitude above the few-µs budget a TC action has per packet
//! at 10 Gbps+. Making `update` faster hits diminishing returns; the
//! architectural answer is to **decouple** the score-on-path from
//! the update-off-path and shed low-value updates before they queue.
//!
//! This module ships two orthogonal building blocks:
//!
//! - [`UpdateSampler`] — stride or per-flow-hash decision
//!   "does this packet contribute to the reservoir?" — discards the
//!   rest before any RCF work. Per-flow sampling keeps the baseline
//!   shape (every flow has *some* representation) while cutting the
//!   update rate by the sampling ratio.
//! - [`channel`] — bounded MPSC channel carrying `[f64; D]` points
//!   from classifier threads to a **single dedicated updater
//!   thread**. The classifier thread `try_enqueue`s non-blockingly
//!   and scores against the *previous* forest snapshot; the updater
//!   thread drains the queue into `RandomCutForest::update` at its
//!   own cadence. Dropped points (queue full) are tallied for ops.
//!
//! Coarser `time_decay` is the third dial, but that's configured
//! directly on [`crate::ForestBuilder::time_decay`]; see
//! `docs/performance.md` for the trade-off.
//!
//! # Example: classifier + updater split
//!
//! ```ignore
//! use rcf_rs::{ForestBuilder, hot_path};
//! use std::thread;
//!
//! let mut forest = ForestBuilder::<16>::new()
//!     .num_trees(100)
//!     .sample_size(256)
//!     .seed(42)
//!     .build()?;
//!
//! let (producer, mut consumer) = hot_path::channel::<16>(4096);
//! let sampler = hot_path::UpdateSampler::new(8); // 1/8 per-flow
//!
//! // Updater thread.
//! thread::spawn(move || loop {
//!     let (ingested, errors) = consumer.try_drain(|p| forest.update(p));
//!     if ingested == 0 && errors == 0 {
//!         std::thread::sleep(std::time::Duration::from_millis(1));
//!     }
//! });
//!
//! // Classifier thread (hot path — runs per packet).
//! fn on_packet(
//!     features: [f64; 16],
//!     flow_hash: u64,
//!     sampler: &hot_path::UpdateSampler,
//!     producer: &hot_path::UpdateProducer<16>,
//! ) {
//!     if sampler.accept_hash(flow_hash) {
//!         let _ = producer.try_enqueue(features);
//!     }
//! }
//! ```

#![cfg(feature = "std")]

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use crate::metrics::{MetricsSink, default_sink, names};

/// Stride-based or per-flow-hash update sampler.
///
/// Accepts `1 / keep_every_n` of the offered updates. The sampler
/// itself never touches the forest — callers invoke `accept_*`
/// before calling [`crate::RandomCutForest::update`].
///
/// `keep_every_n = 0` and `keep_every_n = 1` both disable sampling
/// (every offer is accepted). Values `>= 2` gate proportionally.
#[derive(Debug)]
pub struct UpdateSampler {
    /// Divisor: keep `1 / keep_every_n` offered updates.
    keep_every_n: u32,
    /// Monotonic stride counter for [`Self::accept_stride`].
    counter: AtomicU64,
    /// Running total of accepted offers — observability signal.
    accepted: AtomicU64,
    /// Running total of rejected offers.
    rejected: AtomicU64,
    /// Per-sampler secret multipliers used by [`Self::accept_hash`].
    /// When non-zero the sampler runs a keyed remix of the caller-
    /// supplied `flow_hash` before the modulo decision — makes the
    /// admission boundary unpredictable to an attacker who can
    /// observe or influence their own `flow_hash` value but cannot
    /// learn the sampler secret. Zero-init means "no remix", matches
    /// the historical deterministic behaviour of [`Self::new`].
    mix_k1: u64,
    /// Second secret — XOR'd at the end of the mix to avoid the
    /// multiply by `mix_k1` alone (a structure the attacker could
    /// invert given enough observations).
    mix_k2: u64,
    /// Observability sink — every `accept_*` call emits an
    /// accepted/rejected counter. Defaults to [`crate::NoopSink`].
    metrics: Arc<dyn MetricsSink>,
}

impl UpdateSampler {
    /// Build a sampler keeping 1 offer out of every `keep_every_n`.
    /// `0` and `1` disable sampling (every offer accepted).
    ///
    /// **`accept_hash` admission is deterministic without a
    /// per-sampler secret** — an attacker who can probe the
    /// admission decision on a known `flow_hash` can spray
    /// 5-tuples whose hash lands on the admitted residue class
    /// and poison the reservoir. For internet-facing ingress,
    /// prefer [`Self::new_keyed`] which seeds the secret from
    /// `getrandom` at construction.
    #[must_use]
    pub fn new(keep_every_n: u32) -> Self {
        Self {
            keep_every_n,
            counter: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            mix_k1: 0,
            mix_k2: 0,
            metrics: default_sink(),
        }
    }

    /// Keyed variant — same ratio semantics as [`Self::new`] but
    /// with a per-sampler secret mix applied to every
    /// [`Self::accept_hash`] input. Defeats the deterministic-
    /// admission poisoning vector (MITRE ATLAS `AML.T0020`): the
    /// attacker can't steer their own flow hash into the admitted
    /// residue class without knowing the per-sampler mix keys,
    /// which are seeded from the OS CSPRNG at construction and
    /// never leave the process.
    ///
    /// # Errors
    ///
    /// Propagates [`getrandom::Error`] when the OS entropy source
    /// is unavailable (embedded / chroot without `/dev/urandom`).
    ///
    /// # Panics
    ///
    /// Never in practice — the two `try_into().expect(...)` calls
    /// unwrap a compile-time known 8-byte slice taken from a
    /// 16-byte buffer.
    pub fn new_keyed(keep_every_n: u32) -> Result<Self, getrandom::Error> {
        let mut buf = [0_u8; 16];
        getrandom::fill(&mut buf)?;
        let mix_k1 = u64::from_le_bytes(buf[0..8].try_into().expect("16 bytes"));
        let mix_k2 = u64::from_le_bytes(buf[8..16].try_into().expect("16 bytes"));
        // Ensure `mix_k1` is non-zero and odd so its use as a
        // multiplier is always a bijection.
        let mix_k1 = if mix_k1 == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            mix_k1 | 1
        };
        Ok(Self {
            keep_every_n,
            counter: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            mix_k1,
            mix_k2,
            metrics: default_sink(),
        })
    }

    /// Install a metrics sink — every `accept_*` call emits an
    /// accepted/rejected counter through it. Chain-style builder.
    #[must_use]
    pub fn with_metrics_sink(mut self, sink: Arc<dyn MetricsSink>) -> Self {
        self.metrics = sink;
        self
    }

    /// Read-only handle to the installed sink.
    #[must_use]
    pub fn metrics_sink(&self) -> &Arc<dyn MetricsSink> {
        &self.metrics
    }

    /// Whether this sampler was built with a keyed mix.
    #[must_use]
    pub fn is_keyed(&self) -> bool {
        self.mix_k1 != 0
    }

    /// Configured ratio denominator.
    #[must_use]
    pub fn keep_every_n(&self) -> u32 {
        self.keep_every_n
    }

    /// Stride-based decision — every `keep_every_n`-th offer lands.
    /// Call order-dependent: counter increments on every invocation.
    /// Cheap (one atomic fetch-add) but not flow-aware.
    pub fn accept_stride(&self) -> bool {
        if self.keep_every_n <= 1 {
            self.accepted.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .inc_counter(names::HOT_PATH_SAMPLER_ACCEPTED_TOTAL, 1);
            return true;
        }
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let keep = u64::from(self.keep_every_n);
        let ok = n.is_multiple_of(keep);
        if ok {
            self.accepted.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .inc_counter(names::HOT_PATH_SAMPLER_ACCEPTED_TOTAL, 1);
        } else {
            self.rejected.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .inc_counter(names::HOT_PATH_SAMPLER_REJECTED_TOTAL, 1);
        }
        ok
    }

    /// Per-flow decision — flows with
    /// `keyed_mix(flow_hash) % keep_every_n == 0` are admitted,
    /// every other flow rejected in full. Deterministic across the
    /// sampler's lifetime **for that sampler**: the same flow
    /// always samples the same way, so the baseline keeps
    /// representative coverage of every sampled flow rather than
    /// slicing any single flow.
    ///
    /// The caller supplies `flow_hash` — typically a 64-bit mix of
    /// 5-tuple bytes (`SipHash` / `FxHash` / custom). Quality of
    /// sampling only matters modulo `keep_every_n`.
    ///
    /// When the sampler was built via [`Self::new_keyed`] a
    /// per-sampler secret mix (murmur-style finaliser keyed on
    /// `mix_k1` / `mix_k2`) is applied **before** the modulo so the
    /// admission residue class is unpredictable without the secret
    /// (defends against `AML.T0020` reservoir-poisoning sprays).
    pub fn accept_hash(&self, flow_hash: u64) -> bool {
        if self.keep_every_n <= 1 {
            self.accepted.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .inc_counter(names::HOT_PATH_SAMPLER_ACCEPTED_TOTAL, 1);
            return true;
        }
        let mixed = self.keyed_mix(flow_hash);
        let ok = mixed.is_multiple_of(u64::from(self.keep_every_n));
        if ok {
            self.accepted.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .inc_counter(names::HOT_PATH_SAMPLER_ACCEPTED_TOTAL, 1);
        } else {
            self.rejected.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .inc_counter(names::HOT_PATH_SAMPLER_REJECTED_TOTAL, 1);
        }
        ok
    }

    /// Murmur3-64 finaliser keyed by the sampler secret. Returns
    /// `h` unchanged when the sampler was built unkeyed (via
    /// [`Self::new`]) so the legacy `accept_hash` behaviour is
    /// preserved bit-for-bit.
    #[inline]
    fn keyed_mix(&self, h: u64) -> u64 {
        if self.mix_k1 == 0 {
            return h;
        }
        let mut x = h.wrapping_add(self.mix_k1);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        x ^= x >> 33;
        x ^ self.mix_k2
    }

    /// Running total of accepted offers since construction.
    #[must_use]
    pub fn accepted_total(&self) -> u64 {
        self.accepted.load(Ordering::Relaxed)
    }

    /// Running total of rejected offers since construction.
    #[must_use]
    pub fn rejected_total(&self) -> u64 {
        self.rejected.load(Ordering::Relaxed)
    }
}

/// Producer end of the hot-path update queue. Multi-producer: clone
/// this handle per classifier thread. `try_enqueue` is non-blocking
/// — on queue full it returns `false` and increments `dropped_total`.
#[derive(Debug)]
pub struct UpdateProducer<const D: usize> {
    /// Underlying bounded MPSC sender.
    tx: SyncSender<[f64; D]>,
    /// Capacity the channel was built with — surfaced for gauges.
    capacity: usize,
    /// Lifetime enqueued count.
    enqueued: Arc<AtomicU64>,
    /// Lifetime dropped-on-full count.
    dropped: Arc<AtomicU64>,
    /// Observability sink — shared with every clone of this
    /// producer so every classifier thread emits to the same
    /// endpoint.
    metrics: Arc<dyn MetricsSink>,
}

impl<const D: usize> Clone for UpdateProducer<D> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            capacity: self.capacity,
            enqueued: Arc::clone(&self.enqueued),
            dropped: Arc::clone(&self.dropped),
            metrics: Arc::clone(&self.metrics),
        }
    }
}

impl<const D: usize> UpdateProducer<D> {
    /// Non-blocking enqueue. Returns `true` when the point landed in
    /// the queue, `false` when the queue was full (point dropped,
    /// `dropped_total` incremented).
    #[must_use]
    pub fn try_enqueue(&self, point: [f64; D]) -> bool {
        if self.tx.try_send(point).is_ok() {
            self.enqueued.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .inc_counter(names::HOT_PATH_QUEUE_ENQUEUED_TOTAL, 1);
            true
        } else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .inc_counter(names::HOT_PATH_QUEUE_DROPPED_TOTAL, 1);
            false
        }
    }

    /// Read-only handle to the installed sink.
    #[must_use]
    pub fn metrics_sink(&self) -> &Arc<dyn MetricsSink> {
        &self.metrics
    }

    /// Channel capacity as configured at [`channel`].
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Lifetime count of successfully enqueued points.
    #[must_use]
    pub fn enqueued_total(&self) -> u64 {
        self.enqueued.load(Ordering::Relaxed)
    }

    /// Lifetime count of points dropped because the queue was full.
    /// Non-zero indicates classifier thread is producing faster than
    /// the updater thread is draining — raise alert / widen the
    /// channel / raise the sampler ratio.
    #[must_use]
    pub fn dropped_total(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

/// Consumer end of the hot-path update queue. Single-consumer: hand
/// this handle to the dedicated updater thread.
#[derive(Debug)]
pub struct UpdateConsumer<const D: usize> {
    /// Underlying MPSC receiver.
    rx: Receiver<[f64; D]>,
}

impl<const D: usize> UpdateConsumer<D> {
    /// Drain every point currently queued into `sink`. Returns
    /// `(ingested, errors)` — number of successful sink calls and
    /// number of errored sink calls (typically forest reservoir
    /// errors). The method returns as soon as the queue is empty so
    /// the updater thread can back off or gauge throughput.
    pub fn try_drain<F, E>(&self, mut sink: F) -> (usize, usize)
    where
        F: FnMut([f64; D]) -> Result<(), E>,
    {
        let mut ingested = 0;
        let mut errors = 0;
        while let Ok(p) = self.rx.try_recv() {
            if sink(p).is_ok() {
                ingested += 1;
            } else {
                errors += 1;
            }
        }
        (ingested, errors)
    }

    /// Blocking-take of the next point. Returns `None` when every
    /// [`UpdateProducer`] has been dropped (clean shutdown).
    #[must_use]
    pub fn recv(&self) -> Option<[f64; D]> {
        self.rx.recv().ok()
    }
}

/// Build a bounded MPSC hot-path update channel with the requested
/// capacity. Returns `(producer, consumer)` — clone the producer per
/// classifier thread; hand the consumer to the dedicated updater
/// thread.
///
/// `capacity` is the in-flight queue depth; sizing it ~1 second's
/// worth of expected update rate keeps the updater thread busy
/// without unbounded backpressure under micro-bursts. Drop events
/// (observable via [`UpdateProducer::dropped_total`]) are the ops
/// signal the channel needs widening.
#[must_use]
pub fn channel<const D: usize>(capacity: usize) -> (UpdateProducer<D>, UpdateConsumer<D>) {
    channel_with_sink(capacity, default_sink())
}

/// Same as [`channel`] but with a caller-supplied metrics sink.
/// Every clone of the returned producer shares the same sink.
#[must_use]
pub fn channel_with_sink<const D: usize>(
    capacity: usize,
    sink: Arc<dyn MetricsSink>,
) -> (UpdateProducer<D>, UpdateConsumer<D>) {
    let (tx, rx) = sync_channel::<[f64; D]>(capacity);
    let enqueued = Arc::new(AtomicU64::new(0));
    let dropped = Arc::new(AtomicU64::new(0));
    (
        UpdateProducer {
            tx,
            capacity,
            enqueued,
            dropped,
            metrics: sink,
        },
        UpdateConsumer { rx },
    )
}

/// Fixed-bucket per-prefix rate cap — bounds how many admissions a
/// single source prefix can push into the reservoir within a
/// rolling time window. Defends against the reservoir-poisoning
/// spray where an attacker floods the ingress from one IP prefix
/// hoping a fraction land in the reservoir via
/// [`UpdateSampler::accept_hash`].
///
/// Implementation: 256 atomic `u32` buckets indexed by
/// `prefix_hash & 0xff`. Collisions are soft — the cap bounds
/// across the *bucket*, not the exact prefix. This trades a small
/// amount of cross-prefix interference for O(1) lock-free
/// check-and-record with bounded memory.
///
/// # Example
///
/// ```ignore
/// use rcf_rs::hot_path::PrefixRateCap;
///
/// let cap = PrefixRateCap::new(100, 1_000); // 100 admits / 1 s window
/// let now_ms = /* wall-clock */;
/// if cap.check_and_record(flow_prefix_hash, now_ms) {
///     forest.update(point)?;
/// }
/// ```
#[derive(Debug)]
pub struct PrefixRateCap {
    /// Per-bucket admit counter. 256 buckets keep the struct at
    /// 1 KiB and give tolerable collision rates for typical
    /// /24-prefix spreads.
    buckets: [AtomicU32; Self::BUCKETS],
    /// Epoch-millisecond timestamp at which the current window
    /// opened. The next `check_and_record` past `+ window_ms`
    /// atomically resets the buckets.
    window_start_ms: AtomicU64,
    /// Window length. Cap counts reset every `window_ms`.
    window_ms: u64,
    /// Maximum admits per bucket per window.
    cap_per_window: u32,
    /// Lifetime count of admits that hit the cap and were rejected.
    capped_total: AtomicU64,
    /// Lifetime count of admits passed through.
    admitted_total: AtomicU64,
    /// Observability sink.
    metrics: Arc<dyn MetricsSink>,
}

impl PrefixRateCap {
    /// Number of buckets. Compile-time constant; 256 buckets ×
    /// `AtomicU32` = 1 KiB on 64-bit.
    pub const BUCKETS: usize = 256;

    /// Build a rate cap. `cap_per_window = 0` disables the cap
    /// (every call to `check_and_record` returns `true`);
    /// `window_ms = 0` is rejected as invalid.
    ///
    /// # Panics
    ///
    /// Panics when `window_ms == 0` — a zero-length window is a
    /// programming error (would admit everything on first call).
    #[must_use]
    pub fn new(cap_per_window: u32, window_ms: u64) -> Self {
        assert!(window_ms > 0, "window_ms must be non-zero");
        // Cannot use `[AtomicU32::new(0); 256]` because AtomicU32
        // is !Copy. Build via closure.
        let buckets: [AtomicU32; Self::BUCKETS] = core::array::from_fn(|_| AtomicU32::new(0));
        Self {
            buckets,
            window_start_ms: AtomicU64::new(0),
            window_ms,
            cap_per_window,
            capped_total: AtomicU64::new(0),
            admitted_total: AtomicU64::new(0),
            metrics: default_sink(),
        }
    }

    /// Install a metrics sink — every `check_and_record` call emits
    /// an admitted/capped counter through it.
    #[must_use]
    pub fn with_metrics_sink(mut self, sink: Arc<dyn MetricsSink>) -> Self {
        self.metrics = sink;
        self
    }

    /// Read-only handle to the installed sink.
    #[must_use]
    pub fn metrics_sink(&self) -> &Arc<dyn MetricsSink> {
        &self.metrics
    }

    /// Record an admission attempt and return `true` when the
    /// caller is allowed to proceed. Thread-safe, lock-free.
    pub fn check_and_record(&self, prefix_hash: u64, now_ms: u64) -> bool {
        if self.cap_per_window == 0 {
            self.admitted_total.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .inc_counter(names::HOT_PATH_PREFIX_ADMITTED_TOTAL, 1);
            return true;
        }
        // Atomically roll the window if needed.
        let mut start = self.window_start_ms.load(Ordering::Relaxed);
        if start == 0 || now_ms.saturating_sub(start) >= self.window_ms {
            // Try to claim ownership of the reset. Lost races are
            // fine — any losing thread accepts that its peer reset
            // the window.
            let attempted = self.window_start_ms.compare_exchange(
                start,
                now_ms,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            if attempted.is_ok() {
                for bucket in &self.buckets {
                    bucket.store(0, Ordering::Relaxed);
                }
            }
            start = now_ms;
            let _ = start; // silence unused when optimiser strips.
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = ((prefix_hash & 0xff) as usize) & (Self::BUCKETS - 1);
        let prior = self.buckets[idx].fetch_add(1, Ordering::Relaxed);
        if prior < self.cap_per_window {
            self.admitted_total.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .inc_counter(names::HOT_PATH_PREFIX_ADMITTED_TOTAL, 1);
            true
        } else {
            // Already over — roll back the increment so a late
            // window roll doesn't accumulate forever on this
            // bucket.
            self.buckets[idx].fetch_sub(1, Ordering::Relaxed);
            self.capped_total.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .inc_counter(names::HOT_PATH_PREFIX_CAPPED_TOTAL, 1);
            false
        }
    }

    /// Lifetime admits that passed the cap.
    #[must_use]
    pub fn admitted_total(&self) -> u64 {
        self.admitted_total.load(Ordering::Relaxed)
    }

    /// Lifetime admits rejected because the bucket was at cap.
    #[must_use]
    pub fn capped_total(&self) -> u64 {
        self.capped_total.load(Ordering::Relaxed)
    }

    /// Window length in milliseconds.
    #[must_use]
    pub fn window_ms(&self) -> u64 {
        self.window_ms
    }

    /// Cap per bucket per window.
    #[must_use]
    pub fn cap_per_window(&self) -> u32 {
        self.cap_per_window
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn sampler_disabled_accepts_every_offer() {
        let s = UpdateSampler::new(0);
        for _ in 0..10 {
            assert!(s.accept_stride());
        }
        assert_eq!(s.accepted_total(), 10);
        assert_eq!(s.rejected_total(), 0);
    }

    #[test]
    fn sampler_one_accepts_every_offer() {
        let s = UpdateSampler::new(1);
        for _ in 0..10 {
            assert!(s.accept_stride());
            assert!(s.accept_hash(0xdead_beef_cafe_babe));
        }
    }

    #[test]
    fn sampler_stride_keeps_one_in_n() {
        let s = UpdateSampler::new(4);
        let mut accepted = 0_usize;
        for _ in 0..100 {
            if s.accept_stride() {
                accepted += 1;
            }
        }
        assert_eq!(accepted, 25);
        assert_eq!(s.accepted_total(), 25);
        assert_eq!(s.rejected_total(), 75);
    }

    #[test]
    fn sampler_hash_deterministic_per_flow() {
        let s = UpdateSampler::new(4);
        // Same hash always decides the same way.
        let h = 0xdead_beef_cafe_babe_u64;
        let d1 = s.accept_hash(h);
        let d2 = s.accept_hash(h);
        assert_eq!(d1, d2);
    }

    #[test]
    fn channel_try_enqueue_drops_on_full() {
        let (p, _c) = channel::<2>(2);
        assert!(p.try_enqueue([1.0, 2.0]));
        assert!(p.try_enqueue([3.0, 4.0]));
        assert!(!p.try_enqueue([5.0, 6.0]));
        assert_eq!(p.enqueued_total(), 2);
        assert_eq!(p.dropped_total(), 1);
    }

    #[test]
    fn channel_try_drain_empties_queue() {
        let (p, c) = channel::<2>(8);
        let _ = p.try_enqueue([1.0, 2.0]);
        let _ = p.try_enqueue([3.0, 4.0]);
        let mut sink: Vec<[f64; 2]> = Vec::new();
        let (ing, err) = c.try_drain::<_, ()>(|pt| {
            sink.push(pt);
            Ok(())
        });
        assert_eq!(ing, 2);
        assert_eq!(err, 0);
        assert_eq!(sink.len(), 2);
    }

    #[test]
    fn channel_producer_is_clone_multi_producer() {
        let (p1, c) = channel::<2>(8);
        let p2 = p1.clone();
        let _ = p1.try_enqueue([1.0, 2.0]);
        let _ = p2.try_enqueue([3.0, 4.0]);
        // Counters are shared across clones.
        assert_eq!(p1.enqueued_total(), 2);
        assert_eq!(p2.enqueued_total(), 2);
        let (ing, _) = c.try_drain::<_, ()>(|_| Ok(()));
        assert_eq!(ing, 2);
    }

    #[test]
    fn channel_try_drain_counts_errors() {
        let (p, c) = channel::<2>(8);
        let _ = p.try_enqueue([1.0, 2.0]);
        let _ = p.try_enqueue([3.0, 4.0]);
        let (ing, err) =
            c.try_drain::<_, &'static str>(|pt| if pt[0] > 2.0 { Err("nope") } else { Ok(()) });
        assert_eq!(ing, 1);
        assert_eq!(err, 1);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn end_to_end_hot_path_wires_sampler_producer_consumer() {
        use crate::ForestBuilder;

        let mut forest = ForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(64)
            .seed(2026)
            .build()
            .unwrap();

        let sampler = UpdateSampler::new(3);
        let (producer, consumer) = channel::<2>(16);

        // Simulated classifier hot-path.
        for i in 0..9_u64 {
            if sampler.accept_hash(i) {
                let _ = producer.try_enqueue([i as f64, (i * 2) as f64]);
            }
        }
        drop(producer);

        // Updater thread.
        let (ing, err) = consumer.try_drain(|p| forest.update(p));
        assert!(err == 0);
        // 0, 3, 6 accepted by accept_hash(keep_every_n=3): 3 points.
        assert_eq!(ing, 3);
    }

    #[test]
    fn keyed_sampler_admission_differs_from_unkeyed() {
        // The unkeyed sampler admits h == 0 mod 4.
        let unkeyed = UpdateSampler::new(4);
        assert!(unkeyed.accept_hash(0));
        assert!(unkeyed.accept_hash(4));
        assert!(!unkeyed.accept_hash(1));
        assert!(!unkeyed.accept_hash(2));

        // A keyed sampler shifts the residue class. Same hashes
        // land on a different admission set unless the mix keys
        // happen to map them back — vanishingly unlikely on a
        // 128-bit secret.
        let keyed = UpdateSampler::new_keyed(4).expect("getrandom works");
        assert!(keyed.is_keyed());
        let same_decision = (0..8_u64)
            .filter(|h| unkeyed.accept_hash(*h) == keyed.accept_hash(*h))
            .count();
        // 8 hashes, 2 admission outcomes → expected match rate 50 %
        // under a random mix. Allow a wide range because the mix
        // is a random oracle, not a uniform shuffle; the point of
        // the assertion is that the keyed sampler is *not* the
        // unkeyed one.
        assert!(
            same_decision < 8,
            "keyed sampler accepted every hash exactly like unkeyed — mix ineffective"
        );
    }

    #[test]
    fn keyed_sampler_is_deterministic_within_sampler() {
        // Same flow hash must decide the same way every call on
        // the same sampler — baseline coverage per flow.
        let s = UpdateSampler::new_keyed(4).unwrap();
        let h = 0xdead_beef_cafe_babe_u64;
        let d1 = s.accept_hash(h);
        let d2 = s.accept_hash(h);
        let d3 = s.accept_hash(h);
        assert_eq!(d1, d2);
        assert_eq!(d2, d3);
    }

    #[test]
    fn prefix_rate_cap_allows_up_to_cap() {
        let cap = PrefixRateCap::new(3, 1_000);
        let prefix = 0x1234_5678_u64;
        let now = 1_000_u64;
        assert!(cap.check_and_record(prefix, now));
        assert!(cap.check_and_record(prefix, now));
        assert!(cap.check_and_record(prefix, now));
        // 4th call hits cap → rejected.
        assert!(!cap.check_and_record(prefix, now));
        assert_eq!(cap.admitted_total(), 3);
        assert_eq!(cap.capped_total(), 1);
    }

    #[test]
    fn prefix_rate_cap_zero_disables_gate() {
        let cap = PrefixRateCap::new(0, 1_000);
        for i in 0..100_u64 {
            assert!(cap.check_and_record(i, 0));
        }
        assert_eq!(cap.admitted_total(), 100);
        assert_eq!(cap.capped_total(), 0);
    }

    #[test]
    fn prefix_rate_cap_resets_on_window_roll() {
        let cap = PrefixRateCap::new(2, 1_000);
        let prefix = 0xabcd_u64;
        // Fill the cap inside one window.
        assert!(cap.check_and_record(prefix, 100));
        assert!(cap.check_and_record(prefix, 200));
        assert!(!cap.check_and_record(prefix, 300));
        // Advance past the window — next call resets and admits.
        assert!(cap.check_and_record(prefix, 1_500));
        assert_eq!(cap.admitted_total(), 3);
    }
}
