#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
//! End-to-end integration test for the `hot_path` module: a
//! classifier thread runs `UpdateSampler::accept_hash` + non-
//! blocking `try_enqueue`, a dedicated updater thread drains the
//! MPSC channel into `RandomCutForest::update`. Pins the
//! invariants the unit tests can't exercise because they don't
//! thread:
//!
//! - Multi-producer: N classifier threads clone the producer and
//!   feed the same consumer. Every successfully-enqueued point
//!   lands in the forest exactly once.
//! - Drop-on-full: when the queue saturates the producer
//!   increments `dropped_total` instead of blocking. Count must
//!   equal `emitted_by_classifier − drained_by_updater`.
//! - Sampler determinism: a given `flow_hash` always takes the
//!   same accept/reject decision across clones of the sampler,
//!   so the per-flow admission ratio is stable.

#![cfg(feature = "std")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use anomstream_core::ForestBuilder;
use anomstream_hotpath::{UpdateSampler, update_channel};

const D: usize = 8;

#[test]
fn classifier_producer_updater_consumer_roundtrip() {
    let mut forest = ForestBuilder::<D>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(2026)
        .build()
        .unwrap();

    let sampler = Arc::new(UpdateSampler::new(4));
    let (producer, consumer) = update_channel::<D>(1024);
    // Hold a local clone so we can read `dropped_total()` after
    // classifier threads have dropped their handles — counters are
    // shared across clones via `Arc<AtomicU64>`.
    let observer = producer.clone();
    let stop = Arc::new(AtomicBool::new(false));

    let stop_rx = Arc::clone(&stop);
    let updater = thread::spawn(move || {
        let mut total = 0_usize;
        loop {
            let (ing, _err) = consumer.try_drain(|p| forest.update(p));
            total += ing;
            if ing == 0 {
                if stop_rx.load(Ordering::Relaxed) {
                    break;
                }
                thread::sleep(Duration::from_millis(1));
            }
        }
        total
    });

    // 3 classifier threads, each emitting `n_per_thread` packets.
    let n_per_thread = 5_000_u64;
    let n_threads = 3_usize;
    let mut classifiers = Vec::with_capacity(n_threads);
    for t in 0..n_threads as u64 {
        let p = producer.clone();
        let s = Arc::clone(&sampler);
        classifiers.push(thread::spawn(move || {
            for i in 0..n_per_thread {
                let mut features = [0.0_f64; D];
                for (d, slot) in features.iter_mut().enumerate() {
                    *slot = ((t * n_per_thread + i) as f64) * 0.001 + (d as f64) * 0.1;
                }
                // Hash combines thread + index so flows are distinct
                // per thread but deterministic per hash.
                let flow_hash = (t << 32) | i;
                if s.accept_hash(flow_hash) {
                    let _ = p.try_enqueue(features);
                }
            }
        }));
    }
    for c in classifiers {
        c.join().unwrap();
    }
    drop(producer);
    // Snapshot drop counter BEFORE dropping `observer` — counters
    // live in a shared Arc<AtomicU64>, so any live clone sees the
    // current total.
    let dropped = observer.dropped_total();
    drop(observer);
    stop.store(true, Ordering::Relaxed);
    let ingested = updater.join().unwrap();

    let sampled_total = sampler.accepted_total();
    let rejected_total = sampler.rejected_total();

    // Every classifier offer is either accepted or rejected by the
    // sampler.
    let total_offered = n_per_thread * n_threads as u64;
    assert_eq!(sampled_total + rejected_total, total_offered);

    // Every accepted offer either landed in the queue (ingested)
    // or was dropped because the queue was full. No loss.
    assert_eq!(
        u64::try_from(ingested).unwrap() + dropped,
        sampled_total,
        "accepted={sampled_total}, ingested={ingested}, dropped={dropped}"
    );
}

#[test]
fn sampler_accept_hash_is_deterministic_across_clones() {
    let s = Arc::new(UpdateSampler::new(4));
    let s2 = Arc::clone(&s);

    // Same hash always decides the same way, across Arc clones.
    let h = 0xdead_beef_cafe_babe_u64;
    let d1 = s.accept_hash(h);
    let d2 = s2.accept_hash(h);
    let d3 = s.accept_hash(h);
    assert_eq!(d1, d2);
    assert_eq!(d2, d3);
}

#[test]
fn channel_drop_on_full_increments_counter() {
    let (p, _c) = update_channel::<2>(2);
    assert!(p.try_enqueue([1.0, 2.0]));
    assert!(p.try_enqueue([3.0, 4.0]));
    // Queue full — both of these must drop.
    assert!(!p.try_enqueue([5.0, 6.0]));
    assert!(!p.try_enqueue([7.0, 8.0]));
    assert_eq!(p.enqueued_total(), 2);
    assert_eq!(p.dropped_total(), 2);
}
