#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
//! Minimal runnable demo of the `hot_path` module — classifier
//! thread enqueues sampled features, dedicated updater thread
//! drains into the forest at its own cadence.
//!
//! Models a realistic eBPF TC-action split:
//!
//! - **Classifier** (hot, single-packet path) — runs per packet,
//!   computes a feature vector, invokes `UpdateSampler::accept_hash`
//!   on the flow hash, and non-blockingly `try_enqueue`s admitted
//!   probes into the update channel. Scoring against the current
//!   forest snapshot would happen here in prod; the demo skips it.
//! - **Updater** (cold, background) — owns `&mut forest`, drains
//!   the channel in a loop via `try_drain`, calls `forest.update`.
//!   Its cadence is decoupled from the classifier.
//!
//! Run with: `cargo run --release --example hot_path`

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use anomstream_core::ForestBuilder;
use anomstream_hotpath::{UpdateSampler, update_channel};

const D: usize = 16;

fn main() {
    let mut forest = ForestBuilder::<D>::new()
        .num_trees(100)
        .sample_size(256)
        .seed(2026)
        .build()
        .unwrap();

    // 1-in-8 per-flow admission: classifier calls `accept_hash`
    // with the flow 5-tuple hash; same flow always samples the
    // same way so baseline coverage stays uniform per-flow.
    let sampler = Arc::new(UpdateSampler::new(8));
    let (producer, consumer) = update_channel::<D>(4096);

    // Shutdown flag so the updater thread exits cleanly once the
    // classifier has finished producing.
    let stop = Arc::new(AtomicBool::new(false));

    // Updater thread: drains the channel into `forest.update`.
    // Owns `&mut forest` exclusively. Under prod, this thread also
    // owns forest persistence checkpoints and metric export.
    let stop_rx = Arc::clone(&stop);
    let updater = thread::spawn(move || {
        let mut total_ingested = 0_usize;
        loop {
            let (ingested, errors) = consumer.try_drain(|p| forest.update(p));
            total_ingested += ingested;
            if errors > 0 {
                eprintln!("updater: {errors} errors this batch");
            }
            if ingested == 0 {
                if stop_rx.load(Ordering::Relaxed) {
                    break;
                }
                thread::sleep(Duration::from_millis(1));
            }
        }
        total_ingested
    });

    // Classifier loop: simulate 10k packets with a mix of flows.
    let classifier_sampler = Arc::clone(&sampler);
    let producer_tx = producer.clone();
    let observer = producer.clone(); // keeps a handle for `dropped_total`
    let classifier = thread::spawn(move || {
        for i in 0_u64..10_000 {
            // Simulated per-packet feature vector.
            let mut features = [0.0_f64; D];
            for (d, slot) in features.iter_mut().enumerate() {
                *slot = ((i as f64) * 0.001 + (d as f64) * 0.1).sin();
            }
            // Fake flow hash — in prod this comes from the 5-tuple.
            let flow_hash = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            if classifier_sampler.accept_hash(flow_hash) {
                let _ = producer_tx.try_enqueue(features);
            }
        }
    });

    classifier.join().unwrap();
    // Drop remaining producer handle to let the updater see an
    // empty queue + shutdown flag.
    drop(producer);
    stop.store(true, Ordering::Relaxed);
    let ingested = updater.join().unwrap();

    println!(
        "classifier emitted       = {} admitted / {} rejected by sampler",
        sampler.accepted_total(),
        sampler.rejected_total()
    );
    println!("updater ingested         = {ingested} points");
    println!(
        "classifier → queue drops = {} (queue full events)",
        observer.dropped_total()
    );
}
