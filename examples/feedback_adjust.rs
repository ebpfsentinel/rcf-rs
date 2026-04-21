#![allow(clippy::unwrap_used, clippy::panic, clippy::cast_precision_loss)]
//! Demo of `FeedbackStore` SOC-analyst-label ingestion — warms
//! a forest on a clean baseline, identifies a legitimate-but-
//! outlier probe that scores high, labels it `Benign`, then
//! verifies the adjusted score drops toward baseline on repeat
//! queries.
//!
//! Run: `cargo run --release --example feedback_adjust`

use rcf_rs::{FeedbackLabel, FeedbackStore, ForestBuilder};

const D: usize = 4;

fn main() {
    let mut forest = ForestBuilder::<D>::new()
        .num_trees(100)
        .sample_size(256)
        .seed(2026)
        .build()
        .unwrap();

    // Warm on a periodic-ish baseline.
    let mut t = 0.0_f64;
    for _ in 0..2_000 {
        let p = [t.sin(), (t * 0.3).cos(), (t * 0.7).sin(), (t * 0.2).cos()];
        forest.update(p).unwrap();
        t += 0.1;
    }

    // Probe that sits off-baseline — analyst inspects, confirms
    // benign (legitimate unusual-but-not-malicious traffic).
    let probe = [2.5_f64, 0.2, -1.8, 0.9];
    let raw_before: f64 = forest.score(&probe).unwrap().into();
    println!("raw score before feedback = {raw_before:.3}");

    let mut feedback = FeedbackStore::<D>::default_store();
    feedback.label(probe, FeedbackLabel::Benign).unwrap();

    let raw_after: f64 = forest.score(&probe).unwrap().into();
    let adjusted_after = feedback.adjust(&probe, raw_after);
    println!("raw score after feedback  = {raw_after:.3} (unchanged — forest untouched)");
    println!(
        "adjusted score            = {adjusted_after:.3} (pulled toward baseline by Benign label)"
    );

    // A second probe near the labelled point also gets pulled
    // down — the nearest-neighbour kernel spreads the feedback
    // effect to similar traffic.
    let nearby = [2.55_f64, 0.15, -1.85, 0.95];
    let raw_nearby: f64 = forest.score(&nearby).unwrap().into();
    let adj_nearby = feedback.adjust(&nearby, raw_nearby);
    println!("nearby raw = {raw_nearby:.3}, adjusted = {adj_nearby:.3} (kernel-weighted leak)");

    // A probe far from the labelled point is not influenced —
    // feedback is local, not global.
    let far = [-3.0_f64, -3.0, -3.0, -3.0];
    let raw_far: f64 = forest.score(&far).unwrap().into();
    let adj_far = feedback.adjust(&far, raw_far);
    println!("far    raw = {raw_far:.3}, adjusted = {adj_far:.3} (out of kernel range)");
}
