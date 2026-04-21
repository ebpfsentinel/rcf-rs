#![allow(clippy::unwrap_used, clippy::panic, clippy::cast_precision_loss)]
//! Demo of `DriftAwareForest` + `AdwinDetector`: forest scores a
//! two-regime synthetic stream, ADWIN triggers on the mean shift,
//! the wrapper spawns a shadow forest, warms it on the new
//! regime, and swaps atomically.
//!
//! Run: `cargo run --release --example drift_recovery`

use rcf_rs::{AdwinDetector, DriftAwareForest, DriftRecoveryConfig, ForestBuilder};

const D: usize = 4;

fn main() {
    let builder = ForestBuilder::<D>::new()
        .num_trees(100)
        .sample_size(256)
        .seed(2026);
    let mut detector = DriftAwareForest::new(
        builder,
        DriftRecoveryConfig {
            shadow_warmup: 512,
            min_primary_age: 128,
        },
    )
    .unwrap();
    let mut adwin = AdwinDetector::default_bounded();

    // Regime 1: mean 0.1 on every dim.
    for _ in 0..1_000 {
        let p = [0.1, 0.1, 0.1, 0.1];
        detector.update(p).unwrap();
        let score = detector.score(&p).unwrap();
        let _ = adwin.update(f64::from(score));
    }
    let regime_1_score: f64 = detector.score(&[0.1; D]).unwrap().into();
    println!(
        "regime 1 baseline score = {regime_1_score:.3}, primary_age = {}",
        detector.primary_age()
    );

    // Regime shift: mean jumps to 5.0 (outlier under regime-1
    // baseline but becomes the new normal). The first step is
    // the "trigger moment" — in prod the caller routes an ADWIN /
    // PSI / CUSUM fire into `on_drift`; here we do it explicitly
    // at the shift for a reproducible demo.
    let p_shift = [5.0_f64, 5.0, 5.0, 5.0];
    detector.update(p_shift).unwrap();
    let first_score = detector.score(&p_shift).unwrap();
    let _ = adwin.update(f64::from(first_score));
    let spawned = detector.on_drift().unwrap();
    println!("regime shift detected (score {first_score:.3}), shadow spawned = {spawned}");

    for _ in 0..2_000 {
        let p = [5.0, 5.0, 5.0, 5.0];
        detector.update(p).unwrap();
        let score = detector.score(&p).unwrap();
        let _ = adwin.update(f64::from(score));
    }

    let final_score: f64 = detector.score(&[5.0; D]).unwrap().into();
    println!(
        "regime 2 baseline score = {final_score:.3}, swaps_total = {}, primary_age = {}",
        detector.swaps_total(),
        detector.primary_age()
    );
    println!(
        "is_recovering = {}, shadow_progress = {}",
        detector.is_recovering(),
        detector.shadow_progress()
    );

    if detector.swaps_total() > 0 {
        println!(
            "recovery succeeded — shadow became primary after {} observations",
            detector.primary_age()
        );
    } else {
        println!("no swap completed — raise shadow_warmup / check trigger");
    }
    // Note: both regime baselines score similarly because each
    // forest adapts to whatever distribution it ingests. The
    // swap's observable effect is fresh baseline membership, not
    // score magnitude; inspect forest internals or compare
    // attribution on an overlap probe to see the regime signature.
    let _ = regime_1_score; // avoid unused warning on the comparison.
    let _ = final_score;
}
