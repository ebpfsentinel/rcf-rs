//! Forensic baseline — given an anomalous point, answer
//! "what would dim d have looked like if this window were normal?".
//!
//! Run with `cargo run --example forensic`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{RcfError, ThresholdedForestBuilder};

fn main() -> Result<(), RcfError> {
    let mut detector = ThresholdedForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(128)
        .seed(2026)
        .build()?;
    let mut rng = ChaCha8Rng::seed_from_u64(2026);

    // Train on a tight cluster where dim 0 ~ 50_000 (packet rate),
    // dim 1 ~ 0.5 (protocol ratio), dim 2 ~ 4.0 (entropy),
    // dim 3 ~ 1_000 (cardinality).
    for _ in 0..1024 {
        let p = [
            50_000.0_f64 + rng.random::<f64>() * 2_000.0,
            0.5 + rng.random::<f64>() * 0.1,
            4.0 + rng.random::<f64>() * 0.5,
            1_000.0 + rng.random::<f64>() * 100.0,
        ];
        detector.process(p)?;
    }

    // Anomalous probe: cardinality spike.
    let observed = [50_500.0_f64, 0.55, 4.2, 25_000.0];
    let baseline = detector.forensic_baseline(&observed)?;

    println!("== forensic baseline ==");
    println!("  live_points = {}", baseline.live_points);
    let labels = ["packet_rate", "proto_ratio", "entropy   ", "cardinality"];
    for (d, label) in labels.iter().enumerate() {
        println!(
            "  {label}  observed = {obs:>10.2}  expected = {exp:>10.2}  stddev = {sd:>8.2}  z = {z:+.2}",
            obs = baseline.observed[d],
            exp = baseline.expected[d],
            sd = baseline.stddev[d],
            z = baseline.zscore[d],
        );
    }
    if let Some(idx) = baseline.argmax_abs_zscore() {
        println!();
        println!("driver dim: {label} (|z| = {z:.2})",
            label = labels[idx],
            z = baseline.zscore[idx].abs(),
        );
    }

    Ok(())
}
