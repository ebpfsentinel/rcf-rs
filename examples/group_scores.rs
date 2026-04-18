//! Score decomposition by named feature groups.
//!
//! Instead of a single scalar + a 14-entry per-dim attribution,
//! callers can declare semantic groups (e.g. `rate`, `payload`,
//! `cardinality`) and receive a compact per-group breakdown along
//! with the raw total. Great for SOC-facing triage: *"alert driven
//! by payload, not by rate"* is more actionable than a ranked
//! `DiVector`.
//!
//! Run with `cargo run --example group_scores`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{FeatureGroups, RcfError, ThresholdedForestBuilder};

fn main() -> Result<(), RcfError> {
    // 6-D feature vector grouped into 3 semantic buckets.
    let groups = FeatureGroups::builder()
        .add("rate", [0, 1])
        .add("payload", [2, 3])
        .add("cardinality", [4, 5])
        .build()?;

    let mut detector = ThresholdedForestBuilder::<6>::new()
        .num_trees(100)
        .sample_size(128)
        .z_factor(3.0)
        .min_observations(32)
        .min_threshold(0.1)
        .seed(42)
        .build()?;

    // Train on a tight noisy baseline.
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..512 {
        let p = [
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ];
        detector.process(p)?;
    }

    // Shock only the cardinality dims (indices 4, 5). A SOC analyst
    // should read "cardinality" as the dominant group immediately.
    let probe = [0.05_f64, 0.05, 0.05, 0.05, 50.0, 50.0];
    let decomposition = detector.group_scores(&probe, &groups)?;
    println!("group decomposition:");
    for (name, value) in decomposition.scores() {
        println!("  {name:<12} = {value:+.4}");
    }
    println!("  {name:<12} = {value:+.4}", name = "total", value = decomposition.total());
    println!(
        "  {name:<12} = {value:.2}",
        name = "coverage",
        value = decomposition.coverage()
    );
    if let Some((name, value)) = decomposition.top_group() {
        println!();
        println!("driver group: {name} (contribution {value:.4})");
    }

    Ok(())
}
