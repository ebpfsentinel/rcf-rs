//! Cold-start bootstrap from a simulated TSDB query.
//!
//! Mimics the restart path of a long-running streaming agent: pull
//! the last few hours of feature vectors from the upstream time-series
//! store, hand them to `ThresholdedForest::bootstrap`, and the
//! detector is immediately ready for live traffic — no warmup gap.
//!
//! Run with `cargo run --example bootstrap`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{RcfError, ThresholdedForestBuilder};

/// Stand-in for a TSDB adapter: produce `rows` historical 4-D
/// feature vectors drawn from a tight noisy baseline. A real caller
/// would replace this with a query against `Prometheus` / `Loki` /
/// `InfluxDB` / parquet files.
fn tsdb_history(seed: u64, rows: usize) -> Vec<[f64; 4]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..rows)
        .map(|_| {
            [
                rng.random::<f64>() * 0.1,
                rng.random::<f64>() * 0.1,
                rng.random::<f64>() * 0.1,
                rng.random::<f64>() * 0.1,
            ]
        })
        .collect()
}

fn main() -> Result<(), RcfError> {
    let mut detector = ThresholdedForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(128)
        .z_factor(3.0)
        .min_observations(32)
        .min_threshold(0.1)
        .initial_accept_fraction(0.125)
        .seed(2026)
        .build()?;

    println!("before bootstrap:");
    println!("  observations = {}", detector.stats().observations());
    println!("  threshold    = {:.4}", detector.current_threshold());

    // Pull 512 rows of history from the TSDB.
    let history = tsdb_history(17, 512);
    let report = detector.bootstrap(history)?;

    println!();
    println!("bootstrap report:");
    println!("  points_ingested    = {}", report.points_ingested);
    println!("  points_skipped     = {}", report.points_skipped);
    println!("  final_observations = {}", report.final_observations);
    println!("  final_threshold    = {:.4}", report.final_threshold);
    println!("  is_hot             = {}", report.is_hot());

    // First "live" probe lands on a hot detector.
    let verdict = detector.score_only(&[0.05_f64, 0.05, 0.05, 0.05])?;
    println!();
    println!("first live probe (noisy baseline point):");
    println!("  ready      = {}", verdict.ready());
    println!("  is_anomaly = {}", verdict.is_anomaly());
    println!("  grade      = {:.4}", verdict.grade());

    let outlier = detector.process([50.0_f64, 50.0, 50.0, 50.0])?;
    println!();
    println!("outlier probe (never warmed up live!):");
    println!("  ready      = {}", outlier.ready());
    println!("  is_anomaly = {}", outlier.is_anomaly());
    println!("  grade      = {:.4}", outlier.grade());

    Ok(())
}
