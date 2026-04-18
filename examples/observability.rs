//! Operator-facing observability: wire a `MetricsSink` into every
//! detector layer and plot a `ScoreHistogram` at the end.
//!
//! Demonstrates the production pattern: build once, attach one
//! shared sink, every public op feeds counters / gauges /
//! histograms into it. The sink here is the built-in `TestSink`
//! (in-memory recorder) so the demo is self-contained; a real
//! deployment would wire to a Prometheus registry instead.
//!
//! Run with `cargo run --example observability`.

use std::sync::Arc;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{
    CusumConfig, MetaDriftDetector, RcfError, ScoreHistogram, ThresholdedForestBuilder,
    metrics::{TestSink, names},
};

fn main() -> Result<(), RcfError> {
    let sink = Arc::new(TestSink::new());

    let mut detector = ThresholdedForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(64)
        .min_observations(32)
        .min_threshold(0.1)
        .seed(2026)
        .build()?
        .with_metrics_sink(sink.clone());

    let mut drift = MetaDriftDetector::new(CusumConfig {
        allowance_k: 0.5,
        threshold_h: 6.0,
        min_observations: 32,
        decay: 0.05,
    })?
    .with_metrics_sink(sink.clone());

    let mut score_hist = ScoreHistogram::with_range(0.0, 5.0)?;
    let mut rng = ChaCha8Rng::seed_from_u64(2026);

    // Simulate 2048 baseline windows + a late outlier burst.
    for _ in 0..2048 {
        let p = [rng.random::<f64>() * 0.1, rng.random::<f64>() * 0.1];
        let verdict = detector.process(p)?;
        drift.observe(f64::from(verdict.score()));
        score_hist.record(f64::from(verdict.score()));
    }
    for _ in 0..10 {
        let verdict = detector.process([50.0, 50.0])?;
        drift.observe(f64::from(verdict.score()));
        score_hist.record(f64::from(verdict.score()));
    }

    println!("== sink counters ==");
    println!(
        "  process_total           = {}",
        sink.counter(names::PROCESS_TOTAL)
    );
    println!(
        "  anomalies_fired_total   = {}",
        sink.counter(names::ANOMALIES_FIRED_TOTAL)
    );
    println!(
        "  drift_fires_total       = {}",
        sink.counter(names::DRIFT_FIRES_TOTAL)
    );

    println!();
    println!("== sink gauges ==");
    println!(
        "  threshold_current       = {:.4}",
        sink.gauge(names::THRESHOLD_CURRENT).unwrap_or(0.0)
    );

    println!();
    println!("== score histogram (5 bin snapshot, full has 32) ==");
    let edges = score_hist.bin_edges();
    // Print every 4th bin to keep output compact.
    for (i, (count, (lo, hi))) in score_hist.bins().iter().zip(edges.iter()).enumerate() {
        if i.is_multiple_of(4) {
            let bar = "#".repeat(usize::try_from(*count / 16).unwrap_or(0).min(40));
            println!("  [{lo:>4.2}, {hi:>4.2})  {count:>5}  {bar}");
        }
    }
    println!(
        "  p50 = {:.4}  p95 = {:.4}  p99 = {:.4}",
        score_hist.percentile(0.5).unwrap_or(0.0),
        score_hist.percentile(0.95).unwrap_or(0.0),
        score_hist.percentile(0.99).unwrap_or(0.0),
    );

    Ok(())
}
