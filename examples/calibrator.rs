//! Fit a Platt sigmoid on a labelled calibration set, then map
//! fresh forest scores to `P(anomaly | score) ∈ [0, 1]` — the
//! audit-friendly output for SOC2 / NIS2 paperwork.
//!
//! Run with `cargo run --example calibrator`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{ForestBuilder, PlattCalibrator, PlattFitConfig, RcfError};

fn main() -> Result<(), RcfError> {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(256)
        .seed(2026)
        .build()?;
    let mut rng = ChaCha8Rng::seed_from_u64(2026);

    // Warm the forest on baseline traffic without collecting labels.
    for _ in 0..1024 {
        let p = [
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ];
        f.update(p)?;
    }

    // Collect labelled calibration scores — score BEFORE updating so
    // the forest is probed as if the point were unseen, matching the
    // inference-time semantics.
    let mut calibration: Vec<(f64, bool)> = Vec::new();
    for _ in 0..512 {
        let p = [
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ];
        let s = f64::from(f.score(&p)?);
        calibration.push((s, false));
    }
    for _ in 0..64 {
        let p = [30.0_f64, 30.0, 30.0, 30.0];
        let s = f64::from(f.score(&p)?);
        calibration.push((s, true));
    }

    let calibrator = PlattCalibrator::fit(&calibration, PlattFitConfig::default()).unwrap();
    println!("== fitted Platt calibrator ==");
    println!(
        "  a = {:.4}, b = {:.4}, iters = {}, converged = {}",
        calibrator.a(),
        calibrator.b(),
        calibrator.iters(),
        calibrator.converged(),
    );

    let probes = vec![
        ("baseline", [0.05_f64, 0.05, 0.05, 0.05]),
        ("mild", [0.5_f64, 0.5, 0.5, 0.5]),
        ("outlier", [30.0_f64, 30.0, 30.0, 30.0]),
    ];
    println!();
    println!("== probes ==");
    for (label, p) in &probes {
        let score = f.score(p)?;
        let raw = f64::from(score);
        let prob = calibrator.calibrate(raw);
        println!("  {label:<10}  raw_score = {raw:>7.4}  P(anomaly) = {prob:.3}",);
    }

    Ok(())
}
