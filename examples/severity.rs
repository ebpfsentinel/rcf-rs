//! Severity bands — classify raw anomaly scores into
//! Normal/Low/Medium/High/Critical ordinal labels. Defaults match
//! eBPFsentinel Enterprise ml-detection (2/3/4/5).
//!
//! Run with `cargo run --example severity`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{RcfError, Severity, SeverityBands, ThresholdedForestBuilder};

fn main() -> Result<(), RcfError> {
    let mut detector = ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .min_observations(16)
        .min_threshold(0.1)
        .seed(2026)
        .build()?;
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    for _ in 0..512 {
        let p = [
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ];
        detector.process(p)?;
    }

    // Raw rcf-rs scores use a different scale than eBPFsentinel
    // Z-scores — tune the bands for this forest.
    let bands = SeverityBands::new(0.5, 1.0, 1.5, 2.5)?;

    let probes = vec![
        ("baseline", [0.05_f64, 0.05, 0.05, 0.05]),
        ("mild shift", [0.5_f64, 0.5, 0.5, 0.5]),
        ("clear outlier", [10.0_f64, 10.0, 10.0, 10.0]),
        ("far outlier", [50.0_f64, 50.0, 50.0, 50.0]),
    ];

    println!(
        "bands: low={:.1} medium={:.1} high={:.1} critical={:.1}",
        bands.low, bands.medium, bands.high, bands.critical
    );
    println!();
    for (name, p) in probes {
        let grade = detector.process(p)?;
        let raw: f64 = grade.score().into();
        let sev = grade.severity(&bands);
        let should_page = sev >= Severity::High;
        println!(
            "  {name:<16} raw = {raw:>6.3}  severity = {sev:<8}  page_oncall = {should_page}",
        );
    }

    Ok(())
}
