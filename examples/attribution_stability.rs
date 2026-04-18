//! Attribution dispersion across trees — pick the driver dim whose
//! signal the forest is *confident* about, not just the one with the
//! biggest mean contribution.
//!
//! Two scenarios:
//!
//! 1. A clear single-dim outlier (dim 2). Trees should unanimously
//!    attribute the score to dim 2 → high confidence on dim 2 →
//!    both `argmax_mean` and `argmax_weighted` return `2`.
//! 2. A marginal probe inside the baseline. Per-dim contributions
//!    are close to zero and noisy; `confidence` is still well-defined
//!    and the driver pick is less certain.
//!
//! Run with `cargo run --example attribution_stability`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{RcfError, ThresholdedForestBuilder};

fn main() -> Result<(), RcfError> {
    let mut detector = ThresholdedForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(128)
        .min_observations(32)
        .min_threshold(0.1)
        .seed(42)
        .build()?;

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..512 {
        let p = [
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ];
        detector.process(p)?;
    }

    println!("== scenario 1: clear single-dim outlier on dim 2 ==");
    let outlier = [0.05_f64, 0.05, 50.0, 0.05];
    let stability = detector.attribution_stability(&outlier)?;
    for d in 0..stability.dim() {
        println!(
            "  dim {d}: mean = {mean:+.4}, stddev = {sd:.4}, confidence = {c:.3}",
            mean = stability.mean().per_dim_total(d),
            sd = stability.stddev()[d],
            c = stability.confidence(d),
        );
    }
    println!(
        "  argmax_mean     = {:?}",
        stability.argmax_mean(),
    );
    println!(
        "  argmax_weighted = {:?}",
        stability.argmax_weighted(),
    );

    println!();
    println!("== scenario 2: marginal probe inside baseline ==");
    let marginal = [0.05_f64, 0.05, 0.05, 0.05];
    let stability = detector.attribution_stability(&marginal)?;
    for d in 0..stability.dim() {
        println!(
            "  dim {d}: mean = {mean:+.4}, stddev = {sd:.4}, confidence = {c:.3}",
            mean = stability.mean().per_dim_total(d),
            sd = stability.stddev()[d],
            c = stability.confidence(d),
        );
    }
    println!(
        "  argmax_mean     = {:?}",
        stability.argmax_mean(),
    );
    println!(
        "  argmax_weighted = {:?}",
        stability.argmax_weighted(),
    );

    Ok(())
}
