//! Compare the latency and fidelity of `score` vs
//! `score_early_term` on a warmed forest. Runs 1000 probes through
//! each and reports the savings.
//!
//! Note: the `score` baseline uses rayon parallelism (default
//! `parallel` feature on) while `score_early_term` walks trees
//! sequentially — the early-term path's wall-clock only beats the
//! parallel path when the early-stop rate is high *and* the caller
//! cannot afford to saturate cores (batch jobs, no-std, tight
//! single-thread budget). For a more even comparison add
//! `--no-default-features --features std,postcard,serde` to
//! disable rayon.
//!
//! Run with `cargo run --example early_term --release`.

use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{EarlyTermConfig, ForestBuilder, RcfError};

fn main() -> Result<(), RcfError> {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(256)
        .seed(2026)
        .build()?;
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    for _ in 0..2048 {
        let p = [
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ];
        f.update(p)?;
    }

    // Use a looser threshold so early-stop kicks in on typical
    // baseline traffic — 5 % stderr/mean is tight enough that many
    // forests walk the full ensemble anyway. 15 % lets early-stop
    // trigger on clearly-in-distribution points.
    let cfg = EarlyTermConfig {
        min_trees: 16,
        confidence_threshold: 0.15,
    };

    // Generate a batch of probes: 80% baseline-like, 20% outliers.
    let probes: Vec<[f64; 4]> = (0_u32..1000)
        .map(|i| {
            if i.is_multiple_of(5) {
                [50.0_f64, 50.0, 50.0, 50.0]
            } else {
                [
                    rng.random::<f64>() * 0.1,
                    rng.random::<f64>() * 0.1,
                    rng.random::<f64>() * 0.1,
                    rng.random::<f64>() * 0.1,
                ]
            }
        })
        .collect();

    // Full-ensemble timing.
    let t0 = Instant::now();
    let mut full_acc = 0.0_f64;
    for p in &probes {
        full_acc += f64::from(f.score(p)?);
    }
    let full_elapsed = t0.elapsed();

    // Early-term timing.
    let t0 = Instant::now();
    let mut et_acc = 0.0_f64;
    let mut et_trees_total = 0_usize;
    let mut early_stops = 0_usize;
    for p in &probes {
        let out = f.score_early_term(p, cfg)?;
        et_acc += f64::from(out.score);
        et_trees_total += out.trees_evaluated;
        if out.early_stopped {
            early_stops += 1;
        }
    }
    let et_elapsed = t0.elapsed();

    println!("== 1000 probes, 100-tree forest ==");
    println!(
        "  full-ensemble  : {full_elapsed:?}  mean_score={:.4}",
        full_acc / 1000.0
    );
    println!(
        "  early-term     : {et_elapsed:?}  mean_score={:.4}",
        et_acc / 1000.0
    );
    println!();
    println!("  early-stop rate : {early_stops}/1000");
    #[allow(clippy::cast_precision_loss)]
    let avg_trees = et_trees_total as f64 / 1000.0;
    println!("  avg trees walked: {avg_trees:.1}/100");
    let speedup = full_elapsed.as_secs_f64() / et_elapsed.as_secs_f64();
    println!("  speedup         : {speedup:.2}×");

    Ok(())
}
