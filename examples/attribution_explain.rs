//! Print the top-3 contributing dimensions of the anomaly score for
//! a queried point. Demonstrates [`rcf_rs::RandomCutForest::attribution`]
//! and the [`rcf_rs::DiVector::argmax`] helper.
//!
//! Run with `cargo run --example attribution_explain`.

use rcf_rs::{ForestBuilder, RcfError};

const DIM: usize = 16;
const ANOM_DIM: usize = 5;

fn main() -> Result<(), RcfError> {
    let mut forest = ForestBuilder::new(DIM)
        .num_trees(100)
        .sample_size(128)
        .seed(2026)
        .build()?;

    // Train on points uniformly distributed in [0, 1)^DIM.
    let mut rng = simple_lcg(0x00C0_FFEE);
    for _ in 0..512 {
        let p: Vec<f64> = (0..DIM).map(|_| rng()).collect();
        forest.update(p)?;
    }

    // Query: anomalous on dim ANOM_DIM only.
    let mut query = vec![0.5; DIM];
    query[ANOM_DIM] = 50.0;

    let score = forest.score(&query)?;
    let di = forest.attribution(&query)?;

    let mut totals: Vec<(usize, f64)> = (0..DIM).map(|d| (d, di.high()[d] + di.low()[d])).collect();
    totals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("query        = {query:?}");
    println!("score        = {score}");
    println!("argmax dim   = {:?}", di.argmax());
    println!("top-3 dims (dim, contribution):");
    for (d, v) in totals.iter().take(3) {
        println!("  dim {d:2} → {v:.6}");
    }

    Ok(())
}

/// Tiny linear-congruential RNG so the example has zero non-rcf-rs
/// dependencies — produces uniform `f64` in `[0, 1)`.
fn simple_lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        // Pull 21 high bits of state into a `u32` and normalise — keeps
        // the cast lossless and the distribution close to uniform.
        let frac = u32::try_from(state >> 32).unwrap_or(u32::MAX) >> 11;
        f64::from(frac) / f64::from(1_u32 << 21)
    }
}
