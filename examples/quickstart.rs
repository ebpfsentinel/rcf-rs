//! Minimal `rcf-rs` quickstart — build a forest, stream a few
//! points, and score one. Run with `cargo run --example quickstart`.

use rcf_rs::{AnomalyScore, ForestBuilder, RcfError};

fn main() -> Result<(), RcfError> {
    // AWS-default forest pinned to a deterministic seed.
    let mut forest = ForestBuilder::new(2)
        .num_trees(50)
        .sample_size(64)
        .seed(42)
        .build()?;

    // Stream a tight cluster around the origin.
    for i in 0..200 {
        let v = f64::from(i) * 0.001;
        forest.update(vec![v, v + 0.5])?;
    }

    let normal: AnomalyScore = forest.score(&[0.05, 0.55])?;
    let outlier: AnomalyScore = forest.score(&[10.0, 10.0])?;

    println!("normal point  → score = {normal}");
    println!("far outlier   → score = {outlier}");
    println!(
        "outlier / normal ratio = {:.2}",
        f64::from(outlier) / f64::from(normal).max(f64::EPSILON)
    );
    Ok(())
}
