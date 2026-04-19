//! Tenant similarity index — identify tenants whose TRCF score
//! distributions overlap. Useful for `SaaS` deployments with many
//! tenants: group similar ones for tiered alerting policies or
//! shared calibration.
//!
//! Run with `cargo run --example tenant_similarity`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{RcfError, TenantForestPool, ThresholdedForestBuilder};

fn main() -> Result<(), RcfError> {
    let mut pool: TenantForestPool<&'static str, 2> = TenantForestPool::new(16, || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(32)
            .min_observations(4)
            .seed(2026)
            .build()
    })?;
    let mut rng = ChaCha8Rng::seed_from_u64(2026);

    // Cluster 1: web front-ends — low packet rate, low entropy.
    for tenant in ["web-a", "web-b", "web-c"] {
        for _ in 0..128 {
            pool.process(
                &tenant,
                [rng.random::<f64>() * 0.1, rng.random::<f64>() * 0.1],
            )?;
        }
    }
    // Cluster 2: log shippers — high packet rate, high entropy.
    for tenant in ["log-x", "log-y"] {
        for _ in 0..128 {
            pool.process(
                &tenant,
                [
                    10.0 + rng.random::<f64>() * 0.5,
                    5.0 + rng.random::<f64>() * 0.5,
                ],
            )?;
        }
    }

    println!("== top 3 most similar to web-a ==");
    for (key, sim) in pool.most_similar(&"web-a", 3, 4) {
        println!("  {key:<6}  similarity = {sim:.4}");
    }

    println!();
    println!("== top 3 most similar to log-x ==");
    for (key, sim) in pool.most_similar(&"log-x", 3, 4) {
        println!("  {key:<6}  similarity = {sim:.4}");
    }

    println!();
    println!("== full pairwise similarity ==");
    let mut matrix = pool.similarity_matrix(4);
    matrix.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    for (a, b, sim) in matrix.iter().take(8) {
        println!("  ({a:<6}, {b:<6})  similarity = {sim:.4}");
    }

    Ok(())
}
