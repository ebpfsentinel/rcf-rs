//! Per-tenant detector pool: each tenant owns its own adaptive
//! threshold, drift, and reservoir. A shock on tenant `alice` does
//! not move the baseline of tenant `bob`.
//!
//! Run with `cargo run --example tenant_pool`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{RcfError, TenantForestPool, ThresholdedForestBuilder};

fn main() -> Result<(), RcfError> {
    let mut pool = TenantForestPool::<String, 4>::new(8, || {
        ThresholdedForestBuilder::<4>::new()
            .num_trees(100)
            .sample_size(128)
            .z_factor(3.0)
            .min_observations(32)
            .min_threshold(0.1)
            .initial_accept_fraction(0.125)
            .seed(42)
            .build()
    })?;

    let tenants = ["alice", "bob", "carol"];
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    for tenant in &tenants {
        for _ in 0..256 {
            let p = [
                rng.random::<f64>() * 0.1,
                rng.random::<f64>() * 0.1,
                rng.random::<f64>() * 0.1,
                rng.random::<f64>() * 0.1,
            ];
            pool.process(&(*tenant).to_string(), p)?;
        }
    }
    println!("tenants resident: {:?}", pool.tenants());
    for tenant in &tenants {
        let threshold = pool
            .peek(&(*tenant).to_string())
            .unwrap()
            .current_threshold();
        println!("  {tenant}: threshold = {threshold:.4}");
    }

    // Shock one tenant, leave the others alone.
    println!();
    println!("shocking alice with 20 outliers...");
    for _ in 0..20 {
        pool.process(&"alice".to_string(), [50.0, 50.0, 50.0, 50.0])?;
    }
    for tenant in &tenants {
        let threshold = pool
            .peek(&(*tenant).to_string())
            .unwrap()
            .current_threshold();
        println!("  {tenant}: threshold = {threshold:.4}");
    }

    // New tenant: auto-instantiated, warming-up verdict until it has
    // seen enough data.
    println!();
    let verdict_cold = pool.process(&"dave".to_string(), [0.05, 0.05, 0.05, 0.05])?;
    println!(
        "dave (brand new): ready = {}, is_anomaly = {}, grade = {:.4}",
        verdict_cold.ready(),
        verdict_cold.is_anomaly(),
        verdict_cold.grade(),
    );

    Ok(())
}
