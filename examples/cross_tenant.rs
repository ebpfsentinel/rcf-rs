//! Cross-tenant what-if — score the SAME point against every
//! resident tenant's detector. MSSP threat-intel pattern: a suspect
//! IOC lands in tenant A, we want to know which other tenants'
//! baselines also flag it.
//!
//! Run with `cargo run --example cross_tenant`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{RcfError, TenantForestPool, ThresholdedForestBuilder};

fn main() -> Result<(), RcfError> {
    let mut pool: TenantForestPool<&'static str, 2> = TenantForestPool::new(16, || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(64)
            .min_observations(16)
            .min_threshold(0.1)
            .seed(2026)
            .build()
    })?;
    let mut rng = ChaCha8Rng::seed_from_u64(2026);

    // Four tenants with distinct baselines:
    //   quiet-web : [0.0, 0.0] ish (low traffic)
    //   noisy-api : [5.0, 5.0] ish (moderate)
    //   bulk-ingest: [50.0, 50.0] ish (high)
    //   warming   : only 4 samples — will be filtered out
    for _ in 0..128 {
        pool.process(
            &"quiet-web",
            [rng.random::<f64>() * 0.1, rng.random::<f64>() * 0.1],
        )?;
        pool.process(
            &"noisy-api",
            [5.0 + rng.random::<f64>(), 5.0 + rng.random::<f64>()],
        )?;
        pool.process(
            &"bulk-ingest",
            [50.0 + rng.random::<f64>(), 50.0 + rng.random::<f64>()],
        )?;
    }
    for _ in 0..4 {
        pool.process(&"warming", [1.0, 1.0])?;
    }

    // SOC analyst sees alert on quiet-web for point [10, 10].
    // Question: does this IOC also trigger on other tenants?
    let probe = [10.0_f64, 10.0];
    let ranked = pool.score_across_tenants(&probe)?;
    println!("== cross-tenant scoring of probe {probe:?} ==");
    for (tenant, grade) in ranked {
        println!(
            "  {tenant:<12}  grade = {g:.4}  is_anomaly = {a}  threshold = {t:.3}",
            g = grade.grade(),
            a = grade.is_anomaly(),
            t = grade.threshold(),
        );
    }
    println!();
    println!("(warming tenant filtered out — not-ready verdicts excluded)");

    Ok(())
}
