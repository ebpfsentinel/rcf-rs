//! Cross-tenant what-if scoring end-to-end.
//!
//! Asserts:
//!
//! 1. Same probe scored across multiple tenants returns a sorted
//!    desc list of `(key, grade)`.
//! 2. Warming-up tenants (below `min_observations`) are dropped.
//! 3. Tenant whose baseline matches the probe grades lowest;
//!    tenant whose baseline is far from the probe grades highest.

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{TenantForestPool, ThresholdedForestBuilder};

fn make_pool() -> TenantForestPool<&'static str, 2> {
    TenantForestPool::new(8, || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(32)
            .min_observations(16)
            .min_threshold(0.1)
            .seed(2026)
            .build()
    })
    .unwrap()
}

#[test]
fn ranks_and_filters_warming_up() {
    let mut pool = make_pool();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    // a, b, c — all trained past min_observations.
    for _ in 0..64 {
        pool.process(&"a", [rng.random::<f64>() * 0.1, rng.random::<f64>() * 0.1])
            .unwrap();
        pool.process(&"b", [10.0 + rng.random::<f64>(), 10.0 + rng.random::<f64>()])
            .unwrap();
        pool.process(&"c", [50.0 + rng.random::<f64>(), 50.0 + rng.random::<f64>()])
            .unwrap();
    }
    // d — only 4 observations, still warming-up.
    for _ in 0..4 {
        pool.process(&"d", [rng.random::<f64>() * 0.1, rng.random::<f64>() * 0.1])
            .unwrap();
    }

    let out = pool.score_across_tenants(&[0.05_f64, 0.05]).unwrap();
    // d must be absent.
    assert!(out.iter().all(|(k, _)| *k != "d"));
    // Sorted desc by grade.
    for pair in out.windows(2) {
        assert!(pair[0].1.grade() >= pair[1].1.grade());
    }
}

#[test]
fn probe_inside_a_baseline_grades_low_on_a() {
    let mut pool = make_pool();
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    for _ in 0..128 {
        pool.process(&"a", [rng.random::<f64>() * 0.1, rng.random::<f64>() * 0.1])
            .unwrap();
        pool.process(
            &"far",
            [50.0 + rng.random::<f64>(), 50.0 + rng.random::<f64>()],
        )
        .unwrap();
    }

    let out = pool.score_across_tenants(&[0.05_f64, 0.05]).unwrap();
    let a_grade = out.iter().find(|(k, _)| *k == "a").unwrap().1.grade();
    let far_grade = out.iter().find(|(k, _)| *k == "far").unwrap().1.grade();
    assert!(
        far_grade >= a_grade,
        "probe inside A baseline should grade <= far tenant: a={a_grade} far={far_grade}",
    );
}
