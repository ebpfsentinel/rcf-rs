//! End-to-end behaviour of the feature-group score decomposition.
//!
//! Asserts:
//!
//! 1. A partitioning set of groups explains 100 % of the raw score.
//! 2. An outlier on a specific dim is attributed to the group that
//!    owns that dim — the `top_group` answer matches the shock.
//! 3. Dimension-mismatch is caught at score time with the
//!    documented error.
//! 4. The pool-level entry point preserves tenant isolation.

#![allow(clippy::cast_precision_loss)] // Bounded-counter casts in test setup.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{
    FeatureGroups, ForestBuilder, RcfError, TenantForestPool, ThresholdedForestBuilder,
};

fn noisy(rng: &mut ChaCha8Rng) -> [f64; 4] {
    [
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
    ]
}

fn rate_vs_payload_groups() -> FeatureGroups {
    FeatureGroups::builder()
        .add("rate", [0, 1])
        .add("payload", [2, 3])
        .build()
        .unwrap()
}

#[test]
fn partitioning_groups_fully_explain_raw_score() {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(1)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for _ in 0..256 {
        f.update(noisy(&mut rng)).unwrap();
    }
    let groups = rate_vs_payload_groups();
    let decomposition = f.group_scores(&[50.0, 50.0, 50.0, 50.0], &groups).unwrap();
    assert_eq!(decomposition.len(), 2);
    // Full coverage — explained == total up to f64 precision.
    assert!(
        (decomposition.coverage() - 1.0).abs() < 1e-12,
        "coverage {coverage} should be 1.0 for a partitioning set",
        coverage = decomposition.coverage(),
    );
}

#[test]
fn outlier_on_payload_dims_attributed_to_payload_group() {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(128)
        .seed(2)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    for _ in 0..512 {
        f.update(noisy(&mut rng)).unwrap();
    }
    // Outlier only on dims 2 and 3 — the "payload" group.
    let outlier = [0.05_f64, 0.05, 50.0, 50.0];
    let groups = rate_vs_payload_groups();
    let d = f.group_scores(&outlier, &groups).unwrap();
    let (top_name, top_value) = d.top_group().unwrap();
    assert_eq!(top_name, "payload", "top group should own the shocked dims");
    assert!(top_value > 0.0);
}

#[test]
fn group_scores_rejects_index_out_of_range() {
    let f = ForestBuilder::<4>::new().seed(3).build().unwrap();
    let bad_groups = FeatureGroups::builder()
        .add("rate", [0, 9]) // 9 out of range for D=4
        .build()
        .unwrap();
    let err = f.group_scores(&[0.0, 0.0, 0.0, 0.0], &bad_groups).unwrap_err();
    assert!(matches!(err, RcfError::OutOfBounds { .. }));
}

#[test]
fn thresholded_group_scores_match_bare_forest() {
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .min_observations(32)
        .seed(5)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    for _ in 0..256 {
        d.process(noisy(&mut rng)).unwrap();
    }
    let probe = [0.3_f64, 0.1, 0.1, 0.1];
    let groups = rate_vs_payload_groups();
    let via_trcf = d.group_scores(&probe, &groups).unwrap();
    let via_forest = d.forest().group_scores(&probe, &groups).unwrap();
    assert_eq!(via_trcf, via_forest, "TRCF decomposition should equal forest decomposition");
}

#[test]
fn pool_group_scores_are_per_tenant_isolated() {
    let mut pool: TenantForestPool<&'static str, 4> = TenantForestPool::new(4, || {
        ThresholdedForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(64)
            .min_observations(32)
            .seed(7)
            .build()
    })
    .unwrap();
    let groups = rate_vs_payload_groups();

    // Train tenant A on a baseline where rate dims are always zero,
    // payload dims are zero.
    let mut rng_a = ChaCha8Rng::seed_from_u64(71);
    for _ in 0..512 {
        let p = [
            0.0_f64,
            0.0,
            rng_a.random::<f64>() * 0.1,
            rng_a.random::<f64>() * 0.1,
        ];
        pool.process(&"a", p).unwrap();
    }

    // Train tenant B on a baseline where rate dims are around 10.0.
    for _ in 0..512 {
        let p = [
            10.0_f64 + rng_a.random::<f64>() * 0.1,
            10.0 + rng_a.random::<f64>() * 0.1,
            rng_a.random::<f64>() * 0.1,
            rng_a.random::<f64>() * 0.1,
        ];
        pool.process(&"b", p).unwrap();
    }

    // Query the same probe through both tenants. For tenant A a
    // rate=10 probe is anomalous on the rate group; for tenant B it's
    // routine. The rate-group contribution on tenant A should be
    // strictly larger than on tenant B.
    let probe = [10.0_f64, 10.0, 0.05, 0.05];
    let a = pool.group_scores(&"a", &probe, &groups).unwrap();
    let b = pool.group_scores(&"b", &probe, &groups).unwrap();
    let rate_a = a.scores().iter().find(|(n, _)| n == "rate").unwrap().1;
    let rate_b = b.scores().iter().find(|(n, _)| n == "rate").unwrap().1;
    assert!(
        rate_a > rate_b,
        "rate_a {rate_a} should be > rate_b {rate_b} — pool must isolate baselines",
    );
}
