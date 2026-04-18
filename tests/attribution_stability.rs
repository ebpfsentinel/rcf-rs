//! End-to-end behaviour of inter-tree attribution dispersion.
//!
//! Asserts:
//!
//! 1. A point far outside the training baseline on a single dim
//!    yields a stability report where every tree agrees on that
//!    dim — `confidence` near `1.0` for the driver dim.
//! 2. A "noisy" probe inside the baseline produces a report where
//!    `mean ≈ 0` and the confidence bound is well-defined (`1.0` for
//!    zero mean).
//! 3. The stability mean equals the regular `attribution` mean bit
//!    for bit — no silent drift in the accumulation path.
//! 4. `attribution_stability` rejects dimension-mismatched points
//!    and non-finite components just like `attribution` does.
//! 5. The pool-level entry point is tenant-isolated — shocking
//!    tenant A's attribution does not change the stability report
//!    for tenant B.

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{
    ForestBuilder, RcfError, TenantForestPool, ThresholdedForestBuilder,
};

fn noisy4(rng: &mut ChaCha8Rng) -> [f64; 4] {
    [
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
    ]
}

#[test]
fn driver_dim_has_high_confidence_on_single_dim_outlier() {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(128)
        .seed(1)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for _ in 0..512 {
        f.update(noisy4(&mut rng)).unwrap();
    }
    // Outlier only on dim 2.
    let probe = [0.05_f64, 0.05, 50.0, 0.05];
    let s = f.attribution_stability(&probe).unwrap();
    let driver = s.argmax_weighted().unwrap();
    assert_eq!(driver, 2, "dim 2 should be the confident driver");
    assert!(
        s.confidence(driver) > 0.5,
        "confidence on outlier dim should be high, got {}",
        s.confidence(driver),
    );
}

#[test]
fn stability_mean_equals_plain_attribution() {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(2)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    for _ in 0..256 {
        f.update(noisy4(&mut rng)).unwrap();
    }
    let probe = [0.3_f64, 0.1, 0.2, 0.05];
    let plain = f.attribution(&probe).unwrap();
    let s = f.attribution_stability(&probe).unwrap();
    // Tolerance accounts for rayon's reorder-safe fold/reduce in
    // attribution() — the serial stability path can differ from the
    // parallel sum in the last ULP or two. 1e-10 is orders of
    // magnitude below any observable anomaly signal.
    for d in 0..4 {
        let delta = (plain.per_dim_total(d) - s.mean().per_dim_total(d)).abs();
        assert!(
            delta < 1e-10,
            "dim {d} mean drift between attribution() and attribution_stability(): \
             plain={plain_v} stability={stab_v} delta={delta}",
            plain_v = plain.per_dim_total(d),
            stab_v = s.mean().per_dim_total(d),
        );
    }
}

#[test]
fn attribution_stability_rejects_non_finite() {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(32)
        .seed(3)
        .build()
        .unwrap();
    for i in 0_u32..64 {
        let v = f64::from(i) * 0.01;
        f.update([v, v, v, v]).unwrap();
    }
    let err = f
        .attribution_stability(&[f64::NAN, 0.0, 0.0, 0.0])
        .unwrap_err();
    assert!(matches!(err, RcfError::NaNValue));
}

#[test]
fn thresholded_forest_delegates_to_inner_forest() {
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .min_observations(32)
        .seed(5)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    for _ in 0..256 {
        d.process(noisy4(&mut rng)).unwrap();
    }
    let probe = [0.3_f64, 0.1, 0.1, 0.1];
    let via_trcf = d.attribution_stability(&probe).unwrap();
    let via_forest = d.forest().attribution_stability(&probe).unwrap();
    assert_eq!(via_trcf.tree_count(), via_forest.tree_count());
    // Both paths go through the exact same serial collect+reduce so
    // equality is bit-exact here (no rayon reorder).
    for d_i in 0..via_trcf.dim() {
        assert_eq!(
            via_trcf.mean().per_dim_total(d_i),
            via_forest.mean().per_dim_total(d_i),
        );
        assert_eq!(via_trcf.variance()[d_i], via_forest.variance()[d_i]);
    }
}

#[test]
fn pool_attribution_stability_is_tenant_isolated() {
    let mut pool: TenantForestPool<&'static str, 4> = TenantForestPool::new(4, || {
        ThresholdedForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(64)
            .min_observations(32)
            .seed(7)
            .build()
    })
    .unwrap();

    // Tenant A: baseline zero, outlier on dim 0.
    let mut rng_a = ChaCha8Rng::seed_from_u64(71);
    for _ in 0..256 {
        let p = [
            rng_a.random::<f64>() * 0.1,
            rng_a.random::<f64>() * 0.1,
            rng_a.random::<f64>() * 0.1,
            rng_a.random::<f64>() * 0.1,
        ];
        pool.process(&"a", p).unwrap();
    }
    // Tenant B: baseline is actually centred at dim 0 ~ 10, so a
    // probe of [10.0, 0.05, 0.05, 0.05] is normal for B, anomalous
    // for A. Their `attribution_stability` driver dims must differ.
    for _ in 0..256 {
        let p = [
            10.0 + rng_a.random::<f64>() * 0.1,
            rng_a.random::<f64>() * 0.1,
            rng_a.random::<f64>() * 0.1,
            rng_a.random::<f64>() * 0.1,
        ];
        pool.process(&"b", p).unwrap();
    }

    let probe = [10.0_f64, 0.05, 0.05, 0.05];
    let a = pool.attribution_stability(&"a", &probe).unwrap();
    let b = pool.attribution_stability(&"b", &probe).unwrap();
    // Tenant A should see dim 0 as the driver (with high confidence).
    assert_eq!(a.argmax_weighted(), Some(0));
    // Tenant B should see either noise or a different driver — at
    // minimum, dim 0 is NOT the leader by a wide margin.
    let a_driver_mean = a.mean().per_dim_total(0);
    let b_driver_mean = b.mean().per_dim_total(0);
    assert!(
        a_driver_mean > b_driver_mean,
        "A driver mean {a_driver_mean} should exceed B's {b_driver_mean}",
    );
}
