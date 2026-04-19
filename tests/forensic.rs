//! End-to-end behaviour of imputation-like forensic baseline.
//!
//! Asserts:
//!
//! 1. `forensic_baseline` returns `EmptyForest` on a cold forest.
//! 2. `expected` approximates the baseline mean and `stddev` the
//!    baseline spread for a tight Gaussian-ish training set.
//! 3. `delta` equals `observed − expected` per dim.
//! 4. `argmax_abs_zscore` picks the outlier dim on a single-dim
//!    outlier probe.
//! 5. `feature_scales` are inverted so `expected` lives in raw
//!    caller coordinates.
//! 6. TRCF delegate + pool delegate route through correctly.

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{ForestBuilder, RcfError, TenantForestPool, ThresholdedForestBuilder};

fn noisy(rng: &mut ChaCha8Rng, offset: f64) -> [f64; 4] {
    [
        offset + rng.random::<f64>() * 0.1,
        offset + rng.random::<f64>() * 0.1,
        offset + rng.random::<f64>() * 0.1,
        offset + rng.random::<f64>() * 0.1,
    ]
}

#[test]
fn empty_forest_errors() {
    let f = ForestBuilder::<4>::new().seed(1).build().unwrap();
    let err = f.forensic_baseline(&[0.0, 0.0, 0.0, 0.0]).unwrap_err();
    assert!(matches!(err, RcfError::EmptyForest));
}

#[test]
fn baseline_mean_tracks_training_mean() {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(2)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    for _ in 0..512 {
        f.update(noisy(&mut rng, 5.0)).unwrap();
    }
    let b = f.forensic_baseline(&[5.0, 5.0, 5.0, 5.0]).unwrap();
    for d in 0..4 {
        // Training points are 5.0 + uniform(0, 0.1) → mean near 5.05.
        assert!(
            (b.expected[d] - 5.05).abs() < 0.05,
            "expected[{d}]={} too far from 5.05",
            b.expected[d],
        );
        assert!(b.stddev[d] > 0.0);
    }
    assert!(b.live_points > 0);
}

#[test]
fn delta_is_observed_minus_expected() {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(32)
        .seed(3)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    for _ in 0..128 {
        f.update(noisy(&mut rng, 1.0)).unwrap();
    }
    let observed = [10.0_f64, 1.0, 1.0, 1.0];
    let b = f.forensic_baseline(&observed).unwrap();
    for (d, obs) in observed.iter().enumerate() {
        let expected_delta = obs - b.expected[d];
        assert!((b.delta[d] - expected_delta).abs() < 1e-12);
    }
}

#[test]
fn argmax_abs_zscore_flags_outlier_dim() {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(4)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    for _ in 0..512 {
        f.update(noisy(&mut rng, 0.0)).unwrap();
    }
    // Dim 2 is far outside the baseline, other dims are near baseline.
    let b = f.forensic_baseline(&[0.05, 0.05, 50.0, 0.05]).unwrap();
    assert_eq!(b.argmax_abs_zscore(), Some(2));
}

#[test]
fn feature_scales_invert_for_raw_coords() {
    // Build two forests with the same data but different scales,
    // and check `expected` comes back in raw caller coordinates.
    let mut base = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(5)
        .build()
        .unwrap();
    let mut scaled = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(64)
        .feature_scales([10.0, 1.0])
        .seed(5)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    for _ in 0..256 {
        let v = rng.random::<f64>();
        let p = [v, v + 0.5];
        base.update(p).unwrap();
        scaled.update(p).unwrap();
    }
    let base_b = base.forensic_baseline(&[0.5, 1.0]).unwrap();
    let scaled_b = scaled.forensic_baseline(&[0.5, 1.0]).unwrap();
    // Expected in raw coordinates must be close on both variants —
    // scaled mean * (1/10) ≈ base mean on dim 0.
    for d in 0..2 {
        let drift = (base_b.expected[d] - scaled_b.expected[d]).abs();
        assert!(
            drift < 0.2,
            "dim {d} expected drift {drift}: base={} scaled={}",
            base_b.expected[d],
            scaled_b.expected[d],
        );
    }
}

#[test]
fn thresholded_delegates() {
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(32)
        .min_observations(4)
        .seed(7)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    for _ in 0..64 {
        d.process(noisy(&mut rng, 0.0)).unwrap();
    }
    let b = d.forensic_baseline(&[10.0, 10.0, 10.0, 10.0]).unwrap();
    assert!(b.live_points > 0);
    assert!(b.argmax_abs_zscore().is_some());
}

#[test]
fn pool_returns_none_on_absent_tenant() {
    let mut pool: TenantForestPool<&'static str, 4> = TenantForestPool::new(4, || {
        ThresholdedForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(32)
            .seed(11)
            .build()
    })
    .unwrap();
    let out = pool
        .forensic_baseline(&"unknown", &[0.0, 0.0, 0.0, 0.0])
        .unwrap();
    assert!(out.is_none());
    assert!(!pool.contains(&"unknown"));
}

#[test]
fn pool_returns_some_for_existing_tenant() {
    let mut pool: TenantForestPool<&'static str, 4> = TenantForestPool::new(4, || {
        ThresholdedForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(32)
            .seed(13)
            .build()
    })
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(13);
    for _ in 0..128 {
        pool.process(&"a", noisy(&mut rng, 0.0)).unwrap();
    }
    let out = pool
        .forensic_baseline(&"a", &[10.0, 10.0, 10.0, 10.0])
        .unwrap();
    assert!(out.is_some());
}
