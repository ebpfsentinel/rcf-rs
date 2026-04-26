#![allow(clippy::unwrap_used, clippy::panic)]
//! End-to-end behaviour of bulk scoring APIs.
//!
//! Asserts:
//!
//! 1. `score_many` output order and values match point-by-point
//!    `score` calls.
//! 2. `attribution_many` output dim is `D` per entry and matches
//!    per-point `attribution` output.
//! 3. `score_many_early_term` reports `EarlyTermScore` entries.
//! 4. Empty batch returns empty vec.
//! 5. Single bad point (non-finite) aborts the whole batch.
//! 6. TRCF `score_only_many` matches per-point `score_only`.
//! 7. Pool `score_only_many` returns `None` on absent tenant.

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use anomstream_core::{
    EarlyTermConfig, ForestBuilder, RcfError, TenantForestPool, ThresholdedForestBuilder,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn trained() -> anomstream_core::RandomCutForest<4> {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(2026)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    for _ in 0..256 {
        f.update([
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ])
        .unwrap();
    }
    f
}

#[test]
fn score_many_with_callback_matches_score_many() {
    let f = trained();
    let probes: Vec<[f64; 4]> = (0..32)
        .map(|i| [f64::from(i) * 0.01, 0.05, 0.05, 0.05])
        .collect();
    let mut from_callback: Vec<f64> = vec![0.0; probes.len()];
    f.score_many_with(&probes, |i, s| {
        from_callback[i] = f64::from(s);
    })
    .unwrap();
    for (i, p) in probes.iter().enumerate() {
        let plain = f64::from(f.score(p).unwrap());
        assert_eq!(plain, from_callback[i], "mismatch at idx {i}");
    }
}

#[test]
fn score_many_with_aborts_on_non_finite() {
    let f = trained();
    let probes = vec![[0.0, 0.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0, 0.0]];
    let mut count = 0_usize;
    let err = f
        .score_many_with(&probes, |_, _| {
            count += 1;
        })
        .unwrap_err();
    assert!(matches!(err, RcfError::NaNValue));
    // First probe succeeded before the NaN aborted the batch.
    assert_eq!(count, 1);
}

#[test]
fn score_many_with_empty_batch() {
    let f = trained();
    let mut called = false;
    f.score_many_with(&[], |_, _| called = true).unwrap();
    assert!(!called);
}

#[test]
fn score_many_matches_individual_calls() {
    let f = trained();
    let probes: Vec<[f64; 4]> = (0..32)
        .map(|i| [f64::from(i) * 0.01, 0.05, 0.05, 0.05])
        .collect();

    let bulk = f.score_many(&probes).unwrap();
    assert_eq!(bulk.len(), probes.len());

    for (i, p) in probes.iter().enumerate() {
        let individual: f64 = f.score(p).unwrap().into();
        let from_bulk: f64 = bulk[i].into();
        // Under rayon, parallel scoring may reorder floating
        // accumulations by a ULP — match the tolerance used by
        // `attribution_many_output_matches_per_point`.
        let delta = (individual - from_bulk).abs();
        assert!(
            delta < 1e-10,
            "mismatch at idx {i}: single={individual} bulk={from_bulk} delta={delta}",
        );
    }
}

#[test]
fn attribution_many_output_matches_per_point() {
    let f = trained();
    let probes = vec![[0.05_f64, 0.05, 0.05, 0.05], [50.0, 50.0, 50.0, 50.0]];
    let bulk = f.attribution_many(&probes).unwrap();
    assert_eq!(bulk.len(), probes.len());
    for (i, di) in bulk.iter().enumerate() {
        assert_eq!(di.dim(), 4);
        let single = f.attribution(&probes[i]).unwrap();
        // Under rayon, parallel attribution may reorder floating
        // accumulations by a ULP — 1e-10 is orders of magnitude
        // below any observable signal.
        for d in 0..4 {
            let delta = (di.per_dim_total(d) - single.per_dim_total(d)).abs();
            assert!(delta < 1e-10, "dim {d} drift: {delta}");
        }
    }
}

#[test]
fn score_many_early_term_returns_early_term_scores() {
    let f = trained();
    let probes = vec![
        [0.05_f64, 0.05, 0.05, 0.05],
        [50.0, 50.0, 50.0, 50.0],
        [0.1, 0.2, 0.3, 0.4],
    ];
    let cfg = EarlyTermConfig {
        min_trees: 8,
        confidence_threshold: 0.2,
    };
    let out = f.score_many_early_term(&probes, cfg).unwrap();
    assert_eq!(out.len(), 3);
    for r in &out {
        assert!(r.trees_evaluated >= cfg.min_trees || r.trees_evaluated == r.trees_available);
    }
}

#[test]
fn score_many_empty_batch() {
    let f = trained();
    let out = f.score_many(&[]).unwrap();
    assert!(out.is_empty());
}

#[test]
fn score_many_aborts_on_bad_point() {
    let f = trained();
    let probes = vec![[0.05_f64, 0.05, 0.05, 0.05], [f64::NAN, 0.0, 0.0, 0.0]];
    assert!(matches!(
        f.score_many(&probes).unwrap_err(),
        RcfError::NaNValue
    ));
}

#[test]
fn thresholded_score_only_many_matches_per_point() {
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .min_observations(4)
        .seed(11)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    for _ in 0..128 {
        d.process([
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ])
        .unwrap();
    }
    let probes = vec![[0.05_f64, 0.05, 0.05, 0.05], [50.0, 50.0, 50.0, 50.0]];
    let bulk = d.score_only_many(&probes).unwrap();
    assert_eq!(bulk.len(), 2);
    for (i, p) in probes.iter().enumerate() {
        let single = d.score_only(p).unwrap();
        assert_eq!(f64::from(bulk[i].score()), f64::from(single.score()));
        assert_eq!(bulk[i].ready(), single.ready());
    }
}

#[test]
fn pool_score_only_many_absent_tenant_returns_none() {
    let mut pool: TenantForestPool<&'static str, 4> = TenantForestPool::new(4, || {
        ThresholdedForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(32)
            .seed(13)
            .build()
    })
    .unwrap();
    let probes = vec![[0.0_f64, 0.0, 0.0, 0.0]];
    let out = pool.score_only_many(&"unknown", &probes).unwrap();
    assert!(out.is_none());
    assert!(!pool.contains(&"unknown"));
}

#[test]
fn pool_score_only_many_existing_tenant() {
    let mut pool: TenantForestPool<&'static str, 4> = TenantForestPool::new(4, || {
        ThresholdedForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(32)
            .seed(17)
            .build()
    })
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    for _ in 0..64 {
        pool.process(
            &"a",
            [
                rng.random::<f64>() * 0.1,
                rng.random::<f64>() * 0.1,
                rng.random::<f64>() * 0.1,
                rng.random::<f64>() * 0.1,
            ],
        )
        .unwrap();
    }
    let probes = vec![[0.05_f64, 0.05, 0.05, 0.05], [50.0, 50.0, 50.0, 50.0]];
    let out = pool.score_only_many(&"a", &probes).unwrap();
    assert_eq!(out.map(|v| v.len()), Some(2));
}
