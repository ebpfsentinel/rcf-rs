//! Cold-start bootstrap end-to-end tests.
//!
//! Asserts:
//!
//! 1. Bootstrapping a thresholded forest skips the warmup hole —
//!    the very next live probe returns `ready = true`.
//! 2. A freshly bootstrapped detector fires on an outlier
//!    immediately, proving the adaptive threshold is hot.
//! 3. The pool's per-tenant bootstrap is isolated — bootstrapping
//!    tenant A does not warm tenant B.
//! 4. Non-finite points are skipped without sinking the bootstrap.
//! 5. Bootstrap interoperates with warm reload: bootstrap, persist,
//!    reload; state survives round-trip.

#![allow(clippy::cast_precision_loss, clippy::float_cmp)] // Bit-exact bincode roundtrip asserts + bounded casts.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{ForestBuilder, TenantForestPool, ThresholdedForestBuilder};

fn noisy(rng: &mut ChaCha8Rng) -> [f64; 4] {
    [
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
    ]
}

fn history(seed: u64, count: usize) -> Vec<[f64; 4]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count).map(|_| noisy(&mut rng)).collect()
}

#[test]
fn forest_bootstrap_warms_reservoir() {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(1)
        .build()
        .unwrap();
    assert_eq!(f.updates_seen(), 0);
    let r = f.bootstrap(history(1, 256)).unwrap();
    assert_eq!(r.points_ingested, 256);
    assert_eq!(r.points_skipped, 0);
    assert_eq!(f.updates_seen(), 256);
    // First probe should return a valid score — no EmptyForest.
    let score: f64 = f.score(&[0.05, 0.05, 0.05, 0.05]).unwrap().into();
    assert!(score >= 0.0);
}

#[test]
fn thresholded_bootstrap_eliminates_warmup_hole() {
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .min_observations(32)
        .min_threshold(0.1)
        .seed(2)
        .build()
        .unwrap();
    let r = d.bootstrap(history(2, 512)).unwrap();
    assert!(r.is_hot());
    assert!(r.final_observations >= 32);

    let verdict = d.score_only(&[0.05_f64, 0.05, 0.05, 0.05]).unwrap();
    assert!(verdict.ready(), "bootstrap must remove warmup gate");
}

#[test]
fn thresholded_bootstrap_fires_outlier_without_live_warmup() {
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(100)
        .sample_size(128)
        .z_factor(3.0)
        .min_observations(32)
        .min_threshold(0.1)
        .seed(3)
        .build()
        .unwrap();
    d.bootstrap(history(3, 512)).unwrap();
    // No live warmup — immediately probe an outlier.
    let outlier = d.process([50.0_f64, 50.0, 50.0, 50.0]).unwrap();
    assert!(outlier.ready());
    assert!(outlier.is_anomaly());
    assert!(outlier.grade() > 0.0);
}

#[test]
fn pool_bootstrap_is_per_tenant_isolated() {
    let mut pool: TenantForestPool<&'static str, 4> = TenantForestPool::new(4, || {
        ThresholdedForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(64)
            .min_observations(32)
            .min_threshold(0.1)
            .seed(42)
            .build()
    })
    .unwrap();

    // Bootstrap tenant A with 512 historical points. Leave tenant B
    // completely untrained.
    let r = pool.bootstrap(&"a", history(4, 512)).unwrap();
    assert!(r.is_hot());
    assert!(pool.contains(&"a"));
    assert!(!pool.contains(&"b"));

    // Tenant A is ready; a live B has never been touched.
    let v_a = pool.score_only(&"a", &[0.05_f64, 0.05, 0.05, 0.05]).unwrap();
    let v_b = pool.score_only(&"b", &[0.05_f64, 0.05, 0.05, 0.05]).unwrap();
    assert!(v_a.ready(), "tenant A should be hot");
    assert!(!v_b.ready(), "tenant B should still be warming up");
}

#[test]
fn bootstrap_skips_non_finite_rows() {
    let mut d = ThresholdedForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(16)
        .min_observations(4)
        .seed(5)
        .build()
        .unwrap();
    // Mix of valid + NaN + inf rows — the kind a TSDB query returns
    // when some series have gaps.
    let pts: Vec<[f64; 2]> = vec![
        [0.0, 0.0],
        [f64::NAN, 0.0],
        [1.0, 2.0],
        [0.0, f64::INFINITY],
        [3.0, 4.0],
        [5.0, f64::NEG_INFINITY],
    ];
    let r = d.bootstrap(pts).unwrap();
    assert_eq!(r.points_ingested, 3);
    assert_eq!(r.points_skipped, 3);
}

#[cfg(feature = "bincode")]
#[test]
fn bootstrap_roundtrips_through_persistence() {
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .min_observations(32)
        .min_threshold(0.1)
        .seed(7)
        .build()
        .unwrap();
    d.bootstrap(history(7, 256)).unwrap();
    let bytes = d.to_bytes().unwrap();
    let back = rcf_rs::ThresholdedForest::<4>::from_bytes(&bytes).unwrap();

    // Observation count and threshold survive byte-for-byte.
    assert_eq!(d.stats().observations(), back.stats().observations());
    assert_eq!(d.current_threshold(), back.current_threshold());

    let probe = [0.05_f64, 0.05, 0.05, 0.05];
    let a: f64 = d.score_only(&probe).unwrap().score().into();
    let b: f64 = back.score_only(&probe).unwrap().score().into();
    assert!(
        (a - b).abs() < f64::EPSILON,
        "score drift after persist+reload: {a} vs {b}",
    );
}
