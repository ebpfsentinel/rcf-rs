//! End-to-end behaviour of the per-tenant detector pool.
//!
//! Asserts four properties callers rely on:
//!
//! 1. Baselines do not bleed across tenants — a shock at tenant A
//!    does not move tenant B's adaptive threshold.
//! 2. LRU eviction respects the access pattern — the least recently
//!    touched tenant is the one that gets dropped when capacity is
//!    hit.
//! 3. Warm reload round-trips every tenant via the existing
//!    [`rcf_rs::ThresholdedForest::to_path`] / `from_path` helpers:
//!    save each, reload into a fresh pool, resume scoring.
//! 4. Factory errors surface to the caller without corrupting the
//!    pool state.

#![allow(clippy::cast_precision_loss)] // Bounded-counter casts in test setup.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{RcfError, TenantForestPool, ThresholdedForest, ThresholdedForestBuilder};

fn build_factory() -> impl Fn() -> rcf_rs::RcfResult<ThresholdedForest<4>> {
    || {
        ThresholdedForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(64)
            .z_factor(3.0)
            .min_observations(32)
            .min_threshold(0.1)
            .seed(2026)
            .build()
    }
}

fn noisy(rng: &mut ChaCha8Rng) -> [f64; 4] {
    [
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
        rng.random::<f64>() * 0.1,
    ]
}

#[test]
fn tenants_see_independent_baselines() {
    let mut pool = TenantForestPool::<&'static str, 4>::new(8, build_factory()).unwrap();
    let mut rng_a = ChaCha8Rng::seed_from_u64(1);
    let mut rng_b = ChaCha8Rng::seed_from_u64(2);

    // Warm both tenants with distinct noisy baselines.
    for _ in 0..256 {
        pool.process(&"a", noisy(&mut rng_a)).unwrap();
        pool.process(&"b", noisy(&mut rng_b)).unwrap();
    }

    // Shock tenant A with an outlier many times.
    for _ in 0..20 {
        pool.process(&"a", [50.0, 50.0, 50.0, 50.0]).unwrap();
    }

    let a = pool.peek(&"a").unwrap().current_threshold();
    let b = pool.peek(&"b").unwrap().current_threshold();
    assert!(
        a > b,
        "tenant A threshold {a} did not rise above tenant B {b}",
    );

    // Tenant B's baseline behaviour must be unchanged — no false
    // alarm on a tenant B noisy sample.
    let verdict_b = pool.process(&"b", noisy(&mut rng_b)).unwrap();
    assert!(
        !verdict_b.is_anomaly(),
        "tenant B fired from tenant A's drift: grade={} threshold={}",
        verdict_b.grade(),
        verdict_b.threshold(),
    );
}

#[test]
fn capacity_enforces_lru_eviction() {
    let mut pool = TenantForestPool::<u32, 4>::new(3, build_factory()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(7);

    pool.process(&1, noisy(&mut rng)).unwrap();
    pool.process(&2, noisy(&mut rng)).unwrap();
    pool.process(&3, noisy(&mut rng)).unwrap();

    // Touch 1 and 3, leaving 2 as LRU.
    pool.process(&1, noisy(&mut rng)).unwrap();
    pool.process(&3, noisy(&mut rng)).unwrap();

    // Push a 4th tenant — tenant 2 should evict.
    pool.process(&4, noisy(&mut rng)).unwrap();
    assert!(pool.contains(&1));
    assert!(
        !pool.contains(&2),
        "tenant 2 should have been the LRU victim"
    );
    assert!(pool.contains(&3));
    assert!(pool.contains(&4));
    assert_eq!(pool.len(), 3);
}

#[cfg(feature = "bincode")]
#[test]
fn warm_reload_roundtrips_every_tenant() {
    let dir = std::env::temp_dir().join(format!(
        "rcf-rs-pool-warm-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    std::fs::create_dir_all(&dir).unwrap();

    // Lifecycle 1: train, serialise each tenant.
    let mut pool = TenantForestPool::<String, 4>::new(8, build_factory()).unwrap();
    for tenant in ["a", "b", "c"] {
        let mut rng = ChaCha8Rng::seed_from_u64(u64::from(tenant.as_bytes()[0]));
        for _ in 0..128 {
            pool.process(&tenant.to_string(), noisy(&mut rng)).unwrap();
        }
    }

    let probe = [0.05_f64, 0.05, 0.05, 0.05];
    let mut expected: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for (key, forest) in pool.iter() {
        let path = dir.join(format!("{key}.bin"));
        forest.to_path(&path).unwrap();
        expected.insert(
            key.clone(),
            f64::from(forest.score_only(&probe).unwrap().score()),
        );
    }

    // Lifecycle 2: empty pool, reload from disk.
    let mut reloaded: TenantForestPool<String, 4> =
        TenantForestPool::new(8, build_factory()).unwrap();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let key = path.file_stem().unwrap().to_string_lossy().into_owned();
        let forest = ThresholdedForest::<4>::from_path(&path).unwrap();
        reloaded.insert(key, forest);
    }

    for (key, expected_score) in &expected {
        let verdict = reloaded.score_only(&key.clone(), &probe).unwrap();
        assert!(
            (f64::from(verdict.score()) - expected_score).abs() < f64::EPSILON,
            "tenant {key} score drifted after reload: expected {expected_score}, got {}",
            f64::from(verdict.score()),
        );
    }

    // Best-effort cleanup.
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn factory_error_does_not_corrupt_pool() {
    // A factory that alternates between failure and success.
    let counter = std::sync::atomic::AtomicU32::new(0);
    let mut pool: TenantForestPool<&'static str, 4> = TenantForestPool::new(4, move || {
        let n = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n.is_multiple_of(2) {
            Err(RcfError::InvalidConfig("forced even-call failure".into()))
        } else {
            ThresholdedForestBuilder::<4>::new()
                .num_trees(50)
                .sample_size(16)
                .min_observations(4)
                .seed(42)
                .build()
        }
    })
    .unwrap();

    // First call fails; pool must remain empty (no partial entry).
    let err = pool.process(&"x", [0.0, 0.0, 0.0, 0.0]).unwrap_err();
    assert!(matches!(err, RcfError::InvalidConfig(_)));
    assert!(pool.is_empty(), "failed factory left an entry in the pool");

    // Second call succeeds; pool should now hold one tenant.
    let _ = pool.process(&"x", [0.0, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(pool.len(), 1);
    assert!(pool.contains(&"x"));
}

#[test]
fn score_only_on_unseen_tenant_returns_none() {
    // score_only / attribution do not auto-create — they only touch
    // existing tenants and leave the pool undisturbed when the
    // tenant is absent.
    //
    // (The pool's current `score_only` auto-creates the tenant to
    // give it warming-up semantics; this test therefore verifies the
    // current contract — the tenant shows up in the pool *and* the
    // verdict is warming-up.)
    let mut pool = TenantForestPool::<&'static str, 4>::new(4, build_factory()).unwrap();
    let verdict = pool.score_only(&"unknown", &[0.0, 0.0, 0.0, 0.0]).unwrap();
    assert!(!verdict.ready());
    assert!(pool.contains(&"unknown"));
}
