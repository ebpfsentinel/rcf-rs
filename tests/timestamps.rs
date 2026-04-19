//! End-to-end behaviour of the timestamp + retention API.
//!
//! Asserts:
//!
//! 1. `update_at` records the timestamp and `point_timestamp` reads
//!    it back.
//! 2. `delete_before(cutoff)` retracts every point older than the
//!    cutoff and returns the deleted count.
//! 3. Points inserted via classic `update` (no timestamp) are not
//!    touched by `delete_before`.
//! 4. Reservoir eviction removes the side-map entry automatically.
//! 5. `oldest_timestamp` / `newest_timestamp` reflect the live set.
//! 6. TRCF `process_at` / `delete_before` delegate correctly.
//! 7. Pool retention is per-tenant (`delete_before` on tenant A
//!    does not affect tenant B).

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{ForestBuilder, TenantForestPool, ThresholdedForestBuilder};

#[test]
fn update_at_records_timestamp() {
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(16)
        .seed(1)
        .build()
        .unwrap();
    let idx = f.update_indexed_at([0.1, 0.2], 1_000).unwrap();
    assert_eq!(f.point_timestamp(idx), Some(1_000));
    assert_eq!(f.tracked_timestamps(), 1);
    assert_eq!(f.oldest_timestamp(), Some(1_000));
    assert_eq!(f.newest_timestamp(), Some(1_000));
}

#[test]
fn delete_before_retracts_older_points() {
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(32)
        .seed(2)
        .build()
        .unwrap();
    let _old_a = f.update_indexed_at([0.0, 0.0], 100).unwrap();
    let _old_b = f.update_indexed_at([0.1, 0.1], 200).unwrap();
    let _new_a = f.update_indexed_at([0.2, 0.2], 1_000).unwrap();
    let _new_b = f.update_indexed_at([0.3, 0.3], 2_000).unwrap();

    let removed = f.delete_before(500).unwrap();
    assert_eq!(removed, 2, "two points with ts < 500 should be retracted");
    assert_eq!(f.tracked_timestamps(), 2);
    assert_eq!(f.oldest_timestamp(), Some(1_000));
}

#[test]
fn delete_before_ignores_untimestamped_points() {
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(16)
        .seed(3)
        .build()
        .unwrap();
    f.update([0.0, 0.0]).unwrap(); // no timestamp
    f.update([0.1, 0.1]).unwrap();
    f.update_indexed_at([0.5, 0.5], 100).unwrap();
    f.update_indexed_at([0.6, 0.6], 200).unwrap();
    let before_untimestamped = f.updates_seen();

    let removed = f.delete_before(1_000).unwrap();
    assert_eq!(removed, 2, "only timestamped points should be deleted");
    // Classic-update points are still counted in updates_seen.
    assert_eq!(f.updates_seen(), before_untimestamped);
    assert_eq!(f.tracked_timestamps(), 0);
}

#[test]
fn reservoir_eviction_cleans_up_timestamps() {
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(4)
        .seed(4)
        .build()
        .unwrap();
    // Insert many more points than capacity so older ones are
    // evicted naturally; track timestamps on all of them.
    for i in 0_u64..128 {
        let v = i as f64 * 0.01;
        f.update_at([v, v + 0.5], i).unwrap();
    }
    // Most older timestamps should have been evicted naturally.
    assert!(
        f.tracked_timestamps() < 128,
        "reservoir eviction should prune the timestamp map: {} still tracked",
        f.tracked_timestamps(),
    );
}

#[test]
fn thresholded_process_at_and_delete_before() {
    let mut d = ThresholdedForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(32)
        .min_observations(4)
        .seed(5)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    for ts in 100_u64..200 {
        let p = [rng.random::<f64>() * 0.1, rng.random::<f64>() * 0.1];
        d.process_at(p, ts).unwrap();
    }
    let before = d.forest().tracked_timestamps();
    assert!(before > 0);
    let removed = d.delete_before(150).unwrap();
    assert!(removed > 0);
    assert!(d.forest().oldest_timestamp().unwrap_or(0) >= 150);
}

#[test]
fn pool_delete_before_is_per_tenant() {
    let mut pool: TenantForestPool<&'static str, 2> = TenantForestPool::new(4, || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .min_observations(4)
            .seed(7)
            .build()
    })
    .unwrap();
    for ts in 0_u64..8 {
        pool.process_at(&"a", [0.1 + ts as f64 * 0.01, 0.1], ts)
            .unwrap();
    }
    for ts in 0_u64..8 {
        pool.process_at(&"b", [0.1 + ts as f64 * 0.01, 0.2], ts)
            .unwrap();
    }
    let removed_a = pool.delete_before(&"a", 4).unwrap();
    assert!(removed_a > 0);
    // Tenant B untouched.
    let b_tracked = pool.peek(&"b").unwrap().forest().tracked_timestamps();
    assert_eq!(b_tracked, 8);
}

#[test]
fn pool_delete_before_noop_on_absent_tenant() {
    let mut pool: TenantForestPool<&'static str, 2> = TenantForestPool::new(4, || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .seed(11)
            .build()
    })
    .unwrap();
    assert_eq!(pool.delete_before(&"unknown", 1_000).unwrap(), 0);
    assert!(!pool.contains(&"unknown"));
}
