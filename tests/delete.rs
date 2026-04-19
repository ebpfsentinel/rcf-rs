//! End-to-end behaviour of the retraction API on
//! [`rcf_rs::RandomCutForest`], [`rcf_rs::ThresholdedForest`] and
//! [`rcf_rs::TenantForestPool`].
//!
//! Asserts:
//!
//! 1. `update_indexed` returns a fresh, distinct `point_idx` per
//!    call until eviction reclaims slots.
//! 2. `delete(idx)` removes the point from every tree that held it
//!    and reports whether any tree did.
//! 3. `delete(unknown)` is a no-op returning `false`.
//! 4. `delete_by_value(&point)` removes every resident match,
//!    returning the count.
//! 5. Score + attribution continue to work after deletion.
//! 6. The pool's `delete` / `delete_by_value` do not auto-create
//!    tenants — retraction for an unknown tenant is a no-op.

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use rcf_rs::{ForestBuilder, TenantForestPool, ThresholdedForestBuilder};

#[test]
fn update_indexed_returns_fresh_indices() {
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(1)
        .build()
        .unwrap();
    let idx_a = f.update_indexed([0.0, 0.0]).unwrap();
    let idx_b = f.update_indexed([1.0, 1.0]).unwrap();
    assert_ne!(idx_a, idx_b);
}

#[test]
fn delete_removes_point_from_trees() {
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(2)
        .build()
        .unwrap();
    // Insert a distinctive point we can track.
    let idx = f.update_indexed([7.0, 11.0]).unwrap();
    for i in 0_u32..32 {
        let v = f64::from(i) * 0.01;
        f.update([v, v]).unwrap();
    }
    let was_present = f.delete(idx).unwrap();
    assert!(
        was_present,
        "freshly inserted point should be in at least one tree"
    );
    // After delete, the sampler should not list this idx anywhere.
    for (_, sampler, _) in f.trees() {
        assert!(!sampler.contains(idx), "idx should be gone from every tree");
    }
    // Scoring still works.
    let score: f64 = f.score(&[0.5, 0.5]).unwrap().into();
    assert!(score >= 0.0);
}

#[test]
fn delete_unknown_idx_is_noop() {
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(16)
        .seed(3)
        .build()
        .unwrap();
    for i in 0_u32..8 {
        let v = f64::from(i) * 0.01;
        f.update([v, v]).unwrap();
    }
    let removed = f.delete(9_999_999).unwrap();
    assert!(!removed);
}

#[test]
fn delete_by_value_removes_every_matching_entry() {
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(5)
        .build()
        .unwrap();
    let target = [42.0_f64, 42.0];
    // Insert the same target several times. With sample_size=64 and
    // 50 trees, the reservoirs should absorb most of them.
    for _ in 0..10 {
        f.update(target).unwrap();
    }
    for i in 0_u32..32 {
        let v = f64::from(i) * 0.01;
        f.update([v, v]).unwrap();
    }
    let removed = f.delete_by_value(&target).unwrap();
    assert!(removed > 0, "expected to remove at least one target entry");
    // A second sweep finds none.
    let second = f.delete_by_value(&target).unwrap();
    assert_eq!(second, 0);
}

#[test]
fn thresholded_delete_propagates_to_forest() {
    let mut d = ThresholdedForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(32)
        .min_observations(4)
        .seed(7)
        .build()
        .unwrap();
    let (idx, _verdict) = d.process_indexed([0.1, 0.2]).unwrap();
    for i in 0_u32..32 {
        let v = f64::from(i) * 0.01;
        d.process([v, v]).unwrap();
    }
    let removed = d.delete(idx).unwrap();
    assert!(removed);
}

#[test]
fn pool_delete_does_not_auto_create_tenant() {
    let mut pool: TenantForestPool<&'static str, 2> = TenantForestPool::new(4, || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .min_observations(4)
            .seed(11)
            .build()
    })
    .unwrap();
    assert!(!pool.delete(&"unknown", 42).unwrap());
    assert!(!pool.contains(&"unknown"));
    assert_eq!(pool.delete_by_value(&"unknown", &[0.0, 0.0]).unwrap(), 0);
    assert!(!pool.contains(&"unknown"));
}

#[test]
fn pool_delete_by_value_removes_per_tenant_matches() {
    let mut pool: TenantForestPool<&'static str, 2> = TenantForestPool::new(4, || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(32)
            .min_observations(4)
            .seed(13)
            .build()
    })
    .unwrap();
    let target = [9.0_f64, 9.0];
    for _ in 0..8 {
        pool.process(&"a", target).unwrap();
    }
    for _ in 0..8 {
        pool.process(&"b", [1.0, 1.0]).unwrap();
    }
    let removed_a = pool.delete_by_value(&"a", &target).unwrap();
    let removed_b = pool.delete_by_value(&"b", &target).unwrap();
    assert!(removed_a > 0);
    assert_eq!(removed_b, 0, "tenant B never saw the target value");
}
