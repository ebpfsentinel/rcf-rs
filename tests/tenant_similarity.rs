//! End-to-end behaviour of the tenant similarity index.
//!
//! Asserts:
//!
//! 1. Pairwise similarity matrix includes N*(N-1)/2 entries.
//! 2. Same-shape baselines yield high similarity, divergent
//!    baselines low.
//! 3. `most_similar` sorts descending by similarity.
//! 4. `min_observations` gate filters undertrained tenants.

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{TenantForestPool, ThresholdedForestBuilder};

fn make_pool() -> TenantForestPool<&'static str, 2> {
    TenantForestPool::new(8, || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(32)
            .min_observations(4)
            .seed(42)
            .build()
    })
    .unwrap()
}

#[test]
fn matrix_has_correct_pair_count() {
    let mut pool = make_pool();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for tenant in ["a", "b", "c", "d"] {
        for _ in 0..64 {
            let p = [rng.random::<f64>() * 0.1, rng.random::<f64>() * 0.1];
            pool.process(&tenant, p).unwrap();
        }
    }
    let pairs = pool.similarity_matrix(4);
    assert_eq!(pairs.len(), 4 * 3 / 2, "N=4 → 6 pairs");
}

#[test]
fn similar_baselines_rank_higher() {
    let mut pool = make_pool();
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    // A + C: same baseline distribution.
    for _ in 0..128 {
        let v = rng.random::<f64>() * 0.1;
        pool.process(&"a", [v, v + 0.5]).unwrap();
        pool.process(&"c", [v, v + 0.5]).unwrap();
    }
    // B: very different baseline.
    for _ in 0..128 {
        let v = rng.random::<f64>() * 5.0 + 20.0;
        pool.process(&"b", [v, v + 0.5]).unwrap();
    }
    let ranked = pool.most_similar(&"a", 2, 4);
    assert_eq!(ranked.len(), 2);
    // c should beat b.
    let c_sim = ranked.iter().find(|(k, _)| *k == "c").map(|(_, s)| *s);
    let b_sim = ranked.iter().find(|(k, _)| *k == "b").map(|(_, s)| *s);
    assert!(
        c_sim.unwrap() > b_sim.unwrap(),
        "c should be more similar to a than b is",
    );
}

#[test]
fn most_similar_sorted_descending() {
    let mut pool = make_pool();
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    for tenant in ["a", "b", "c", "d"] {
        for _ in 0..64 {
            let p = [rng.random::<f64>() * 0.1, rng.random::<f64>() * 0.1];
            pool.process(&tenant, p).unwrap();
        }
    }
    let ranked = pool.most_similar(&"a", 10, 4);
    for pair in ranked.windows(2) {
        assert!(pair[0].1 >= pair[1].1, "not sorted desc");
    }
}

#[test]
fn min_observations_gate_filters_undertrained() {
    let mut pool = make_pool();
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    for _ in 0..128 {
        let p = [rng.random::<f64>() * 0.1, rng.random::<f64>() * 0.1];
        pool.process(&"a", p).unwrap();
    }
    for _ in 0..3 {
        let p = [rng.random::<f64>() * 0.1, rng.random::<f64>() * 0.1];
        pool.process(&"b", p).unwrap();
    }
    // min_observations=64 filters b (only 3 stats observations).
    let pairs = pool.similarity_matrix(64);
    assert!(pairs.is_empty());
}
