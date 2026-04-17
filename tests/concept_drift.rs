//! Concept-drift integration test for [`rcf_rs::RandomCutForest`].
//!
//! Story RCF.9 AC #3: a stream that switches from cluster A to
//! cluster B should initially flag cluster B as anomalous, then
//! settle back to baseline once the reservoir refreshes.

#![allow(clippy::cast_precision_loss)]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rcf_rs::ForestBuilder;

#[test]
fn distribution_shift_flagged_then_normalised() {
    const PER_PHASE: usize = 2000;
    let mut rng = ChaCha8Rng::seed_from_u64(2026);

    let mut forest = ForestBuilder::new(2)
        .num_trees(50)
        .sample_size(64)
        .seed(7)
        .build()
        .expect("AWS-conformant config");

    // Phase 1: cluster A near the origin (uniform [-0.05, 0.05]).
    for _ in 0..PER_PHASE {
        let p = vec![
            <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng) * 0.1 - 0.05,
            <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng) * 0.1 - 0.05,
        ];
        forest.update(p).expect("update succeeds");
    }

    // Score a fresh cluster-B sample BEFORE we feed any cluster-B
    // points — cluster B should look anomalous to the forest now.
    let probe_b = vec![5.0, 5.0];
    let initial_b: f64 = forest.score(&probe_b).unwrap().into();

    // Phase 2: cluster B around (5, 5). Stream enough points that
    // the reservoir refreshes (PER_PHASE > num_trees × sample_size).
    for _ in 0..PER_PHASE {
        let p = vec![
            5.0 + <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng) * 0.1 - 0.05,
            5.0 + <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng) * 0.1 - 0.05,
        ];
        forest.update(p).expect("update succeeds");
    }

    let final_b: f64 = forest.score(&probe_b).unwrap().into();

    // After absorbing cluster B the same probe should score
    // strictly lower than it did before the shift.
    assert!(
        final_b < initial_b,
        "drift not absorbed: initial_b={initial_b}, final_b={final_b}"
    );

    // Cluster A is now the historical anomaly — verify it scores
    // higher than cluster B at the new equilibrium.
    let probe_a = vec![0.0, 0.0];
    let final_a: f64 = forest.score(&probe_a).unwrap().into();
    assert!(
        final_a > final_b,
        "post-drift: A should look more anomalous than B (a={final_a}, b={final_b})"
    );
}
