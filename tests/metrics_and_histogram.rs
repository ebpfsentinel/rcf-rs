//! End-to-end tests for the observability surface:
//!
//! - [`rcf_rs::MetricsSink`] receives the documented events from
//!   `RandomCutForest`, `ThresholdedForest`, `TenantForestPool`,
//!   `MetaDriftDetector`.
//! - [`rcf_rs::ScoreHistogram`] bins a streamed score distribution
//!   into well-formed buckets.

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use std::sync::Arc;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{
    CusumConfig, ForestBuilder, MetaDriftDetector, ScoreHistogram, TenantForestPool,
    ThresholdedForestBuilder,
    metrics::{TestSink, names},
};

fn noisy(rng: &mut ChaCha8Rng) -> [f64; 2] {
    [rng.random::<f64>() * 0.1, rng.random::<f64>() * 0.1]
}

#[test]
fn forest_sink_records_updates_scores_deletes() {
    let sink = Arc::new(TestSink::new());
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(16)
        .seed(1)
        .build()
        .unwrap()
        .with_metrics_sink(sink.clone());

    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for _ in 0..20 {
        f.update(noisy(&mut rng)).unwrap();
    }
    for _ in 0..5 {
        let _ = f.score(&[0.05, 0.05]).unwrap();
    }
    let idx = f.update_indexed([9.0, 9.0]).unwrap();
    assert!(f.delete(idx).unwrap());

    assert_eq!(sink.counter(names::UPDATES_TOTAL), 21);
    assert_eq!(sink.counter(names::DELETES_TOTAL), 1);
    assert_eq!(sink.histogram(names::SCORE_OBSERVATION).len(), 5);
    assert_eq!(sink.gauge(names::FOREST_TREES), Some(50.0));
}

#[test]
fn thresholded_sink_records_process_grade_threshold() {
    let sink = Arc::new(TestSink::new());
    let mut d = ThresholdedForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(16)
        .min_observations(4)
        .min_threshold(0.1)
        .seed(2)
        .build()
        .unwrap()
        .with_metrics_sink(sink.clone());

    let mut rng = ChaCha8Rng::seed_from_u64(2);
    for _ in 0..32 {
        d.process(noisy(&mut rng)).unwrap();
    }
    let _ = d.process([50.0, 50.0]).unwrap();

    let processed = sink.counter(names::PROCESS_TOTAL);
    assert_eq!(processed, 33);
    assert!(sink.counter(names::ANOMALIES_FIRED_TOTAL) >= 1);
    assert_eq!(sink.histogram(names::GRADE_OBSERVATION).len(), 33);
    assert!(
        sink.gauge(names::THRESHOLD_CURRENT).is_some(),
        "threshold gauge should have been set at least once",
    );
}

#[test]
fn pool_sink_records_evictions_and_resident_gauge() {
    let sink = Arc::new(TestSink::new());
    let mut pool: TenantForestPool<u32, 2> = TenantForestPool::new(2, || {
        ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .seed(3)
            .build()
    })
    .unwrap()
    .with_metrics_sink(sink.clone());

    let mut rng = ChaCha8Rng::seed_from_u64(3);
    pool.process(&1, noisy(&mut rng)).unwrap();
    pool.process(&2, noisy(&mut rng)).unwrap();
    pool.process(&3, noisy(&mut rng)).unwrap(); // Triggers LRU eviction
    assert_eq!(sink.counter(names::TENANT_EVICTIONS_TOTAL), 1);
    assert_eq!(sink.gauge(names::TENANTS_RESIDENT), Some(2.0));
}

#[test]
fn drift_sink_records_cusum_and_fires() {
    let sink = Arc::new(TestSink::new());
    let mut drift = MetaDriftDetector::new(CusumConfig {
        allowance_k: 0.5,
        threshold_h: 3.0,
        min_observations: 8,
        decay: 0.1,
    })
    .unwrap()
    .with_metrics_sink(sink.clone());

    for _ in 0..64 {
        drift.observe(1.0);
    }
    for _ in 0..200 {
        let v = drift.observe(5.0);
        if v.drift.is_some() {
            break;
        }
    }
    let s_high_obs = sink.histogram(names::DRIFT_S_HIGH);
    assert!(!s_high_obs.is_empty());
    assert!(sink.counter(names::DRIFT_FIRES_TOTAL) >= 1);
}

#[test]
fn score_histogram_bins_streamed_grades() {
    let mut h = ScoreHistogram::with_range(0.0, 1.0).unwrap();
    // Bimodal: 80% near 0.1 (normal), 20% near 0.9 (anomaly).
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    for _ in 0..800 {
        h.record(rng.random::<f64>() * 0.2);
    }
    for _ in 0..200 {
        h.record(0.8 + rng.random::<f64>() * 0.2);
    }
    assert_eq!(h.total(), 1000);
    assert_eq!(h.underflow(), 0);
    // The lower half should hold the 800 normal observations, the
    // upper half the 200 anomaly observations.
    let mid = h.bins().len() / 2;
    let lower: u64 = h.bins()[..mid].iter().sum();
    let upper: u64 = h.bins()[mid..].iter().sum();
    assert!(lower > upper);
    let p50 = h.percentile(0.5).unwrap();
    assert!(
        p50 < 0.3,
        "median of bimodal lower mode should be < 0.3, got {p50}"
    );
}

#[test]
fn histogram_merge_produces_union_distribution() {
    let mut a = ScoreHistogram::with_range(0.0, 1.0).unwrap();
    let mut b = ScoreHistogram::with_range(0.0, 1.0).unwrap();
    for _ in 0..100 {
        a.record(0.1);
    }
    for _ in 0..200 {
        b.record(0.9);
    }
    a.merge(&b).unwrap();
    assert_eq!(a.total(), 300);
}
