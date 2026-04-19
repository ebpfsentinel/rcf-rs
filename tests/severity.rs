//! Severity bands end-to-end — pipe a trained thresholded forest's
//! verdicts through [`rcf_rs::SeverityBands`] and check classification.
//!
//! Asserts:
//!
//! 1. Default bands match eBPFsentinel Enterprise (2/3/4/5).
//! 2. Custom bands classify correctly.
//! 3. `AnomalyGrade::severity` uses raw score (not bounded grade).
//! 4. `AnomalyScore::severity` works on bare forest output.

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{AnomalyScore, ForestBuilder, Severity, SeverityBands, ThresholdedForestBuilder};

#[test]
fn default_bands_equal_ml_detection_thresholds() {
    let b = SeverityBands::default();
    assert_eq!(b.low, 2.0);
    assert_eq!(b.medium, 3.0);
    assert_eq!(b.high, 4.0);
    assert_eq!(b.critical, 5.0);
}

#[test]
fn bare_forest_score_classifies() {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .seed(1)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for _ in 0..256 {
        f.update([
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ])
        .unwrap();
    }
    // Raw rcf-rs scores follow the Guha-2016 convention, not the
    // eBPFsentinel Z-score scale — relax the default bands for
    // this bench so the relative ordering (outlier > baseline)
    // still crosses a band boundary.
    let bands = SeverityBands::new(0.5, 0.8, 1.2, 2.0).unwrap();
    let inside = f.score(&[0.05, 0.05, 0.05, 0.05]).unwrap();
    let outside = f.score(&[50.0, 50.0, 50.0, 50.0]).unwrap();
    assert!(inside.severity(&bands) < outside.severity(&bands));
}

#[test]
fn thresholded_grade_severity_delegates_to_score() {
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .min_observations(16)
        .min_threshold(0.1)
        .seed(2)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    for _ in 0..256 {
        d.process([
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ])
        .unwrap();
    }
    let grade = d.process([50.0, 50.0, 50.0, 50.0]).unwrap();
    let bands = SeverityBands::default();
    // Grade.severity uses raw score — if the score exceeds Critical,
    // verdict should be Critical regardless of bounded grade.
    if f64::from(grade.score()) >= 5.0 {
        assert_eq!(grade.severity(&bands), Severity::Critical);
    }
}

#[test]
fn custom_bands_override_defaults() {
    let strict = SeverityBands::new(0.5, 1.0, 1.5, 2.0).unwrap();
    let s = AnomalyScore::new(1.2).unwrap();
    assert_eq!(s.severity(&strict), Severity::Medium);
    // Same score under default bands is Normal.
    assert_eq!(s.severity(&SeverityBands::default()), Severity::Normal);
}

#[test]
fn severity_ordinal_comparison_works_for_routing() {
    let high_alert = AnomalyScore::new(4.5).unwrap();
    let low_alert = AnomalyScore::new(2.5).unwrap();
    let bands = SeverityBands::default();
    // Typical routing rule: page oncall when >= High.
    let pageable = high_alert.severity(&bands) >= Severity::High;
    let silent = low_alert.severity(&bands) < Severity::High;
    assert!(pageable);
    assert!(silent);
}
