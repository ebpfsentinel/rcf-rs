//! End-to-end persistence roundtrip tests for [`rcf_rs::RandomCutForest`].
//!
//! Covers RCF.8 acceptance criteria #2, #4, #5, #8: bit-exact score
//! roundtrip for binary + JSON, version-mismatch rejection,
//! malformed-payload rejection, and the four matrix scenarios
//! (empty, trained, time-decay, custom seed).
//!
//! Only compiled with `--features bincode,serde_json`.

#![cfg(all(feature = "bincode", feature = "serde_json"))]
#![allow(clippy::cast_precision_loss, clippy::float_cmp)] // Roundtrip asserts bit-exact f64 equality.

use rcf_rs::{ForestBuilder, RandomCutForest, RcfError};

fn trained(
    seed: u64,
    dim: usize,
    sample_size: usize,
    updates: usize,
    decay: f64,
) -> RandomCutForest {
    let mut f = ForestBuilder::new(dim)
        .num_trees(50)
        .sample_size(sample_size)
        .time_decay(decay)
        .seed(seed)
        .build()
        .unwrap();
    for i in 0..updates {
        let v = i as f64 * 0.01;
        let p: Vec<f64> = (0..dim).map(|d| v + d as f64 * 0.1).collect();
        f.update(p).unwrap();
    }
    f
}

#[test]
fn binary_roundtrip_empty_forest() {
    let f = ForestBuilder::new(4)
        .num_trees(50)
        .sample_size(8)
        .seed(1)
        .build()
        .unwrap();
    let bytes = f.to_bytes().unwrap();
    let back = RandomCutForest::from_bytes(&bytes).unwrap();
    assert_eq!(back.dimension(), 4);
    assert_eq!(back.num_trees(), 50);
    assert_eq!(back.updates_seen(), 0);
}

#[test]
fn binary_roundtrip_trained_forest() {
    let f = trained(2026, 4, 32, 500, 0.0);
    let bytes = f.to_bytes().unwrap();
    let back = RandomCutForest::from_bytes(&bytes).unwrap();
    let probe = vec![0.5, 1.5, 2.5, 3.5];
    let s_orig: f64 = f.score(&probe).unwrap().into();
    let s_back: f64 = back.score(&probe).unwrap().into();
    assert_eq!(s_orig, s_back, "score must be bit-exact after roundtrip");
}

#[test]
fn binary_roundtrip_with_time_decay() {
    let f = trained(7, 3, 16, 200, 0.05);
    let bytes = f.to_bytes().unwrap();
    let back = RandomCutForest::from_bytes(&bytes).unwrap();
    assert!((back.config().time_decay - 0.05).abs() < f64::EPSILON);
    let probe = vec![5.0, 5.0, 5.0];
    assert_eq!(
        f64::from(f.score(&probe).unwrap()),
        f64::from(back.score(&probe).unwrap())
    );
}

#[test]
fn binary_roundtrip_with_non_default_seed() {
    let f = trained(12_345_678, 2, 16, 100, 0.0);
    let bytes = f.to_bytes().unwrap();
    let back = RandomCutForest::from_bytes(&bytes).unwrap();
    assert_eq!(f.config().seed, back.config().seed);
}

#[test]
fn json_roundtrip_preserves_score() {
    let f = trained(99, 3, 16, 80, 0.0);
    let json = f.to_json().unwrap();
    let back = RandomCutForest::from_json(&json).unwrap();
    let probe = vec![1.0, 2.0, 3.0];
    let s_orig: f64 = f.score(&probe).unwrap().into();
    let s_back: f64 = back.score(&probe).unwrap().into();
    assert_eq!(s_orig, s_back);
}

#[test]
fn binary_truncated_payload_rejected() {
    assert!(matches!(
        RandomCutForest::from_bytes(&[]).unwrap_err(),
        RcfError::DeserializationFailed(_)
    ));
}

#[test]
fn binary_version_mismatch_rejected() {
    let f = trained(42, 2, 8, 20, 0.0);
    let mut bytes = f.to_bytes().unwrap();
    bytes[0] = bytes[0].wrapping_add(1);
    assert!(matches!(
        RandomCutForest::from_bytes(&bytes).unwrap_err(),
        RcfError::IncompatibleVersion { .. }
    ));
}

#[test]
fn json_version_mismatch_rejected() {
    let f = trained(42, 2, 8, 20, 0.0);
    let json = f.to_json().unwrap();
    let bogus = json.replacen("\"version\":1", "\"version\":42", 1);
    assert!(matches!(
        RandomCutForest::from_json(&bogus).unwrap_err(),
        RcfError::IncompatibleVersion {
            found: 42,
            expected: 1
        }
    ));
}

#[test]
fn json_malformed_rejected() {
    assert!(matches!(
        RandomCutForest::from_json("{ invalid").unwrap_err(),
        RcfError::DeserializationFailed(_)
    ));
}

#[test]
fn updates_after_roundtrip_continue_consistently() {
    let mut f = trained(2026, 2, 16, 50, 0.0);
    let bytes = f.to_bytes().unwrap();
    let mut back = RandomCutForest::from_bytes(&bytes).unwrap();
    // Both forests receive the same new point.
    let extra = vec![100.0, 100.0];
    f.update(extra.clone()).unwrap();
    back.update(extra.clone()).unwrap();
    assert_eq!(f.updates_seen(), back.updates_seen());
    let s1: f64 = f.score(&[50.0, 50.0]).unwrap().into();
    let s2: f64 = back.score(&[50.0, 50.0]).unwrap().into();
    assert_eq!(s1, s2);
}
