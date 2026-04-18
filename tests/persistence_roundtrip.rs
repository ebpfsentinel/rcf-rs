//! End-to-end persistence roundtrip tests for [`rcf_rs::RandomCutForest`].
//!
//! Bit-exact score roundtrip for binary + JSON, version-mismatch
//! rejection, malformed-payload rejection, and the four matrix
//! scenarios (empty, trained, time-decay, custom seed).
//!
//! Only compiled with `--features postcard,serde_json`.

#![cfg(all(feature = "postcard", feature = "serde_json"))]
#![allow(clippy::cast_precision_loss, clippy::float_cmp)] // Roundtrip asserts bit-exact f64 equality.

use rcf_rs::{ForestBuilder, RandomCutForest, RcfError};

fn trained_2(seed: u64, sample_size: usize, updates: usize, decay: f64) -> RandomCutForest<2> {
    let mut f = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(sample_size)
        .time_decay(decay)
        .seed(seed)
        .build()
        .unwrap();
    for i in 0..updates {
        let v = i as f64 * 0.01;
        f.update([v, v + 0.1]).unwrap();
    }
    f
}

fn trained_3(seed: u64, sample_size: usize, updates: usize, decay: f64) -> RandomCutForest<3> {
    let mut f = ForestBuilder::<3>::new()
        .num_trees(50)
        .sample_size(sample_size)
        .time_decay(decay)
        .seed(seed)
        .build()
        .unwrap();
    for i in 0..updates {
        let v = i as f64 * 0.01;
        f.update([v, v + 0.1, v + 0.2]).unwrap();
    }
    f
}

fn trained_4(seed: u64, sample_size: usize, updates: usize, decay: f64) -> RandomCutForest<4> {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(sample_size)
        .time_decay(decay)
        .seed(seed)
        .build()
        .unwrap();
    for i in 0..updates {
        let v = i as f64 * 0.01;
        f.update([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
    }
    f
}

#[test]
fn binary_roundtrip_empty_forest() {
    let f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(8)
        .seed(1)
        .build()
        .unwrap();
    let bytes = f.to_bytes().unwrap();
    let back = RandomCutForest::<4>::from_bytes(&bytes).unwrap();
    assert_eq!(back.dimension(), 4);
    assert_eq!(back.num_trees(), 50);
    assert_eq!(back.updates_seen(), 0);
}

#[test]
fn binary_roundtrip_trained_forest() {
    let f = trained_4(2026, 32, 500, 0.0);
    let bytes = f.to_bytes().unwrap();
    let back = RandomCutForest::<4>::from_bytes(&bytes).unwrap();
    let probe: [f64; 4] = [0.5, 1.5, 2.5, 3.5];
    let s_orig: f64 = f.score(&probe).unwrap().into();
    let s_back: f64 = back.score(&probe).unwrap().into();
    assert_eq!(s_orig, s_back, "score must be bit-exact after roundtrip");
}

#[test]
fn binary_roundtrip_with_time_decay() {
    let f = trained_3(7, 16, 200, 0.05);
    let bytes = f.to_bytes().unwrap();
    let back = RandomCutForest::<3>::from_bytes(&bytes).unwrap();
    assert!((back.config().time_decay - 0.05).abs() < f64::EPSILON);
    let probe: [f64; 3] = [5.0, 5.0, 5.0];
    assert_eq!(
        f64::from(f.score(&probe).unwrap()),
        f64::from(back.score(&probe).unwrap())
    );
}

#[test]
fn binary_roundtrip_with_non_default_seed() {
    let f = trained_2(12_345_678, 16, 100, 0.0);
    let bytes = f.to_bytes().unwrap();
    let back = RandomCutForest::<2>::from_bytes(&bytes).unwrap();
    assert_eq!(f.config().seed, back.config().seed);
}

#[test]
fn json_roundtrip_preserves_score() {
    let f = trained_3(99, 16, 80, 0.0);
    let json = f.to_json().unwrap();
    let back = RandomCutForest::<3>::from_json(&json).unwrap();
    let probe: [f64; 3] = [1.0, 2.0, 3.0];
    let s_orig: f64 = f.score(&probe).unwrap().into();
    let s_back: f64 = back.score(&probe).unwrap().into();
    assert_eq!(s_orig, s_back);
}

#[test]
fn binary_truncated_payload_rejected() {
    assert!(matches!(
        RandomCutForest::<2>::from_bytes(&[]).unwrap_err(),
        RcfError::DeserializationFailed(_)
    ));
}

#[test]
fn binary_version_mismatch_rejected() {
    let f = trained_2(42, 8, 20, 0.0);
    let mut bytes = f.to_bytes().unwrap();
    bytes[0] = bytes[0].wrapping_add(1);
    assert!(matches!(
        RandomCutForest::<2>::from_bytes(&bytes).unwrap_err(),
        RcfError::IncompatibleVersion { .. }
    ));
}

#[test]
fn json_version_mismatch_rejected() {
    let f = trained_2(42, 8, 20, 0.0);
    let json = f.to_json().unwrap();
    let bogus = json.replacen("\"version\":1", "\"version\":42", 1);
    assert!(matches!(
        RandomCutForest::<2>::from_json(&bogus).unwrap_err(),
        RcfError::IncompatibleVersion {
            found: 42,
            expected: 1
        }
    ));
}

#[test]
fn json_malformed_rejected() {
    assert!(matches!(
        RandomCutForest::<2>::from_json("{ invalid").unwrap_err(),
        RcfError::DeserializationFailed(_)
    ));
}

#[test]
fn updates_after_roundtrip_continue_consistently() {
    let mut f = trained_2(2026, 16, 50, 0.0);
    let bytes = f.to_bytes().unwrap();
    let mut back = RandomCutForest::<2>::from_bytes(&bytes).unwrap();
    let extra: [f64; 2] = [100.0, 100.0];
    f.update(extra).unwrap();
    back.update(extra).unwrap();
    assert_eq!(f.updates_seen(), back.updates_seen());
    let s1: f64 = f.score(&[50.0, 50.0]).unwrap().into();
    let s2: f64 = back.score(&[50.0, 50.0]).unwrap().into();
    assert_eq!(s1, s2);
}
