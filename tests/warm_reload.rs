//! Warm-reload end-to-end tests covering
//! [`rcf_rs::RandomCutForest`] and [`rcf_rs::ThresholdedForest`]
//! persisted to disk and resumed later.
//!
//! Asserts:
//!
//! 1. `to_path` / `from_path` round-trip for both forest and
//!    thresholded detector (bit-exact score + grade preservation).
//! 2. The atomic write replaces the previous snapshot and leaves no
//!    leftover tmp file on success.
//! 3. A pre-existing snapshot survives intact when the new save is
//!    observed by the reader only after `from_path` succeeds against
//!    the final path (rename-after-fsync discipline).
//! 4. `from_path` on a missing file / truncated file / wrong version
//!    returns the documented errors.
//! 5. A save/reload/resume cycle extends training without breaking
//!    the graded-verdict path.

#![cfg(all(feature = "postcard", feature = "std"))]
#![allow(clippy::cast_precision_loss, clippy::float_cmp)] // bit-exact roundtrip asserts + bounded casts.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rcf_rs::{ForestBuilder, RandomCutForest, RcfError, ThresholdedForestBuilder};

/// Build a path inside the OS temp dir that is unique to the running
/// test. Tests run in parallel so a shared filename would race.
fn unique_tmp_path(tag: &str) -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut p = std::env::temp_dir();
    p.push(format!("rcf-rs-{tag}-{nanos}-{seq}.bin"));
    p
}

/// Best-effort cleanup — ignore failures (another test may have
/// already unlinked the file, or it may never have existed).
fn cleanup(path: &Path) {
    let _ = fs::remove_file(path);
    let mut tmp: OsString = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let _ = fs::remove_file(PathBuf::from(tmp));
}

fn trained_forest(seed: u64, updates: u32) -> RandomCutForest<4> {
    let mut f = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(32)
        .seed(seed)
        .build()
        .unwrap();
    for i in 0_u32..updates {
        let v = f64::from(i) * 0.01;
        f.update([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
    }
    f
}

#[test]
fn forest_path_roundtrip_preserves_score() {
    let path = unique_tmp_path("forest-rt");
    let f = trained_forest(2026, 200);
    f.to_path(&path).unwrap();
    let back = RandomCutForest::<4>::from_path(&path).unwrap();
    let probe = [0.5_f64, 1.5, 2.5, 3.5];
    assert_eq!(
        f64::from(f.score(&probe).unwrap()),
        f64::from(back.score(&probe).unwrap())
    );
    cleanup(&path);
}

#[test]
fn forest_atomic_write_leaves_no_tmp_on_success() {
    let path = unique_tmp_path("forest-atomic");
    let f = trained_forest(7, 100);
    f.to_path(&path).unwrap();
    assert!(path.exists(), "final snapshot must exist on success");

    let mut tmp: OsString = path.as_os_str().to_owned();
    tmp.push(".tmp");
    assert!(
        !PathBuf::from(&tmp).exists(),
        "tmp file must be renamed away on success",
    );
    cleanup(&path);
}

#[test]
fn forest_from_path_missing_file_returns_deserialization_error() {
    let path = unique_tmp_path("forest-missing");
    // Path guaranteed not to exist yet.
    assert!(!path.exists());
    let err = RandomCutForest::<4>::from_path(&path).unwrap_err();
    assert!(
        matches!(err, RcfError::DeserializationFailed(_)),
        "expected DeserializationFailed on missing file, got {err:?}",
    );
}

#[test]
fn forest_from_path_truncated_file_rejected() {
    let path = unique_tmp_path("forest-trunc");
    fs::write(&path, [0_u8; 2]).unwrap();
    let err = RandomCutForest::<4>::from_path(&path).unwrap_err();
    assert!(matches!(err, RcfError::DeserializationFailed(_)));
    cleanup(&path);
}

#[test]
fn forest_save_over_existing_snapshot_replaces_it() {
    let path = unique_tmp_path("forest-replace");
    let first = trained_forest(1, 50);
    let second = trained_forest(2, 300);
    first.to_path(&path).unwrap();
    let first_bytes = fs::read(&path).unwrap();
    second.to_path(&path).unwrap();
    let second_bytes = fs::read(&path).unwrap();
    assert_ne!(first_bytes, second_bytes, "snapshot was not overwritten");
    cleanup(&path);
}

#[test]
fn thresholded_path_roundtrip_preserves_grade() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let path = unique_tmp_path("trcf-rt");
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(64)
        .z_factor(3.0)
        .min_observations(32)
        .min_threshold(0.1)
        .seed(42)
        .build()
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..300 {
        let p = [
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ];
        d.process(p).unwrap();
    }

    d.to_path(&path).unwrap();
    let back = rcf_rs::ThresholdedForest::<4>::from_path(&path).unwrap();

    // Stats must survive byte-for-byte.
    assert_eq!(d.stats().mean(), back.stats().mean());
    assert_eq!(d.stats().stddev(), back.stats().stddev());
    assert_eq!(d.stats().observations(), back.stats().observations());
    assert_eq!(d.current_threshold(), back.current_threshold());

    // Graded verdict must survive byte-for-byte on a fresh probe.
    let probe = [0.05_f64, 0.05, 0.05, 0.05];
    let g1 = d.score_only(&probe).unwrap();
    let g2 = back.score_only(&probe).unwrap();
    assert_eq!(f64::from(g1.score()), f64::from(g2.score()));
    assert_eq!(g1.grade(), g2.grade());
    assert_eq!(g1.threshold(), g2.threshold());
    assert_eq!(g1.is_anomaly(), g2.is_anomaly());
    assert_eq!(g1.ready(), g2.ready());
    cleanup(&path);
}

#[test]
fn thresholded_resume_training_after_reload() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let path = unique_tmp_path("trcf-resume");
    // First lifecycle: warm up, save.
    {
        let mut d = ThresholdedForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(64)
            .min_observations(16)
            .seed(11)
            .build()
            .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        for _ in 0..256 {
            let p = [
                rng.random::<f64>() * 0.1,
                rng.random::<f64>() * 0.1,
                rng.random::<f64>() * 0.1,
                rng.random::<f64>() * 0.1,
            ];
            d.process(p).unwrap();
        }
        d.to_path(&path).unwrap();
    }

    // Second lifecycle: reload, resume training, score an outlier.
    let mut d = rcf_rs::ThresholdedForest::<4>::from_path(&path).unwrap();
    assert!(d.stats().observations() > 0, "reloaded detector lost stats");
    let obs_before = d.stats().observations();
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    for _ in 0..32 {
        let p = [
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
            rng.random::<f64>() * 0.1,
        ];
        d.process(p).unwrap();
    }
    assert!(
        d.stats().observations() > obs_before,
        "resumed training did not advance stats counter",
    );
    let outlier = d.process([50.0, 50.0, 50.0, 50.0]).unwrap();
    assert!(
        outlier.ready() && outlier.is_anomaly(),
        "outlier after reload did not fire: ready={} is_anomaly={} grade={}",
        outlier.ready(),
        outlier.is_anomaly(),
        outlier.grade(),
    );
    cleanup(&path);
}

#[test]
fn thresholded_from_path_wrong_version_rejected() {
    let path = unique_tmp_path("trcf-version");
    let mut d = ThresholdedForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(16)
        .seed(3)
        .build()
        .unwrap();
    for i in 0_u32..20 {
        let v = f64::from(i) * 0.01;
        d.process([v, v, v, v]).unwrap();
    }
    d.to_path(&path).unwrap();
    // Corrupt the version prefix in-place.
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] = bytes[0].wrapping_add(1);
    fs::write(&path, &bytes).unwrap();

    let err = rcf_rs::ThresholdedForest::<4>::from_path(&path).unwrap_err();
    assert!(
        matches!(err, RcfError::IncompatibleVersion { .. }),
        "expected IncompatibleVersion, got {err:?}",
    );
    cleanup(&path);
}
