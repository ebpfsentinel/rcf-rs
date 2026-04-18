//! Optional persistence helpers for [`crate::RandomCutForest`] and
//! [`crate::ThresholdedForest`].
//!
//! Gated behind the `serde` cargo feature. Four flavours are
//! exposed:
//!
//! - **Binary bytes** (`to_bytes` / `from_bytes`, gated on `postcard`):
//!   a compact `postcard` payload prefixed with a 4-byte
//!   little-endian version field. Use this for on-disk snapshots or
//!   to ship forests over a network socket. (`postcard` replaced
//!   `bincode` in persistence format v2 after the `bincode` crate
//!   was marked unmaintained by `RustSec` in 2025.)
//! - **JSON text** (`to_json` / `from_json`, gated on `serde_json`):
//!   a human-readable text encoding wrapping the same versioned
//!   envelope. Useful for debugging or for callers who already pipe
//!   JSON elsewhere.
//! - **Atomic file path** (`to_path` / `from_path`, gated on
//!   `postcard + std`): write-tmp-then-rename + `fsync` so a crash or
//!   power-loss mid-save cannot corrupt the snapshot on disk. Pair
//!   with periodic checkpointing for **warm reload** — the detector
//!   resumes exactly where it left off across restarts.
//! - **JSON file path** (`to_json_path` / `from_json_path`, gated on
//!   `serde_json + std`): same atomic write discipline, human-readable
//!   payload.
//!
//! The version prefix lives **outside** the serialised payload so a
//! version skew is detected before any third-party deserialiser runs
//! against arbitrary bytes — a defence against malformed
//! payload-driven panics.
//!
//! Both encodings preserve the per-point dimensionality `D` at the
//! type level — callers must deserialise into a type with the same
//! compile-time `D` that produced the payload.

#[cfg(any(feature = "postcard", feature = "serde_json"))]
use crate::error::{RcfError, RcfResult};
use crate::forest::RandomCutForest;
#[cfg(feature = "serde")]
use crate::thresholded::ThresholdedForest;

/// Persistence format version for [`RandomCutForest`]. Bump on any
/// breaking layout change. Version `2` is `postcard`; version `1`
/// was the original `bincode 2` payload — the serialiser changed
/// when `RustSec` flagged `bincode` as unmaintained.
pub const PERSISTENCE_VERSION: u32 = 2;

/// Persistence format version for [`ThresholdedForest`]. Distinct
/// from [`PERSISTENCE_VERSION`] because the threshold envelope carries
/// additional state (EMA stats, threshold config) that evolves on its
/// own cadence. Version `2` switched the underlying binary serialiser
/// from `bincode` to `postcard`.
pub const THRESHOLDED_PERSISTENCE_VERSION: u32 = 2;

/// Number of bytes reserved for the version prefix.
pub const VERSION_PREFIX_BYTES: usize = 4;

/// Decode the first four bytes of `bytes` as the persistence version.
///
/// # Errors
///
/// Returns [`RcfError::DeserializationFailed`] when `bytes` is shorter
/// than [`VERSION_PREFIX_BYTES`].
#[cfg(feature = "postcard")]
fn read_version_prefix(bytes: &[u8]) -> RcfResult<u32> {
    if bytes.len() < VERSION_PREFIX_BYTES {
        return Err(RcfError::DeserializationFailed(format!(
            "payload too short: {} byte(s), need at least {VERSION_PREFIX_BYTES}",
            bytes.len()
        )));
    }
    let mut v = [0_u8; VERSION_PREFIX_BYTES];
    v.copy_from_slice(&bytes[..VERSION_PREFIX_BYTES]);
    Ok(u32::from_le_bytes(v))
}

/// Path helpers for atomic write-tmp-rename persistence.
///
/// The tmp suffix is appended to the caller-supplied path so the temp
/// file lives in the same filesystem — rename is only atomic within a
/// single filesystem. The file is `fsync`'d before the rename so a
/// power-loss between `write` and `rename` cannot leave a partially
/// written snapshot on disk.
#[cfg(all(feature = "std", any(feature = "postcard", feature = "serde_json")))]
mod atomic {
    use std::ffi::OsString;
    use std::fs::{File, rename};
    use std::io::Write;
    use std::path::{Path, PathBuf};

    use crate::error::{RcfError, RcfResult};

    /// Compute the temporary path used for the atomic write.
    pub(super) fn tmp_path(path: &Path) -> PathBuf {
        let mut s: OsString = path.as_os_str().to_owned();
        s.push(".tmp");
        PathBuf::from(s)
    }

    /// Write `bytes` to `path` atomically: tmp file first, fsync,
    /// then rename onto the target.
    pub(super) fn write_atomic(path: &Path, bytes: &[u8]) -> RcfResult<()> {
        let tmp = tmp_path(path);
        let mut f = File::create(&tmp)
            .map_err(|e| RcfError::SerializationFailed(format!("create {}: {e}", tmp.display())))?;
        f.write_all(bytes)
            .map_err(|e| RcfError::SerializationFailed(format!("write {}: {e}", tmp.display())))?;
        f.sync_all()
            .map_err(|e| RcfError::SerializationFailed(format!("fsync {}: {e}", tmp.display())))?;
        drop(f);
        rename(&tmp, path).map_err(|e| {
            RcfError::SerializationFailed(format!(
                "rename {} -> {}: {e}",
                tmp.display(),
                path.display()
            ))
        })?;
        Ok(())
    }

    /// Read the full byte content of `path`.
    #[cfg(feature = "postcard")]
    pub(super) fn read_all(path: &Path) -> RcfResult<Vec<u8>> {
        std::fs::read(path)
            .map_err(|e| RcfError::DeserializationFailed(format!("read {}: {e}", path.display())))
    }

    /// Read the full text content of `path`.
    #[cfg(feature = "serde_json")]
    pub(super) fn read_all_string(path: &Path) -> RcfResult<String> {
        std::fs::read_to_string(path)
            .map_err(|e| RcfError::DeserializationFailed(format!("read {}: {e}", path.display())))
    }
}

impl<const D: usize> RandomCutForest<D> {
    /// Serialise the forest into a versioned binary blob.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::SerializationFailed`] when the underlying
    /// `postcard` encoder rejects the payload.
    #[cfg(feature = "postcard")]
    pub fn to_bytes(&self) -> RcfResult<Vec<u8>> {
        let mut out = Vec::with_capacity(VERSION_PREFIX_BYTES + 4096);
        out.extend_from_slice(&PERSISTENCE_VERSION.to_le_bytes());
        let payload = postcard::to_allocvec(self)
            .map_err(|e| RcfError::SerializationFailed(e.to_string()))?;
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Reload a forest previously produced by [`to_bytes`](Self::to_bytes).
    ///
    /// # Errors
    ///
    /// - [`RcfError::DeserializationFailed`] when the byte slice is
    ///   too short to hold the version prefix or the `postcard`
    ///   payload is malformed.
    /// - [`RcfError::IncompatibleVersion`] when the embedded version
    ///   does not match [`PERSISTENCE_VERSION`].
    #[cfg(feature = "postcard")]
    pub fn from_bytes(bytes: &[u8]) -> RcfResult<Self> {
        let version = read_version_prefix(bytes)?;
        if version != PERSISTENCE_VERSION {
            return Err(RcfError::IncompatibleVersion {
                found: version,
                expected: PERSISTENCE_VERSION,
            });
        }
        let forest: Self = postcard::from_bytes(&bytes[VERSION_PREFIX_BYTES..])
            .map_err(|e| RcfError::DeserializationFailed(e.to_string()))?;
        Ok(forest)
    }

    /// Atomically serialise the forest to `path` using the binary
    /// encoding. Writes `<path>.tmp`, `fsync`s it, then renames onto
    /// `path` — a mid-write crash leaves the previous snapshot
    /// intact.
    ///
    /// # Errors
    ///
    /// - [`RcfError::SerializationFailed`] for any filesystem or
    ///   encoder failure.
    #[cfg(all(feature = "postcard", feature = "std"))]
    pub fn to_path(&self, path: impl AsRef<std::path::Path>) -> RcfResult<()> {
        let bytes = self.to_bytes()?;
        atomic::write_atomic(path.as_ref(), &bytes)
    }

    /// Reload a forest from `path` using the binary encoding.
    ///
    /// # Errors
    ///
    /// - [`RcfError::DeserializationFailed`] when the file cannot be
    ///   read or the payload is malformed.
    /// - [`RcfError::IncompatibleVersion`] when the embedded version
    ///   does not match [`PERSISTENCE_VERSION`].
    #[cfg(all(feature = "postcard", feature = "std"))]
    pub fn from_path(path: impl AsRef<std::path::Path>) -> RcfResult<Self> {
        let bytes = atomic::read_all(path.as_ref())?;
        Self::from_bytes(&bytes)
    }

    /// Serialise the forest as JSON. The version field lives at
    /// `"version"` alongside the payload at `"forest"`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::SerializationFailed`] when `serde_json`
    /// rejects the payload.
    #[cfg(feature = "serde_json")]
    pub fn to_json(&self) -> RcfResult<String> {
        let envelope = JsonEnvelope {
            version: PERSISTENCE_VERSION,
            forest: self,
        };
        serde_json::to_string(&envelope).map_err(|e| RcfError::SerializationFailed(e.to_string()))
    }

    /// Reload a forest from JSON produced by [`to_json`](Self::to_json).
    ///
    /// # Errors
    ///
    /// - [`RcfError::DeserializationFailed`] when the JSON is malformed.
    /// - [`RcfError::IncompatibleVersion`] when the embedded version
    ///   does not match [`PERSISTENCE_VERSION`].
    #[cfg(feature = "serde_json")]
    pub fn from_json(json: &str) -> RcfResult<Self> {
        let envelope: JsonEnvelopeOwned<D> = serde_json::from_str(json)
            .map_err(|e| RcfError::DeserializationFailed(e.to_string()))?;
        if envelope.version != PERSISTENCE_VERSION {
            return Err(RcfError::IncompatibleVersion {
                found: envelope.version,
                expected: PERSISTENCE_VERSION,
            });
        }
        Ok(envelope.forest)
    }

    /// Atomically write the forest as JSON to `path`. Same atomic
    /// write discipline as [`to_path`](Self::to_path).
    ///
    /// # Errors
    ///
    /// - [`RcfError::SerializationFailed`] for any filesystem or
    ///   encoder failure.
    #[cfg(all(feature = "serde_json", feature = "std"))]
    pub fn to_json_path(&self, path: impl AsRef<std::path::Path>) -> RcfResult<()> {
        let json = self.to_json()?;
        atomic::write_atomic(path.as_ref(), json.as_bytes())
    }

    /// Reload a forest from a JSON file at `path`.
    ///
    /// # Errors
    ///
    /// - [`RcfError::DeserializationFailed`] when the file cannot be
    ///   read or the JSON is malformed.
    /// - [`RcfError::IncompatibleVersion`] when the embedded version
    ///   does not match [`PERSISTENCE_VERSION`].
    #[cfg(all(feature = "serde_json", feature = "std"))]
    pub fn from_json_path(path: impl AsRef<std::path::Path>) -> RcfResult<Self> {
        let json = atomic::read_all_string(path.as_ref())?;
        Self::from_json(&json)
    }
}

impl<const D: usize> ThresholdedForest<D> {
    /// Serialise the thresholded detector into a versioned binary blob.
    ///
    /// The payload carries the underlying forest, the threshold
    /// configuration, and the EMA statistics — enough for a receiver
    /// to resume scoring and emitting graded verdicts without a
    /// warmup gap.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::SerializationFailed`] when the underlying
    /// `postcard` encoder rejects the payload.
    #[cfg(feature = "postcard")]
    pub fn to_bytes(&self) -> RcfResult<Vec<u8>> {
        let mut out = Vec::with_capacity(VERSION_PREFIX_BYTES + 4096);
        out.extend_from_slice(&THRESHOLDED_PERSISTENCE_VERSION.to_le_bytes());
        let payload = postcard::to_allocvec(self)
            .map_err(|e| RcfError::SerializationFailed(e.to_string()))?;
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Reload a thresholded detector previously produced by
    /// [`to_bytes`](Self::to_bytes).
    ///
    /// # Errors
    ///
    /// - [`RcfError::DeserializationFailed`] when the byte slice is
    ///   too short to hold the version prefix or the `postcard`
    ///   payload is malformed.
    /// - [`RcfError::IncompatibleVersion`] when the embedded version
    ///   does not match [`THRESHOLDED_PERSISTENCE_VERSION`].
    #[cfg(feature = "postcard")]
    pub fn from_bytes(bytes: &[u8]) -> RcfResult<Self> {
        let version = read_version_prefix(bytes)?;
        if version != THRESHOLDED_PERSISTENCE_VERSION {
            return Err(RcfError::IncompatibleVersion {
                found: version,
                expected: THRESHOLDED_PERSISTENCE_VERSION,
            });
        }
        let detector: Self = postcard::from_bytes(&bytes[VERSION_PREFIX_BYTES..])
            .map_err(|e| RcfError::DeserializationFailed(e.to_string()))?;
        Ok(detector)
    }

    /// Atomically serialise the thresholded detector to `path`. Same
    /// atomic write discipline as [`RandomCutForest::to_path`].
    ///
    /// # Errors
    ///
    /// - [`RcfError::SerializationFailed`] for any filesystem or
    ///   encoder failure.
    #[cfg(all(feature = "postcard", feature = "std"))]
    pub fn to_path(&self, path: impl AsRef<std::path::Path>) -> RcfResult<()> {
        let bytes = self.to_bytes()?;
        atomic::write_atomic(path.as_ref(), &bytes)
    }

    /// Reload a thresholded detector from `path`.
    ///
    /// # Errors
    ///
    /// - [`RcfError::DeserializationFailed`] when the file cannot be
    ///   read or the payload is malformed.
    /// - [`RcfError::IncompatibleVersion`] when the embedded version
    ///   does not match [`THRESHOLDED_PERSISTENCE_VERSION`].
    #[cfg(all(feature = "postcard", feature = "std"))]
    pub fn from_path(path: impl AsRef<std::path::Path>) -> RcfResult<Self> {
        let bytes = atomic::read_all(path.as_ref())?;
        Self::from_bytes(&bytes)
    }

    /// Serialise the thresholded detector as JSON.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::SerializationFailed`] when `serde_json`
    /// rejects the payload.
    #[cfg(feature = "serde_json")]
    pub fn to_json(&self) -> RcfResult<String> {
        let envelope = ThresholdedJsonEnvelope {
            version: THRESHOLDED_PERSISTENCE_VERSION,
            detector: self,
        };
        serde_json::to_string(&envelope).map_err(|e| RcfError::SerializationFailed(e.to_string()))
    }

    /// Reload a thresholded detector from JSON.
    ///
    /// # Errors
    ///
    /// - [`RcfError::DeserializationFailed`] when the JSON is malformed.
    /// - [`RcfError::IncompatibleVersion`] when the embedded version
    ///   does not match [`THRESHOLDED_PERSISTENCE_VERSION`].
    #[cfg(feature = "serde_json")]
    pub fn from_json(json: &str) -> RcfResult<Self> {
        let envelope: ThresholdedJsonEnvelopeOwned<D> = serde_json::from_str(json)
            .map_err(|e| RcfError::DeserializationFailed(e.to_string()))?;
        if envelope.version != THRESHOLDED_PERSISTENCE_VERSION {
            return Err(RcfError::IncompatibleVersion {
                found: envelope.version,
                expected: THRESHOLDED_PERSISTENCE_VERSION,
            });
        }
        Ok(envelope.detector)
    }

    /// Atomically write the thresholded detector as JSON to `path`.
    ///
    /// # Errors
    ///
    /// - [`RcfError::SerializationFailed`] for any filesystem or
    ///   encoder failure.
    #[cfg(all(feature = "serde_json", feature = "std"))]
    pub fn to_json_path(&self, path: impl AsRef<std::path::Path>) -> RcfResult<()> {
        let json = self.to_json()?;
        atomic::write_atomic(path.as_ref(), json.as_bytes())
    }

    /// Reload a thresholded detector from a JSON file at `path`.
    ///
    /// # Errors
    ///
    /// - [`RcfError::DeserializationFailed`] when the file cannot be
    ///   read or the JSON is malformed.
    /// - [`RcfError::IncompatibleVersion`] when the embedded version
    ///   does not match [`THRESHOLDED_PERSISTENCE_VERSION`].
    #[cfg(all(feature = "serde_json", feature = "std"))]
    pub fn from_json_path(path: impl AsRef<std::path::Path>) -> RcfResult<Self> {
        let json = atomic::read_all_string(path.as_ref())?;
        Self::from_json(&json)
    }
}

/// JSON envelope used by [`RandomCutForest::to_json`] — borrows the
/// forest to avoid an unnecessary clone during serialisation.
#[cfg(feature = "serde_json")]
#[derive(serde::Serialize)]
struct JsonEnvelope<'a, const D: usize> {
    /// Persistence format version embedded alongside the payload.
    version: u32,
    /// Borrowed forest to be serialised.
    forest: &'a RandomCutForest<D>,
}

/// JSON envelope used by [`RandomCutForest::from_json`] — owns the
/// reconstructed forest.
#[cfg(feature = "serde_json")]
#[derive(serde::Deserialize)]
struct JsonEnvelopeOwned<const D: usize> {
    /// Persistence format version embedded alongside the payload.
    version: u32,
    /// Reconstructed forest owned by the envelope.
    forest: RandomCutForest<D>,
}

/// JSON envelope for [`ThresholdedForest::to_json`].
#[cfg(feature = "serde_json")]
#[derive(serde::Serialize)]
struct ThresholdedJsonEnvelope<'a, const D: usize> {
    /// Persistence format version embedded alongside the payload.
    version: u32,
    /// Borrowed detector to be serialised.
    detector: &'a ThresholdedForest<D>,
}

/// JSON envelope for [`ThresholdedForest::from_json`].
#[cfg(feature = "serde_json")]
#[derive(serde::Deserialize)]
struct ThresholdedJsonEnvelopeOwned<const D: usize> {
    /// Persistence format version embedded alongside the payload.
    version: u32,
    /// Reconstructed detector owned by the envelope.
    detector: ThresholdedForest<D>,
}

#[cfg(all(test, feature = "postcard"))]
#[allow(clippy::float_cmp, clippy::cast_precision_loss, clippy::cast_lossless)] // Roundtrip asserts bit-exact equality + small bounded counters.
mod binary_tests {
    use super::*;
    use crate::ForestBuilder;

    fn trained_forest(seed: u64, updates: usize) -> RandomCutForest<2> {
        let mut f = ForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .seed(seed)
            .build()
            .unwrap();
        for i in 0..updates {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 * 0.01;
            f.update([v, v + 0.5]).unwrap();
        }
        f
    }

    #[test]
    fn version_prefix_present() {
        let f = trained_forest(2026, 10);
        let bytes = f.to_bytes().unwrap();
        assert!(bytes.len() >= VERSION_PREFIX_BYTES);
        let mut v = [0_u8; 4];
        v.copy_from_slice(&bytes[..4]);
        assert_eq!(u32::from_le_bytes(v), PERSISTENCE_VERSION);
    }

    #[test]
    fn empty_forest_roundtrip() {
        let f = ForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(16)
            .seed(1)
            .build()
            .unwrap();
        let bytes = f.to_bytes().unwrap();
        let back = RandomCutForest::<4>::from_bytes(&bytes).unwrap();
        assert_eq!(back.num_trees(), f.num_trees());
        assert_eq!(back.sample_size(), f.sample_size());
        assert_eq!(back.dimension(), f.dimension());
    }

    #[test]
    fn trained_forest_score_roundtrip() {
        let f = trained_forest(7, 200);
        let bytes = f.to_bytes().unwrap();
        let back = RandomCutForest::<2>::from_bytes(&bytes).unwrap();
        let probe = [1.5_f64, 2.0];
        let s1: f64 = f.score(&probe).unwrap().into();
        let s2: f64 = back.score(&probe).unwrap().into();
        assert_eq!(s1, s2);
    }

    #[test]
    fn time_decay_roundtrip() {
        let mut f = ForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .time_decay(0.05)
            .seed(11)
            .build()
            .unwrap();
        for i in 0..100 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64;
            f.update([v, v]).unwrap();
        }
        let bytes = f.to_bytes().unwrap();
        let back = RandomCutForest::<2>::from_bytes(&bytes).unwrap();
        assert_eq!(f.config().time_decay, back.config().time_decay);
        let probe = [10.0_f64, 10.0];
        assert_eq!(
            f64::from(f.score(&probe).unwrap()),
            f64::from(back.score(&probe).unwrap())
        );
    }

    #[test]
    fn truncated_bytes_rejected() {
        let bytes = [0_u8; 2];
        let err = RandomCutForest::<2>::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, RcfError::DeserializationFailed(_)));
    }

    #[test]
    fn version_mismatch_rejected() {
        let f = trained_forest(2026, 5);
        let mut bytes = f.to_bytes().unwrap();
        let bogus_version = (PERSISTENCE_VERSION + 99).to_le_bytes();
        bytes[..VERSION_PREFIX_BYTES].copy_from_slice(&bogus_version);
        let err = RandomCutForest::<2>::from_bytes(&bytes).unwrap_err();
        match err {
            RcfError::IncompatibleVersion { found, expected } => {
                assert_eq!(found, PERSISTENCE_VERSION + 99);
                assert_eq!(expected, PERSISTENCE_VERSION);
            }
            other => panic!("expected IncompatibleVersion, got {other:?}"),
        }
    }

    #[test]
    fn malformed_payload_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&PERSISTENCE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&[0xFF; 16]);
        let err = RandomCutForest::<2>::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, RcfError::DeserializationFailed(_)));
    }

    #[test]
    fn updates_seen_counter_roundtrips() {
        let f = trained_forest(42, 75);
        let before = f.updates_seen();
        let bytes = f.to_bytes().unwrap();
        let back = RandomCutForest::<2>::from_bytes(&bytes).unwrap();
        assert_eq!(back.updates_seen(), before);
    }
}

#[cfg(all(test, feature = "serde_json"))]
#[allow(clippy::float_cmp, clippy::cast_precision_loss, clippy::cast_lossless)]
mod json_tests {
    use super::*;
    use crate::ForestBuilder;

    fn small_trained() -> RandomCutForest<2> {
        let mut f = ForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(8)
            .seed(2026)
            .build()
            .unwrap();
        for i in 0..30 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64;
            f.update([v, v + 1.0]).unwrap();
        }
        f
    }

    #[test]
    fn json_roundtrip_preserves_score() {
        let f = small_trained();
        let json = f.to_json().unwrap();
        let back = RandomCutForest::<2>::from_json(&json).unwrap();
        let probe = [3.0_f64, 4.0];
        let s1: f64 = f.score(&probe).unwrap().into();
        let s2: f64 = back.score(&probe).unwrap().into();
        assert_eq!(s1, s2);
    }

    #[test]
    fn json_envelope_carries_version_field() {
        let f = small_trained();
        let json = f.to_json().unwrap();
        assert!(json.contains("\"version\""));
        assert!(json.contains(&format!(":{PERSISTENCE_VERSION}")));
    }

    #[test]
    fn json_version_mismatch_rejected() {
        let f = small_trained();
        let json = f.to_json().unwrap();
        let bogus = json.replace(
            &format!("\"version\":{PERSISTENCE_VERSION}"),
            &format!("\"version\":{}", PERSISTENCE_VERSION + 99),
        );
        let err = RandomCutForest::<2>::from_json(&bogus).unwrap_err();
        assert!(matches!(err, RcfError::IncompatibleVersion { .. }));
    }

    #[test]
    fn json_malformed_rejected() {
        assert!(matches!(
            RandomCutForest::<2>::from_json("not json").unwrap_err(),
            RcfError::DeserializationFailed(_)
        ));
    }
}
