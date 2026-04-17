//! Optional persistence helpers for [`crate::RandomCutForest`].
//!
//! Gated behind the `serde` cargo feature. Two encodings are
//! exposed:
//!
//! - **Binary** (`RandomCutForest::to_bytes` / `from_bytes`, gated
//!   on the `bincode` feature): a compact `bincode 2` payload
//!   prefixed with a 4-byte little-endian version field. Use this
//!   for on-disk snapshots.
//! - **JSON** (`RandomCutForest::to_json` / `from_json`, gated on
//!   the `serde_json` feature): a human-readable text encoding
//!   wrapping the same versioned envelope. Useful for debugging or
//!   for callers who already pipe JSON elsewhere.
//!
//! The version prefix lives **outside** the serialised payload so a
//! version skew is detected before any third-party deserialiser
//! runs against arbitrary bytes — a defence against malformed
//! payload-driven panics.

#[cfg(any(feature = "bincode", feature = "serde_json"))]
use crate::error::{RcfError, RcfResult};
use crate::forest::RandomCutForest;

/// Persistence format version. Bump on any breaking layout change.
pub const PERSISTENCE_VERSION: u32 = 1;

/// Number of bytes reserved for the version prefix.
pub const VERSION_PREFIX_BYTES: usize = 4;

impl RandomCutForest {
    /// Serialise the forest into a versioned binary blob.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::SerializationFailed`] when the underlying
    /// `bincode` encoder rejects the payload.
    #[cfg(feature = "bincode")]
    pub fn to_bytes(&self) -> RcfResult<Vec<u8>> {
        let mut out = Vec::with_capacity(VERSION_PREFIX_BYTES + 4096);
        out.extend_from_slice(&PERSISTENCE_VERSION.to_le_bytes());
        let payload = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| RcfError::SerializationFailed(e.to_string()))?;
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Reload a forest previously produced by [`to_bytes`](Self::to_bytes).
    ///
    /// # Errors
    ///
    /// - [`RcfError::DeserializationFailed`] when the byte slice is
    ///   too short to hold the version prefix or the bincode payload
    ///   is malformed.
    /// - [`RcfError::IncompatibleVersion`] when the embedded version
    ///   does not match [`PERSISTENCE_VERSION`].
    #[cfg(feature = "bincode")]
    pub fn from_bytes(bytes: &[u8]) -> RcfResult<Self> {
        if bytes.len() < VERSION_PREFIX_BYTES {
            return Err(RcfError::DeserializationFailed(format!(
                "payload too short: {} byte(s), need at least {VERSION_PREFIX_BYTES}",
                bytes.len()
            )));
        }
        let mut version_bytes = [0_u8; VERSION_PREFIX_BYTES];
        version_bytes.copy_from_slice(&bytes[..VERSION_PREFIX_BYTES]);
        let version = u32::from_le_bytes(version_bytes);
        if version != PERSISTENCE_VERSION {
            return Err(RcfError::IncompatibleVersion {
                found: version,
                expected: PERSISTENCE_VERSION,
            });
        }
        let (forest, _consumed) = bincode::serde::decode_from_slice::<Self, _>(
            &bytes[VERSION_PREFIX_BYTES..],
            bincode::config::standard(),
        )
        .map_err(|e| RcfError::DeserializationFailed(e.to_string()))?;
        Ok(forest)
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
        let envelope: JsonEnvelopeOwned = serde_json::from_str(json)
            .map_err(|e| RcfError::DeserializationFailed(e.to_string()))?;
        if envelope.version != PERSISTENCE_VERSION {
            return Err(RcfError::IncompatibleVersion {
                found: envelope.version,
                expected: PERSISTENCE_VERSION,
            });
        }
        Ok(envelope.forest)
    }
}

/// JSON envelope used by [`RandomCutForest::to_json`] — borrows the
/// forest to avoid an unnecessary clone during serialisation.
#[cfg(feature = "serde_json")]
#[derive(serde::Serialize)]
struct JsonEnvelope<'a> {
    /// Persistence format version embedded alongside the payload.
    version: u32,
    /// Borrowed forest to be serialised.
    forest: &'a RandomCutForest,
}

/// JSON envelope used by [`RandomCutForest::from_json`] — owns the
/// reconstructed forest.
#[cfg(feature = "serde_json")]
#[derive(serde::Deserialize)]
struct JsonEnvelopeOwned {
    /// Persistence format version embedded alongside the payload.
    version: u32,
    /// Reconstructed forest owned by the envelope.
    forest: RandomCutForest,
}

#[cfg(all(test, feature = "bincode"))]
#[allow(clippy::float_cmp, clippy::cast_precision_loss, clippy::cast_lossless)] // Roundtrip asserts bit-exact equality + small bounded counters.
mod bincode_tests {
    use super::*;
    use crate::ForestBuilder;

    fn trained_forest(seed: u64, updates: usize) -> RandomCutForest {
        let mut f = ForestBuilder::new(2)
            .num_trees(50)
            .sample_size(16)
            .seed(seed)
            .build()
            .unwrap();
        for i in 0..updates {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 * 0.01;
            f.update(vec![v, v + 0.5]).unwrap();
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
        let f = ForestBuilder::new(4)
            .num_trees(50)
            .sample_size(16)
            .seed(1)
            .build()
            .unwrap();
        let bytes = f.to_bytes().unwrap();
        let back = RandomCutForest::from_bytes(&bytes).unwrap();
        assert_eq!(back.num_trees(), f.num_trees());
        assert_eq!(back.sample_size(), f.sample_size());
        assert_eq!(back.dimension(), f.dimension());
    }

    #[test]
    fn trained_forest_score_roundtrip() {
        let f = trained_forest(7, 200);
        let bytes = f.to_bytes().unwrap();
        let back = RandomCutForest::from_bytes(&bytes).unwrap();
        // Bit-exact score equality on a probe point.
        let probe = [1.5, 2.0];
        let s1: f64 = f.score(&probe).unwrap().into();
        let s2: f64 = back.score(&probe).unwrap().into();
        assert_eq!(s1, s2);
    }

    #[test]
    fn time_decay_roundtrip() {
        let mut f = ForestBuilder::new(2)
            .num_trees(50)
            .sample_size(16)
            .time_decay(0.05)
            .seed(11)
            .build()
            .unwrap();
        for i in 0..100 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64;
            f.update(vec![v, v]).unwrap();
        }
        let bytes = f.to_bytes().unwrap();
        let back = RandomCutForest::from_bytes(&bytes).unwrap();
        assert_eq!(f.config().time_decay, back.config().time_decay);
        let probe = [10.0, 10.0];
        assert_eq!(
            f64::from(f.score(&probe).unwrap()),
            f64::from(back.score(&probe).unwrap())
        );
    }

    #[test]
    fn truncated_bytes_rejected() {
        let bytes = [0_u8; 2];
        let err = RandomCutForest::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, RcfError::DeserializationFailed(_)));
    }

    #[test]
    fn version_mismatch_rejected() {
        let f = trained_forest(2026, 5);
        let mut bytes = f.to_bytes().unwrap();
        // Flip the version prefix to something the runtime does not know.
        let bogus_version = (PERSISTENCE_VERSION + 99).to_le_bytes();
        bytes[..VERSION_PREFIX_BYTES].copy_from_slice(&bogus_version);
        let err = RandomCutForest::from_bytes(&bytes).unwrap_err();
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
        bytes.extend_from_slice(&[0xFF; 16]); // garbage payload
        let err = RandomCutForest::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, RcfError::DeserializationFailed(_)));
    }

    #[test]
    fn updates_seen_counter_roundtrips() {
        let f = trained_forest(42, 75);
        let before = f.updates_seen();
        let bytes = f.to_bytes().unwrap();
        let back = RandomCutForest::from_bytes(&bytes).unwrap();
        assert_eq!(back.updates_seen(), before);
    }
}

#[cfg(all(test, feature = "serde_json"))]
#[allow(clippy::float_cmp, clippy::cast_precision_loss, clippy::cast_lossless)]
mod json_tests {
    use super::*;
    use crate::ForestBuilder;

    fn small_trained() -> RandomCutForest {
        let mut f = ForestBuilder::new(2)
            .num_trees(50)
            .sample_size(8)
            .seed(2026)
            .build()
            .unwrap();
        for i in 0..30 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64;
            f.update(vec![v, v + 1.0]).unwrap();
        }
        f
    }

    #[test]
    fn json_roundtrip_preserves_score() {
        let f = small_trained();
        let json = f.to_json().unwrap();
        let back = RandomCutForest::from_json(&json).unwrap();
        let probe = [3.0, 4.0];
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
        let err = RandomCutForest::from_json(&bogus).unwrap_err();
        assert!(matches!(err, RcfError::IncompatibleVersion { .. }));
    }

    #[test]
    fn json_malformed_rejected() {
        assert!(matches!(
            RandomCutForest::from_json("not json").unwrap_err(),
            RcfError::DeserializationFailed(_)
        ));
    }
}
