//! Crate-local serde adapters.
//!
//! Rust's stable `serde` ships `Deserialize` for `[T; N]` only up to
//! `N = 32`, so every const-generic `[f64; D]` field has to route
//! through an adapter that round-trips via `Vec<f64>`. Factored out
//! so `BoundingBox`, `AlertRecord` and friends share one path.

/// Snapshot `[f64; D]` to / from a `Vec<f64>` payload.
pub(crate) mod fixed_array_f64 {
    use alloc::format;
    use alloc::vec::Vec;

    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

    /// Write the array as a borrowed slice — the downstream encoder
    /// decides the wire shape (JSON array, postcard varint length,
    /// etc.).
    pub fn serialize<S, const D: usize>(arr: &[f64; D], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        arr.as_slice().serialize(serializer)
    }

    /// Reconstitute a `[f64; D]` from a decoded `Vec<f64>`, rejecting
    /// payloads whose length does not match `D`.
    pub fn deserialize<'de, D2, const D: usize>(deserializer: D2) -> Result<[f64; D], D2::Error>
    where
        D2: Deserializer<'de>,
    {
        let v: Vec<f64> = Vec::deserialize(deserializer)?;
        v.try_into().map_err(|got: Vec<f64>| {
            D2::Error::custom(format!(
                "array length mismatch: expected {D}, got {}",
                got.len()
            ))
        })
    }
}
