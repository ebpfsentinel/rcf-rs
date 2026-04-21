//! Error types used across the crate.
//!
//! [`RcfError`] is the canonical error returned by every fallible
//! operation in `rcf-rs`. Each variant carries enough context for the
//! caller to act without re-fetching state. [`RcfResult`] is the
//! convenient `Result` alias used in public signatures.

use alloc::string::String;

use thiserror::Error;

/// Errors produced by `rcf-rs`.
///
/// Variants are stable across `0.x` patch releases — adding a new
/// variant is a minor-version change.
///
/// # Examples
///
/// ```
/// use rcf_rs::{ForestBuilder, RcfError};
///
/// let err = ForestBuilder::<4>::new().num_trees(10).build().unwrap_err();
/// assert!(matches!(err, RcfError::InvalidConfig(_)));
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RcfError {
    /// A point with the wrong dimensionality was supplied.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Dimensionality the forest was configured with.
        expected: usize,
        /// Dimensionality of the offending input.
        got: usize,
    },

    /// A configuration value falls outside the AWS `SageMaker` spec
    /// bounds enforced by `ForestBuilder`.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// An operation that requires a non-empty forest was attempted on
    /// an empty one (e.g. scoring before any `update` call).
    #[error("forest is empty")]
    EmptyForest,

    /// A bounding box operation was requested on an empty box.
    #[error("bounding box is empty")]
    EmptyBoundingBox,

    /// A floating-point input contained `NaN`, which would break the
    /// total ordering required by every algorithm in the crate.
    #[error("input contains NaN")]
    NaNValue,

    /// An indexed access fell outside the live range.
    #[error("index {index} out of bounds (len={len})")]
    OutOfBounds {
        /// Index that was attempted.
        index: usize,
        /// Current length of the underlying collection.
        len: usize,
    },

    /// Persistence: serialising the forest failed.
    #[error("serialization failed: {0}")]
    SerializationFailed(String),

    /// Persistence: deserialising the forest failed (truncated bytes,
    /// malformed JSON, version-skew payload, etc).
    #[error("deserialization failed: {0}")]
    DeserializationFailed(String),

    /// Persistence: the encoded version prefix does not match the
    /// running library's expected version.
    #[error("incompatible persistence version: found {found}, expected {expected}")]
    IncompatibleVersion {
        /// Version embedded in the loaded payload.
        found: u32,
        /// Version the running library understands.
        expected: u32,
    },
}

/// Convenience alias for `Result<T, RcfError>`.
///
/// # Examples
///
/// ```
/// use rcf_rs::{RcfError, RcfResult};
///
/// fn check(n: u64) -> RcfResult<u64> {
///     if n == 0 { Err(RcfError::EmptyForest) } else { Ok(n) }
/// }
/// assert_eq!(check(7).unwrap(), 7);
/// assert!(check(0).is_err());
/// ```
pub type RcfResult<T> = Result<T, RcfError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_mismatch_renders_both_values() {
        let err = RcfError::DimensionMismatch {
            expected: 4,
            got: 7,
        };
        let msg = err.to_string();
        assert!(msg.contains('4'));
        assert!(msg.contains('7'));
    }

    #[test]
    fn invalid_config_carries_message() {
        let err = RcfError::InvalidConfig("num_trees=42 below minimum 50".into());
        assert!(err.to_string().contains("num_trees"));
    }

    #[test]
    fn out_of_bounds_renders_index_and_len() {
        let err = RcfError::OutOfBounds { index: 12, len: 10 };
        let msg = err.to_string();
        assert!(msg.contains("12"));
        assert!(msg.contains("10"));
    }

    #[test]
    fn empty_variants_render_static_message() {
        assert_eq!(RcfError::EmptyForest.to_string(), "forest is empty");
        assert_eq!(
            RcfError::EmptyBoundingBox.to_string(),
            "bounding box is empty"
        );
        assert_eq!(RcfError::NaNValue.to_string(), "input contains NaN");
    }

    #[test]
    fn rcf_result_alias_aliases_correctly() {
        let ok: RcfResult<u32> = Ok(7);
        let err: RcfResult<u32> = Err(RcfError::EmptyForest);
        assert!(matches!(ok, Ok(7)));
        assert!(err.is_err());
    }
}
