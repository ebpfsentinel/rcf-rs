//! Point type alias and dimensionality helpers.
//!
//! A [`Point`] is just `[f64]` — keeping the type as a slice/`Vec`
//! avoids friction at every call site. Validation (dim equality,
//! finiteness) happens through [`ensure_dim`] and [`ensure_finite`]
//! at the future `RandomCutForest::update` / `RandomCutForest::score`
//! boundary functions instead of inside every helper.

use alloc::vec::Vec;

use crate::error::{RcfError, RcfResult};

/// A point in `d`-dimensional Euclidean space.
///
/// Held as an owned `Vec<f64>` when the forest needs to retain the
/// data, or as a `&[f64]` slice on the ephemeral query path.
pub type Point = Vec<f64>;

/// Validate that `point` has the expected `dimension`.
///
/// # Errors
///
/// Returns [`RcfError::DimensionMismatch`] when the lengths differ.
///
/// # Examples
///
/// ```
/// use rcf_rs::domain::ensure_dim;
/// assert!(ensure_dim(&[1.0, 2.0, 3.0], 3).is_ok());
/// assert!(ensure_dim(&[1.0, 2.0], 3).is_err());
/// ```
pub fn ensure_dim(point: &[f64], dimension: usize) -> RcfResult<()> {
    if point.len() != dimension {
        return Err(RcfError::DimensionMismatch {
            expected: dimension,
            got: point.len(),
        });
    }
    Ok(())
}

/// Validate that every coordinate of `point` is finite (no `NaN`,
/// no `±∞`).
///
/// # Errors
///
/// Returns [`RcfError::NaNValue`] on the first non-finite component.
///
/// # Examples
///
/// ```
/// use rcf_rs::domain::ensure_finite;
/// assert!(ensure_finite(&[1.0, 2.0, 3.0]).is_ok());
/// assert!(ensure_finite(&[1.0, f64::NAN]).is_err());
/// assert!(ensure_finite(&[1.0, f64::INFINITY]).is_err());
/// ```
pub fn ensure_finite(point: &[f64]) -> RcfResult<()> {
    for &v in point {
        if !v.is_finite() {
            return Err(RcfError::NaNValue);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_dim_ok_when_match() {
        ensure_dim(&[1.0, 2.0, 3.0], 3).unwrap();
    }

    #[test]
    fn ensure_dim_err_when_short() {
        let err = ensure_dim(&[1.0, 2.0], 3).unwrap_err();
        match err {
            RcfError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn ensure_dim_err_when_long() {
        let err = ensure_dim(&[1.0, 2.0, 3.0, 4.0], 3).unwrap_err();
        assert!(matches!(
            err,
            RcfError::DimensionMismatch {
                expected: 3,
                got: 4
            }
        ));
    }

    #[test]
    fn ensure_finite_passes_for_normal_floats() {
        ensure_finite(&[-1.0, 0.0, 1.5e9, f64::MIN, f64::MAX]).unwrap();
    }

    #[test]
    fn ensure_finite_rejects_nan() {
        assert!(matches!(
            ensure_finite(&[1.0, f64::NAN, 2.0]).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn ensure_finite_rejects_positive_infinity() {
        assert!(matches!(
            ensure_finite(&[f64::INFINITY]).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn ensure_finite_rejects_negative_infinity() {
        assert!(matches!(
            ensure_finite(&[f64::NEG_INFINITY]).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn ensure_finite_passes_for_empty_slice() {
        ensure_finite(&[]).unwrap();
    }
}
