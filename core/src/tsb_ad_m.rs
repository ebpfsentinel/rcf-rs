//! TSB-AD-M — CSV loader for the multivariate split of the
//! Time-Series Benchmark for Anomaly Detection (Liu & Paparrizos,
//! `NeurIPS` 2024).
//!
//! Datasets ship as numeric CSV files with a one-row header:
//!
//! ```text
//! col_0,col_1,col_2,...,col_k,Label
//! 0.123,0.456,...,0,
//! 0.789,...,0
//! ```
//!
//! Every non-label column is a feature (`f64`); the final column
//! carries a binary anomaly flag encoded as `0` / `1` (or `0.0` /
//! `1.0`). Timestamp columns, when present, are treated as
//! ordinary features — TSB-AD-M leaves time handling to the
//! caller.
//!
//! The loader is dependency-free on purpose: TSB-AD-M files are
//! numeric-only CSV with no quoting, no escaping, and a fixed
//! comma delimiter, so a hand-written splitter is sufficient and
//! keeps the crate's dep graph small.
//!
//! Gated behind `std` — needs `std::fs` for [`TsbAdMDataset::load_csv`].
//!
//! # Reference
//!
//! Q. Liu, J. Paparrizos, "The Elephant in the Room: Towards A
//! Reliable Time-Series Anomaly Detection Benchmark", `NeurIPS`
//! Datasets & Benchmarks 2024.

use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use std::fs;
use std::io;
use std::path::Path;

use crate::error::{RcfError, RcfResult};

/// One TSB-AD-M dataset — features + per-timestamp labels.
///
/// `features[i]` is the `D`-dim feature vector at timestamp `i`.
/// `labels[i]` is `true` iff timestamp `i` is flagged anomalous
/// in the dataset's ground truth.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TsbAdMDataset {
    /// Column headers for the feature columns (label column is
    /// stripped).
    pub feature_headers: Vec<String>,
    /// Per-timestamp feature rows. `features.len() == labels.len()`.
    pub features: Vec<Vec<f64>>,
    /// Per-timestamp binary anomaly labels.
    pub labels: Vec<bool>,
}

impl TsbAdMDataset {
    /// Timestamp count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// `true` when the dataset holds zero timestamps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Feature dimensionality (number of non-label columns).
    #[must_use]
    pub fn feature_dim(&self) -> usize {
        self.feature_headers.len()
    }

    /// Count of anomalous timestamps.
    #[must_use]
    pub fn positive_count(&self) -> usize {
        self.labels.iter().filter(|&&b| b).count()
    }

    /// Load a TSB-AD-M CSV from disk.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] on I/O failure, empty
    /// file, missing label column, or malformed rows.
    pub fn load_csv<P: AsRef<Path>>(path: P) -> RcfResult<Self> {
        let raw = fs::read_to_string(path.as_ref()).map_err(|e| io_err(&e))?;
        Self::parse_csv(&raw)
    }

    /// Parse a TSB-AD-M CSV from an in-memory string. Useful for
    /// tests + already-downloaded buffers.
    ///
    /// # Errors
    ///
    /// See [`Self::load_csv`].
    pub fn parse_csv(raw: &str) -> RcfResult<Self> {
        let mut lines = raw.lines().filter(|l| !l.trim().is_empty());
        let header = lines
            .next()
            .ok_or_else(|| RcfError::InvalidConfig("TSB-AD-M: empty CSV".to_string()))?;
        let columns: Vec<&str> = header.split(',').map(str::trim).collect();
        let label_idx = columns
            .iter()
            .rposition(|h| h.eq_ignore_ascii_case("label"))
            .ok_or_else(|| {
                RcfError::InvalidConfig("TSB-AD-M: no 'Label' column in header".to_string())
            })?;
        if columns.len() < 2 {
            return Err(RcfError::InvalidConfig(
                "TSB-AD-M: header must contain at least one feature + Label".to_string(),
            ));
        }
        let feature_headers: Vec<String> = columns
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != label_idx)
            .map(|(_, h)| (*h).to_string())
            .collect();

        let mut features = Vec::with_capacity(1024);
        let mut labels = Vec::with_capacity(1024);

        for (row_idx, line) in lines.enumerate() {
            let cells: Vec<&str> = line.split(',').map(str::trim).collect();
            if cells.len() != columns.len() {
                return Err(RcfError::InvalidConfig(alloc::format!(
                    "TSB-AD-M: row {} has {} cells, expected {}",
                    row_idx,
                    cells.len(),
                    columns.len()
                )));
            }
            let label_cell = cells[label_idx];
            let label = parse_label(label_cell).ok_or_else(|| {
                RcfError::InvalidConfig(alloc::format!(
                    "TSB-AD-M: row {row_idx} label '{label_cell}' not binary"
                ))
            })?;
            let mut feats = Vec::with_capacity(columns.len() - 1);
            for (i, cell) in cells.iter().enumerate() {
                if i == label_idx {
                    continue;
                }
                let v: f64 = cell.parse().map_err(|_| {
                    RcfError::InvalidConfig(alloc::format!(
                        "TSB-AD-M: row {row_idx} col {i} value '{cell}' is not f64"
                    ))
                })?;
                feats.push(v);
            }
            features.push(feats);
            labels.push(label);
        }

        if features.is_empty() {
            return Err(RcfError::InvalidConfig(
                "TSB-AD-M: CSV has no data rows".to_string(),
            ));
        }
        Ok(Self {
            feature_headers,
            features,
            labels,
        })
    }

    /// Column `c` projected as a flat `Vec<f64>`. Handy for
    /// univariate detectors (e.g. `ShingledForest`, `MatrixProfile`).
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] on out-of-range index.
    pub fn column(&self, c: usize) -> RcfResult<Vec<f64>> {
        if c >= self.feature_dim() {
            return Err(RcfError::InvalidConfig(alloc::format!(
                "TSB-AD-M: column {c} out of range (dim = {})",
                self.feature_dim()
            )));
        }
        Ok(self.features.iter().map(|row| row[c]).collect())
    }
}

/// Parse a TSB-AD-M label cell — accepts common binary encodings.
fn parse_label(cell: &str) -> Option<bool> {
    match cell {
        "0" | "0.0" | "false" | "False" | "FALSE" => Some(false),
        "1" | "1.0" | "true" | "True" | "TRUE" => Some(true),
        _ => None,
    }
}

/// Wrap an `io::Error` into an [`RcfError::InvalidConfig`] with a
/// TSB-AD-M prefix.
fn io_err(e: &io::Error) -> RcfError {
    RcfError::InvalidConfig(alloc::format!("TSB-AD-M: I/O error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
col_a,col_b,Label
0.1,0.2,0
0.3,0.4,1
0.5,0.6,0
0.7,0.8,1
";

    #[test]
    fn parse_sample_csv() {
        let ds = TsbAdMDataset::parse_csv(SAMPLE).unwrap();
        assert_eq!(ds.len(), 4);
        assert_eq!(ds.feature_dim(), 2);
        assert_eq!(ds.positive_count(), 2);
        assert_eq!(ds.feature_headers, vec!["col_a", "col_b"]);
        assert!((ds.features[0][0] - 0.1).abs() < 1e-12);
        assert!((ds.features[3][1] - 0.8).abs() < 1e-12);
        assert_eq!(ds.labels, vec![false, true, false, true]);
    }

    #[test]
    fn column_projection() {
        let ds = TsbAdMDataset::parse_csv(SAMPLE).unwrap();
        let c0 = ds.column(0).unwrap();
        assert_eq!(c0, vec![0.1, 0.3, 0.5, 0.7]);
        assert!(ds.column(2).is_err());
    }

    #[test]
    fn rejects_missing_label_column() {
        let csv = "a,b,c\n0.1,0.2,0.3\n";
        assert!(TsbAdMDataset::parse_csv(csv).is_err());
    }

    #[test]
    fn rejects_empty_input() {
        assert!(TsbAdMDataset::parse_csv("").is_err());
        assert!(TsbAdMDataset::parse_csv("\n\n").is_err());
    }

    #[test]
    fn rejects_ragged_rows() {
        let bad = "a,b,Label\n0.1,0.2,0\n0.3,1\n";
        assert!(TsbAdMDataset::parse_csv(bad).is_err());
    }

    #[test]
    fn rejects_non_binary_label() {
        let bad = "a,Label\n0.1,2\n";
        assert!(TsbAdMDataset::parse_csv(bad).is_err());
    }

    #[test]
    fn accepts_float_labels() {
        let csv = "a,Label\n0.1,0.0\n0.2,1.0\n";
        let ds = TsbAdMDataset::parse_csv(csv).unwrap();
        assert_eq!(ds.labels, vec![false, true]);
    }

    #[test]
    fn accepts_label_mid_header() {
        // Older dumps put the label column first.
        let csv = "Label,a,b\n0,0.1,0.2\n1,0.3,0.4\n";
        let ds = TsbAdMDataset::parse_csv(csv).unwrap();
        assert_eq!(ds.feature_dim(), 2);
        assert_eq!(ds.labels, vec![false, true]);
        assert!((ds.features[1][0] - 0.3).abs() < 1e-12);
    }
}
