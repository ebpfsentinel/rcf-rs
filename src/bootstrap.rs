//! Cold-start bootstrap: warm a detector from historical data before
//! exposing it to live traffic.
//!
//! A freshly-built [`crate::RandomCutForest`] or
//! [`crate::ThresholdedForest`] has an empty reservoir and (for the
//! thresholded variant) an EMA of the anomaly-score stream with zero
//! observations. Real scoring only becomes meaningful once the
//! detector has seen enough points to populate the reservoir and
//! converge the adaptive threshold. In a production streaming agent
//! this warmup window is a coverage hole at every restart — unless
//! the caller can replay a slice of recent history (from a TSDB,
//! Kafka topic, S3 parquet dump…) before going live.
//!
//! [`RandomCutForest::bootstrap`], [`ThresholdedForest::bootstrap`]
//! and [`crate::TenantForestPool::bootstrap`] accept any
//! [`IntoIterator`] of `[f64; D]` points and ingest them through the
//! normal `update` / `process` path, returning a [`BootstrapReport`]
//! so the caller can confirm the detector is hot (observations past
//! the configured warmup window, threshold above the floor).
//!
//! Points containing non-finite components (`NaN`, `±∞`) are
//! **skipped** and tallied in the report rather than aborting the
//! whole bootstrap — historical TSDB query results routinely contain
//! gaps, and a single bad row should not sink the restart.

use crate::error::{RcfError, RcfResult};
use crate::forest::RandomCutForest;
use crate::thresholded::ThresholdedForest;

/// Summary of a bootstrap replay — what went in, what was filtered
/// out, and where the detector's warmup stands afterwards.
///
/// Used by callers to check the detector is ready for live traffic
/// before the streaming pipeline is switched on. A rule of thumb:
/// `final_observations >= min_observations` (for a thresholded
/// forest) means the threshold is adaptive, not the floor.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BootstrapReport {
    /// Number of points successfully folded into the detector.
    pub points_ingested: u64,
    /// Number of non-finite points skipped without mutating the
    /// detector (NaN / ±∞ components).
    pub points_skipped: u64,
    /// Observation count of the detector's EMA stats after the
    /// replay (`0` for [`RandomCutForest`] which has no threshold
    /// layer).
    pub final_observations: u64,
    /// Adaptive threshold at the end of the replay (`0.0` for
    /// [`RandomCutForest`] which has no threshold layer).
    pub final_threshold: f64,
}

impl BootstrapReport {
    /// Empty report — zero points, threshold at the configured floor.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            points_ingested: 0,
            points_skipped: 0,
            final_observations: 0,
            final_threshold: 0.0,
        }
    }

    /// Whether any historical point actually made it into the
    /// detector. A `false` return means the iterator was empty or
    /// every row was non-finite — the detector is still cold and
    /// should be treated as warming-up by downstream consumers.
    #[must_use]
    pub fn is_hot(&self) -> bool {
        self.points_ingested > 0
    }
}

impl Default for BootstrapReport {
    fn default() -> Self {
        Self::empty()
    }
}

/// Skip policy applied to non-finite inputs during bootstrap.
fn is_finite_point<const D: usize>(p: &[f64; D]) -> bool {
    p.iter().all(|x| x.is_finite())
}

impl<const D: usize> RandomCutForest<D> {
    /// Replay historical `points` through the forest without
    /// exposing any score — warms the reservoir so subsequent
    /// [`Self::score`] calls return meaningful values from the first
    /// live point.
    ///
    /// Non-finite points are silently skipped and tallied in the
    /// report. All other errors from [`Self::update`] are propagated.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::update`] failures other than
    /// [`RcfError::NaNValue`] (which is absorbed and counted as a
    /// skip).
    pub fn bootstrap<I>(&mut self, points: I) -> RcfResult<BootstrapReport>
    where
        I: IntoIterator<Item = [f64; D]>,
    {
        let mut ingested: u64 = 0;
        let mut skipped: u64 = 0;
        for p in points {
            if !is_finite_point(&p) {
                skipped = skipped.saturating_add(1);
                continue;
            }
            match self.update(p) {
                Ok(()) => ingested = ingested.saturating_add(1),
                Err(RcfError::NaNValue) => skipped = skipped.saturating_add(1),
                Err(other) => return Err(other),
            }
        }
        Ok(BootstrapReport {
            points_ingested: ingested,
            points_skipped: skipped,
            final_observations: self.updates_seen(),
            final_threshold: 0.0,
        })
    }
}

impl<const D: usize> ThresholdedForest<D> {
    /// Replay historical `points` through the thresholded detector,
    /// folding each one into the forest *and* the score-stream EMA
    /// so the adaptive threshold is hot before the first live point.
    ///
    /// Graded verdicts produced during the replay are discarded —
    /// they would be misleading for historical data. The detector
    /// is ready for live traffic as soon as
    /// [`BootstrapReport::final_observations`] passes the configured
    /// `min_observations` threshold.
    ///
    /// Non-finite points are skipped and tallied in the report.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::process`] failures other than
    /// [`RcfError::NaNValue`] (absorbed and counted as a skip).
    pub fn bootstrap<I>(&mut self, points: I) -> RcfResult<BootstrapReport>
    where
        I: IntoIterator<Item = [f64; D]>,
    {
        let mut ingested: u64 = 0;
        let mut skipped: u64 = 0;
        for p in points {
            if !is_finite_point(&p) {
                skipped = skipped.saturating_add(1);
                continue;
            }
            match self.process(p) {
                Ok(_) => ingested = ingested.saturating_add(1),
                Err(RcfError::NaNValue) => skipped = skipped.saturating_add(1),
                Err(other) => return Err(other),
            }
        }
        Ok(BootstrapReport {
            points_ingested: ingested,
            points_skipped: skipped,
            final_observations: self.stats().observations(),
            final_threshold: self.current_threshold(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert bounds on closed-form quantities.
mod tests {
    use super::*;
    use crate::{ForestBuilder, ThresholdedForestBuilder};

    #[test]
    fn bootstrap_report_empty_defaults() {
        let r = BootstrapReport::empty();
        assert_eq!(r.points_ingested, 0);
        assert_eq!(r.points_skipped, 0);
        assert_eq!(r.final_observations, 0);
        assert_eq!(r.final_threshold, 0.0);
        assert!(!r.is_hot());
        assert_eq!(r, BootstrapReport::default());
    }

    #[test]
    fn forest_bootstrap_from_empty_iter_is_noop() {
        let mut f = ForestBuilder::<2>::new().seed(1).build().unwrap();
        let r = f.bootstrap(std::iter::empty::<[f64; 2]>()).unwrap();
        assert_eq!(r.points_ingested, 0);
        assert!(!r.is_hot());
        assert_eq!(f.updates_seen(), 0);
    }

    #[test]
    fn forest_bootstrap_counts_ingested_and_skipped() {
        let mut f = ForestBuilder::<2>::new().seed(1).build().unwrap();
        let pts: Vec<[f64; 2]> = vec![
            [0.0, 0.0],
            [1.0, 1.0],
            [f64::NAN, 0.0], // skip
            [2.0, 2.0],
            [0.0, f64::INFINITY], // skip
        ];
        let r = f.bootstrap(pts).unwrap();
        assert_eq!(r.points_ingested, 3);
        assert_eq!(r.points_skipped, 2);
        assert_eq!(f.updates_seen(), 3);
        assert_eq!(r.final_observations, 3);
    }

    #[test]
    fn thresholded_bootstrap_makes_detector_ready() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut d = ThresholdedForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(64)
            .min_observations(32)
            .min_threshold(0.1)
            .seed(42)
            .build()
            .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let history: Vec<[f64; 4]> = (0..512)
            .map(|_| {
                [
                    rng.random::<f64>() * 0.1,
                    rng.random::<f64>() * 0.1,
                    rng.random::<f64>() * 0.1,
                    rng.random::<f64>() * 0.1,
                ]
            })
            .collect();

        let r = d.bootstrap(history).unwrap();
        assert_eq!(r.points_ingested, 512);
        assert!(r.is_hot());
        assert!(r.final_observations >= 32, "should be past warmup");
        assert!(r.final_threshold > 0.1, "threshold should be adaptive");

        // First live probe produces a ready verdict, no warming-up.
        let verdict = d.score_only(&[0.05, 0.05, 0.05, 0.05]).unwrap();
        assert!(verdict.ready(), "detector must be hot after bootstrap");
    }

    #[test]
    fn thresholded_bootstrap_detects_outlier_immediately() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut d = ThresholdedForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(64)
            .min_observations(32)
            .min_threshold(0.1)
            .seed(3)
            .build()
            .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let history: Vec<[f64; 4]> = (0..512)
            .map(|_| {
                [
                    rng.random::<f64>() * 0.1,
                    rng.random::<f64>() * 0.1,
                    rng.random::<f64>() * 0.1,
                    rng.random::<f64>() * 0.1,
                ]
            })
            .collect();
        d.bootstrap(history).unwrap();
        let outlier = d.process([50.0, 50.0, 50.0, 50.0]).unwrap();
        assert!(outlier.ready());
        assert!(outlier.is_anomaly());
        assert!(outlier.grade() > 0.0);
    }

    #[test]
    fn thresholded_bootstrap_skips_non_finite() {
        let mut d = ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .min_observations(4)
            .seed(1)
            .build()
            .unwrap();
        let pts: Vec<[f64; 2]> = vec![
            [0.0, 0.0],
            [f64::NAN, 0.0],
            [0.5, 0.5],
            [f64::NEG_INFINITY, 1.0],
        ];
        let r = d.bootstrap(pts).unwrap();
        assert_eq!(r.points_ingested, 2);
        assert_eq!(r.points_skipped, 2);
    }
}
