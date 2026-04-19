//! Adaptive-threshold wrapper around [`RandomCutForest`].
//!
//! [`ThresholdedForest`] composes a [`RandomCutForest`] with an EMA of
//! the anomaly-score stream ([`EmaStats`]) and derives a continuously
//! updated threshold:
//!
//! ```text
//! threshold  = max(min_threshold, mean + z_factor · stddev)
//! is_anomaly = ready && score > threshold
//! grade      = clamp01( (score − threshold) / (z_factor · stddev) )  if ready
//!            | 0.0                                                   otherwise
//! ```
//!
//! where `ready` means the detector has seen at least
//! `min_observations` points *and* the running stddev is strictly
//! positive. The `ready` flag guards the cold-start period so callers
//! never see spurious anomaly verdicts on the first few points (the
//! EMA has not yet converged and the bootstrap variance is exactly
//! zero).
//!
//! # Scoring protocol
//!
//! `process(point)` evaluates the point *before* inserting it into
//! the forest. This avoids a self-referential bias where the freshly
//! inserted point would be scored against a forest that already
//! contains it — always shallow, always low-anomaly. The stats EMA
//! is updated with the pre-insert score so the threshold adapts to
//! the distribution of scores the forest would assign to unseen
//! points.

use crate::config::RcfConfig;
use crate::domain::point::ensure_finite;
use crate::domain::{AnomalyScore, DiVector};
use crate::error::{RcfError, RcfResult};
use crate::forest::RandomCutForest;
use crate::thresholded::config::ThresholdedConfig;
use crate::thresholded::grade::AnomalyGrade;
use crate::thresholded::stats::EmaStats;

/// Adaptive-threshold detector composed of a [`RandomCutForest`] plus
/// a running EMA of the anomaly-score stream.
///
/// Instantiate via [`crate::ThresholdedForestBuilder`]. The type
/// parameter `D` is the per-point dimensionality, pinned at compile
/// time exactly like the bare [`RandomCutForest`].
///
/// # Examples
///
/// ```
/// use rcf_rs::ThresholdedForestBuilder;
///
/// let mut detector = ThresholdedForestBuilder::<2>::new()
///     .num_trees(50)
///     .sample_size(64)
///     .min_observations(4)
///     .seed(42)
///     .build()
///     .unwrap();
/// for i in 0..64 {
///     let v = f64::from(i) * 0.01;
///     let _ = detector.process([v, v + 0.5]).unwrap();
/// }
/// let verdict = detector.process([10.0, 10.0]).unwrap();
/// assert!(verdict.ready());
/// ```
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThresholdedForest<const D: usize> {
    /// Underlying random cut forest.
    forest: RandomCutForest<D>,
    /// Threshold-layer configuration.
    thresholded: ThresholdedConfig,
    /// Running mean/variance of the per-point anomaly scores.
    stats: EmaStats,
    /// Observability sink for threshold-layer events. Distinct from
    /// the inner forest's sink so wrapper-only metrics
    /// (`rcf_process_total`, `rcf_anomalies_fired_total`,
    /// `rcf_threshold_current`, `rcf_grade`) do not duplicate the
    /// forest's counters.
    #[cfg(feature = "std")]
    #[cfg_attr(feature = "serde", serde(skip, default = "crate::metrics::default_sink"))]
    metrics: std::sync::Arc<dyn crate::metrics::MetricsSink>,
}

impl<const D: usize> ThresholdedForest<D> {
    /// Low-level constructor used by [`crate::ThresholdedForestBuilder::build`].
    ///
    /// Both `forest` and `thresholded` are expected to have been
    /// validated upstream; this function only wires them together and
    /// constructs the EMA.
    ///
    /// # Errors
    ///
    /// Propagates [`EmaStats::new`] failures (non-finite decay etc.).
    pub fn from_parts(
        forest: RandomCutForest<D>,
        thresholded: ThresholdedConfig,
    ) -> RcfResult<Self> {
        thresholded.validate()?;
        let stats = EmaStats::new(thresholded.score_decay)?;
        Ok(Self {
            forest,
            thresholded,
            stats,
            #[cfg(feature = "std")]
            metrics: crate::metrics::default_sink(),
        })
    }

    /// Install a [`crate::MetricsSink`] — every subsequent
    /// `process` / `score_only` call emits counters and histograms
    /// into it. Does **not** propagate to the underlying forest;
    /// install on the forest separately if you also want low-level
    /// `rcf_updates_total` / `rcf_score` / `rcf_deletes_total`
    /// events.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_metrics_sink(
        mut self,
        sink: std::sync::Arc<dyn crate::metrics::MetricsSink>,
    ) -> Self {
        self.metrics = sink;
        self
    }

    /// Read-only handle to the installed threshold-layer sink.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn metrics_sink(&self) -> &std::sync::Arc<dyn crate::metrics::MetricsSink> {
        &self.metrics
    }

    /// Read-only access to the underlying forest.
    #[must_use]
    pub fn forest(&self) -> &RandomCutForest<D> {
        &self.forest
    }

    /// Read-only access to the forest configuration.
    #[must_use]
    pub fn forest_config(&self) -> &RcfConfig {
        self.forest.config()
    }

    /// Threshold-layer configuration.
    #[must_use]
    pub fn thresholded_config(&self) -> &ThresholdedConfig {
        &self.thresholded
    }

    /// Running statistics of the anomaly-score stream.
    #[must_use]
    pub fn stats(&self) -> &EmaStats {
        &self.stats
    }

    /// Current adaptive threshold. Clamped to the configured floor
    /// whenever the detector has not yet accumulated enough
    /// observations to trust the running stddev.
    #[must_use]
    pub fn current_threshold(&self) -> f64 {
        let ready = self.stats.observations() >= self.thresholded.min_observations
            && self.stats.stddev() > 0.0;
        if ready {
            let adaptive = self.stats.mean() + self.thresholded.z_factor * self.stats.stddev();
            adaptive.max(self.thresholded.min_threshold)
        } else {
            self.thresholded.min_threshold
        }
    }

    /// Score `point` against the *current* forest, grade it against
    /// the adaptive threshold, insert it into the forest, then fold
    /// the score into the running statistics.
    ///
    /// The first call returns a warming-up verdict (`ready = false`,
    /// `is_anomaly = false`) because the forest holds no leaves yet.
    /// Subsequent calls within the `min_observations` warmup window
    /// also return `ready = false`.
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] when the point contains a non-finite
    ///   component.
    /// - Any error bubbled up from [`RandomCutForest::update`] or
    ///   [`RandomCutForest::score`].
    pub fn process(&mut self, point: [f64; D]) -> RcfResult<AnomalyGrade> {
        ensure_finite(&point)?;

        // Score against the forest BEFORE the insert — a post-insert
        // score would bias the result toward "seen" for the freshly
        // inserted point and distort the threshold-driving statistics.
        let score = match self.forest.score(&point) {
            Ok(s) => s,
            Err(RcfError::EmptyForest) => {
                // Cold start: no leaves yet. Record the insert,
                // emit a warming-up verdict, do not update stats.
                self.forest.update(point)?;
                let verdict = AnomalyGrade::new(
                    AnomalyScore::new(0.0)?,
                    self.thresholded.min_threshold,
                    0.0,
                    false,
                    false,
                )?;
                #[cfg(feature = "std")]
                self.emit_process_metrics(&verdict);
                return Ok(verdict);
            }
            Err(other) => return Err(other),
        };

        self.forest.update(point)?;

        let verdict = self.grade_from_score(score)?;
        self.stats.update(f64::from(score));
        #[cfg(feature = "std")]
        self.emit_process_metrics(&verdict);
        Ok(verdict)
    }

    /// Score `point` and grade it without touching the forest or the
    /// running statistics. Useful for re-evaluating a point against a
    /// snapshot of the model without contaminating the training
    /// stream.
    ///
    /// On an empty forest (no points have been inserted yet), returns
    /// a warming-up verdict (`ready = false`, `is_anomaly = false`)
    /// rather than an error — mirrors [`Self::process`]'s cold-start
    /// handling so callers can query the detector during the warmup
    /// window without special-casing the empty case.
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] when the point contains a non-finite
    ///   component.
    /// - Any other error bubbled up from [`RandomCutForest::score`].
    pub fn score_only(&self, point: &[f64; D]) -> RcfResult<AnomalyGrade> {
        match self.forest.score(point) {
            Ok(score) => self.grade_from_score(score),
            Err(RcfError::EmptyForest) => AnomalyGrade::new(
                AnomalyScore::new(0.0)?,
                self.thresholded.min_threshold,
                0.0,
                false,
                false,
            ),
            Err(other) => Err(other),
        }
    }

    /// Compute the per-feature attribution of `point`'s anomaly score
    /// against the underlying forest. Forwarded to
    /// [`RandomCutForest::attribution`]; the threshold layer has no
    /// bearing on attribution.
    ///
    /// # Errors
    ///
    /// Same as [`RandomCutForest::attribution`].
    pub fn attribution(&self, point: &[f64; D]) -> RcfResult<DiVector> {
        self.forest.attribution(point)
    }

    /// Bulk-score a batch of points without touching the threshold
    /// layer's stats. Returns [`AnomalyGrade`]s graded against the
    /// current adaptive threshold — identical to what
    /// [`Self::score_only`] would emit per point, but parallelised
    /// across the batch via rayon when the `parallel` feature is
    /// enabled.
    ///
    /// # Errors
    ///
    /// Propagates any [`Self::score_only`] error hit while
    /// processing the batch.
    pub fn score_only_many(&self, points: &[[f64; D]]) -> RcfResult<Vec<AnomalyGrade>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            points
                .par_iter()
                .map(|p| self.score_only(p))
                .collect::<RcfResult<Vec<_>>>()
        }
        #[cfg(not(feature = "parallel"))]
        {
            points.iter().map(|p| self.score_only(p)).collect()
        }
    }

    /// Bulk per-feature attribution. Delegates to
    /// [`RandomCutForest::attribution_many`].
    ///
    /// # Errors
    ///
    /// Same as [`RandomCutForest::attribution_many`].
    pub fn attribution_many(&self, points: &[[f64; D]]) -> RcfResult<Vec<DiVector>> {
        self.forest.attribution_many(points)
    }

    /// Imputation-like forensic baseline. Delegates to
    /// [`RandomCutForest::forensic_baseline`].
    ///
    /// # Errors
    ///
    /// Same as [`RandomCutForest::forensic_baseline`].
    pub fn forensic_baseline(
        &self,
        point: &[f64; D],
    ) -> RcfResult<crate::forensic::ForensicBaseline<D>> {
        self.forest.forensic_baseline(point)
    }

    /// Bulk early-termination scoring. Delegates to
    /// [`RandomCutForest::score_many_early_term`] — the threshold
    /// layer does not alter the scoring path.
    ///
    /// # Errors
    ///
    /// Same as [`RandomCutForest::score_many_early_term`].
    pub fn score_many_early_term(
        &self,
        points: &[[f64; D]],
        config: crate::early_term::EarlyTermConfig,
    ) -> RcfResult<Vec<crate::early_term::EarlyTermScore>> {
        self.forest.score_many_early_term(points, config)
    }

    /// Early-termination variant of the scoring path — delegates to
    /// [`RandomCutForest::score_early_term`]. Does not update the
    /// thresholded layer's stats (this is a read path, not a
    /// training path).
    ///
    /// # Errors
    ///
    /// Same as [`RandomCutForest::score_early_term`].
    pub fn score_early_term(
        &self,
        point: &[f64; D],
        config: crate::early_term::EarlyTermConfig,
    ) -> RcfResult<crate::early_term::EarlyTermScore> {
        self.forest.score_early_term(point, config)
    }

    /// Drop every statistic and warm-up sample. The underlying forest
    /// is left untouched — callers who want a full reset should
    /// rebuild via the builder. Used by tests and by callers that
    /// want to re-enter a warmup phase after a major regime change.
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    /// Retract a previously-observed point from the underlying forest
    /// by its `point_idx`. Delegates to
    /// [`RandomCutForest::delete`] — the threshold layer's stats are
    /// left untouched (they already reflect the score that was
    /// emitted when the point was processed).
    ///
    /// # Errors
    ///
    /// Same as [`RandomCutForest::delete`].
    pub fn delete(&mut self, point_idx: usize) -> RcfResult<bool> {
        self.forest.delete(point_idx)
    }

    /// Retract every point whose stored value bit-matches `point`.
    /// Delegates to [`RandomCutForest::delete_by_value`].
    ///
    /// # Errors
    ///
    /// Same as [`RandomCutForest::delete_by_value`].
    pub fn delete_by_value(&mut self, point: &[f64; D]) -> RcfResult<usize> {
        self.forest.delete_by_value(point)
    }

    /// Same as [`Self::process`] but returns the `point_idx` the
    /// underlying forest assigned to the fresh observation, paired
    /// with the usual graded verdict. Callers that want to later
    /// retract the observation via [`Self::delete`] should store the
    /// index from this call.
    ///
    /// # Errors
    ///
    /// Same as [`Self::process`].
    pub fn process_indexed(
        &mut self,
        point: [f64; D],
    ) -> RcfResult<(usize, AnomalyGrade)> {
        ensure_finite(&point)?;

        let score = match self.forest.score(&point) {
            Ok(s) => s,
            Err(RcfError::EmptyForest) => {
                // Cold start: bypass the normal scoring path, mirror
                // `process`'s warming-up verdict, and return the
                // fresh point_idx from the underlying insert.
                let idx = self.forest.update_indexed(point)?;
                let grade = AnomalyGrade::new(
                    AnomalyScore::new(0.0)?,
                    self.thresholded.min_threshold,
                    0.0,
                    false,
                    false,
                )?;
                #[cfg(feature = "std")]
                self.emit_process_metrics(&grade);
                return Ok((idx, grade));
            }
            Err(other) => return Err(other),
        };

        let idx = self.forest.update_indexed(point)?;
        let verdict = self.grade_from_score(score)?;
        self.stats.update(f64::from(score));
        #[cfg(feature = "std")]
        self.emit_process_metrics(&verdict);
        Ok((idx, verdict))
    }

    /// Timestamped variant of [`Self::process`] — tags the freshly
    /// inserted point with `timestamp` so callers can later prune
    /// history via [`RandomCutForest::delete_before`]. Returns the
    /// same graded verdict as [`Self::process`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::process`].
    pub fn process_at(
        &mut self,
        point: [f64; D],
        timestamp: u64,
    ) -> RcfResult<AnomalyGrade> {
        let (_, verdict) = self.process_indexed_at(point, timestamp)?;
        Ok(verdict)
    }

    /// Timestamped variant of [`Self::process_indexed`] — records
    /// the caller-supplied `timestamp` against the fresh `point_idx`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::process_indexed`].
    pub fn process_indexed_at(
        &mut self,
        point: [f64; D],
        timestamp: u64,
    ) -> RcfResult<(usize, AnomalyGrade)> {
        let (idx, verdict) = self.process_indexed(point)?;
        // process_indexed may have called update_indexed, so the
        // side-map entry we tag here is attached to the correct
        // fresh point_idx — even when the call path went through
        // the cold-start warming-up branch.
        if self.forest.point_store().ref_count(idx) > 0 {
            self.forest.set_point_timestamp(idx, timestamp);
        }
        Ok((idx, verdict))
    }

    /// Retract every point whose timestamp is strictly less than
    /// `cutoff`. Forwards to [`RandomCutForest::delete_before`].
    ///
    /// # Errors
    ///
    /// Propagates [`RandomCutForest::delete_before`] failures.
    pub fn delete_before(&mut self, cutoff: u64) -> RcfResult<usize> {
        self.forest.delete_before(cutoff)
    }

    /// Emit the counters / gauges / histograms associated with a
    /// completed `process` call. Called once per public process
    /// entry so cold-start warming-up verdicts are counted too.
    #[cfg(feature = "std")]
    fn emit_process_metrics(&self, verdict: &AnomalyGrade) {
        use crate::metrics::names;
        self.metrics.inc_counter(names::PROCESS_TOTAL, 1);
        self.metrics
            .observe_histogram(names::GRADE_OBSERVATION, verdict.grade());
        if verdict.is_anomaly() {
            self.metrics.inc_counter(names::ANOMALIES_FIRED_TOTAL, 1);
        }
        self.metrics
            .set_gauge(names::THRESHOLD_CURRENT, self.current_threshold());
    }

    /// Translate a raw anomaly score into a graded verdict using the
    /// current running statistics.
    fn grade_from_score(&self, score: AnomalyScore) -> RcfResult<AnomalyGrade> {
        let stddev = self.stats.stddev();
        let ready = self.stats.observations() >= self.thresholded.min_observations && stddev > 0.0;

        if !ready {
            return AnomalyGrade::new(score, self.thresholded.min_threshold, 0.0, false, false);
        }

        let adaptive = self.stats.mean() + self.thresholded.z_factor * stddev;
        let threshold = adaptive.max(self.thresholded.min_threshold);
        let raw = f64::from(score);

        if raw <= threshold {
            return AnomalyGrade::new(score, threshold, 0.0, false, true);
        }

        // Grade: linearly scaled between `threshold` (0) and
        // `threshold + z_factor · stddev` (1). This keeps the grade
        // interpretable in "sigmas above threshold" units regardless
        // of the absolute score magnitude.
        let span = self.thresholded.z_factor * stddev;
        let grade = if span > 0.0 {
            ((raw - threshold) / span).clamp(0.0, 1.0)
        } else {
            1.0
        };
        AnomalyGrade::new(score, threshold, grade, true, true)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert bounds on closed-form quantities.
mod tests {
    use super::*;
    use crate::thresholded::config::ThresholdedForestBuilder;

    fn detector<const D: usize>(min_obs: u64) -> ThresholdedForest<D> {
        ThresholdedForestBuilder::<D>::new()
            .num_trees(50)
            .sample_size(64)
            .min_observations(min_obs)
            .min_threshold(0.0)
            .seed(42)
            .build()
            .unwrap()
    }

    #[test]
    fn cold_start_emits_warming_up_verdict() {
        let mut d = detector::<2>(8);
        let v = d.process([0.1, 0.2]).unwrap();
        assert!(!v.ready());
        assert!(!v.is_anomaly());
        assert_eq!(v.grade(), 0.0);
    }

    #[test]
    fn warmup_period_always_not_ready() {
        let mut d = detector::<2>(32);
        for i in 0..20 {
            let v = f64::from(i) * 0.01;
            let verdict = d.process([v, v + 0.5]).unwrap();
            assert!(!verdict.ready(), "should still be warming up at i={i}");
        }
    }

    #[test]
    fn becomes_ready_after_min_observations() {
        let mut d = detector::<2>(8);
        for i in 0..64 {
            let v = f64::from(i) * 0.01;
            d.process([v, v + 0.5]).unwrap();
        }
        // Probe an existing-ish point; observations > min_obs and
        // stddev should be > 0 after 64 updates.
        let verdict = d.process([0.64, 1.14]).unwrap();
        assert!(verdict.ready());
    }

    #[test]
    fn rejects_non_finite_point() {
        let mut d = detector::<2>(8);
        assert!(matches!(
            d.process([f64::NAN, 0.0]).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn score_only_does_not_mutate_stats() {
        let mut d = detector::<2>(4);
        for i in 0..32 {
            let v = f64::from(i) * 0.01;
            d.process([v, v + 0.5]).unwrap();
        }
        let obs_before = d.stats().observations();
        let _ = d.score_only(&[10.0, 10.0]).unwrap();
        assert_eq!(d.stats().observations(), obs_before);
    }

    #[test]
    fn outlier_grades_above_cluster_member() {
        let mut d = detector::<2>(8);
        for i in 0..128 {
            let v = f64::from(i) * 0.01;
            d.process([v, v + 0.5]).unwrap();
        }
        let cluster = d.score_only(&[0.3, 0.8]).unwrap();
        let outlier = d.score_only(&[20.0, 20.0]).unwrap();
        assert!(f64::from(outlier.score()) > f64::from(cluster.score()));
    }

    #[test]
    fn current_threshold_respects_min_floor_during_warmup() {
        let d = detector::<2>(16);
        assert_eq!(d.current_threshold(), 0.0);
    }

    #[test]
    fn current_threshold_above_floor_when_stddev_positive() {
        use rand::{Rng, SeedableRng};
        // Use a non-zero min_threshold and confirm the adaptive
        // threshold rises above it once stats converge.
        let mut d = ThresholdedForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(64)
            .min_observations(8)
            .min_threshold(0.01)
            .seed(3)
            .build()
            .unwrap();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(17);
        for _ in 0..256 {
            let a: f64 = rng.random();
            let b: f64 = rng.random();
            d.process([a, b]).unwrap();
        }
        assert!(d.current_threshold() >= 0.01);
    }

    #[test]
    fn attribution_forwards_to_forest() {
        let mut d = detector::<2>(4);
        for i in 0..32 {
            let v = f64::from(i) * 0.01;
            d.process([v, v + 0.5]).unwrap();
        }
        let di = d.attribution(&[10.0, 10.0]).unwrap();
        assert_eq!(di.dim(), 2);
    }

    #[test]
    fn reset_stats_sends_detector_back_to_warmup() {
        let mut d = detector::<2>(4);
        for i in 0..32 {
            let v = f64::from(i) * 0.01;
            d.process([v, v + 0.5]).unwrap();
        }
        assert!(d.stats().observations() > 0);
        d.reset_stats();
        assert_eq!(d.stats().observations(), 0);
        // Next verdict should be warming-up again.
        let v = d.process([0.5, 1.0]).unwrap();
        assert!(!v.ready());
    }

    #[test]
    fn accessors_expose_inner_state() {
        let d = detector::<4>(8);
        assert_eq!(d.forest().num_trees(), 50);
        assert_eq!(d.forest_config().sample_size, 64);
        assert_eq!(d.thresholded_config().min_observations, 8);
        assert_eq!(d.stats().observations(), 0);
    }
}
