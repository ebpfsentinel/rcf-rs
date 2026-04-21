//! Shadow-forest drift recovery — wraps a live
//! [`crate::RandomCutForest`] plus an optional shadow that warms
//! on the post-drift stream, then atomically replaces the primary
//! once the shadow has seen enough observations.
//!
//! Pairs with any upstream drift trigger — [`crate::AdwinDetector`]
//! on the score stream, [`crate::FeatureDriftDetector`] PSI alert
//! level, or [`crate::MetaDriftDetector`] CUSUM fire. The trigger
//! logic lives outside this type; callers call
//! [`DriftAwareForest::on_drift`] when they want a shadow to spawn.
//!
//! ```ignore
//! use rcf_rs::{AdwinDetector, DriftAwareForest, DriftRecoveryConfig, ForestBuilder};
//!
//! let builder = ForestBuilder::<16>::new()
//!     .num_trees(100)
//!     .sample_size(256)
//!     .seed(42);
//! let mut detector = DriftAwareForest::new(
//!     builder,
//!     DriftRecoveryConfig::default(),
//! )?;
//! let mut adwin = AdwinDetector::default_bounded();
//!
//! for point in stream_of_points {
//!     detector.update(point)?;
//!     let score = detector.score(&point)?;
//!     if adwin.update(f64::from(score)) {
//!         detector.on_drift()?;           // spawn shadow
//!     }
//! }
//! # Ok::<(), rcf_rs::RcfError>(())
//! ```

#![cfg(feature = "std")]

use std::sync::Arc;

use crate::config::ForestBuilder;
use crate::domain::{AnomalyScore, DiVector};
use crate::error::RcfResult;
use crate::forest::RandomCutForest;
use crate::metrics::{MetricsSink, default_sink, names};

/// Policy parameters for the shadow swap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DriftRecoveryConfig {
    /// Shadow must ingest at least this many observations before it
    /// replaces the primary. Sized to fill the reservoir twice over
    /// so the new baseline is stable when the swap lands.
    pub shadow_warmup: u64,
    /// Minimum observations a newly-swapped primary has to see
    /// before [`DriftAwareForest::on_drift`] can spawn another
    /// shadow. Prevents flap-loops from a noisy trigger.
    pub min_primary_age: u64,
}

impl Default for DriftRecoveryConfig {
    fn default() -> Self {
        Self {
            shadow_warmup: 1_024,
            min_primary_age: 512,
        }
    }
}

/// Stateful shadow forest — accumulates post-drift observations
/// alongside the primary, then replaces it on warmup completion.
#[derive(Debug)]
struct ShadowState<const D: usize> {
    /// Parallel forest being warmed on recent traffic.
    forest: RandomCutForest<D>,
    /// Observations ingested into the shadow since it was spawned.
    seen: u64,
}

/// Forest wrapper that handles drift recovery via a shadow swap.
///
/// The primary forest handles every `score` / `score_many` call —
/// this type is a drop-in facade for the hot-path. Drift recovery
/// is entirely opt-in through [`Self::on_drift`]; without a
/// trigger call the wrapper behaves exactly like a bare
/// [`RandomCutForest`].
#[derive(Debug)]
pub struct DriftAwareForest<const D: usize> {
    /// Live forest — every `score` reads from here.
    primary: RandomCutForest<D>,
    /// Optional shadow — `Some` between `on_drift` and the swap.
    shadow: Option<ShadowState<D>>,
    /// Observations the current primary has ingested.
    primary_age: u64,
    /// Builder template used to spawn fresh shadows.
    builder: ForestBuilder<D>,
    /// Recovery policy.
    config: DriftRecoveryConfig,
    /// Lifetime count of completed shadow swaps — observability.
    swaps: u64,
    /// Observability sink.
    metrics: Arc<dyn MetricsSink>,
}

impl<const D: usize> DriftAwareForest<D> {
    /// Build a drift-aware wrapper from a prepared [`ForestBuilder`].
    /// The builder is cloned internally to spawn shadow forests on
    /// demand.
    ///
    /// # Errors
    ///
    /// Propagates [`ForestBuilder::build`] failures.
    pub fn new(builder: ForestBuilder<D>, config: DriftRecoveryConfig) -> RcfResult<Self> {
        let primary = builder.clone().build()?;
        Ok(Self {
            primary,
            shadow: None,
            primary_age: 0,
            builder,
            config,
            swaps: 0,
            metrics: default_sink(),
        })
    }

    /// Install a metrics sink — `on_drift` / swap emit counters,
    /// shadow activity emits a gauge.
    #[must_use]
    pub fn with_metrics_sink(mut self, sink: Arc<dyn MetricsSink>) -> Self {
        self.metrics = sink;
        self
    }

    /// Read-only handle to the installed sink.
    #[must_use]
    pub fn metrics_sink(&self) -> &Arc<dyn MetricsSink> {
        &self.metrics
    }

    /// Read-only access to the live primary forest.
    #[must_use]
    pub fn forest(&self) -> &RandomCutForest<D> {
        &self.primary
    }

    /// Whether a shadow is currently warming.
    #[must_use]
    pub fn is_recovering(&self) -> bool {
        self.shadow.is_some()
    }

    /// Number of observations the shadow has seen since spawn
    /// (`0` when no shadow is active).
    #[must_use]
    pub fn shadow_progress(&self) -> u64 {
        self.shadow.as_ref().map_or(0, |s| s.seen)
    }

    /// Observations the current primary has ingested since the
    /// last shadow swap (or construction).
    #[must_use]
    pub fn primary_age(&self) -> u64 {
        self.primary_age
    }

    /// Lifetime shadow swaps completed.
    #[must_use]
    pub fn swaps_total(&self) -> u64 {
        self.swaps
    }

    /// Policy knobs.
    #[must_use]
    pub fn config(&self) -> DriftRecoveryConfig {
        self.config
    }

    /// Fold `point` into the primary and, when present, into the
    /// shadow. Triggers an atomic swap if the shadow has reached
    /// `config.shadow_warmup`.
    ///
    /// # Errors
    ///
    /// Propagates [`RandomCutForest::update`] failures from either
    /// path; on shadow error the shadow is discarded so the
    /// primary stays healthy.
    pub fn update(&mut self, point: [f64; D]) -> RcfResult<()> {
        self.primary.update(point)?;
        self.primary_age = self.primary_age.saturating_add(1);

        if let Some(shadow) = self.shadow.as_mut() {
            match shadow.forest.update(point) {
                Ok(()) => {
                    shadow.seen = shadow.seen.saturating_add(1);
                }
                Err(e) => {
                    // Drop the shadow — primary path must stay
                    // clean. The caller can re-arm via on_drift.
                    self.shadow = None;
                    self.metrics
                        .set_gauge(names::DRIFT_AWARE_SHADOW_ACTIVE, 0.0);
                    return Err(e);
                }
            }
            if self
                .shadow
                .as_ref()
                .is_some_and(|s| s.seen >= self.config.shadow_warmup)
            {
                self.swap_shadow_into_primary();
            }
        }
        Ok(())
    }

    /// Score `point` against the primary. Shadow is not consulted
    /// — scoring stays on the stable baseline until the swap lands.
    ///
    /// # Errors
    ///
    /// Propagates [`RandomCutForest::score`] failures.
    pub fn score(&self, point: &[f64; D]) -> RcfResult<AnomalyScore> {
        self.primary.score(point)
    }

    /// Attribution against the primary.
    ///
    /// # Errors
    ///
    /// Propagates [`RandomCutForest::attribution`] failures.
    pub fn attribution(&self, point: &[f64; D]) -> RcfResult<DiVector> {
        self.primary.attribution(point)
    }

    /// Spawn a shadow forest to train on the post-drift stream.
    /// No-op when a shadow is already warming, or when the primary
    /// has not yet reached `config.min_primary_age` (anti-flap
    /// guard).
    ///
    /// # Errors
    ///
    /// Propagates [`ForestBuilder::build`] failures.
    pub fn on_drift(&mut self) -> RcfResult<bool> {
        if self.shadow.is_some() {
            return Ok(false);
        }
        if self.primary_age < self.config.min_primary_age {
            return Ok(false);
        }
        let fresh = self.builder.clone().build()?;
        self.shadow = Some(ShadowState {
            forest: fresh,
            seen: 0,
        });
        self.metrics
            .inc_counter(names::DRIFT_AWARE_ON_DRIFT_TOTAL, 1);
        self.metrics
            .set_gauge(names::DRIFT_AWARE_SHADOW_ACTIVE, 1.0);
        Ok(true)
    }

    /// Cancel the current shadow (if any) without swapping. Used
    /// when the trigger retracts its alert or the operator wants
    /// to abort recovery.
    pub fn abort_shadow(&mut self) {
        self.shadow = None;
        self.metrics
            .set_gauge(names::DRIFT_AWARE_SHADOW_ACTIVE, 0.0);
    }

    /// Promote shadow → primary. Callers never invoke this
    /// directly — [`Self::update`] handles the swap once the
    /// shadow reaches `shadow_warmup`.
    fn swap_shadow_into_primary(&mut self) {
        if let Some(shadow) = self.shadow.take() {
            self.primary = shadow.forest;
            self.primary_age = shadow.seen;
            self.swaps = self.swaps.saturating_add(1);
            self.metrics.inc_counter(names::DRIFT_AWARE_SWAPS_TOTAL, 1);
            self.metrics
                .set_gauge(names::DRIFT_AWARE_SHADOW_ACTIVE, 0.0);
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;

    fn small_builder() -> ForestBuilder<2> {
        ForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(64)
            .seed(2026)
    }

    #[test]
    fn fresh_wrapper_has_no_shadow() {
        let d = DriftAwareForest::new(small_builder(), DriftRecoveryConfig::default()).unwrap();
        assert!(!d.is_recovering());
        assert_eq!(d.shadow_progress(), 0);
        assert_eq!(d.swaps_total(), 0);
    }

    #[test]
    fn on_drift_requires_min_primary_age() {
        let mut d = DriftAwareForest::new(
            small_builder(),
            DriftRecoveryConfig {
                shadow_warmup: 10,
                min_primary_age: 50,
            },
        )
        .unwrap();
        // Only a handful of updates — below min_primary_age → no-op.
        for _ in 0..10 {
            d.update([0.1, 0.2]).unwrap();
        }
        assert!(!d.on_drift().unwrap());
        assert!(!d.is_recovering());
    }

    #[test]
    fn on_drift_spawns_shadow_when_primary_mature() {
        let mut d = DriftAwareForest::new(
            small_builder(),
            DriftRecoveryConfig {
                shadow_warmup: 100,
                min_primary_age: 50,
            },
        )
        .unwrap();
        for _ in 0..60 {
            d.update([0.1, 0.2]).unwrap();
        }
        assert!(d.on_drift().unwrap());
        assert!(d.is_recovering());
        assert_eq!(d.shadow_progress(), 0);
        // A second on_drift during recovery is a no-op.
        assert!(!d.on_drift().unwrap());
    }

    #[test]
    fn shadow_promotes_after_warmup() {
        let mut d = DriftAwareForest::new(
            small_builder(),
            DriftRecoveryConfig {
                shadow_warmup: 30,
                min_primary_age: 10,
            },
        )
        .unwrap();
        for _ in 0..20 {
            d.update([0.1, 0.2]).unwrap();
        }
        d.on_drift().unwrap();
        for i in 0..30 {
            let v = f64::from(i) * 0.01;
            d.update([v, v + 0.5]).unwrap();
        }
        assert!(!d.is_recovering());
        assert_eq!(d.swaps_total(), 1);
        // Primary age should reset to the shadow's warm-up count.
        assert_eq!(d.primary_age(), 30);
    }

    #[test]
    fn abort_shadow_discards_recovery() {
        let mut d = DriftAwareForest::new(
            small_builder(),
            DriftRecoveryConfig {
                shadow_warmup: 100,
                min_primary_age: 10,
            },
        )
        .unwrap();
        for _ in 0..20 {
            d.update([0.1, 0.2]).unwrap();
        }
        d.on_drift().unwrap();
        assert!(d.is_recovering());
        d.abort_shadow();
        assert!(!d.is_recovering());
        assert_eq!(d.swaps_total(), 0);
    }

    #[test]
    fn score_uses_primary_forest_always() {
        let mut d = DriftAwareForest::new(small_builder(), DriftRecoveryConfig::default()).unwrap();
        for i in 0..100 {
            let v = f64::from(i) * 0.01;
            d.update([v, v + 0.5]).unwrap();
        }
        // Even during recovery the public `score` reads from the
        // primary (stable baseline).
        let s_before: f64 = d.score(&[0.5, 1.0]).unwrap().into();
        d.on_drift().unwrap();
        let s_during: f64 = d.score(&[0.5, 1.0]).unwrap().into();
        assert_eq!(s_before, s_during);
    }
}
