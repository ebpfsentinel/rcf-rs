//! Metrics sink abstraction for observability wiring.
//!
//! [`MetricsSink`] is a narrow trait exposed by the crate so
//! long-running agents can drain counters / gauges / histograms
//! into `Prometheus`, `StatsD`, `OpenTelemetry`, or any other aggregator
//! without the rcf-rs internals pulling a concrete metrics crate.
//! Three event types cover everything the forest / thresholded /
//! pool / meta-drift layers emit:
//!
//! - **counter** — monotonically increasing event tallies
//!   (`rcf_updates_total`, `rcf_anomalies_fired_total`, …).
//! - **gauge** — point-in-time values (`rcf_tenants_resident`,
//!   `rcf_threshold_current`, …).
//! - **histogram observation** — an `f64` sample the sink should
//!   bucket on its own (`rcf_score`, `rcf_grade`, …).
//!
//! Implementations must be `Send + Sync` so the sink can be shared
//! across threads — every detector type holds an `Arc<dyn
//! MetricsSink>`. [`NoopSink`] is the default zero-cost fallback
//! (every call is a `#[inline]` no-op).
//!
//! # Wiring
//!
//! Every detector exposes a consuming builder method
//! `.with_metrics_sink(Arc<dyn MetricsSink>)` that installs a sink.
//! For [`crate::TenantForestPool`] the sink applies to the pool
//! itself (tenant evictions, resident count); per-tenant detectors
//! inherit nothing automatically — callers who want per-tenant
//! observability should install a sink on each detector through the
//! pool's factory closure.

#[cfg(feature = "std")]
use std::sync::Arc;

/// Narrow observability interface exposed by rcf-rs detectors.
pub trait MetricsSink: Send + Sync + core::fmt::Debug {
    /// Increment a named monotonic counter by `value`.
    fn inc_counter(&self, name: &str, value: u64);
    /// Set a named gauge to `value`.
    fn set_gauge(&self, name: &str, value: f64);
    /// Record a histogram observation of `value` under `name`.
    fn observe_histogram(&self, name: &str, value: f64);
}

/// Zero-cost [`MetricsSink`] implementation — every call is an
/// inlined no-op. Default sink every detector ships with.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopSink;

impl MetricsSink for NoopSink {
    #[inline]
    fn inc_counter(&self, _name: &str, _value: u64) {}
    #[inline]
    fn set_gauge(&self, _name: &str, _value: f64) {}
    #[inline]
    fn observe_histogram(&self, _name: &str, _value: f64) {}
}

/// Build an `Arc<dyn MetricsSink>` backed by a fresh [`NoopSink`].
/// Used by the crate's own default-sink paths.
#[cfg(feature = "std")]
#[must_use]
pub fn default_sink() -> Arc<dyn MetricsSink> {
    Arc::new(NoopSink)
}

/// In-memory testing sink that records every observation into
/// caller-inspectable maps. Useful in tests and benches that want
/// to assert on what the forest / detector emitted during a
/// scenario.
#[cfg(feature = "std")]
#[derive(Debug, Default)]
pub struct TestSink {
    /// Thread-safe recorded events.
    inner: std::sync::Mutex<TestSinkInner>,
}

/// Recorded state of a [`TestSink`] at inspection time.
#[cfg(feature = "std")]
#[derive(Debug, Default, Clone)]
pub struct TestSinkInner {
    /// Cumulative counter totals keyed by name.
    pub counters: std::collections::HashMap<String, u64>,
    /// Latest gauge value keyed by name.
    pub gauges: std::collections::HashMap<String, f64>,
    /// Per-name list of histogram observations (ordered by arrival).
    pub histograms: std::collections::HashMap<String, Vec<f64>>,
}

#[cfg(feature = "std")]
impl TestSink {
    /// Build a fresh empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot the current recorded state.
    ///
    /// # Panics
    ///
    /// Panics if another thread panicked while holding the internal
    /// lock — the mutex is poisoned in that case and recovery is a
    /// test-only concern.
    #[must_use]
    pub fn snapshot(&self) -> TestSinkInner {
        self.lock_inner().clone()
    }

    /// Counter total for `name`, `0` when unseen.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn counter(&self, name: &str) -> u64 {
        *self.lock_inner().counters.get(name).unwrap_or(&0)
    }

    /// Latest gauge value for `name`, `None` when unseen.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn gauge(&self, name: &str) -> Option<f64> {
        self.lock_inner().gauges.get(name).copied()
    }

    /// Histogram observations for `name`, cloned.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn histogram(&self, name: &str) -> Vec<f64> {
        self.lock_inner()
            .histograms
            .get(name)
            .cloned()
            .unwrap_or_default()
    }

    /// Shared helper — acquires the inner guard and surfaces poison
    /// with an explicit message instead of an opaque `unwrap`.
    /// Poison can only happen if another thread panicked while the
    /// lock was held; callers already document this in `# Panics`.
    fn lock_inner(&self) -> std::sync::MutexGuard<'_, TestSinkInner> {
        self.inner
            .lock()
            .expect("TestSink mutex poisoned — another thread panicked holding it")
    }
}

#[cfg(feature = "std")]
impl MetricsSink for TestSink {
    fn inc_counter(&self, name: &str, value: u64) {
        let mut guard = self.lock_inner();
        *guard.counters.entry(name.to_string()).or_insert(0) = guard
            .counters
            .get(name)
            .copied()
            .unwrap_or(0)
            .saturating_add(value);
    }
    fn set_gauge(&self, name: &str, value: f64) {
        let mut guard = self.lock_inner();
        guard.gauges.insert(name.to_string(), value);
    }
    fn observe_histogram(&self, name: &str, value: f64) {
        let mut guard = self.lock_inner();
        guard
            .histograms
            .entry(name.to_string())
            .or_default()
            .push(value);
    }
}

/// Canonical metric names emitted by the crate. Exposed as
/// constants so downstream dashboards can pin label expectations
/// without stringly-typing.
pub mod names {
    /// Counter: every [`crate::RandomCutForest::update`] call.
    pub const UPDATES_TOTAL: &str = "rcf_updates_total";
    /// Counter: every [`crate::ThresholdedForest::process`] call.
    pub const PROCESS_TOTAL: &str = "rcf_process_total";
    /// Counter: every [`crate::ThresholdedForest::process`] call
    /// whose verdict was flagged `is_anomaly`.
    pub const ANOMALIES_FIRED_TOTAL: &str = "rcf_anomalies_fired_total";
    /// Counter: every [`crate::MetaDriftDetector::observe`] call
    /// that returned `Some(DriftKind::*)` — aggregate of up + down.
    pub const DRIFT_FIRES_TOTAL: &str = "rcf_drift_fires_total";
    /// Counter: CUSUM upward drift fires (`DriftKind::Upward`).
    pub const DRIFT_UP_TOTAL: &str = "rcf_drift_up_total";
    /// Counter: CUSUM downward drift fires (`DriftKind::Downward`).
    pub const DRIFT_DOWN_TOTAL: &str = "rcf_drift_down_total";
    /// Counter: every [`crate::RandomCutForest::delete`] call that
    /// actually removed a point.
    pub const DELETES_TOTAL: &str = "rcf_deletes_total";
    /// Counter: every [`crate::RandomCutForest::attribution`] call
    /// that returned successfully.
    pub const ATTRIBUTION_TOTAL: &str = "rcf_attribution_total";
    /// Counter: inputs rejected because they contained a non-finite
    /// component (NaN / ±inf). Bumped once per rejected public call
    /// — upstream data-quality signal for SOC dashboards.
    pub const REJECTED_NAN_TOTAL: &str = "rcf_rejected_nan_total";
    /// Counter: [`crate::RandomCutForest::score_early_term`] calls
    /// that short-circuited (walked fewer than `num_trees`). Pair
    /// with the call site's total to derive the early-stop ratio.
    pub const EARLY_TERM_STOPPED_TOTAL: &str = "rcf_early_term_stopped_total";
    /// Counter: every [`crate::TenantForestPool`] eviction (LRU + TTL
    /// paths combined). Pair with [`TENANT_IDLE_EVICTIONS_TOTAL`] to
    /// derive the pressure-driven share.
    pub const TENANT_EVICTIONS_TOTAL: &str = "rcf_tenant_evictions_total";
    /// Counter: idle / TTL-driven evictions from
    /// [`crate::TenantForestPool::evict_idle`]. Subset of
    /// [`TENANT_EVICTIONS_TOTAL`].
    pub const TENANT_IDLE_EVICTIONS_TOTAL: &str = "rcf_tenant_idle_evictions_total";
    /// Counter: pool-factory invocations — a fresh tenant entered
    /// the resident set. Diverges from `TENANT_EVICTIONS_TOTAL` so
    /// churn (create − evict) is observable.
    pub const TENANT_CREATED_TOTAL: &str = "rcf_tenant_created_total";
    /// Counter: bootstrap/replay points successfully ingested.
    pub const BOOTSTRAP_POINTS_TOTAL: &str = "rcf_bootstrap_points_total";
    /// Counter: bootstrap/replay points skipped for being non-finite.
    pub const BOOTSTRAP_SKIPPED_TOTAL: &str = "rcf_bootstrap_skipped_total";
    /// Counter: every `FeatureDriftDetector::observe` call.
    pub const FEATURE_DRIFT_OBSERVED_TOTAL: &str = "rcf_feature_drift_observed_total";
    /// Counter: every `AlertClusterer::observe` call — total alerts
    /// ingested, pre-dedup.
    pub const ALERTS_OBSERVED_TOTAL: &str = "rcf_alerts_observed_total";
    /// Counter: alerts that opened a brand-new cluster (no existing
    /// cluster passed the similarity threshold).
    pub const ALERT_CLUSTERS_NEW_TOTAL: &str = "rcf_alert_clusters_new_total";
    /// Counter: alerts merged into an existing cluster — the dedup
    /// win. Pair with `ALERTS_OBSERVED_TOTAL` to derive the dedup
    /// ratio.
    pub const ALERT_CLUSTERS_JOINED_TOTAL: &str = "rcf_alert_clusters_joined_total";
    /// Counter: clusters pruned because their most recent alert fell
    /// outside the sliding window.
    pub const ALERT_CLUSTERS_PRUNED_TOTAL: &str = "rcf_alert_clusters_pruned_total";

    /// Gauge: number of trees held by a forest.
    pub const FOREST_TREES: &str = "rcf_forest_trees";
    /// Gauge: current adaptive threshold
    /// ([`crate::ThresholdedForest::current_threshold`]).
    pub const THRESHOLD_CURRENT: &str = "rcf_threshold_current";
    /// Gauge: EMA mean of the score stream on a `ThresholdedForest`.
    pub const EMA_MEAN: &str = "rcf_ema_mean";
    /// Gauge: EMA stddev of the score stream on a `ThresholdedForest`.
    pub const EMA_STDDEV: &str = "rcf_ema_stddev";
    /// Gauge: observations folded into the TRCF score-stream EMA.
    /// Combined with the configured `min_observations`, exposes the
    /// warmup progress of a fresh detector.
    pub const OBSERVATIONS_SEEN: &str = "rcf_observations_seen";
    /// Gauge: number of tenants resident in a
    /// [`crate::TenantForestPool`] after each public op.
    pub const TENANTS_RESIDENT: &str = "rcf_tenants_resident";
    /// Gauge: configured tenant capacity of a
    /// [`crate::TenantForestPool`]. Static after construction but
    /// dashboards benefit from being able to chart
    /// `TENANTS_RESIDENT / TENANT_CAPACITY` pressure ratios.
    pub const TENANT_CAPACITY: &str = "rcf_tenant_capacity";
    /// Gauge: active clusters held by an `AlertClusterer`.
    pub const ALERT_CLUSTERS_ACTIVE: &str = "rcf_alert_clusters_active";
    /// Gauge: maximum per-dim PSI of a
    /// `FeatureDriftDetector`. Set on every `psi()` call.
    pub const FEATURE_DRIFT_MAX_PSI: &str = "rcf_feature_drift_max_psi";

    /// Histogram: raw anomaly score per scored point.
    pub const SCORE_OBSERVATION: &str = "rcf_score";
    /// Histogram: graded verdict (`[0, 1]`) per processed point.
    pub const GRADE_OBSERVATION: &str = "rcf_grade";
    /// Histogram: upward CUSUM accumulator after each observation.
    pub const DRIFT_S_HIGH: &str = "rcf_drift_s_high";
    /// Histogram: downward CUSUM accumulator after each observation.
    pub const DRIFT_S_LOW: &str = "rcf_drift_s_low";
    /// Histogram: trees actually walked per
    /// [`crate::RandomCutForest::score_early_term`] call — use with
    /// [`FOREST_TREES`] to compute the latency savings distribution.
    pub const EARLY_TERM_TREES: &str = "rcf_early_term_trees";

    // --- hot_path ------------------------------------------------

    /// Counter: [`crate::UpdateSampler`] `accept_*` calls that
    /// admitted the offer.
    pub const HOT_PATH_SAMPLER_ACCEPTED_TOTAL: &str = "rcf_hot_path_sampler_accepted_total";
    /// Counter: [`crate::UpdateSampler`] `accept_*` calls that
    /// rejected the offer (stride / per-flow-hash residue mismatch).
    pub const HOT_PATH_SAMPLER_REJECTED_TOTAL: &str = "rcf_hot_path_sampler_rejected_total";
    /// Counter: points successfully enqueued through a
    /// [`crate::UpdateProducer::try_enqueue`] call.
    pub const HOT_PATH_QUEUE_ENQUEUED_TOTAL: &str = "rcf_hot_path_queue_enqueued_total";
    /// Counter: points dropped because the hot-path MPSC queue was
    /// full. Non-zero indicates classifier > updater throughput.
    pub const HOT_PATH_QUEUE_DROPPED_TOTAL: &str = "rcf_hot_path_queue_dropped_total";
    /// Counter: [`crate::PrefixRateCap::check_and_record`] calls that
    /// admitted the offer.
    pub const HOT_PATH_PREFIX_ADMITTED_TOTAL: &str = "rcf_hot_path_prefix_admitted_total";
    /// Counter: [`crate::PrefixRateCap::check_and_record`] calls that
    /// capped the offer (bucket at limit).
    pub const HOT_PATH_PREFIX_CAPPED_TOTAL: &str = "rcf_hot_path_prefix_capped_total";

    // --- drift_aware ---------------------------------------------

    /// Counter: [`crate::DriftAwareForest`] shadow → primary swaps.
    pub const DRIFT_AWARE_SWAPS_TOTAL: &str = "rcf_drift_aware_swaps_total";
    /// Counter: [`crate::DriftAwareForest::on_drift`] calls that
    /// actually spawned a shadow (not gated out by
    /// `min_primary_age` or in-flight shadow).
    pub const DRIFT_AWARE_ON_DRIFT_TOTAL: &str = "rcf_drift_aware_on_drift_total";
    /// Gauge: `1.0` while a shadow is warming, `0.0` otherwise.
    pub const DRIFT_AWARE_SHADOW_ACTIVE: &str = "rcf_drift_aware_shadow_active";

    // --- adwin ---------------------------------------------------

    /// Counter: finite values folded into an
    /// [`crate::AdwinDetector::update`] window.
    pub const ADWIN_OBSERVED_TOTAL: &str = "rcf_adwin_observed_total";
    /// Counter: [`crate::AdwinDetector::update`] calls that
    /// detected drift and shrank the window.
    pub const ADWIN_DRIFT_FIRES_TOTAL: &str = "rcf_adwin_drift_fires_total";

    // --- lsh_cluster ---------------------------------------------

    /// Counter: every [`crate::LshAlertClusterer::observe`] call.
    pub const LSH_ALERTS_OBSERVED_TOTAL: &str = "rcf_lsh_alerts_observed_total";
    /// Counter: LSH alerts that opened a brand-new bucket.
    pub const LSH_CLUSTERS_NEW_TOTAL: &str = "rcf_lsh_clusters_new_total";
    /// Counter: LSH alerts merged into an existing bucket.
    pub const LSH_CLUSTERS_JOINED_TOTAL: &str = "rcf_lsh_clusters_joined_total";
    /// Gauge: distinct active LSH cluster hashes.
    pub const LSH_CLUSTERS_ACTIVE: &str = "rcf_lsh_clusters_active";

    // --- feedback ------------------------------------------------

    /// Counter: every [`crate::FeedbackStore::label`] call that
    /// accepted the point.
    pub const FEEDBACK_LABELS_OBSERVED_TOTAL: &str = "rcf_feedback_labels_observed_total";
    /// Counter: feedback labels of kind
    /// [`crate::FeedbackLabel::Benign`].
    pub const FEEDBACK_LABELS_BENIGN_TOTAL: &str = "rcf_feedback_labels_benign_total";
    /// Counter: feedback labels of kind
    /// [`crate::FeedbackLabel::Confirmed`].
    pub const FEEDBACK_LABELS_CONFIRMED_TOTAL: &str = "rcf_feedback_labels_confirmed_total";

    // --- univariate_spot -----------------------------------------

    /// Counter: every finite value folded via
    /// [`crate::PotDetector::record`].
    pub const SPOT_OBSERVATIONS_TOTAL: &str = "rcf_spot_observations_total";
    /// Counter: peaks accumulated above the SPOT/DSPOT threshold `u`.
    pub const SPOT_PEAKS_TOTAL: &str = "rcf_spot_peaks_total";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_sink_is_noop() {
        let s = NoopSink;
        s.inc_counter("x", 1);
        s.set_gauge("y", 2.0);
        s.observe_histogram("z", 3.0);
    }

    #[test]
    fn default_sink_builds_noop_arc() {
        let s = default_sink();
        s.inc_counter("x", 1);
    }

    #[test]
    fn test_sink_records_counter_gauge_histogram() {
        let s = TestSink::new();
        s.inc_counter("a", 3);
        s.inc_counter("a", 4);
        s.set_gauge("b", 1.25);
        s.set_gauge("b", 2.5);
        s.observe_histogram("c", 0.1);
        s.observe_histogram("c", 0.2);
        assert_eq!(s.counter("a"), 7);
        assert_eq!(s.gauge("b"), Some(2.5));
        assert_eq!(s.histogram("c"), vec![0.1, 0.2]);
    }

    #[test]
    fn test_sink_unseen_metrics_default() {
        let s = TestSink::new();
        assert_eq!(s.counter("nope"), 0);
        assert!(s.gauge("nope").is_none());
        assert!(s.histogram("nope").is_empty());
    }
}
