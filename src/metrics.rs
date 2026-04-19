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
        self.inner.lock().unwrap().clone()
    }

    /// Counter total for `name`, `0` when unseen.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn counter(&self, name: &str) -> u64 {
        *self.inner.lock().unwrap().counters.get(name).unwrap_or(&0)
    }

    /// Latest gauge value for `name`, `None` when unseen.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn gauge(&self, name: &str) -> Option<f64> {
        self.inner.lock().unwrap().gauges.get(name).copied()
    }

    /// Histogram observations for `name`, cloned.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn histogram(&self, name: &str) -> Vec<f64> {
        self.inner
            .lock()
            .unwrap()
            .histograms
            .get(name)
            .cloned()
            .unwrap_or_default()
    }
}

#[cfg(feature = "std")]
impl MetricsSink for TestSink {
    fn inc_counter(&self, name: &str, value: u64) {
        let mut guard = self.inner.lock().unwrap();
        *guard.counters.entry(name.to_string()).or_insert(0) = guard
            .counters
            .get(name)
            .copied()
            .unwrap_or(0)
            .saturating_add(value);
    }
    fn set_gauge(&self, name: &str, value: f64) {
        let mut guard = self.inner.lock().unwrap();
        guard.gauges.insert(name.to_string(), value);
    }
    fn observe_histogram(&self, name: &str, value: f64) {
        let mut guard = self.inner.lock().unwrap();
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
    /// that returned `Some(DriftKind::*)`.
    pub const DRIFT_FIRES_TOTAL: &str = "rcf_drift_fires_total";
    /// Counter: every [`crate::RandomCutForest::delete`] call that
    /// actually removed a point.
    pub const DELETES_TOTAL: &str = "rcf_deletes_total";
    /// Counter: every [`crate::TenantForestPool`] LRU eviction.
    pub const TENANT_EVICTIONS_TOTAL: &str = "rcf_tenant_evictions_total";

    /// Gauge: number of trees held by a forest.
    pub const FOREST_TREES: &str = "rcf_forest_trees";
    /// Gauge: current adaptive threshold
    /// ([`crate::ThresholdedForest::current_threshold`]).
    pub const THRESHOLD_CURRENT: &str = "rcf_threshold_current";
    /// Gauge: number of tenants resident in a
    /// [`crate::TenantForestPool`] after each public op.
    pub const TENANTS_RESIDENT: &str = "rcf_tenants_resident";

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
