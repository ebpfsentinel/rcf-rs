//! LSH-based alert clustering — buckets near-duplicate alerts via
//! a locality-sensitive hash on the attribution `DiVector`.
//!
//! `AlertClusterer` (the cosine-similarity variant in
//! `src/alert_cluster.rs`) groups by pairwise similarity — O(N)
//! comparisons per new alert. For MSSP-scale volumes (tens of
//! thousands of alerts per tenant per day) the linear scan
//! dominates. `LshAlertClusterer` quantises each per-dim
//! attribution into a 4-bit symbol and uses the concatenated hex
//! string as the bucket key — O(1) lookup, collision-safe under
//! attribution similarity.
//!
//! The hash mirrors the TLSH spirit (Oliver et al. 2013) without
//! the bigram-frequency machinery: similar attribution vectors
//! land in the same bucket because their quantised per-dim values
//! agree on every dim. Exact duplicates are always co-bucketed;
//! near-duplicates co-bucket when every dim's magnitude agrees
//! modulo the quantisation step.

#![cfg(feature = "std")]

use std::collections::HashMap;
use std::sync::Arc;

use crate::audit::AlertRecord;
use crate::domain::DiVector;
use crate::error::{RcfError, RcfResult};
use crate::metrics::{MetricsSink, default_sink, names};

/// Number of quantisation buckets per per-dim attribution value.
/// 16 buckets → 4 bits per dim, packed into the hash string.
pub const DEFAULT_BUCKETS_PER_DIM: usize = 16;

/// Default attribution-magnitude cap used by the quantiser —
/// matches the typical post-normaliser range of `DiVector::total`.
pub const DEFAULT_ATTR_CAP: f64 = 8.0;

/// Cluster decision emitted by [`LshAlertClusterer::observe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LshClusterDecision {
    /// No bucket held a prior alert with this hash — new cluster
    /// opened at the returned hash.
    NewCluster,
    /// A prior alert in the same bucket was found; the new alert
    /// was merged into it.
    Joined,
}

/// Quantise-and-bucket alert clusterer. Thread-`Send` / `Sync`
/// friendly — holds a `HashMap<String, u64>` counter keyed by the
/// quantised hash.
#[derive(Debug, Clone)]
pub struct LshAlertClusterer {
    /// Per-hash count of alerts merged into the cluster.
    buckets: HashMap<String, u64>,
    /// Number of buckets to quantise each per-dim attribution
    /// into. 16 is typical; higher → more clusters, less merging.
    buckets_per_dim: usize,
    /// Attribution magnitude cap used by the quantiser.
    attr_cap: f64,
    /// Lifetime alerts observed.
    observed_total: u64,
    /// Lifetime merges (existing bucket hit).
    joined_total: u64,
    /// Lifetime cluster openings.
    new_cluster_total: u64,
    /// Observability sink.
    metrics: Arc<dyn MetricsSink>,
}

impl Default for LshAlertClusterer {
    fn default() -> Self {
        Self {
            buckets: HashMap::new(),
            buckets_per_dim: DEFAULT_BUCKETS_PER_DIM,
            attr_cap: DEFAULT_ATTR_CAP,
            observed_total: 0,
            joined_total: 0,
            new_cluster_total: 0,
            metrics: default_sink(),
        }
    }
}

impl LshAlertClusterer {
    /// Build with caller-configured quantisation settings.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] on `buckets_per_dim < 2`,
    /// `buckets_per_dim > 256` (single hex / 2-hex overflow), or
    /// non-finite / non-positive `attr_cap`.
    pub fn new(buckets_per_dim: usize, attr_cap: f64) -> RcfResult<Self> {
        if !(2..=256).contains(&buckets_per_dim) {
            return Err(RcfError::InvalidConfig(format!(
                "LshAlertClusterer: buckets_per_dim must be in [2, 256], got {buckets_per_dim}"
            )));
        }
        if !attr_cap.is_finite() || attr_cap <= 0.0 {
            return Err(RcfError::InvalidConfig(format!(
                "LshAlertClusterer: attr_cap must be finite and > 0, got {attr_cap}"
            )));
        }
        Ok(Self {
            buckets: HashMap::new(),
            buckets_per_dim,
            attr_cap,
            observed_total: 0,
            joined_total: 0,
            new_cluster_total: 0,
            metrics: default_sink(),
        })
    }

    /// Install a metrics sink — every `observe` call emits
    /// observed/new/joined counters + an active-clusters gauge.
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

    /// Convenience constructor — [`DEFAULT_BUCKETS_PER_DIM`] /
    /// [`DEFAULT_ATTR_CAP`].
    ///
    /// # Panics
    ///
    /// Never — the defaults pass [`Self::new`]'s validation.
    #[must_use]
    pub fn default_lsh() -> Self {
        Self::new(DEFAULT_BUCKETS_PER_DIM, DEFAULT_ATTR_CAP).expect("defaults valid")
    }

    /// Observe an alert. Hashes its attribution via
    /// [`Self::hash_divector`], increments the bucket counter,
    /// returns the bucket hash + the cluster decision.
    pub fn observe<K, const D: usize>(
        &mut self,
        record: &AlertRecord<K, D>,
    ) -> (String, LshClusterDecision)
    where
        K: Clone,
    {
        let hash = self.hash_divector(&record.attribution);
        self.observed_total = self.observed_total.saturating_add(1);
        self.metrics
            .inc_counter(names::LSH_ALERTS_OBSERVED_TOTAL, 1);
        let entry = self.buckets.entry(hash.clone()).or_insert(0);
        let is_new = *entry == 0;
        *entry = entry.saturating_add(1);
        let decision = if is_new {
            self.new_cluster_total = self.new_cluster_total.saturating_add(1);
            self.metrics.inc_counter(names::LSH_CLUSTERS_NEW_TOTAL, 1);
            LshClusterDecision::NewCluster
        } else {
            self.joined_total = self.joined_total.saturating_add(1);
            self.metrics
                .inc_counter(names::LSH_CLUSTERS_JOINED_TOTAL, 1);
            LshClusterDecision::Joined
        };
        #[allow(clippy::cast_precision_loss)]
        self.metrics
            .set_gauge(names::LSH_CLUSTERS_ACTIVE, self.buckets.len() as f64);
        (hash, decision)
    }

    /// LSH hash of a [`DiVector`]: per-dim signed attribution
    /// (`high - low`) quantised into `buckets_per_dim` buckets on
    /// `[-attr_cap, +attr_cap]`, concatenated as a fixed-length
    /// hex string.
    ///
    /// Similar alerts hash to the same string because every dim's
    /// signed attribution falls in the same bucket modulo the
    /// quantisation step.
    #[must_use]
    pub fn hash_divector(&self, di: &DiVector) -> String {
        let dims = di.dim();
        let mut out = String::with_capacity(dims.saturating_mul(2));
        for d in 0..dims {
            let signed = di.high()[d] - di.low()[d];
            let bucket = self.quantise(signed);
            if self.buckets_per_dim <= 16 {
                let _ = std::fmt::Write::write_fmt(&mut out, format_args!("{bucket:x}"));
            } else {
                let _ = std::fmt::Write::write_fmt(&mut out, format_args!("{bucket:02x}"));
            }
        }
        out
    }

    /// Count of alerts merged into the cluster keyed by `hash`.
    /// `0` when the hash has never been observed.
    #[must_use]
    pub fn cluster_size(&self, hash: &str) -> u64 {
        self.buckets.get(hash).copied().unwrap_or(0)
    }

    /// Distinct active cluster hashes.
    #[must_use]
    pub fn cluster_count(&self) -> usize {
        self.buckets.len()
    }

    /// Lifetime alerts observed.
    #[must_use]
    pub fn observed_total(&self) -> u64 {
        self.observed_total
    }

    /// Lifetime merges (existing bucket hit).
    #[must_use]
    pub fn joined_total(&self) -> u64 {
        self.joined_total
    }

    /// Lifetime cluster openings.
    #[must_use]
    pub fn new_cluster_total(&self) -> u64 {
        self.new_cluster_total
    }

    /// Drop every bucket; counters preserved.
    pub fn clear_buckets(&mut self) {
        self.buckets.clear();
        self.metrics.set_gauge(names::LSH_CLUSTERS_ACTIVE, 0.0);
    }

    /// Map a signed attribution value to `[0, buckets_per_dim)`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn quantise(&self, value: f64) -> u32 {
        if !value.is_finite() {
            return 0;
        }
        let cap = self.attr_cap;
        let clamped = value.clamp(-cap, cap);
        // Map [-cap, cap] linearly to [0, buckets_per_dim − 1].
        #[allow(clippy::cast_precision_loss)]
        let n = self.buckets_per_dim as f64;
        let normalised = (clamped + cap) / (2.0 * cap); // ∈ [0, 1]
        let bucket = (normalised * n).floor();
        #[allow(clippy::cast_precision_loss)]
        let max_bucket = (self.buckets_per_dim - 1) as f64;
        bucket.clamp(0.0, max_bucket) as u32
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
    use crate::audit::AlertRecord;
    use crate::domain::{AnomalyScore, DiVector};
    use crate::forensic::ForensicBaseline;

    fn record_with_di(di: DiVector) -> AlertRecord<u32, 4> {
        AlertRecord {
            version: crate::audit::ALERT_RECORD_VERSION,
            tenant: Some(1_u32),
            timestamp_ms: 0,
            point: [0.0; 4],
            score: AnomalyScore::new(1.0).unwrap(),
            grade: None,
            severity: None,
            attribution: di,
            baseline: ForensicBaseline::<4> {
                observed: [0.0; 4],
                expected: [0.0; 4],
                stddev: [0.0; 4],
                delta: [0.0; 4],
                zscore: [0.0; 4],
                live_points: 0,
            },
        }
    }

    #[test]
    fn new_rejects_invalid_params() {
        assert!(LshAlertClusterer::new(1, 1.0).is_err());
        assert!(LshAlertClusterer::new(16, 0.0).is_err());
        assert!(LshAlertClusterer::new(16, f64::NAN).is_err());
    }

    #[test]
    fn identical_divectors_share_bucket() {
        let mut c = LshAlertClusterer::default_lsh();
        let mut di = DiVector::zeros(4);
        di.add_high(0, 3.0).unwrap();
        di.add_low(1, 2.0).unwrap();
        let r1 = record_with_di(di.clone());
        let r2 = record_with_di(di);
        let (h1, d1) = c.observe(&r1);
        let (h2, d2) = c.observe(&r2);
        assert_eq!(h1, h2);
        assert_eq!(d1, LshClusterDecision::NewCluster);
        assert_eq!(d2, LshClusterDecision::Joined);
        assert_eq!(c.cluster_size(&h1), 2);
    }

    #[test]
    fn far_divectors_land_in_distinct_buckets() {
        let mut c = LshAlertClusterer::default_lsh();
        let mut di_a = DiVector::zeros(4);
        di_a.add_high(0, 3.0).unwrap();
        let mut di_b = DiVector::zeros(4);
        di_b.add_low(0, 3.0).unwrap();
        let (h_a, _) = c.observe(&record_with_di(di_a));
        let (h_b, _) = c.observe(&record_with_di(di_b));
        assert_ne!(h_a, h_b);
        assert_eq!(c.cluster_count(), 2);
    }

    #[test]
    fn hash_length_matches_dim() {
        let c = LshAlertClusterer::new(16, 4.0).unwrap();
        let di = DiVector::zeros(8);
        let h = c.hash_divector(&di);
        // 16 buckets → 1 hex char per dim → dim 8 → 8 chars.
        assert_eq!(h.len(), 8);
    }

    #[test]
    fn counters_reflect_traffic() {
        let mut c = LshAlertClusterer::default_lsh();
        let mut di = DiVector::zeros(2);
        di.add_high(0, 1.0).unwrap();
        let r = record_with_di(di.clone());
        c.observe(&record_with_di(di.clone()));
        c.observe(&record_with_di(di.clone()));
        c.observe(&record_with_di(di));
        let _ = r;
        assert_eq!(c.observed_total(), 3);
        assert_eq!(c.new_cluster_total(), 1);
        assert_eq!(c.joined_total(), 2);
    }

    #[test]
    fn clear_buckets_preserves_counters() {
        let mut c = LshAlertClusterer::default_lsh();
        let mut di = DiVector::zeros(2);
        di.add_high(0, 1.0).unwrap();
        c.observe(&record_with_di(di));
        c.clear_buckets();
        assert_eq!(c.cluster_count(), 0);
        assert_eq!(c.observed_total(), 1);
    }
}
