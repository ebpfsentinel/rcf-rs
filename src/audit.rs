//! Structured, serialisable audit records for compliance / SOC2 / NIS2.
//!
//! Live `RandomCutForest::score` + `attribution` + `forensic_baseline`
//! return ephemeral analytic values. Compliance regimes want
//! durable evidence that an alert fired, with enough context to
//! reconstruct *why* during a later audit. [`AlertRecord`] packages
//! every analytic output plus provenance fields (tenant, timestamp,
//! point) into one `serde`-roundtrippable struct.
//!
//! # Typical flow
//!
//! ```
//! # #[cfg(feature = "serde")] {
//! use rcf_rs::{ForestBuilder, audit::{AlertRecord, AlertContext}};
//!
//! let mut forest = ForestBuilder::<4>::new()
//!     .num_trees(50).sample_size(16).seed(42).build().unwrap();
//! for i in 0..32 {
//!     let v = f64::from(i) * 0.01;
//!     forest.update([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
//! }
//! let probe = [5.0, 5.0, 5.0, 5.0];
//! let ctx = AlertContext::<String> {
//!     tenant: Some("tenant-42".into()),
//!     timestamp_ms: 1_700_000_000_000,
//! };
//! let rec = AlertRecord::from_forest(&forest, &probe, &ctx).unwrap();
//! assert_eq!(rec.tenant.as_deref(), Some("tenant-42"));
//! # }
//! ```
//!
//! Emit the record to a SIEM / object-store / WORM log via any
//! `serde` sink — `postcard` for compact per-event bytes, JSON for
//! pipelines that want self-describing records.

use alloc::string::String;

use crate::domain::{AnomalyScore, DiVector};
use crate::error::RcfResult;
use crate::forensic::ForensicBaseline;
use crate::forest::RandomCutForest;
use crate::severity::{Severity, SeverityBands};
use crate::thresholded::{AnomalyGrade, ThresholdedForest};

/// Current `AlertRecord` schema version. Consumers reject payloads
/// whose version falls outside the supported range so incompatible
/// schema changes surface early instead of silently drifting.
pub const ALERT_RECORD_VERSION: u32 = 1;

/// Caller-supplied provenance fields — the detector does not track
/// wall-clock time or tenant identity, so the audit pipeline passes
/// them in explicitly on each emission.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AlertContext<K = String>
where
    K: Clone,
{
    /// Optional tenant / customer / source identifier. `None` for
    /// bare single-tenant agents.
    pub tenant: Option<K>,
    /// Unix epoch milliseconds at which the observation was ingested
    /// by the caller. Detector-internal timestamps (`update_at`)
    /// are unrelated — this is the audit wall clock.
    pub timestamp_ms: u64,
}

impl<K: Clone> AlertContext<K> {
    /// Untenanted context helper.
    #[must_use]
    pub fn untenanted(timestamp_ms: u64) -> Self {
        Self {
            tenant: None,
            timestamp_ms,
        }
    }

    /// Tenanted context helper.
    #[must_use]
    pub fn for_tenant(tenant: K, timestamp_ms: u64) -> Self {
        Self {
            tenant: Some(tenant),
            timestamp_ms,
        }
    }
}

/// Durable evidence of an anomaly evaluation — serialisable under
/// the `serde` feature, round-trips bit-exact through postcard or
/// JSON.
///
/// `K` is the tenant key type (`String`, `u64`, UUID newtype…);
/// defaults to `String` for JSON-friendly audit sinks. `D` is the
/// per-point dimensionality, identical to the producing detector.
///
/// Fields are **all public** — audit consumers frequently need ad-hoc
/// access in reporting pipelines, and the struct is already a
/// caller-facing data carrier with no invariants beyond what the
/// producing detector guaranteed.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AlertRecord<K = String, const D: usize = 4>
where
    K: Clone,
{
    /// Schema version. Producers always set [`ALERT_RECORD_VERSION`];
    /// consumers compare and reject unknown versions.
    pub version: u32,
    /// Optional tenant identifier. Mirrors `AlertContext::tenant`.
    pub tenant: Option<K>,
    /// Unix-epoch milliseconds at which the observation occurred.
    pub timestamp_ms: u64,
    /// Raw observed point in caller coordinates.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_util::fixed_array_f64"))]
    pub point: [f64; D],
    /// Raw RCF score for `point`.
    pub score: AnomalyScore,
    /// TRCF graded verdict when the producer is a
    /// [`ThresholdedForest`]; `None` for bare `RandomCutForest`
    /// producers.
    pub grade: Option<AnomalyGrade>,
    /// Severity band mapped from `score`. `None` when the caller did
    /// not supply `SeverityBands`.
    pub severity: Option<Severity>,
    /// Per-feature attribution `DiVector` for `point`.
    pub attribution: DiVector,
    /// Per-dim imputation-like baseline for `point`.
    pub baseline: ForensicBaseline<D>,
}

impl<K: Clone, const D: usize> AlertRecord<K, D> {
    /// Build an audit record from a bare [`RandomCutForest`]. No
    /// TRCF grade is produced — [`Self::grade`] stays `None`.
    ///
    /// # Errors
    ///
    /// Propagates failures from [`RandomCutForest::score`],
    /// [`RandomCutForest::attribution`] and
    /// [`RandomCutForest::forensic_baseline`].
    pub fn from_forest(
        forest: &RandomCutForest<D>,
        point: &[f64; D],
        ctx: &AlertContext<K>,
    ) -> RcfResult<Self> {
        let score = forest.score(point)?;
        let attribution = forest.attribution(point)?;
        let baseline = forest.forensic_baseline(point)?;
        Ok(Self {
            version: ALERT_RECORD_VERSION,
            tenant: ctx.tenant.clone(),
            timestamp_ms: ctx.timestamp_ms,
            point: *point,
            score,
            grade: None,
            severity: None,
            attribution,
            baseline,
        })
    }

    /// Build an audit record from a [`ThresholdedForest`] — emits
    /// the TRCF grade alongside the raw analytic outputs.
    ///
    /// # Errors
    ///
    /// Propagates failures from
    /// [`ThresholdedForest::score_only`],
    /// [`ThresholdedForest::attribution`] and
    /// [`ThresholdedForest::forensic_baseline`].
    pub fn from_thresholded(
        detector: &mut ThresholdedForest<D>,
        point: &[f64; D],
        ctx: &AlertContext<K>,
    ) -> RcfResult<Self> {
        let grade = detector.score_only(point)?;
        let score = grade.score();
        let attribution = detector.attribution(point)?;
        let baseline = detector.forensic_baseline(point)?;
        Ok(Self {
            version: ALERT_RECORD_VERSION,
            tenant: ctx.tenant.clone(),
            timestamp_ms: ctx.timestamp_ms,
            point: *point,
            score,
            grade: Some(grade),
            severity: None,
            attribution,
            baseline,
        })
    }

    /// Attach a severity band derived from `score` under `bands`.
    /// Chain-call after `from_forest` / `from_thresholded`.
    #[must_use]
    pub fn with_severity(mut self, bands: &SeverityBands) -> Self {
        self.severity = Some(bands.classify(f64::from(self.score)));
        self
    }
}

#[cfg(all(test, feature = "serde", feature = "postcard", feature = "serde_json"))]
#[allow(clippy::unwrap_used, clippy::panic, clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::{ForestBuilder, ThresholdedForestBuilder};

    fn warm_forest() -> RandomCutForest<4> {
        let mut f = ForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(16)
            .seed(2026)
            .build()
            .unwrap();
        for i in 0..32_u32 {
            let v = f64::from(i) * 0.01;
            f.update([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
        }
        f
    }

    #[test]
    fn from_forest_captures_every_analytic() {
        let f = warm_forest();
        let ctx = AlertContext::<String>::for_tenant("t1".into(), 1_700_000_000_000);
        let rec = AlertRecord::from_forest(&f, &[5.0, 5.0, 5.0, 5.0], &ctx).unwrap();
        assert_eq!(rec.version, ALERT_RECORD_VERSION);
        assert_eq!(rec.tenant.as_deref(), Some("t1"));
        assert_eq!(rec.timestamp_ms, 1_700_000_000_000);
        assert_eq!(rec.point, [5.0, 5.0, 5.0, 5.0]);
        assert!(f64::from(rec.score) > 0.0);
        assert!(rec.grade.is_none());
        assert!(rec.severity.is_none());
        assert_eq!(rec.attribution.dim(), 4);
        assert_eq!(rec.baseline.observed, [5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn from_thresholded_emits_grade() {
        let mut d = ThresholdedForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(16)
            .min_observations(4)
            .seed(2026)
            .build()
            .unwrap();
        for i in 0..32_u32 {
            let v = f64::from(i) * 0.01;
            d.process([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
        }
        let ctx = AlertContext::<String>::untenanted(1_700_000_000_000);
        let rec = AlertRecord::from_thresholded(&mut d, &[5.0, 5.0, 5.0, 5.0], &ctx).unwrap();
        assert!(rec.grade.is_some());
        assert!(rec.tenant.is_none());
    }

    #[test]
    fn with_severity_maps_score() {
        let f = warm_forest();
        let ctx = AlertContext::<String>::untenanted(0);
        let bands = SeverityBands::default();
        let rec = AlertRecord::from_forest(&f, &[5.0, 5.0, 5.0, 5.0], &ctx)
            .unwrap()
            .with_severity(&bands);
        assert!(rec.severity.is_some());
    }

    #[test]
    fn postcard_roundtrip_preserves_record() {
        let f = warm_forest();
        let ctx = AlertContext::<String>::for_tenant("t1".into(), 42);
        let rec = AlertRecord::from_forest(&f, &[5.0, 5.0, 5.0, 5.0], &ctx).unwrap();
        let bytes = postcard::to_allocvec(&rec).unwrap();
        let back: AlertRecord<String, 4> = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(rec, back);
    }

    #[test]
    fn json_roundtrip_preserves_record() {
        let f = warm_forest();
        let ctx = AlertContext::<String>::for_tenant("t1".into(), 42);
        let rec = AlertRecord::from_forest(&f, &[5.0, 5.0, 5.0, 5.0], &ctx).unwrap();
        let json = serde_json::to_string(&rec).unwrap();
        let back: AlertRecord<String, 4> = serde_json::from_str(&json).unwrap();
        // ryu shortest-roundtrip may drift by 1 ULP on derived
        // stddev / zscore values; non-fp fields must be bit-exact.
        assert_eq!(rec.version, back.version);
        assert_eq!(rec.tenant, back.tenant);
        assert_eq!(rec.timestamp_ms, back.timestamp_ms);
        assert_eq!(rec.point, back.point);
        assert_eq!(rec.score, back.score);
        assert_eq!(rec.grade, back.grade);
        assert_eq!(rec.severity, back.severity);
        assert_eq!(rec.baseline.live_points, back.baseline.live_points);
        assert_eq!(rec.baseline.observed, back.baseline.observed);
    }

    #[test]
    fn untenanted_helper_sets_tenant_none() {
        let ctx = AlertContext::<String>::untenanted(123);
        assert!(ctx.tenant.is_none());
        assert_eq!(ctx.timestamp_ms, 123);
    }
}
