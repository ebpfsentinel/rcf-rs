//! Smoke test — verify the `anomstream` facade re-exports from
//! all three workspace members are reachable under several
//! feature combinations and that types compose the way a real
//! consumer would use them.
//!
//! Keeps the all-features path **and** every feature subset that
//! CI exercises green at the facade level so breakages in member
//! re-export surfaces surface here rather than deep in a
//! downstream consumer lockfile. Each test is `cfg`-gated so
//! `cargo test --no-default-features --features <combo>` runs
//! only the tests applicable to that combo — the same matrix CI
//! drives.

#![allow(clippy::unwrap_used, clippy::panic)]

// --- Default-features smoke ----------------------------------
//
// These tests have NO `#[cfg(feature = ...)]` gate so they run
// under the crate's default feature set (`["core", "std"]`) AND
// under any superset including `full`. The point is to make
// breakage of the default surface visible without having to
// guess which combo CI happens to be running.

/// Default features (`core + std`) must let consumers build a
/// forest, score against it, and reach the bare cross-cuts
/// (`MetricsSink`, `SeverityBands`) — the absolute minimum the
/// crate's `default = ["core", "std"]` declaration commits to.
#[test]
fn default_features_build_score_via_facade() {
    use anomstream::{
        ForestBuilder, MetricsSink, NoopSink, RandomCutForest, Severity, SeverityBands,
    };
    let mut forest: RandomCutForest<2> = ForestBuilder::<2>::new()
        .num_trees(50)
        .sample_size(16)
        .seed(2026)
        .build()
        .unwrap();
    for i in 0..32_u32 {
        let v = f64::from(i) * 0.01;
        forest.update([v, v + 0.1]).unwrap();
    }
    let score = forest.score(&[5.0, 5.0]).unwrap();
    let bands = SeverityBands::default();
    let _: Severity = bands.classify(f64::from(score));
    let _: &dyn MetricsSink = &NoopSink;
}

// --- Core-only smoke -----------------------------------------
//
// Exercised under every feature combo that includes `core` —
// even the bare `core` feature without `std` must let consumers
// instantiate the bare detectors and the cross-cutting
// `MetricsSink` / `Severity` vocabulary.

/// Detector primitive + cross-cut error type — the absolute
/// minimum the facade promises under bare `core`.
#[cfg(feature = "core")]
#[test]
fn core_error_type_reachable_via_facade() {
    use anomstream::RcfError;
    let err = RcfError::InvalidConfig("smoke".into());
    let _: &dyn core::fmt::Debug = &err;
}

/// `MetricsSink` + `NoopSink` cross-cut: `NoopSink` reachable at
/// root, implements the trait alias.
#[cfg(feature = "core")]
#[test]
fn metrics_sink_reachable_via_facade() {
    use anomstream::{MetricsSink, NoopSink};
    let sink = NoopSink;
    let _: &dyn MetricsSink = &sink;
}

/// `Severity` + `SeverityBands` classification primitives.
#[cfg(feature = "core")]
#[test]
fn severity_classification_via_facade() {
    use anomstream::{Severity, SeverityBands};
    let bands = SeverityBands::default();
    assert_eq!(bands.classify(0.5), Severity::Normal);
}

// --- Core + std smoke ----------------------------------------
//
// `RandomCutForest` is `std`-free, so `ForestBuilder` works
// under bare `core`. `core_forest_roundtrip_via_facade` covers
// the build path; under `std` we additionally check the
// `std`-gated detectors (`AdwinDetector`, `ShingledForest`,
// `BloomFilter`) are reachable.

/// Core re-exports resolve: build a forest, confirm
/// `ForestSnapshot` trait bounds survive through the facade.
#[cfg(feature = "core")]
#[test]
fn core_forest_roundtrip_via_facade() {
    use anomstream::{ForestBuilder, ForestSnapshot, RandomCutForest};
    let forest: RandomCutForest<4> = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(32)
        .seed(2026)
        .build()
        .unwrap();
    assert_eq!(forest.snapshot_num_trees(), 50);
    assert_eq!(forest.snapshot_dimension(), 4);
    assert_eq!(forest.snapshot_updates_seen(), 0);
}

/// Per-feature detector + severity classifier (core cross-cuts).
#[cfg(feature = "core")]
#[test]
fn core_per_feature_cusum_plus_severity_via_facade() {
    use anomstream::{PerFeatureCusum, PerFeatureCusumConfig, Severity, SeverityBands};
    let mut det = PerFeatureCusum::<2>::new(PerFeatureCusumConfig {
        slack: 0.5,
        threshold: 5.0,
    });
    let _ = det.observe(&[100.0, 200.0]);
    for _ in 0..20 {
        let _ = det.observe(&[105.0, 200.0]);
    }
    let bands = SeverityBands::default();
    assert_eq!(bands.classify(0.5), Severity::Normal);
}

/// `std`-gated detectors (Adwin, Bloom, Shingled) reachable
/// when `std` is on. Only checks the type names compile —
/// behaviour is exercised in the owning member's own test
/// suite.
#[cfg(all(feature = "core", feature = "std"))]
#[test]
fn core_std_gated_types_reachable_via_facade() {
    use anomstream::{
        AdwinDetector, BloomFilter, CountMinSketch, HyperLogLog, ShingledForest,
        ShingledForestBuilder, SpaceSaving,
    };
    let _: ShingledForestBuilder<4> = ShingledForestBuilder::<4>::new();
    let _ = BloomFilter::new(64, 0.01).unwrap();
    let _: CountMinSketch = CountMinSketch::new(16, 4).unwrap();
    let _ = HyperLogLog::new(10).unwrap();
    let _: SpaceSaving<u32> = SpaceSaving::new(8).unwrap();
    let _ = AdwinDetector::new(1.0, 0.01, 1024).unwrap();
    let _: Option<ShingledForest<4>> = None;
}

// --- Triage smoke --------------------------------------------
//
// Triage types reachable through the facade root: full
// pipeline (forest build → alert record → cluster → calibrate)
// composes through `anomstream::...` imports only.

#[cfg(all(
    feature = "core",
    feature = "triage",
    feature = "std",
    feature = "serde",
    feature = "postcard"
))]
#[test]
fn triage_clusterer_plus_calibrator_plus_audit_via_facade() {
    use anomstream::{
        AlertClusterer, AlertContext, AlertRecord, ClusterDecision, ForestBuilder, PlattCalibrator,
        PlattFitConfig,
    };
    let mut forest = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(16)
        .seed(42)
        .build()
        .unwrap();
    for i in 0..32 {
        let v = f64::from(i) * 0.01;
        forest.update([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
    }
    let ctx = AlertContext::<String>::untenanted(1_000);
    let rec: AlertRecord<String, 4> = AlertRecord::from_forest(&forest, &[5.0; 4], &ctx).unwrap();
    let mut clusterer: AlertClusterer<String, 4> = AlertClusterer::new(0.95, 60_000).unwrap();
    let decision = clusterer.observe(rec);
    assert!(matches!(decision, ClusterDecision::NewCluster(_)));

    let calibration = vec![(0.5, false), (4.0, true), (0.6, false), (3.9, true)];
    let cal = PlattCalibrator::fit(&calibration, PlattFitConfig::default()).unwrap();
    let p = cal.calibrate(3.5);
    assert!((0.0..=1.0).contains(&p));
}

/// `std`-gated triage types (LSH clusterer, feedback ledger,
/// SAGE) reachable.
#[cfg(all(feature = "triage", feature = "std"))]
#[test]
fn triage_std_gated_types_reachable_via_facade() {
    use anomstream::{
        FEEDBACK_DEFAULT_CAPACITY, FeedbackLabel, FeedbackStore, LshAlertClusterer,
        SAGE_DEFAULT_PERMUTATIONS, SageEstimator,
    };
    let _: LshAlertClusterer = LshAlertClusterer::default_lsh();
    let _: FeedbackStore<4> = FeedbackStore::new(FEEDBACK_DEFAULT_CAPACITY, 1.0, 1.0).unwrap();
    let _ = FeedbackLabel::Confirmed;
    let _: SageEstimator<4> = SageEstimator::default_anchor([0.0_f64; 4]).unwrap();
    const { assert!(SAGE_DEFAULT_PERMUTATIONS > 0) };
}

// --- Hot-path smoke ------------------------------------------
//
// Hot-path primitives reachable through `anomstream::hot_path::*`.

#[cfg(feature = "hotpath")]
#[test]
fn hot_path_sampler_plus_rate_cap_via_facade() {
    use anomstream::hot_path::{PrefixRateCap, UpdateSampler};
    use core::num::{NonZeroU32, NonZeroU64};
    let sampler = UpdateSampler::new(4);
    // Stride sampler: 1st call true (counter = 0 % 4 == 0), next 3 false.
    assert!(sampler.accept_stride());
    assert!(!sampler.accept_stride());

    let cap = PrefixRateCap::new(
        NonZeroU32::new(10).unwrap(),
        NonZeroU64::new(1_000).unwrap(),
    );
    assert!(cap.check_and_record(0x1234_5678_u64, 0));
}

/// `update_channel` re-export reachable.
#[cfg(feature = "hotpath")]
#[test]
fn hot_path_update_channel_reachable_via_facade() {
    use anomstream::hot_path::{UpdateConsumer, UpdateProducer, update_channel};
    let (_p, _c): (UpdateProducer<4>, UpdateConsumer<4>) = update_channel(8);
}

/// `try_update_channel` Result variant + `MAX_CHANNEL_CAPACITY`
/// const reachable through the facade.
#[cfg(feature = "hotpath")]
#[test]
fn hot_path_try_update_channel_via_facade() {
    use anomstream::hot_path::{MAX_CHANNEL_CAPACITY, try_update_channel};
    assert!(try_update_channel::<4>(0).is_err());
    assert!(try_update_channel::<4>(MAX_CHANNEL_CAPACITY + 1).is_err());
    let (p, _c) = try_update_channel::<4>(8).unwrap();
    p.flush_metrics();
}

// --- Audit-integrity smoke -----------------------------------
//
// HMAC-SHA256 chain re-exports reachable through the facade
// when `audit-integrity` is on. Round-trip a single-entry
// chain to confirm the wiring composes end-to-end.

#[cfg(feature = "audit-integrity")]
#[test]
fn audit_chain_round_trip_via_facade() {
    use anomstream::{
        AUDIT_CHAIN_GENESIS_PREV, AUDIT_CHAIN_MIN_KEY_LEN, AlertContext, AlertRecord, AuditChain,
        ForestBuilder, verify_audit_chain,
    };
    let mut forest = ForestBuilder::<4>::new()
        .num_trees(50)
        .sample_size(16)
        .seed(42)
        .build()
        .unwrap();
    for i in 0..32 {
        let v = f64::from(i) * 0.01;
        forest.update([v, v + 0.1, v + 0.2, v + 0.3]).unwrap();
    }
    let key = [0x42u8; AUDIT_CHAIN_MIN_KEY_LEN];
    let mut chain: AuditChain<String, 4> = AuditChain::new(&key).unwrap();
    let ctx = AlertContext::<String>::untenanted(1_000);
    let rec: AlertRecord<String, 4> = AlertRecord::from_forest(&forest, &[5.0; 4], &ctx).unwrap();
    let entry = chain.append(rec).unwrap();
    verify_audit_chain(
        core::slice::from_ref(&entry),
        &key,
        &AUDIT_CHAIN_GENESIS_PREV,
    )
    .unwrap();
}

// --- Namespace escape hatches --------------------------------
//
// Verify `core_lib`, `triage_lib`, `hotpath_lib` reach deep
// module paths that the explicit root re-exports do not surface.

#[cfg(feature = "core")]
#[test]
fn core_lib_namespace_reaches_deep_paths() {
    let _: anomstream::core_lib::error::RcfError =
        anomstream::core_lib::error::RcfError::InvalidConfig("smoke".into());
}

#[cfg(feature = "triage")]
#[test]
fn triage_lib_namespace_reaches_deep_paths() {
    let _: anomstream::triage_lib::audit::AlertContext<String> =
        anomstream::triage_lib::audit::AlertContext::<String>::untenanted(0);
}

#[cfg(feature = "hotpath")]
#[test]
fn hotpath_lib_namespace_reaches_deep_paths() {
    use core::num::{NonZeroU32, NonZeroU64};
    let _: anomstream::hotpath_lib::PrefixRateCap = anomstream::hotpath_lib::PrefixRateCap::new(
        NonZeroU32::new(1).unwrap(),
        NonZeroU64::new(1).unwrap(),
    );
}
