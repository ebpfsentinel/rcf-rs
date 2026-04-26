//! `anomstream` — streaming anomaly detection toolkit (facade).
//!
//! Umbrella crate that re-exports the three workspace members
//! under feature gates so consumers get a single import path
//! regardless of which layers they pull in:
//!
//! ```toml
//! [dependencies]
//! anomstream = { version = "0.2", default-features = false, features = ["core", "triage"] }
//! ```
//!
//! Feature matrix:
//!
//! | Feature | Pulls in | Purpose |
//! |---|---|---|
//! | `core` | `anomstream-core` | Detectors + streaming primitives + `MetricsSink` + `SeverityBands` |
//! | `triage` | `anomstream-triage` | Platt, SAGE, alert clustering, feedback, audit records |
//! | `hotpath` | `anomstream-hotpath` | eBPF-style ingress `UpdateSampler` / `PrefixRateCap` / `update_channel` |
//!
//! `triage` and `hotpath` both depend on `core`; enabling them
//! implies `core`. Default feature set enables all three plus
//! `std` / `parallel` / `serde` / `postcard` passthroughs.
//!
//! # Consumer DX
//!
//! Core + triage types are re-exported at the crate root with the
//! same spellings as the owning member crate, so consumers write
//! one `use anomstream::...;` regardless of which layer they
//! depend on:
//!
//! ```ignore
//! use anomstream::{ForestBuilder, PerFeatureCusum, PerFeatureCusumConfig};
//! #[cfg(feature = "triage")]
//! use anomstream::{AlertClusterer, PlattCalibrator};
//! #[cfg(feature = "hotpath")]
//! use anomstream::hot_path::{PrefixRateCap, UpdateSampler};
//! ```
//!
//! Hot-path primitives live under a [`hot_path`] submodule to
//! preserve the `anomstream::hot_path::*` import path used by
//! pre-workspace-split callers and to flag the opinionated nature
//! of that layer.
//!
//! Sibling member namespaces also stay accessible verbatim via
//! [`core_lib`], [`triage_lib`], [`hotpath_lib`] when a consumer
//! needs to spell a deep module path (e.g.
//! `anomstream::core_lib::persistence::Snapshot`,
//! `anomstream::core_lib::serde_util::*`,
//! `anomstream::triage_lib::audit::AlertRecordShadow`) that the
//! root re-exports do not surface.
//!
//! # `SemVer` scope
//!
//! Every `pub use` below is the **complete committed public
//! surface** of the `anomstream` crate. Re-exports are spelled
//! out item by item rather than via glob — `pub use
//! member::*` would silently forward every future addition,
//! private-ish helper, or accidentally-public escape hatch
//! shipped by a member crate, and `cargo` does not read the
//! disclaimer comments around it. Enumerating the surface
//! explicitly means a member-crate item promoted to `pub` is
//! invisible from `anomstream::...` until a follow-up release
//! adds the matching `pub use` here, and `SemVer` guarantees
//! travel through `cargo publish` rather than through prose.
//!
//! Items reachable only via [`core_lib`] / [`triage_lib`] /
//! [`hotpath_lib`] are **not** part of that committed surface —
//! their stability follows the owning member crate's `SemVer`
//! contract directly. Catalogue + member ownership table:
//! `docs/features.md`.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
// Every member-crate doc link is either full-path
// (`[crate::Foo]`) or uses an in-scope identifier, so rustdoc
// resolves them cleanly under the facade namespace. The old
// blanket allow for `rustdoc::broken_intra_doc_links` is no
// longer needed and CI (`RUSTDOCFLAGS="-D
// rustdoc::broken_intra_doc_links"`) fails on new breakage.

// -- Core re-exports ------------------------------------------
//
// Mirrors `anomstream_core/src/lib.rs` root re-exports verbatim.
// `cfg` gates here mirror the gates on the owning module in
// core: anything `cfg(feature = "std")` in core stays gated on
// `feature = "std"` here, and so on. Every entry corresponds
// 1:1 to a `pub use` in the owning member.

#[cfg(all(feature = "core", feature = "std"))]
#[doc(inline)]
pub use anomstream_core::{
    ADWIN_DEFAULT_DELTA, ADWIN_DEFAULT_WINDOW, AdwinDetector, BLOOM_DEFAULT_FPR, BLOOM_MAX_HASHES,
    BloomFilter, CountMinSketch, DriftAwareForest, DriftRecoveryConfig, DynamicForest,
    HLL_DEFAULT_PRECISION, HLL_MAX_PRECISION, HLL_MIN_PRECISION, HeavyHitter, HeavyHitterEntry,
    HyperLogLog, MATRIX_PROFILE_MIN_WINDOW, MatrixProfile, PotDetector, ReadinessSummary,
    SPACE_SAVING_DEFAULT_CAPACITY, SPOT_DEFAULT_ALERT_P, SPOT_DEFAULT_QUANTILE, ShingledForest,
    ShingledForestBuilder, SpaceSaving, TenantForestPool, TsbAdMDataset, VUS_PR_DEFAULT_MAX_BUFFER,
    chi_squared_survival_even, fisher_combine, range_auc_pr, vus_pr, vus_pr_with_buffer,
};

#[cfg(feature = "core")]
#[doc(inline)]
pub use anomstream_core::{
    AnomalyGrade, AnomalyScore, AttributionStability, AttributionVisitor, BootstrapReport,
    BoundingBox, Centroid, CusumConfig, Cut, DEFAULT_CI_Z_FACTOR, DiVector, DriftDirection,
    DriftKind, DriftLevel, DriftVerdict, EarlyTermConfig, EarlyTermScore, EmaStats,
    EwmaAccumulator, FeatureDriftDetector, FeatureGroup, FeatureGroups, FeatureGroupsBuilder,
    ForensicBaseline, ForestBuilder, ForestSnapshot, GroupScores, HistogramConfig, InternalData,
    LeafData, MetaDriftDetector, MetricsSink, NodeRef, NodeStore, NodeView, NodeViewMut, NoopSink,
    NormParams, NormStrategy, Normalizer, OnlineStats, PerFeatureCusum, PerFeatureCusumAccumulator,
    PerFeatureCusumAlert, PerFeatureCusumConfig, PerFeatureCusumResult, PerFeatureEwma,
    PerFeatureEwmaConfig, PerFeatureEwmaResult, Point, PointAccessor, PointStore, RandomCutForest,
    RandomCutTree, RcfConfig, RcfError, RcfResult, ReservoirSampler, SamplerOp, ScalarScoreVisitor,
    ScoreAttributionVisitor, ScoreHistogram, ScoreWithConfidence, Severity, SeverityBands,
    TDIGEST_DEFAULT_COMPRESSION, TDigest, ThresholdMode, ThresholdedConfig, ThresholdedForest,
    ThresholdedForestBuilder, Visitor,
};

// -- Triage re-exports ----------------------------------------
//
// Mirrors `anomstream_triage/src/lib.rs` root re-exports.

#[cfg(feature = "triage")]
#[doc(inline)]
pub use anomstream_triage::{
    ALERT_RECORD_VERSION, AlertCluster, AlertClusterer, AlertContext, AlertRecord, ClusterDecision,
    PlattCalibrator, PlattFitConfig,
};

#[cfg(all(feature = "triage", feature = "std"))]
#[doc(inline)]
pub use anomstream_triage::{
    FEEDBACK_DEFAULT_CAPACITY, FEEDBACK_DEFAULT_SIGMA, FEEDBACK_DEFAULT_STRENGTH, FeedbackLabel,
    FeedbackStore, LshAlertClusterer, LshClusterDecision, SAGE_DEFAULT_PERMUTATIONS,
    SAGE_DEFAULT_SEED, SageEstimator, SageExplanation,
};

// -- Hot-path re-exports --------------------------------------
//
// Mirrors `anomstream_hotpath/src/lib.rs` root surface. Lives
// under a `hot_path` submodule to keep the
// `anomstream::hot_path::*` spelling that pre-workspace-split
// callers depend on.

/// eBPF-style ingress primitives (`UpdateSampler`, `PrefixRateCap`,
/// `update_channel`) — re-exported as a submodule so the path
/// `anomstream::hot_path::UpdateSampler` matches the pre-split
/// `anomstream_core::hot_path::UpdateSampler` spelling. Like the
/// crate-root re-exports above, every entry is enumerated rather
/// than glob-imported so `cargo` (not the docstring) is the
/// authority on what counts as the committed surface.
#[cfg(feature = "hotpath")]
pub mod hot_path {
    #[doc(inline)]
    pub use anomstream_hotpath::{
        PrefixRateCap, UpdateConsumer, UpdateProducer, UpdateSampler, update_channel,
        update_channel_with_sink,
    };
}

// -- Named namespace escape hatches ---------------------------
//
// Direct re-exports of the member crate roots so deep module
// paths (`anomstream::core_lib::persistence::Snapshot`,
// `anomstream::triage_lib::audit::AlertRecordShadow`,
// `anomstream::hotpath_lib::PrefixRateCap`) stay reachable
// without taking on a direct member-crate dependency. These
// namespaces are `SemVer`-scoped to the member crate, NOT to
// the facade.

/// Direct access to the `anomstream-core` crate namespace.
#[cfg(feature = "core")]
#[doc(inline)]
pub use anomstream_core as core_lib;

/// Direct access to the `anomstream-triage` crate namespace.
#[cfg(feature = "triage")]
#[doc(inline)]
pub use anomstream_triage as triage_lib;

/// Direct access to the `anomstream-hotpath` crate namespace.
#[cfg(feature = "hotpath")]
#[doc(inline)]
pub use anomstream_hotpath as hotpath_lib;
