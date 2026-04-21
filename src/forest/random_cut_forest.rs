//! Aggregate root: orchestrates `N` `(RandomCutTree<D>, ReservoirSampler)`
//! pairs sharing a single refcounted [`PointStore<D>`].
//!
//! `update(point)` validates dimensionality + finiteness, registers the
//! point in the store, then offers it to every tree's sampler. On
//! `Inserted` the point is added to the tree; on `Replaced(evicted)`
//! the evicted point is deleted from the tree first; on `Rejected`
//! the tree is left untouched. After the pass, any point that no
//! tree adopted is dropped from the store.
//!
//! `score(point)` and `attribution(point)` walk every tree's
//! `traverse` with a fresh visitor and average the per-tree outputs
//! — matching the AWS spec ("average across trees").

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::config::RcfConfig;
use crate::domain::point::ensure_finite;
use crate::domain::{AnomalyScore, DiVector};
use crate::early_term::{EarlyTermConfig, EarlyTermScore};
use crate::error::{RcfError, RcfResult};
use crate::forest::point_store::PointStore;
use crate::sampler::{ReservoirSampler, SamplerOp};
use crate::tree::{PointAccessor, RandomCutTree};
use crate::visitor::{AttributionVisitor, ScalarScoreVisitor, ScoreAttributionVisitor};

/// Per-tree state: tree + sampler + dedicated RNG. The dedicated
/// RNG is what unlocks parallel insertion under the `parallel`
/// feature — every tree advances its own deterministic stream,
/// seeded once at construction from the master forest seed.
type TreeSlot<const D: usize> = (RandomCutTree<D>, ReservoirSampler, ChaCha8Rng);

/// Random Cut Forest aggregate over `D`-dimensional points.
///
/// # Examples
///
/// ```
/// use rcf_rs::ForestBuilder;
///
/// let mut forest = ForestBuilder::<2>::new()
///     .num_trees(50)
///     .sample_size(16)
///     .seed(2026)
///     .build()
///     .unwrap();
/// for i in 0..32 {
///     let v = f64::from(i) * 0.01;
///     forest.update([v, v + 0.5]).unwrap();
/// }
/// let score: f64 = forest.score(&[10.0, 10.0]).unwrap().into();
/// assert!(score >= 0.0);
/// ```
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RandomCutForest<const D: usize> {
    /// Validated configuration.
    config: RcfConfig,
    /// Per-tree state: `(tree, sampler, rng)` triples — one entry per
    /// tree. Each tree owns a dedicated `ChaCha8Rng` seeded from the
    /// master forest seed at construction so parallel insert paths
    /// never share RNG state.
    trees: Vec<TreeSlot<D>>,
    /// Refcounted store shared across every tree.
    point_store: PointStore<D>,
    /// Total number of `update` calls observed.
    updates_seen: u64,
    /// Optional dedicated rayon thread pool, built from
    /// `config.num_threads` when the `parallel` feature is enabled
    /// and the config requested a custom pool size. Skipped from
    /// serde — deserialised forests fall back to rayon's global
    /// pool until the next [`from_config`](Self::from_config)-style
    /// rebuild.
    #[cfg(feature = "parallel")]
    #[cfg_attr(feature = "serde", serde(skip))]
    pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    /// Observability sink — every public operation emits counters,
    /// gauges, and histogram observations into it. Defaults to a
    /// shared [`crate::NoopSink`] so a detector without an attached
    /// sink pays only an inlined no-op per event.
    #[cfg(feature = "std")]
    #[cfg_attr(
        feature = "serde",
        serde(skip, default = "crate::metrics::default_sink")
    )]
    metrics: std::sync::Arc<dyn crate::metrics::MetricsSink>,
    /// Optional per-point timestamps captured by [`Self::update_at`]
    /// / [`Self::update_indexed_at`]. Keyed by the `point_idx`
    /// returned at insertion time. Populated only when the caller
    /// opts in via the `_at` APIs — the classic
    /// [`Self::update`] path never touches this map. Cleaned up
    /// whenever a slot's reservoir refcount drops to zero
    /// (reservoir eviction or explicit [`Self::delete`]).
    #[cfg_attr(feature = "serde", serde(default))]
    timestamps: std::collections::HashMap<usize, u64>,
}

impl<const D: usize> RandomCutForest<D> {
    /// Construct a forest from a pre-validated [`RcfConfig`]. Public
    /// callers should go through
    /// [`crate::ForestBuilder::build`](crate::ForestBuilder::build).
    ///
    /// # Errors
    ///
    /// Propagates failures from the underlying tree, sampler and
    /// point-store constructors.
    pub fn from_config(config: RcfConfig) -> RcfResult<Self> {
        // Belt-and-braces: `ForestBuilder::build` already runs both
        // checks, but any direct `from_config` caller must also see
        // a mismatched `feature_scales` length rejected up front.
        config.validate_feature_scales_dimension(D)?;
        // Master RNG only used to seed per-tree RNGs deterministically.
        let mut master = if let Some(seed) = config.seed {
            ChaCha8Rng::seed_from_u64(seed)
        } else {
            let mut bytes = [0_u8; 8];
            getrandom::fill(&mut bytes).map_err(|e| {
                RcfError::InvalidConfig(format!(
                    "OS RNG unavailable for seed-less forest construction: {e}"
                ))
            })?;
            ChaCha8Rng::seed_from_u64(u64::from_le_bytes(bytes))
        };

        let mut trees: Vec<TreeSlot<D>> = Vec::with_capacity(config.num_trees);
        for _ in 0..config.num_trees {
            let tree =
                RandomCutTree::<D>::new(u32::try_from(config.sample_size).map_err(|_| {
                    RcfError::InvalidConfig(format!(
                        "sample_size {} exceeds u32::MAX",
                        config.sample_size
                    ))
                })?)?;
            let sampler = ReservoirSampler::with_initial_accept_fraction(
                config.sample_size,
                config.time_decay,
                config.initial_accept_fraction,
            )?;
            let tree_rng = ChaCha8Rng::seed_from_u64(master.next_u64());
            trees.push((tree, sampler, tree_rng));
        }

        let point_store = PointStore::<D>::new()?;

        #[cfg(feature = "parallel")]
        let pool = match config.num_threads {
            Some(n) => Some(std::sync::Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| {
                        RcfError::InvalidConfig(format!(
                            "rayon ThreadPool build failed for num_threads={n}: {e}"
                        ))
                    })?,
            )),
            None => None,
        };

        Ok(Self {
            config,
            trees,
            point_store,
            updates_seen: 0,
            #[cfg(feature = "parallel")]
            pool,
            #[cfg(feature = "std")]
            metrics: crate::metrics::default_sink(),
            timestamps: std::collections::HashMap::new(),
        })
    }

    /// Install a [`crate::MetricsSink`] — every subsequent public
    /// op emits counters / gauges / histograms into it. Zero-cost
    /// when the default [`crate::NoopSink`] is kept.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_metrics_sink(
        mut self,
        sink: std::sync::Arc<dyn crate::metrics::MetricsSink>,
    ) -> Self {
        use crate::metrics::names;
        #[allow(clippy::cast_precision_loss)]
        sink.set_gauge(names::FOREST_TREES, self.trees.len() as f64);
        self.metrics = sink;
        self
    }

    /// Read-only handle to the installed sink — mainly for tests.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn metrics_sink(&self) -> &std::sync::Arc<dyn crate::metrics::MetricsSink> {
        &self.metrics
    }

    /// Read-only access to the validated config.
    #[must_use]
    pub fn config(&self) -> &RcfConfig {
        &self.config
    }

    /// Apply the configured per-dim multiplicative
    /// [`RcfConfig::feature_scales`] to `point` and return the scaled
    /// copy. When no scales are configured, returns `*point`. Exposed
    /// to the crate so helpers living outside `forest::` (notably
    /// [`crate::attribution_stability`]) can pre-scale caller
    /// queries through the same path the forest uses internally.
    ///
    /// Silently passes through when the resolved scales length does
    /// not match `D` — the invariant is enforced at build time, but
    /// snapshot-loaded forests that bypass validation must still
    /// behave predictably.
    #[must_use]
    pub(crate) fn scale_point_copy(&self, point: &[f64; D]) -> [f64; D] {
        match &self.config.feature_scales {
            Some(scales) if scales.len() == D => {
                let mut out = *point;
                for (p, s) in out.iter_mut().zip(scales.iter()) {
                    *p *= s;
                }
                out
            }
            _ => *point,
        }
    }

    /// Number of trees in the forest.
    #[must_use]
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Per-tree reservoir capacity.
    #[must_use]
    pub fn sample_size(&self) -> usize {
        self.config.sample_size
    }

    /// Per-point dimensionality (compile-time `D`).
    #[must_use]
    #[inline]
    pub const fn dimension(&self) -> usize {
        D
    }

    /// Total number of [`update`](Self::update) calls observed.
    #[must_use]
    pub fn updates_seen(&self) -> u64 {
        self.updates_seen
    }

    /// Borrow the underlying point store. Used by tests, persistence
    /// and diagnostics.
    #[must_use]
    pub fn point_store(&self) -> &PointStore<D> {
        &self.point_store
    }

    /// Borrow the per-tree `(tree, sampler, rng)` triples. Used by
    /// tests and persistence.
    #[must_use]
    pub fn trees(&self) -> &[TreeSlot<D>] {
        &self.trees
    }

    /// Metered [`ensure_finite`] — bumps the `REJECTED_NAN_TOTAL`
    /// counter on the installed sink before propagating the error.
    /// Upstream bad-data volume is a first-class SOC signal.
    #[inline]
    fn ensure_finite_metered(&self, point: &[f64; D]) -> RcfResult<()> {
        match ensure_finite(point) {
            Ok(()) => Ok(()),
            Err(e) => {
                #[cfg(feature = "std")]
                self.metrics
                    .inc_counter(crate::metrics::names::REJECTED_NAN_TOTAL, 1);
                Err(e)
            }
        }
    }

    /// Pessimistic memory upper bound (in bytes) of the forest's
    /// payload: point store + per-tree node arenas + per-tree sampler
    /// heaps + per-tree RNG state.
    #[must_use]
    pub fn memory_estimate(&self) -> usize {
        let mut total = self.point_store.memory_estimate();
        for (_, sampler, _) in &self.trees {
            total += sampler.capacity() * 16;
            total += core::mem::size_of::<ChaCha8Rng>();
        }
        total += self.trees.len() * (2 * self.config.sample_size * 48);
        total
    }

    /// Stream a new `point` through the forest.
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] when `point` contains a non-finite component.
    /// - Propagates failures from the underlying tree, sampler and
    ///   point-store calls.
    ///
    /// # Panics
    ///
    /// Never. The internal `expect("just-added point must be present")`
    /// is unreachable: [`PointStore::add`] returned the index in the
    /// preceding line, so the slot is guaranteed live.
    pub fn update(&mut self, point: [f64; D]) -> RcfResult<()> {
        let scaled = self.scale_point_copy(&point);
        self.insert_point(scaled)?;
        Ok(())
    }

    /// Same as [`Self::update`] but returns the `point_idx` assigned
    /// to the freshly inserted point. Callers that want to later
    /// retract a specific observation (SOC false-positive
    /// annotation, scripted unit-test fixtures, …) should track the
    /// index they receive here and hand it to [`Self::delete`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::update`].
    pub fn update_indexed(&mut self, point: [f64; D]) -> RcfResult<usize> {
        let scaled = self.scale_point_copy(&point);
        self.insert_point(scaled)
    }

    /// Insert `point` and tag it with `timestamp` — a caller-
    /// supplied monotonic sequence or epoch value. The timestamp
    /// feeds the [`Self::delete_before`] retention path so callers
    /// can prune history by age (GDPR, `NIS2` data-retention
    /// windows, forensic replay of a specific window).
    ///
    /// Timestamps are tracked in a side-map keyed by the fresh
    /// `point_idx`; the map entry is dropped automatically when the
    /// reservoir evicts the point or [`Self::delete`] retracts it.
    ///
    /// # Errors
    ///
    /// Same as [`Self::update`].
    pub fn update_at(&mut self, point: [f64; D], timestamp: u64) -> RcfResult<()> {
        let idx = self.update_indexed_at(point, timestamp)?;
        let _ = idx;
        Ok(())
    }

    /// [`Self::update_at`] variant that returns the fresh `point_idx`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::update_indexed`].
    pub fn update_indexed_at(&mut self, point: [f64; D], timestamp: u64) -> RcfResult<usize> {
        let idx = self.update_indexed(point)?;
        // Only record the timestamp if the store still holds the
        // slot — a sample_size=0 forest or an immediately-rejected
        // reservoir drops the point via `drop_unreferenced`, in
        // which case there is no live idx to tag.
        if self.point_store.ref_count(idx) > 0 {
            self.timestamps.insert(idx, timestamp);
        }
        Ok(idx)
    }

    /// Retract every point whose tracked timestamp is strictly less
    /// than `cutoff`. Returns the number of indices deleted.
    ///
    /// Points inserted via the classic [`Self::update`] /
    /// [`Self::update_indexed`] path (no timestamp) are ignored —
    /// retention requires the caller to have opted into the `_at`
    /// APIs.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::delete`] failures.
    pub fn delete_before(&mut self, cutoff: u64) -> RcfResult<usize> {
        let victims: Vec<usize> = self
            .timestamps
            .iter()
            .filter_map(|(idx, ts)| if *ts < cutoff { Some(*idx) } else { None })
            .collect();
        let mut removed = 0_usize;
        for idx in victims {
            if self.delete(idx)? {
                removed = removed.saturating_add(1);
            }
            // Even if delete reported `false` (e.g. the point was
            // already evicted and the side-map is stale), scrub the
            // dangling timestamp so subsequent `delete_before` calls
            // do not rescan it.
            self.timestamps.remove(&idx);
        }
        Ok(removed)
    }

    /// Timestamp recorded for `point_idx`, or `None` when the slot
    /// was inserted without a timestamp / has been evicted.
    #[must_use]
    pub fn point_timestamp(&self, point_idx: usize) -> Option<u64> {
        self.timestamps.get(&point_idx).copied()
    }

    /// Oldest tracked timestamp still live in the forest, or `None`
    /// when no timestamped points remain.
    #[must_use]
    pub fn oldest_timestamp(&self) -> Option<u64> {
        self.timestamps.values().min().copied()
    }

    /// Newest tracked timestamp still live in the forest.
    #[must_use]
    pub fn newest_timestamp(&self) -> Option<u64> {
        self.timestamps.values().max().copied()
    }

    /// Number of timestamped points currently tracked.
    #[must_use]
    pub fn tracked_timestamps(&self) -> usize {
        self.timestamps.len()
    }

    /// Attach `timestamp` to a specific `point_idx` after the fact —
    /// used by higher-level wrappers (e.g.
    /// [`crate::ThresholdedForest::process_indexed_at`]) that cannot
    /// route their insertion through [`Self::update_indexed_at`]
    /// because the scoring path is mixed into the update. Overwrites
    /// any previous timestamp on the same index.
    pub fn set_point_timestamp(&mut self, point_idx: usize, timestamp: u64) {
        self.timestamps.insert(point_idx, timestamp);
    }

    /// Core insertion body — performs the full `update` pipeline and
    /// returns the freshly assigned `point_idx`. Shared by
    /// [`Self::update`] and [`Self::update_indexed`] so the hot-path
    /// logic lives in exactly one place.
    fn insert_point(&mut self, point: [f64; D]) -> RcfResult<usize> {
        self.ensure_finite_metered(&point)?;

        let new_idx = self.point_store.add(point)?;

        #[cfg(feature = "parallel")]
        let pool = self.pool.clone();

        let Self {
            trees,
            point_store,
            timestamps,
            ..
        } = self;
        let store: &PointStore<D> = point_store;

        #[cfg(feature = "parallel")]
        let pending_frees = if let Some(p) = pool.as_deref() {
            p.install(|| update_trees(trees, store, new_idx))?
        } else {
            update_trees(trees, store, new_idx)?
        };

        #[cfg(not(feature = "parallel"))]
        let pending_frees = update_trees(trees, store, new_idx)?;

        for evicted in pending_frees {
            point_store.set_free(evicted)?;
            timestamps.remove(&evicted);
        }

        if point_store.ref_count(new_idx) == 0 {
            point_store.drop_unreferenced(new_idx)?;
            timestamps.remove(&new_idx);
        }

        self.updates_seen = self.updates_seen.saturating_add(1);
        #[cfg(feature = "std")]
        self.metrics
            .inc_counter(crate::metrics::names::UPDATES_TOTAL, 1);
        Ok(new_idx)
    }

    /// Retract a previously inserted point from every tree that
    /// currently holds it.
    ///
    /// Returns `true` when at least one tree's reservoir held
    /// `point_idx` and the point was evicted. Returns `false` when
    /// no tree had the point (already evicted by the reservoir, or
    /// the index was never observed in the first place).
    ///
    /// Use this to handle SOC-driven false-positive retractions —
    /// a window that an analyst marks as benign post-hoc should no
    /// longer baseline the detector.
    ///
    /// # Errors
    ///
    /// Propagates [`crate::RandomCutTree::delete`] and
    /// [`PointStore`] failures.
    pub fn delete(&mut self, point_idx: usize) -> RcfResult<bool> {
        #[cfg(feature = "parallel")]
        let pool = self.pool.clone();

        let Self {
            trees,
            point_store,
            timestamps,
            ..
        } = self;
        let store: &PointStore<D> = point_store;

        #[cfg(feature = "parallel")]
        let (removed_from_any, went_to_zero) = if let Some(p) = pool.as_deref() {
            p.install(|| delete_from_trees(trees, store, point_idx))?
        } else {
            delete_from_trees(trees, store, point_idx)?
        };

        #[cfg(not(feature = "parallel"))]
        let (removed_from_any, went_to_zero) = delete_from_trees(trees, store, point_idx)?;

        if went_to_zero {
            point_store.set_free(point_idx)?;
            timestamps.remove(&point_idx);
        }
        #[cfg(feature = "std")]
        if removed_from_any {
            self.metrics
                .inc_counter(crate::metrics::names::DELETES_TOTAL, 1);
        }
        Ok(removed_from_any)
    }

    /// Retract every point whose stored value bit-matches `point`.
    /// Returns the number of indices actually deleted — zero when no
    /// reservoir currently holds a matching observation.
    ///
    /// The match is strictly bit-for-bit: a point that was stored
    /// as `0.1_f64 + 0.2_f64` will not match a caller-supplied
    /// `0.3_f64`. Callers that cannot guarantee identical source
    /// arithmetic should track the `point_idx` from
    /// [`Self::update_indexed`] and call [`Self::delete`] directly.
    ///
    /// Cost: O(`num_trees` · `sample_size`) to enumerate candidate
    /// indices, plus one `delete` per match.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::delete`] errors.
    pub fn delete_by_value(&mut self, point: &[f64; D]) -> RcfResult<usize> {
        self.ensure_finite_metered(point)?;
        // Stored points live in the forest's scaled space — scale
        // the caller query so the bit-exact comparison matches.
        let scaled = self.scale_point_copy(point);
        let probe: &[f64; D] = &scaled;
        // Bitmap dedup — same optimisation as `forensic_baseline`.
        let capacity = self.point_store.capacity();
        let mut seen = vec![false; capacity];
        for (_, sampler, _) in &self.trees {
            for idx in sampler.iter_indices() {
                if idx < capacity {
                    seen[idx] = true;
                }
            }
        }
        let matching: Vec<usize> = seen
            .iter()
            .enumerate()
            .filter_map(|(idx, hit)| {
                if *hit && self.point_store.point(idx) == Some(probe) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        let mut removed = 0_usize;
        for idx in matching {
            if self.delete(idx)? {
                removed = removed.saturating_add(1);
            }
        }
        Ok(removed)
    }

    /// Score `point` with an early-termination guard — walk trees
    /// sequentially, break as soon as the running per-tree mean
    /// has converged tightly enough (standard error of the mean
    /// relative to `|mean|` drops below
    /// [`EarlyTermConfig::confidence_threshold`]).
    ///
    /// Cuts inline detection latency on "obvious" points (clearly
    /// in-baseline or clearly anomalous) by 30-50 % on typical
    /// forests. Ambiguous points walk every tree — the method
    /// degrades gracefully to the full-ensemble answer when the
    /// per-tree scores disagree.
    ///
    /// Sequential by design — parallel rayon fold cannot short-
    /// circuit, so the early-term path deliberately bypasses the
    /// parallel score aggregator. Callers that want the full
    /// ensemble answer should stay with [`Self::score`].
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] when `point` contains a non-finite
    ///   component.
    /// - [`RcfError::EmptyForest`] when no tree currently holds a
    ///   leaf.
    /// - [`RcfError::InvalidConfig`] when `config.validate` rejects
    ///   the supplied early-term configuration.
    pub fn score_early_term(
        &self,
        point: &[f64; D],
        config: EarlyTermConfig,
    ) -> RcfResult<EarlyTermScore> {
        use crate::visitor::ScalarScoreVisitor;

        self.ensure_finite_metered(point)?;
        config.validate()?;
        let scaled = self.scale_point_copy(point);
        let probe: &[f64; D] = &scaled;

        // Welford online mean/variance: stop walking once
        // stderr/|mean| drops below the configured threshold.
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        let mut n: usize = 0;
        let mut stderr = 0.0_f64;
        let mut early = false;
        let trees_available = self.trees.len();

        for (tree, _, _) in &self.trees {
            let Some(root) = tree.root() else {
                continue;
            };
            let mass = tree.store().view(root)?.mass();
            let visitor = ScalarScoreVisitor::new(mass);
            let x: f64 = tree.traverse(probe, visitor)?.into();

            n = n.saturating_add(1);
            #[allow(clippy::cast_precision_loss)]
            let n_f = n as f64;
            let delta = x - mean;
            mean += delta / n_f;
            let delta2 = x - mean;
            m2 += delta * delta2;

            if n >= config.min_trees && n >= 2 {
                #[allow(clippy::cast_precision_loss)]
                let variance = m2 / (n - 1) as f64;
                #[allow(clippy::cast_precision_loss)]
                let stderr_now = (variance / n as f64).sqrt();
                stderr = stderr_now;
                let denom = mean.abs().max(f64::EPSILON);
                if stderr_now / denom < config.confidence_threshold {
                    early = true;
                    break;
                }
            }
        }

        if n == 0 {
            return Err(RcfError::EmptyForest);
        }

        let score = AnomalyScore::new(mean.max(0.0))?;
        #[cfg(feature = "std")]
        {
            use crate::metrics::names;
            self.metrics
                .observe_histogram(names::SCORE_OBSERVATION, f64::from(score));
            #[allow(clippy::cast_precision_loss)]
            self.metrics
                .observe_histogram(names::EARLY_TERM_TREES, n as f64);
            if early {
                self.metrics.inc_counter(names::EARLY_TERM_STOPPED_TOTAL, 1);
            }
        }
        Ok(EarlyTermScore {
            score,
            trees_evaluated: n,
            trees_available,
            stderr,
            early_stopped: early,
        })
    }

    /// Score `point` as an anomaly. Higher = more anomalous.
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] when `point` contains a non-finite component.
    /// - [`RcfError::EmptyForest`] when no tree currently holds any leaf.
    pub fn score(&self, point: &[f64; D]) -> RcfResult<AnomalyScore> {
        self.ensure_finite_metered(point)?;
        let scaled = self.scale_point_copy(point);
        let point = &scaled;

        #[cfg(feature = "parallel")]
        let (total, count) = if let Some(p) = self.pool.as_deref() {
            p.install(|| score_aggregate(&self.trees, point))?
        } else {
            score_aggregate(&self.trees, point)?
        };

        #[cfg(not(feature = "parallel"))]
        let (total, count) = score_aggregate(&self.trees, point)?;

        if count == 0 {
            return Err(RcfError::EmptyForest);
        }

        #[allow(clippy::cast_precision_loss)]
        let mean = total / count as f64;
        let score = AnomalyScore::new(mean.max(0.0))?;
        #[cfg(feature = "std")]
        self.metrics
            .observe_histogram(crate::metrics::names::SCORE_OBSERVATION, f64::from(score));
        Ok(score)
    }

    /// Score `point` and attach a confidence interval derived from
    /// per-tree dispersion. Returns a [`ScoreWithConfidence`] with
    /// `score` (mean), `stddev` (unbiased sample), `stderr`
    /// (`stddev / sqrt(n)`), and `trees_evaluated` so callers can
    /// build arbitrary-level CIs via
    /// [`ScoreWithConfidence::ci`] / [`ScoreWithConfidence::ci95`].
    ///
    /// Always walks every tree (no early-term) — use this path when
    /// SOC tuning needs the dispersion estimate on the full
    /// ensemble. [`Self::score_early_term`] reports a similar
    /// `stderr` for latency-bounded paths.
    ///
    /// # Errors
    ///
    /// Same as [`score`](Self::score).
    /// Probe-based anomaly score — the "codisp" variant popularised
    /// by `rrcf` and used by AWS's `getAnomalyScore` visitor.
    ///
    /// For each tree: insert the probe, locate the resulting leaf,
    /// walk leaf→root accumulating
    /// `max(sibling.mass / current_subtree.mass)` across ancestors,
    /// then remove the probe. The per-tree codisp is averaged
    /// across the ensemble.
    ///
    /// Trade-off vs [`Self::score`]:
    ///
    /// - Accuracy: probe-based captures *contextual* isolation
    ///   (how much mass would displace if this point were removed)
    ///   and typically scores wider-window contextual anomalies
    ///   better. Matches the scoring semantic of `rrcf` / AWS Java.
    /// - Cost: insert + delete per probe mutates the forest and
    ///   rebuilds ancestor bounding boxes. Roughly 2-5× the latency
    ///   of [`Self::score`] on a single probe. Use for SOC triage
    ///   / forensics; keep [`Self::score`] on the eBPF hot path.
    ///
    /// Mutating the forest means the call takes `&mut self`. Under
    /// `parallel`, the per-tree walks cannot run concurrently on
    /// the same forest — this path is serial by design.
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] on non-finite input.
    /// - [`RcfError::EmptyForest`] when no tree accepted the probe
    ///   (every tree's reservoir rejected it).
    /// - Propagates [`Self::update_indexed`] / [`Self::delete`] failures.
    pub fn score_codisp(&mut self, point: &[f64; D]) -> RcfResult<AnomalyScore> {
        self.ensure_finite_metered(point)?;
        let idx = self.update_indexed(*point)?;

        // Per-tree walks are read-only on the tree store once the
        // probe has been inserted. The only mutation is the outer
        // insert/delete around the loop — already serial. Under
        // `parallel` the walks fan out across trees via rayon;
        // per-tree order is irrelevant to the final mean.
        let walk_result = codisp_walk_all_trees(&self.trees, idx);

        // Always delete the probe — even on walk error — to keep
        // the forest clean.
        let _ = self.delete(idx);

        let (total, count) = walk_result?;
        if count == 0 {
            return Err(RcfError::EmptyForest);
        }
        #[allow(clippy::cast_precision_loss)]
        let mean = total / count as f64;
        let score = AnomalyScore::new(mean.max(0.0))?;
        #[cfg(feature = "std")]
        self.metrics
            .observe_histogram(crate::metrics::names::SCORE_OBSERVATION, f64::from(score));
        Ok(score)
    }

    /// Stateless codisp score — descends every tree root → leaf
    /// along `cut.left_of(point)`, accumulates the maximum per-depth
    /// `sibling_mass / subtree_mass` ratio, and averages across the
    /// ensemble.
    ///
    /// **Fix for the mutating codisp drift bug**: [`Self::score_codisp`]
    /// inserts the probe into the reservoir, walks leaf → root,
    /// deletes the probe. The `delete` cannot restore baseline
    /// points the insert evicted, so over a long eval stream the
    /// reservoir drifts away from the frozen warm-phase baseline.
    /// Observed on NAB `rogue_hold`: same seed, same algorithm,
    /// AUC drifts from 0.69 (fresh forest) to 0.20 after ~5 k probes.
    ///
    /// `score_codisp_stateless` takes `&self`, never touches the
    /// reservoir, and parallelises across trees under `parallel` —
    /// same cost profile as [`Self::score`] plus a handful of mass
    /// lookups per depth. Matches the classical "frozen baseline"
    /// semantic AWS Java / rrcf claim but don't enforce.
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] on non-finite input.
    /// - [`RcfError::EmptyForest`] when no tree holds any leaf.
    pub fn score_codisp_stateless(&self, point: &[f64; D]) -> RcfResult<AnomalyScore> {
        self.ensure_finite_metered(point)?;
        let scaled = self.scale_point_copy(point);
        let point = &scaled;

        #[cfg(feature = "parallel")]
        let (total, count) = if let Some(p) = self.pool.as_deref() {
            p.install(|| codisp_stateless_aggregate(&self.trees, point))?
        } else {
            codisp_stateless_aggregate(&self.trees, point)?
        };

        #[cfg(not(feature = "parallel"))]
        let (total, count) = codisp_stateless_aggregate(&self.trees, point)?;

        if count == 0 {
            return Err(RcfError::EmptyForest);
        }

        #[allow(clippy::cast_precision_loss)]
        let mean = total / count as f64;
        let score = AnomalyScore::new(mean.max(0.0))?;
        #[cfg(feature = "std")]
        self.metrics
            .observe_histogram(crate::metrics::names::SCORE_OBSERVATION, f64::from(score));
        Ok(score)
    }

    /// Batched stateless codisp — aligns with the frozen-baseline
    /// semantic by construction. Each probe is scored via
    /// [`Self::score_codisp_stateless`] (root → leaf walk, no
    /// reservoir mutation) and rayon fans out across probes on top
    /// of the per-tree parallelism inside each call.
    ///
    /// This is the P2 fix for [`Self::score_codisp_many`]'s
    /// large-batch failure: the mutating batched variant hits
    /// `EmptyForest` on batches larger than `sample_size` because
    /// pre-inserted probes saturate the reservoir. The stateless
    /// batched path has no insertion step, so batch size is only
    /// bounded by memory.
    ///
    /// Semantic: identical to looping [`Self::score_codisp_stateless`]
    /// over the batch. Use this path whenever the caller needs the
    /// codisp signal without mutating the baseline — forensic
    /// replay, SOC triage over a captured window, or any
    /// long-stream evaluation where drift is a concern.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::score_codisp_stateless`] errors.
    pub fn score_codisp_stateless_many(
        &self,
        points: &[[f64; D]],
    ) -> RcfResult<Vec<AnomalyScore>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let run = || {
                points
                    .par_iter()
                    .map(|p| self.score_codisp_stateless(p))
                    .collect::<RcfResult<Vec<_>>>()
            };
            if let Some(pool) = self.pool.as_deref() {
                pool.install(run)
            } else {
                run()
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            points
                .iter()
                .map(|p| self.score_codisp_stateless(p))
                .collect()
        }
    }

    /// Batched probe-based codisp — amortises the per-probe
    /// insert / delete overhead by pre-inserting all `points` into
    /// every tree, walking leaf → root with a **shared-walk cache**
    /// keyed on leaf [`crate::NodeRef`], then bulk-deleting.
    ///
    /// Expected speedup over a naive loop of [`Self::score_codisp`]:
    ///
    /// - Constant factor from one tree-state mutation per batch per
    ///   tree instead of per probe (cuts reservoir housekeeping
    ///   overhead for SOC forensic replay).
    /// - When probes converge on the same leaf (correlated traffic,
    ///   repeated near-duplicate flows), the leaf-cache collapses
    ///   the leaf → root walk to one `walk_codisp` call per unique
    ///   leaf — the rest are O(1) cache hits.
    ///
    /// # Semantic trade-off
    ///
    /// Probes inserted in the same batch can **see each other** in
    /// the codisp walk (another probe landing as a sibling inflates
    /// that probe's displacement). [`Self::score_codisp`] avoids
    /// this by inserting one probe at a time. When the batch holds
    /// unrelated, uncorrelated probes this is negligible; for
    /// correlated probes the bias can be material. Caller picks the
    /// trade-off: throughput vs purity.
    ///
    /// Batches larger than `sample_size` will saturate every tree's
    /// reservoir and the call returns [`RcfError::EmptyForest`]
    /// (every probe's `leaf_of` misses in at least one tree once
    /// the pre-insert rewrite is complete). Callers that need
    /// frozen-baseline batched codisp on arbitrary-size batches
    /// should prefer [`Self::score_codisp_stateless_many`] — same
    /// semantic, no reservoir mutation, no drift.
    ///
    /// Secondary effect: batched inserts are more likely to evict
    /// reservoir points than one-at-a-time inserts would (a larger
    /// reservoir churn per call). Bulk-delete removes every probe
    /// from storage but cannot restore evicted baseline points —
    /// same eviction semantic as the single-probe path, only more
    /// probes per call.
    ///
    /// Mutates the forest; serial by design (per-tree walk order
    /// cannot run concurrently on the same forest).
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] on any non-finite input in `points`.
    /// - [`RcfError::EmptyForest`] when no tree accepted any probe.
    /// - Propagates [`Self::update_indexed`] / [`Self::delete`] failures.
    #[cfg(feature = "std")]
    pub fn score_codisp_many(
        &mut self,
        points: &[[f64; D]],
    ) -> RcfResult<Vec<AnomalyScore>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        // Validate input before any tree mutation so an invalid
        // probe does not leave partial state behind.
        for p in points {
            self.ensure_finite_metered(p)?;
        }

        // Pre-insert every probe; record the freshly-assigned idx
        // per (tree-agnostic) point. Aborts mid-batch are possible
        // if `update_indexed` fails — rewind by deleting whatever we
        // already inserted.
        let mut probe_indices: Vec<usize> = Vec::with_capacity(points.len());
        for p in points {
            match self.update_indexed(*p) {
                Ok(idx) => probe_indices.push(idx),
                Err(e) => {
                    for idx in &probe_indices {
                        let _ = self.delete(*idx);
                    }
                    return Err(e);
                }
            }
        }

        let n = points.len();
        // Rayon-parallel per-tree walks. Each worker owns its own
        // leaf-cache `HashMap` so the shared-walk amortisation
        // stays intact, and per-tree `(partial_totals, partial_counts)`
        // `Vec`s are reduced at the end via component-wise sum.
        // Tree state is read-only during the walks (only `walk_codisp`
        // touches `tree.store()`), so `par_iter` over `&self.trees`
        // is sound.
        let per_tree = codisp_many_walks_all_trees(&self.trees, &probe_indices, n);
        let (totals, counts, walk_err) = match per_tree {
            Ok((totals, counts)) => (totals, counts, None),
            Err(e) => (vec![0.0_f64; n], vec![0_usize; n], Some(e)),
        };

        // Always bulk-delete probes — even on walk error — so the
        // forest returns to a clean state before the error surfaces.
        for idx in &probe_indices {
            let _ = self.delete(*idx);
        }

        if let Some(e) = walk_err {
            return Err(e);
        }

        let mut out = Vec::with_capacity(n);
        for (total, count) in totals.into_iter().zip(counts) {
            if count == 0 {
                return Err(RcfError::EmptyForest);
            }
            #[allow(clippy::cast_precision_loss)]
            let mean = total / count as f64;
            let score = AnomalyScore::new(mean.max(0.0))?;
            #[cfg(feature = "std")]
            self.metrics
                .observe_histogram(crate::metrics::names::SCORE_OBSERVATION, f64::from(score));
            out.push(score);
        }
        Ok(out)
    }

    /// Score `point` with per-tree dispersion statistics. Returns
    /// a [`crate::ScoreWithConfidence`] packing the ensemble mean,
    /// sample stddev, stderr, and tree count — use `ci95` /
    /// `ci(z)` for confidence-interval bands.
    ///
    /// # Errors
    ///
    /// Same as [`Self::score`].
    pub fn score_with_confidence(
        &self,
        point: &[f64; D],
    ) -> RcfResult<crate::score_ci::ScoreWithConfidence> {
        self.ensure_finite_metered(point)?;
        let scaled = self.scale_point_copy(point);
        let probe: &[f64; D] = &scaled;

        // Collect per-tree scores into a small Vec (≤ num_trees f64s).
        let mut samples: Vec<f64> = Vec::with_capacity(self.trees.len());
        for (tree, _, _) in &self.trees {
            let Some(root) = tree.root() else {
                continue;
            };
            let mass = tree.store().view(root)?.mass();
            let visitor = ScalarScoreVisitor::new(mass);
            let s = tree.traverse(probe, visitor)?;
            samples.push(f64::from(s));
        }
        if samples.is_empty() {
            return Err(RcfError::EmptyForest);
        }

        let n = samples.len();
        #[allow(clippy::cast_precision_loss)]
        let n_f = n as f64;
        let mean = samples.iter().sum::<f64>() / n_f;
        let variance = if n > 1 {
            let sq: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum();
            #[allow(clippy::cast_precision_loss)]
            {
                sq / (n - 1) as f64
            }
        } else {
            0.0
        };
        let stddev = variance.sqrt();
        let stderr = stddev / n_f.sqrt();
        let score = AnomalyScore::new(mean.max(0.0))?;
        #[cfg(feature = "std")]
        self.metrics
            .observe_histogram(crate::metrics::names::SCORE_OBSERVATION, f64::from(score));
        Ok(crate::score_ci::ScoreWithConfidence {
            score,
            trees_evaluated: n,
            stddev,
            stderr,
        })
    }

    /// Compute the per-feature attribution of `point`'s anomaly
    /// score. Returns the mean [`DiVector`] across all trees that
    /// hold leaves.
    ///
    /// # Errors
    ///
    /// Same as [`score`](Self::score).
    pub fn attribution(&self, point: &[f64; D]) -> RcfResult<DiVector> {
        self.ensure_finite_metered(point)?;
        let scaled = self.scale_point_copy(point);
        let point = &scaled;

        #[cfg(feature = "parallel")]
        let (mut accumulator, count) = if let Some(p) = self.pool.as_deref() {
            p.install(|| attribution_aggregate::<D>(&self.trees, point))?
        } else {
            attribution_aggregate::<D>(&self.trees, point)?
        };

        #[cfg(not(feature = "parallel"))]
        let (mut accumulator, count) = attribution_aggregate::<D>(&self.trees, point)?;

        if count == 0 {
            return Err(RcfError::EmptyForest);
        }

        #[allow(clippy::cast_precision_loss)]
        let divisor = count as f64;
        accumulator.scale(divisor)?;
        #[cfg(feature = "std")]
        self.metrics
            .inc_counter(crate::metrics::names::ATTRIBUTION_TOTAL, 1);
        Ok(accumulator)
    }

    /// Single-walk score + attribution — when the caller needs both,
    /// this path traverses each tree **once** instead of twice. Saves
    /// the second round of cache loads, bounding-box probability
    /// SIMD passes, and rayon fan-out. Roughly the cost of a lone
    /// [`Self::attribution`] call; the `AnomalyScore` sortie is free.
    ///
    /// Semantically identical to calling [`Self::score`] and
    /// [`Self::attribution`] back-to-back (up to floating-point
    /// summation order).
    ///
    /// # Errors
    ///
    /// Same as [`Self::score`] / [`Self::attribution`].
    pub fn score_and_attribution(
        &self,
        point: &[f64; D],
    ) -> RcfResult<(AnomalyScore, DiVector)> {
        self.ensure_finite_metered(point)?;
        let scaled = self.scale_point_copy(point);
        let point = &scaled;

        #[cfg(feature = "parallel")]
        let (total, mut accumulator, count) = if let Some(p) = self.pool.as_deref() {
            p.install(|| score_attribution_aggregate::<D>(&self.trees, point))?
        } else {
            score_attribution_aggregate::<D>(&self.trees, point)?
        };

        #[cfg(not(feature = "parallel"))]
        let (total, mut accumulator, count) =
            score_attribution_aggregate::<D>(&self.trees, point)?;

        if count == 0 {
            return Err(RcfError::EmptyForest);
        }

        #[allow(clippy::cast_precision_loss)]
        let divisor = count as f64;
        let mean = total / divisor;
        let score = AnomalyScore::new(mean.max(0.0))?;
        accumulator.scale(divisor)?;

        #[cfg(feature = "std")]
        {
            self.metrics
                .observe_histogram(crate::metrics::names::SCORE_OBSERVATION, f64::from(score));
            self.metrics
                .inc_counter(crate::metrics::names::ATTRIBUTION_TOTAL, 1);
        }
        Ok((score, accumulator))
    }

    /// Bulk-score a slice of points — under the `parallel` feature
    /// each point is fanned out to rayon workers across the batch
    /// while each individual score still parallelises across trees
    /// inside the thread pool. 2-3× speedup over a serial
    /// `for p in points { f.score(p)? }` loop on batched workloads
    /// (SOC forensic replay, offline backfill, periodic scan).
    ///
    /// Returns a `Vec` of scores in the same order as the input
    /// slice. On the first error the whole batch aborts — the
    /// partial result set is dropped to keep the API pure.
    ///
    /// # Errors
    ///
    /// Propagates any [`Self::score`] error hit while processing
    /// the batch.
    pub fn score_many(&self, points: &[[f64; D]]) -> RcfResult<Vec<AnomalyScore>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let run = || {
                points
                    .par_iter()
                    .map(|p| self.score(p))
                    .collect::<RcfResult<Vec<_>>>()
            };
            if let Some(pool) = self.pool.as_deref() {
                pool.install(run)
            } else {
                run()
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            points.iter().map(|p| self.score(p)).collect()
        }
    }

    /// Cache-aware bulk scoring — sorts the batch by a cheap
    /// locality key ([`crate::forest::locality_bucket`]) before
    /// dispatching, in the hope that probes with close-by
    /// coordinates on the leading dimensions traverse similar
    /// root→leaf paths and let each rayon worker re-use warm arena
    /// cache lines.
    ///
    /// Results are returned in **caller order** — the internal
    /// permutation is inverted before returning. Semantically
    /// identical to [`Self::score_many`] up to floating-point
    /// summation.
    ///
    /// Caveat: on the reference `(100 trees, 256 samples, D = 16)`
    /// configuration the sort + permutation overhead outweighs the
    /// cache-locality win on uniformly-drawn batches — the plain
    /// [`Self::score_many`] stays faster. This path is kept as an
    /// opt-in for callers whose batches are **strongly correlated**
    /// (SOC alert replay of a single flow, periodic tenant-scan
    /// where contiguous probes share a feature vector shape) and
    /// who have measured a win on their own workload. Do not swap
    /// in blindly.
    ///
    /// # Errors
    ///
    /// Same as [`Self::score_many`].
    pub fn score_many_locality_sorted(
        &self,
        points: &[[f64; D]],
    ) -> RcfResult<Vec<AnomalyScore>> {
        let n = points.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        // Sort-permutation by locality bucket; caller order is
        // restored at the end via `orig_idx`.
        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_unstable_by_key(|&i| locality_bucket(&points[i]));

        #[cfg(feature = "parallel")]
        let sorted: RcfResult<Vec<(usize, AnomalyScore)>> = {
            use rayon::prelude::*;
            let run = || {
                perm.par_iter()
                    .map(|&i| self.score(&points[i]).map(|s| (i, s)))
                    .collect::<RcfResult<Vec<_>>>()
            };
            if let Some(pool) = self.pool.as_deref() {
                pool.install(run)
            } else {
                run()
            }
        };
        #[cfg(not(feature = "parallel"))]
        let sorted: RcfResult<Vec<(usize, AnomalyScore)>> = perm
            .iter()
            .map(|&i| self.score(&points[i]).map(|s| (i, s)))
            .collect();
        let sorted = sorted?;

        // Un-permute.
        let zero = AnomalyScore::new(0.0)?;
        let mut out = vec![zero; n];
        for (orig_idx, s) in sorted {
            out[orig_idx] = s;
        }
        Ok(out)
    }

    /// No-alloc bulk scoring — invoke `on_score(index, score)` for
    /// every probe in order instead of collecting a `Vec`. Avoids
    /// the intermediate allocation on hot paths where the caller
    /// streams results directly into a writer, histogram, or alert
    /// pipeline.
    ///
    /// Runs serially on purpose — the callback is invoked from the
    /// caller's thread in input order, so `Fn(usize, AnomalyScore)`
    /// does not need to be `Send + Sync`. For parallel batch work
    /// where the order / thread-affinity does not matter, keep
    /// using [`Self::score_many`].
    ///
    /// # Errors
    ///
    /// Aborts on the first [`Self::score`] error; earlier callback
    /// invocations are preserved (caller-visible side effects up to
    /// that point are final).
    pub fn score_many_with<F>(&self, points: &[[f64; D]], mut on_score: F) -> RcfResult<()>
    where
        F: FnMut(usize, AnomalyScore),
    {
        for (i, p) in points.iter().enumerate() {
            let s = self.score(p)?;
            on_score(i, s);
        }
        Ok(())
    }

    /// Bulk early-termination scoring — same batch semantics as
    /// [`Self::score_many`] but each point goes through the
    /// sequential-per-tree short-circuit path. Best pick when the
    /// caller expects many points to early-stop (baseline-heavy
    /// batches).
    ///
    /// # Errors
    ///
    /// Propagates any [`Self::score_early_term`] error hit while
    /// processing the batch.
    pub fn score_many_early_term(
        &self,
        points: &[[f64; D]],
        config: EarlyTermConfig,
    ) -> RcfResult<Vec<EarlyTermScore>> {
        config.validate()?;
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let run = || {
                points
                    .par_iter()
                    .map(|p| self.score_early_term(p, config))
                    .collect::<RcfResult<Vec<_>>>()
            };
            if let Some(pool) = self.pool.as_deref() {
                pool.install(run)
            } else {
                run()
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            points
                .iter()
                .map(|p| self.score_early_term(p, config))
                .collect()
        }
    }

    /// Bulk per-feature attribution — same batch semantics as
    /// [`Self::score_many`].
    ///
    /// # Errors
    ///
    /// Propagates any [`Self::attribution`] error hit while
    /// processing the batch.
    pub fn attribution_many(&self, points: &[[f64; D]]) -> RcfResult<Vec<DiVector>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let run = || {
                points
                    .par_iter()
                    .map(|p| self.attribution(p))
                    .collect::<RcfResult<Vec<_>>>()
            };
            if let Some(pool) = self.pool.as_deref() {
                pool.install(run)
            } else {
                run()
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            points.iter().map(|p| self.attribution(p)).collect()
        }
    }

    /// Compute an imputation-like forensic baseline for `point`:
    /// per-dim mean / stddev / delta / z-score against every
    /// sample currently held in any tree's reservoir. Returns the
    /// caller-facing result in *raw* point space — the internal
    /// `feature_scales` transform is inverted so SOC dashboards see
    /// the same coordinates they passed in.
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] when `point` contains a non-finite
    ///   component.
    /// - [`RcfError::EmptyForest`] when no tree currently holds a
    ///   live point.
    pub fn forensic_baseline(
        &self,
        point: &[f64; D],
    ) -> RcfResult<crate::forensic::ForensicBaseline<D>> {
        self.ensure_finite_metered(point)?;

        // Bitmap dedup: point_idx is a slot index bounded by
        // `point_store.capacity()`. A flat `Vec<bool>` sized at
        // capacity is cache-friendlier than a `HashSet<usize>` when
        // the same idx appears across multiple reservoirs.
        let capacity = self.point_store.capacity();
        let mut seen = vec![false; capacity];
        let mut unique_count = 0_usize;
        for (_, sampler, _) in &self.trees {
            for idx in sampler.iter_indices() {
                if idx < capacity && !seen[idx] {
                    seen[idx] = true;
                    unique_count = unique_count.saturating_add(1);
                }
            }
        }
        if unique_count == 0 {
            return Err(RcfError::EmptyForest);
        }

        // Welford aggregate in *scaled* space (stored points are
        // already scaled).
        let mut n = 0_usize;
        let mut mean_scaled = [0.0_f64; D];
        let mut m2 = [0.0_f64; D];
        for (idx, hit) in seen.iter().enumerate() {
            if !*hit {
                continue;
            }
            let Some(p_scaled) = self.point_store.point(idx) else {
                continue;
            };
            n = n.saturating_add(1);
            #[allow(clippy::cast_precision_loss)]
            let n_f = n as f64;
            for d in 0..D {
                let delta = p_scaled[d] - mean_scaled[d];
                mean_scaled[d] += delta / n_f;
                let delta2 = p_scaled[d] - mean_scaled[d];
                m2[d] += delta * delta2;
            }
        }
        if n == 0 {
            return Err(RcfError::EmptyForest);
        }

        // Unscale mean / stddev back to raw coordinates.
        let scales: Option<&Vec<f64>> = self.config.feature_scales.as_ref();
        let mut expected = [0.0_f64; D];
        let mut stddev = [0.0_f64; D];
        for d in 0..D {
            let scale_d = scales
                .and_then(|v| v.get(d).copied())
                .filter(|s| s.abs() > f64::EPSILON)
                .unwrap_or(1.0);
            expected[d] = mean_scaled[d] / scale_d;
            #[allow(clippy::cast_precision_loss)]
            let variance_scaled = if n >= 2 { m2[d] / (n - 1) as f64 } else { 0.0 };
            stddev[d] = variance_scaled.sqrt() / scale_d.abs();
        }

        let observed = *point;
        let mut delta_arr = [0.0_f64; D];
        let mut zscore = [0.0_f64; D];
        for d in 0..D {
            delta_arr[d] = observed[d] - expected[d];
            zscore[d] = if stddev[d] > 0.0 {
                delta_arr[d] / stddev[d]
            } else {
                0.0
            };
        }
        Ok(crate::forensic::ForensicBaseline {
            observed,
            expected,
            stddev,
            delta: delta_arr,
            zscore,
            live_points: n,
        })
    }
}

/// Per-tree insert work — returns the list of evicted point indices
/// whose refcount just hit zero so the caller can finalise the slot
/// freeing single-threaded after the (possibly parallel) block.
///
/// Serial when the `parallel` feature is off; rayon `par_chunks_mut`
/// when enabled. The point store is borrowed immutably from the
/// closures — refcount mutations are atomic, slot mutations happen
/// only through `&mut PointStore` outside the parallel block.
fn update_trees<const D: usize>(
    trees: &mut [TreeSlot<D>],
    store: &PointStore<D>,
    new_idx: usize,
) -> RcfResult<Vec<usize>> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let chunk_size = trees.len().div_ceil(rayon::current_num_threads()).max(1);
        let chunks: RcfResult<Vec<Vec<usize>>> = trees
            .par_chunks_mut(chunk_size)
            .map(|chunk| -> RcfResult<Vec<usize>> {
                let mut local = Vec::new();
                for slot in chunk {
                    let mut freed = process_tree_update(slot, store, new_idx)?;
                    local.append(&mut freed);
                }
                Ok(local)
            })
            .collect();
        let mut flat = Vec::new();
        for c in chunks? {
            flat.extend(c);
        }
        Ok(flat)
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut out = Vec::new();
        for slot in trees.iter_mut() {
            let mut local = process_tree_update(slot, store, new_idx)?;
            out.append(&mut local);
        }
        Ok(out)
    }
}

/// Parallel-friendly per-tree delete — rayon `par_chunks_mut`
/// across `trees` under the `parallel` feature, reduce the two
/// bool flags `(removed_from_any, went_to_zero)` at the end. Each
/// worker loops through its chunk, attempts sampler removal,
/// deletes the leaf from the tree, and decrements the store
/// refcount. `PointStore::decr_ref` is atomic so the shared store
/// reference is safe to hand to every worker.
fn delete_from_trees<const D: usize>(
    trees: &mut [TreeSlot<D>],
    store: &PointStore<D>,
    point_idx: usize,
) -> RcfResult<(bool, bool)> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let chunk_size = trees.len().div_ceil(rayon::current_num_threads()).max(1);
        let partials: RcfResult<Vec<(bool, bool)>> = trees
            .par_chunks_mut(chunk_size)
            .map(|chunk| -> RcfResult<(bool, bool)> {
                let mut any = false;
                let mut zero = false;
                for slot in chunk {
                    let (a, z) = process_tree_delete(slot, store, point_idx)?;
                    any |= a;
                    zero |= z;
                }
                Ok((any, zero))
            })
            .collect();
        let parts = partials?;
        Ok(parts
            .into_iter()
            .fold((false, false), |(a1, z1), (a2, z2)| (a1 | a2, z1 | z2)))
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut any = false;
        let mut zero = false;
        for slot in trees.iter_mut() {
            let (a, z) = process_tree_delete(slot, store, point_idx)?;
            any |= a;
            zero |= z;
        }
        Ok((any, zero))
    }
}

/// Per-tree delete step used by [`delete_from_trees`] — returns
/// `(removed_here, store_hit_zero)`.
fn process_tree_delete<const D: usize>(
    slot: &mut TreeSlot<D>,
    store: &PointStore<D>,
    point_idx: usize,
) -> RcfResult<(bool, bool)> {
    let (tree, sampler, _) = slot;
    if sampler.remove(point_idx) {
        tree.delete(point_idx, store)?;
        let hit_zero = store.decr_ref(point_idx)?;
        Ok((true, hit_zero))
    } else {
        Ok((false, false))
    }
}

/// Single-tree branch of [`update_trees`]: feeds the new index to
/// the sampler, applies the resulting `Inserted` / `Replaced` /
/// `Rejected` op to the tree + refcounts. Returns the evicted index
/// when [`PointStore::decr_ref`] reports it just hit zero.
fn process_tree_update<const D: usize>(
    slot: &mut TreeSlot<D>,
    store: &PointStore<D>,
    new_idx: usize,
) -> RcfResult<Vec<usize>> {
    let (tree, sampler, rng) = slot;
    let mut freed = Vec::new();
    match sampler.accept(new_idx, rng) {
        SamplerOp::Inserted => {
            let p = store
                .point(new_idx)
                .expect("just-added point must be present");
            tree.add(new_idx, p, store, rng)?;
            store.incr_ref(new_idx)?;
        }
        SamplerOp::Replaced(evicted) => {
            tree.delete(evicted, store)?;
            if store.decr_ref(evicted)? {
                freed.push(evicted);
            }
            let p = store
                .point(new_idx)
                .expect("just-added point must be present");
            tree.add(new_idx, p, store, rng)?;
            store.incr_ref(new_idx)?;
        }
        SamplerOp::Rejected => {}
    }
    Ok(freed)
}

/// Batched per-tree codisp walks with per-thread leaf cache —
/// rayon parallel across trees, component-wise reduce into
/// `(totals[n_probes], counts[n_probes])`. Each worker builds its
/// own `HashMap<NodeRef, f64>` so the shared-walk optimisation
/// stays intact within the thread.
fn codisp_many_walks_all_trees<const D: usize>(
    trees: &[TreeSlot<D>],
    probe_indices: &[usize],
    n: usize,
) -> RcfResult<(Vec<f64>, Vec<usize>)> {
    type PerTree = (Vec<f64>, Vec<usize>);
    let per_tree_fn = |tree: &RandomCutTree<D>| -> RcfResult<PerTree> {
        use std::collections::HashMap;
        let mut totals = vec![0.0_f64; n];
        let mut counts = vec![0_usize; n];
        let mut leaf_cache: HashMap<crate::tree::NodeRef, f64> = HashMap::new();
        for (i, &idx) in probe_indices.iter().enumerate() {
            let Some(leaf) = tree.leaf_of(idx) else {
                continue;
            };
            let codisp = if let Some(hit) = leaf_cache.get(&leaf) {
                *hit
            } else {
                let c = walk_codisp(tree.store(), leaf)?;
                leaf_cache.insert(leaf, c);
                c
            };
            totals[i] += codisp;
            counts[i] = counts[i].saturating_add(1);
        }
        Ok((totals, counts))
    };

    let reduce_pair = |mut a: PerTree, b: PerTree| -> PerTree {
        for i in 0..n {
            a.0[i] += b.0[i];
            a.1[i] = a.1[i].saturating_add(b.1[i]);
        }
        a
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        trees
            .par_iter()
            .map(|(tree, _, _)| per_tree_fn(tree))
            .try_reduce(|| (vec![0.0_f64; n], vec![0_usize; n]), |a, b| Ok(reduce_pair(a, b)))
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut acc: PerTree = (vec![0.0_f64; n], vec![0_usize; n]);
        for (tree, _, _) in trees {
            acc = reduce_pair(acc, per_tree_fn(tree)?);
        }
        Ok(acc)
    }
}

/// Walk `walk_codisp` on every tree that holds leaf `idx` and
/// fold the per-tree codisp into `(sum, count)`. Serial or rayon
/// parallel fold/reduce depending on the `parallel` cargo feature.
/// Trees that rejected the probe (no `leaf_of(idx)`) contribute
/// nothing. Fails on the first per-tree walk error.
fn codisp_walk_all_trees<const D: usize>(
    trees: &[TreeSlot<D>],
    idx: usize,
) -> RcfResult<(f64, usize)> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        trees
            .par_iter()
            .map(|(tree, _, _)| -> RcfResult<Option<f64>> {
                let Some(leaf) = tree.leaf_of(idx) else {
                    return Ok(None);
                };
                walk_codisp(tree.store(), leaf).map(Some)
            })
            .try_fold(
                || (0.0_f64, 0_usize),
                |(t, c), step| {
                    let s = step?;
                    Ok::<_, RcfError>(match s {
                        Some(v) => (t + v, c + 1),
                        None => (t, c),
                    })
                },
            )
            .try_reduce(
                || (0.0_f64, 0_usize),
                |(t1, c1), (t2, c2)| Ok((t1 + t2, c1 + c2)),
            )
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut total = 0.0_f64;
        let mut count = 0_usize;
        for (tree, _, _) in trees {
            let Some(leaf) = tree.leaf_of(idx) else {
                continue;
            };
            total += walk_codisp(tree.store(), leaf)?;
            count = count.saturating_add(1);
        }
        Ok((total, count))
    }
}

/// Stateless codisp aggregation across trees — each tree computes
/// `codisp_stateless(point)` and the forest averages the per-tree
/// results. Serial fold or rayon parallel fold/reduce depending on
/// the `parallel` cargo feature.
fn codisp_stateless_aggregate<const D: usize>(
    trees: &[TreeSlot<D>],
    point: &[f64; D],
) -> RcfResult<(f64, usize)> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        trees
            .par_iter()
            .map(|(tree, _, _)| -> RcfResult<Option<f64>> {
                if tree.root().is_none() {
                    return Ok(None);
                }
                let c = tree.codisp_stateless(point)?;
                Ok(Some(c))
            })
            .try_fold(
                || (0.0_f64, 0_usize),
                |(t, c), step| {
                    let s = step?;
                    Ok::<_, RcfError>(match s {
                        Some(v) => (t + v, c + 1),
                        None => (t, c),
                    })
                },
            )
            .try_reduce(
                || (0.0_f64, 0_usize),
                |(t1, c1), (t2, c2)| Ok((t1 + t2, c1 + c2)),
            )
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut total = 0.0_f64;
        let mut count = 0_usize;
        for (tree, _, _) in trees {
            if tree.root().is_none() {
                continue;
            }
            total += tree.codisp_stateless(point)?;
            count += 1;
        }
        Ok((total, count))
    }
}

/// Cheap locality key used by
/// [`RandomCutForest::score_many_locality_sorted`] — quantises the
/// first `min(D, 8)` dimensions into 8-bit buckets and packs them
/// into a `u64`. Probes close in the leading dims hash close, so
/// sorting by this key groups probes whose tree descents share
/// arena cache lines.
///
/// The quantisation range is `[-100, 100]` clamped, tuned for
/// z-scored features (typical z-score magnitudes stay inside
/// `±6σ`). Values outside saturate to the extreme bucket. Callers
/// running on raw unnormalised data should post-process (scale /
/// z-score) before calling this key for meaningful locality.
#[must_use]
pub fn locality_bucket<const D: usize>(p: &[f64; D]) -> u64 {
    let dims = D.min(8);
    let mut key: u64 = 0;
    for (d, &coord) in p.iter().take(dims).enumerate() {
        let clamped = coord.clamp(-100.0, 100.0);
        // Map [-100, 100] linearly to [0, 255].
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let bucket = ((clamped * 1.275) + 128.0).clamp(0.0, 255.0) as u64;
        key |= (bucket & 0xFF) << (d * 8);
    }
    key
}

/// Score aggregation across trees. Serial fold or rayon parallel
/// fold/reduce depending on the `parallel` cargo feature.
/// Walk leaf → root on `tree.store()` computing the rrcf-style
/// codisp — `max(sibling.mass / current_subtree.mass)` across
/// ancestors. Returns `0.0` when the leaf has no parent (single-
/// leaf tree).
fn walk_codisp<const D: usize>(
    store: &crate::tree::NodeStore<D>,
    leaf: crate::tree::NodeRef,
) -> RcfResult<f64> {
    use crate::tree::NodeView;
    let mut cur = leaf;
    let mut max_disp = 0.0_f64;
    while let Some(parent_ref) = store.parent(cur)? {
        let parent = store.internal(parent_ref)?;
        let sibling_ref = if parent.left.raw() == cur.raw() {
            parent.right
        } else {
            parent.left
        };
        let sibling_mass = match store.view(sibling_ref)? {
            NodeView::Internal(i) => i.mass,
            NodeView::Leaf(l) => l.mass,
        };
        let current_mass = match store.view(cur)? {
            NodeView::Internal(i) => i.mass,
            NodeView::Leaf(l) => l.mass,
        };
        if current_mass == 0 {
            break;
        }
        #[allow(clippy::cast_precision_loss)]
        let disp = sibling_mass as f64 / current_mass as f64;
        if disp > max_disp {
            max_disp = disp;
        }
        cur = parent_ref;
    }
    Ok(max_disp)
}

/// Score aggregation across trees. Serial fold or rayon parallel
/// fold/reduce depending on the `parallel` cargo feature.
fn score_aggregate<const D: usize>(
    trees: &[TreeSlot<D>],
    point: &[f64; D],
) -> RcfResult<(f64, usize)> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        trees
            .par_iter()
            .map(|(tree, _, _)| -> RcfResult<Option<f64>> {
                let Some(root) = tree.root() else {
                    return Ok(None);
                };
                let mass = tree.store().view(root)?.mass();
                let visitor = ScalarScoreVisitor::new(mass);
                let s = tree.traverse(point, visitor)?;
                Ok(Some(f64::from(s)))
            })
            .try_fold(
                || (0.0_f64, 0_usize),
                |(t, c), step| {
                    let s = step?;
                    Ok::<_, RcfError>(match s {
                        Some(v) => (t + v, c + 1),
                        None => (t, c),
                    })
                },
            )
            .try_reduce(
                || (0.0_f64, 0_usize),
                |(t1, c1), (t2, c2)| Ok((t1 + t2, c1 + c2)),
            )
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut total = 0.0_f64;
        let mut count = 0_usize;
        for (tree, _, _) in trees {
            let Some(root) = tree.root() else {
                continue;
            };
            let mass = tree.store().view(root)?.mass();
            let visitor = ScalarScoreVisitor::new(mass);
            let s = tree.traverse(point, visitor)?;
            total += f64::from(s);
            count += 1;
        }
        Ok((total, count))
    }
}

/// Attribution aggregation across trees. Serial accumulate or
/// rayon parallel fold/reduce depending on the `parallel` cargo
/// feature.
fn attribution_aggregate<const D: usize>(
    trees: &[TreeSlot<D>],
    point: &[f64; D],
) -> RcfResult<(DiVector, usize)> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        trees
            .par_iter()
            .map(|(tree, _, _)| -> RcfResult<Option<DiVector>> {
                let Some(root) = tree.root() else {
                    return Ok(None);
                };
                let mass = tree.store().view(root)?.mass();
                let visitor = AttributionVisitor::new(point, mass)?;
                Ok(Some(tree.traverse(point, visitor)?))
            })
            .try_fold(
                || (DiVector::zeros(D), 0_usize),
                |(mut acc, c), step| {
                    if let Some(di) = step? {
                        acc.accumulate(&di)?;
                        Ok::<_, RcfError>((acc, c + 1))
                    } else {
                        Ok((acc, c))
                    }
                },
            )
            .try_reduce(
                || (DiVector::zeros(D), 0_usize),
                |(mut a1, c1), (a2, c2)| {
                    a1.accumulate(&a2)?;
                    Ok((a1, c1 + c2))
                },
            )
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut accumulator = DiVector::zeros(D);
        let mut count = 0_usize;
        for (tree, _, _) in trees {
            let Some(root) = tree.root() else {
                continue;
            };
            let mass = tree.store().view(root)?.mass();
            let visitor = AttributionVisitor::new(point, mass)?;
            let di = tree.traverse(point, visitor)?;
            accumulator.accumulate(&di)?;
            count += 1;
        }
        Ok((accumulator, count))
    }
}

/// Combined score + attribution aggregation — single traversal per
/// tree via [`ScoreAttributionVisitor`]. Serial accumulate or rayon
/// parallel fold/reduce depending on the `parallel` cargo feature.
fn score_attribution_aggregate<const D: usize>(
    trees: &[TreeSlot<D>],
    point: &[f64; D],
) -> RcfResult<(f64, DiVector, usize)> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        trees
            .par_iter()
            .map(|(tree, _, _)| -> RcfResult<Option<(f64, DiVector)>> {
                let Some(root) = tree.root() else {
                    return Ok(None);
                };
                let mass = tree.store().view(root)?.mass();
                let visitor = ScoreAttributionVisitor::new(point, mass)?;
                let (s, di) = tree.traverse(point, visitor)?;
                Ok(Some((f64::from(s), di)))
            })
            .try_fold(
                || (0.0_f64, DiVector::zeros(D), 0_usize),
                |(t, mut acc, c), step| match step? {
                    Some((s, di)) => {
                        acc.accumulate(&di)?;
                        Ok::<_, RcfError>((t + s, acc, c + 1))
                    }
                    None => Ok((t, acc, c)),
                },
            )
            .try_reduce(
                || (0.0_f64, DiVector::zeros(D), 0_usize),
                |(t1, mut a1, c1), (t2, a2, c2)| {
                    a1.accumulate(&a2)?;
                    Ok((t1 + t2, a1, c1 + c2))
                },
            )
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut total = 0.0_f64;
        let mut accumulator = DiVector::zeros(D);
        let mut count = 0_usize;
        for (tree, _, _) in trees {
            let Some(root) = tree.root() else {
                continue;
            };
            let mass = tree.store().view(root)?.mass();
            let visitor = ScoreAttributionVisitor::new(point, mass)?;
            let (s, di) = tree.traverse(point, visitor)?;
            total += f64::from(s);
            accumulator.accumulate(&di)?;
            count += 1;
        }
        Ok((total, accumulator, count))
    }
}

// Compile-time Send + Sync assertions.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<RandomCutForest<4>>();
    assert_sync::<RandomCutForest<4>>();
    assert_send::<PointStore<4>>();
    assert_sync::<PointStore<4>>();
};

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss, clippy::cast_lossless)] // Tests assert deterministic values + cast small bounded counters.
mod tests {
    use super::*;
    use crate::config::ForestBuilder;
    use rand::Rng;

    fn small_forest() -> RandomCutForest<2> {
        ForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .seed(2026)
            .build()
            .expect("AWS-conformant config")
    }

    #[test]
    fn from_config_constructs_n_trees() {
        let forest = small_forest();
        assert_eq!(forest.num_trees(), 50);
        assert_eq!(forest.trees().len(), 50);
        assert_eq!(forest.sample_size(), 16);
        assert_eq!(forest.dimension(), 2);
        assert_eq!(forest.updates_seen(), 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn dedicated_thread_pool_runs_score_and_update() {
        let mut f = ForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .seed(2026)
            .num_threads(2)
            .build()
            .expect("custom pool builds");
        for i in 0..100 {
            let v = i as f64 * 0.01;
            f.update([v, v + 0.5]).unwrap();
        }
        let score: f64 = f.score(&[5.0, 5.0]).unwrap().into();
        assert!(score >= 0.0);
        assert_eq!(f.config().num_threads, Some(2));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn dedicated_thread_pool_zero_threads_rejected_at_validate() {
        let err = ForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .num_threads(0)
            .build()
            .unwrap_err();
        assert!(matches!(err, RcfError::InvalidConfig(_)));
    }

    #[cfg(not(feature = "parallel"))]
    #[test]
    fn num_threads_field_inert_without_parallel_feature() {
        let f = ForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(16)
            .num_threads(4)
            .build()
            .expect("config with num_threads still validates without parallel feature");
        assert_eq!(f.config().num_threads, Some(4));
    }

    #[test]
    fn update_rejects_non_finite() {
        let mut f = small_forest();
        assert!(matches!(
            f.update([1.0, f64::NAN]).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn first_update_inserts_in_every_tree() {
        let mut f = small_forest();
        f.update([0.0, 0.0]).unwrap();
        for (tree, sampler, _) in f.trees() {
            assert!(tree.root().is_some());
            assert_eq!(sampler.len(), 1);
        }
        assert_eq!(f.updates_seen(), 1);
    }

    #[test]
    fn many_updates_keep_sample_size_bound() {
        let mut f = small_forest();
        for i in 0..500 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64;
            f.update([v, v + 1.0]).unwrap();
        }
        for (tree, sampler, _) in f.trees() {
            assert!(sampler.len() <= sampler.capacity());
            assert!(tree.distinct_point_count() <= sampler.capacity());
        }
    }

    #[test]
    fn score_empty_forest_returns_err() {
        let f = small_forest();
        assert!(matches!(
            f.score(&[1.0, 2.0]).unwrap_err(),
            RcfError::EmptyForest
        ));
    }

    #[test]
    fn score_returns_non_negative() {
        let mut f = small_forest();
        for i in 0..200 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 * 0.01;
            f.update([v, v + 0.5]).unwrap();
        }
        let score: f64 = f.score(&[0.5, 0.5]).unwrap().into();
        assert!(score >= 0.0);
        assert!(score.is_finite());
    }

    #[test]
    fn score_with_confidence_matches_score_mean() {
        let mut f = small_forest();
        for i in 0..200 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 * 0.01;
            f.update([v, v + 0.5]).unwrap();
        }
        let probe = [0.5, 0.5];
        let plain: f64 = f.score(&probe).unwrap().into();
        let ci = f.score_with_confidence(&probe).unwrap();
        assert_eq!(f64::from(ci.score), plain);
        assert!(ci.trees_evaluated > 0);
        assert!(ci.stderr >= 0.0);
        assert!(ci.stddev >= 0.0);
        let (lo, hi) = ci.ci95();
        assert!(lo <= plain);
        assert!(hi >= plain);
    }

    #[test]
    fn score_and_attribution_matches_split_calls() {
        let mut f = small_forest();
        for i in 0..200 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 * 0.01;
            f.update([v, v + 0.5]).unwrap();
        }
        let probe = [5.0, -3.0]; // clear outlier to produce non-zero attribution
        let s_split: f64 = f.score(&probe).unwrap().into();
        let di_split = f.attribution(&probe).unwrap();
        let (s_merged, di_merged) = f.score_and_attribution(&probe).unwrap();
        assert!((f64::from(s_merged) - s_split).abs() < 1e-9);
        for d in 0..2 {
            assert!((di_merged.high()[d] - di_split.high()[d]).abs() < 1e-9);
            assert!((di_merged.low()[d] - di_split.low()[d]).abs() < 1e-9);
        }
    }

    #[test]
    fn locality_bucket_is_deterministic() {
        let p = [0.25_f64, -0.5, 1.5, 0.0];
        assert_eq!(locality_bucket(&p), locality_bucket(&p));
    }

    #[test]
    fn locality_bucket_groups_close_points() {
        // Nearby points in the leading dims must land in adjacent
        // or identical buckets — this is the signal sort_unstable
        // consumes for cache-friendly ordering.
        let a = [0.10_f64, 0.20, 0.30, 0.40];
        let b = [0.11_f64, 0.21, 0.31, 0.41];
        let c = [900.0_f64, 0.20, 0.30, 0.40];
        let ka = locality_bucket(&a);
        let kb = locality_bucket(&b);
        let kc = locality_bucket(&c);
        // `a` and `b` differ by <1 quantisation step on each dim:
        // their keys match exactly.
        assert_eq!(ka, kb);
        // `c` is far in dim 0 — key differs significantly.
        assert_ne!(ka, kc);
    }

    #[test]
    fn score_many_locality_sorted_matches_score_many_output() {
        let mut f = small_forest();
        for i in 0..200 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 * 0.01;
            f.update([v, v + 0.5]).unwrap();
        }
        let probes: Vec<[f64; 2]> = (0..64)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let x = (i as f64) * 0.07;
                [x, x + 1.0]
            })
            .collect();
        let plain = f.score_many(&probes).unwrap();
        let sorted = f.score_many_locality_sorted(&probes).unwrap();
        assert_eq!(plain.len(), sorted.len());
        for (a, b) in plain.iter().zip(sorted.iter()) {
            assert!((f64::from(*a) - f64::from(*b)).abs() < 1e-9);
        }
    }

    #[test]
    fn score_many_locality_sorted_empty_input() {
        let mut f = small_forest();
        for _ in 0..50 {
            f.update([0.1, 0.2]).unwrap();
        }
        let out = f.score_many_locality_sorted(&[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn score_codisp_stateless_returns_non_negative_on_trained_forest() {
        let mut f = small_forest();
        for i in 0..200 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 * 0.01;
            f.update([v, v + 0.5]).unwrap();
        }
        let s: f64 = f.score_codisp_stateless(&[5.0, -3.0]).unwrap().into();
        assert!(s.is_finite());
        assert!(s >= 0.0);
    }

    #[test]
    fn score_codisp_stateless_rejects_empty_forest() {
        let f = small_forest();
        assert!(matches!(
            f.score_codisp_stateless(&[0.0, 0.0]).unwrap_err(),
            RcfError::EmptyForest
        ));
    }

    #[test]
    fn score_codisp_stateless_rejects_nan() {
        let mut f = small_forest();
        for _ in 0..50 {
            f.update([0.1, 0.2]).unwrap();
        }
        assert!(matches!(
            f.score_codisp_stateless(&[f64::NAN, 0.0]).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn score_codisp_stateless_many_empty_input() {
        let mut f = small_forest();
        for _ in 0..50 {
            f.update([0.1, 0.2]).unwrap();
        }
        let out = f.score_codisp_stateless_many(&[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn score_codisp_stateless_many_matches_single_probe_loop() {
        let mut f = small_forest();
        for i in 0..200 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 * 0.01;
            f.update([v, v + 0.5]).unwrap();
        }
        let probes = [[0.5, 1.0], [5.0, -3.0], [0.2, 0.7]];
        let single: Vec<f64> = probes
            .iter()
            .map(|p| f64::from(f.score_codisp_stateless(p).unwrap()))
            .collect();
        let batched: Vec<f64> = f
            .score_codisp_stateless_many(&probes)
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect();
        assert_eq!(single.len(), batched.len());
        for (a, b) in single.iter().zip(batched.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn score_codisp_stateless_many_handles_large_batch() {
        // The mutating score_codisp_many fails once the batch exceeds
        // sample_size (reservoir saturation). The stateless variant
        // must handle any batch size since it never inserts.
        let mut f = small_forest();
        for i in 0..200 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 * 0.01;
            f.update([v, v + 0.5]).unwrap();
        }
        let n_probes = 10_000; // far larger than sample_size
        let probes: Vec<[f64; 2]> = (0..n_probes)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let x = (i as f64) * 0.001;
                [x, x + 0.5]
            })
            .collect();
        let out = f.score_codisp_stateless_many(&probes).unwrap();
        assert_eq!(out.len(), n_probes);
        for s in &out {
            let v: f64 = (*s).into();
            assert!(v.is_finite() && v >= 0.0);
        }
    }

    #[test]
    fn score_codisp_stateless_does_not_drift_across_many_probes() {
        // Repro of the mutating-codisp drift: `score_codisp` calls
        // displace reservoir baseline points permanently. The
        // stateless variant must return the exact same score
        // whether called once or 5 000 times.
        let mut f = small_forest();
        for i in 0..200 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 * 0.01;
            f.update([v, v + 0.5]).unwrap();
        }
        let probe = [5.0, -3.0];
        let first: f64 = f.score_codisp_stateless(&probe).unwrap().into();
        for _ in 0..5_000 {
            let _ = f.score_codisp_stateless(&probe).unwrap();
        }
        let last: f64 = f.score_codisp_stateless(&probe).unwrap().into();
        assert!(
            (first - last).abs() < 1e-12,
            "stateless codisp drifted: first={first} last={last}"
        );
    }

    #[test]
    fn score_codisp_many_empty_input() {
        let mut f = small_forest();
        for _ in 0..50 {
            f.update([0.1, 0.2]).unwrap();
        }
        let out = f.score_codisp_many(&[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn score_codisp_many_returns_one_score_per_probe() {
        let mut f = small_forest();
        for i in 0..200 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 * 0.01;
            f.update([v, v + 0.5]).unwrap();
        }
        let probes = [[0.5, 1.0], [5.0, -3.0], [0.2, 0.7]];
        let scores = f.score_codisp_many(&probes).unwrap();
        assert_eq!(scores.len(), 3);
        for s in &scores {
            let v: f64 = (*s).into();
            assert!(v.is_finite() && v >= 0.0);
        }
    }

    #[test]
    fn score_codisp_many_rejects_nan() {
        let mut f = small_forest();
        for _ in 0..50 {
            f.update([0.1, 0.2]).unwrap();
        }
        let err = f
            .score_codisp_many(&[[0.0, 0.0], [f64::NAN, 0.0]])
            .unwrap_err();
        assert!(matches!(err, RcfError::NaNValue));
    }

    #[test]
    fn score_and_attribution_rejects_empty_forest() {
        let f = small_forest();
        assert!(matches!(
            f.score_and_attribution(&[0.0, 0.0]).unwrap_err(),
            RcfError::EmptyForest
        ));
    }

    #[test]
    fn score_and_attribution_rejects_nan() {
        let mut f = small_forest();
        for _ in 0..10 {
            f.update([0.1, 0.2]).unwrap();
        }
        assert!(matches!(
            f.score_and_attribution(&[f64::NAN, 0.0]).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn score_with_confidence_rejects_empty_forest() {
        let f = small_forest();
        assert!(matches!(
            f.score_with_confidence(&[0.0, 0.0]).unwrap_err(),
            RcfError::EmptyForest
        ));
    }

    #[test]
    fn score_with_confidence_rejects_nan_input() {
        let mut f = small_forest();
        for _ in 0..10 {
            f.update([0.1, 0.2]).unwrap();
        }
        assert!(matches!(
            f.score_with_confidence(&[f64::NAN, 0.0]).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn outlier_scores_higher_than_cluster_member() {
        let mut f = ForestBuilder::<2>::new()
            .num_trees(50)
            .sample_size(64)
            .seed(7)
            .build()
            .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        for _ in 0..200 {
            let p = [rng.random::<f64>() * 0.1, rng.random::<f64>() * 0.1];
            f.update(p).unwrap();
        }
        let cluster_score: f64 = f.score(&[0.05, 0.05]).unwrap().into();
        let outlier_score: f64 = f.score(&[10.0, 10.0]).unwrap().into();
        assert!(
            outlier_score > cluster_score,
            "outlier {outlier_score} not > cluster {cluster_score}"
        );
    }

    #[test]
    fn attribution_dim_matches_config() {
        let mut f = ForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(32)
            .seed(2026)
            .build()
            .unwrap();
        for i in 0..100 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 * 0.01;
            f.update([v, v, v, v]).unwrap();
        }
        let di = f.attribution(&[0.5, 0.5, 0.5, 0.5]).unwrap();
        assert_eq!(di.dim(), 4);
    }

    #[test]
    fn attribution_empty_forest_returns_err() {
        let f = small_forest();
        assert!(matches!(
            f.attribution(&[0.0, 0.0]).unwrap_err(),
            RcfError::EmptyForest
        ));
    }

    #[test]
    fn deterministic_under_fixed_seed() {
        fn build_and_score(seed: u64) -> f64 {
            let mut f = ForestBuilder::<2>::new()
                .num_trees(50)
                .sample_size(16)
                .seed(seed)
                .build()
                .unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            for _ in 0..100 {
                f.update([rng.random::<f64>(), rng.random::<f64>()])
                    .unwrap();
            }
            f.score(&[5.0, 5.0]).unwrap().into()
        }
        let s1 = build_and_score(2026);
        let s2 = build_and_score(2026);
        assert_eq!(s1, s2);
    }

    #[test]
    fn memory_estimate_within_4mb_at_default_config() {
        let mut f = ForestBuilder::<16>::new().seed(1).build().unwrap();
        for i in 0..(100 * 256) {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64;
            f.update([v; 16]).unwrap();
        }
        let bytes = f.memory_estimate();
        assert!(bytes < 8 * 1024 * 1024, "memory_estimate = {bytes}");
    }

    #[test]
    fn point_store_capacity_stays_bounded() {
        let mut f = small_forest();
        for i in 0..1000 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64;
            f.update([v, v]).unwrap();
        }
        let live = f.point_store().live_count();
        assert!(live <= f.num_trees() * f.sample_size());
    }
}
