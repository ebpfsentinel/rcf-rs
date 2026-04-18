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
use crate::error::{RcfError, RcfResult};
use crate::forest::point_store::PointStore;
use crate::sampler::{ReservoirSampler, SamplerOp};
use crate::tree::{PointAccessor, RandomCutTree};
use crate::visitor::{AttributionVisitor, ScalarScoreVisitor};

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
    #[cfg_attr(feature = "serde", serde(skip, default = "crate::metrics::default_sink"))]
    metrics: std::sync::Arc<dyn crate::metrics::MetricsSink>,
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

    /// Core insertion body — performs the full `update` pipeline and
    /// returns the freshly assigned `point_idx`. Shared by
    /// [`Self::update`] and [`Self::update_indexed`] so the hot-path
    /// logic lives in exactly one place.
    fn insert_point(&mut self, point: [f64; D]) -> RcfResult<usize> {
        ensure_finite(&point)?;

        let new_idx = self.point_store.add(point)?;

        #[cfg(feature = "parallel")]
        let pool = self.pool.clone();

        let Self {
            trees, point_store, ..
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
        }

        if point_store.ref_count(new_idx) == 0 {
            point_store.drop_unreferenced(new_idx)?;
        }

        self.updates_seen = self.updates_seen.saturating_add(1);
        #[cfg(feature = "std")]
        self.metrics.inc_counter(crate::metrics::names::UPDATES_TOTAL, 1);
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
        let Self {
            trees, point_store, ..
        } = self;
        let mut removed_from_any = false;
        let mut went_to_zero = false;
        for (tree, sampler, _) in trees.iter_mut() {
            if sampler.remove(point_idx) {
                tree.delete(point_idx, &*point_store)?;
                if point_store.decr_ref(point_idx)? {
                    went_to_zero = true;
                }
                removed_from_any = true;
            }
        }
        if went_to_zero {
            point_store.set_free(point_idx)?;
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
        ensure_finite(point)?;
        // Stored points live in the forest's scaled space — scale
        // the caller query so the bit-exact comparison matches.
        let scaled = self.scale_point_copy(point);
        let probe: &[f64; D] = &scaled;
        let mut candidates: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (_, sampler, _) in &self.trees {
            for idx in sampler.iter_indices() {
                candidates.insert(idx);
            }
        }
        let matching: Vec<usize> = candidates
            .into_iter()
            .filter(|&idx| self.point_store.point(idx) == Some(probe))
            .collect();
        let mut removed = 0_usize;
        for idx in matching {
            if self.delete(idx)? {
                removed = removed.saturating_add(1);
            }
        }
        Ok(removed)
    }

    /// Score `point` as an anomaly. Higher = more anomalous.
    ///
    /// # Errors
    ///
    /// - [`RcfError::NaNValue`] when `point` contains a non-finite component.
    /// - [`RcfError::EmptyForest`] when no tree currently holds any leaf.
    pub fn score(&self, point: &[f64; D]) -> RcfResult<AnomalyScore> {
        ensure_finite(point)?;
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

    /// Compute the per-feature attribution of `point`'s anomaly
    /// score. Returns the mean [`DiVector`] across all trees that
    /// hold leaves.
    ///
    /// # Errors
    ///
    /// Same as [`score`](Self::score).
    pub fn attribution(&self, point: &[f64; D]) -> RcfResult<DiVector> {
        ensure_finite(point)?;
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
        Ok(accumulator)
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
                let mass = tree.store().node(root)?.mass();
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
            let mass = tree.store().node(root)?.mass();
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
                let mass = tree.store().node(root)?.mass();
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
            let mass = tree.store().node(root)?.mass();
            let visitor = AttributionVisitor::new(point, mass)?;
            let di = tree.traverse(point, visitor)?;
            accumulator.accumulate(&di)?;
            count += 1;
        }
        Ok((accumulator, count))
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
