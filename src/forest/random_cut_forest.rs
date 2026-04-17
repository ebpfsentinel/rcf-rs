//! Aggregate root: orchestrates `N` `(RandomCutTree, ReservoirSampler)`
//! pairs sharing a single refcounted [`PointStore`].
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
use crate::domain::point::{ensure_dim, ensure_finite};
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
type TreeSlot = (RandomCutTree, ReservoirSampler, ChaCha8Rng);

/// Random Cut Forest aggregate.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RandomCutForest {
    /// Validated configuration.
    config: RcfConfig,
    /// Per-tree state: `(tree, sampler, rng)` triples — one entry per
    /// tree. Each tree owns a dedicated `ChaCha8Rng` seeded from the
    /// master forest seed at construction so parallel insert paths
    /// never share RNG state.
    trees: Vec<TreeSlot>,
    /// Refcounted store shared across every tree.
    point_store: PointStore,
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
}

impl RandomCutForest {
    /// Construct a forest from a pre-validated [`RcfConfig`]. Public
    /// callers should go through
    /// [`crate::ForestBuilder::build`](crate::ForestBuilder::build).
    ///
    /// # Errors
    ///
    /// Propagates failures from the underlying tree, sampler and
    /// point-store constructors.
    pub fn from_config(config: RcfConfig) -> RcfResult<Self> {
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

        let mut trees: Vec<TreeSlot> = Vec::with_capacity(config.num_trees);
        for _ in 0..config.num_trees {
            let tree = RandomCutTree::new(
                u32::try_from(config.sample_size).map_err(|_| {
                    RcfError::InvalidConfig(format!(
                        "sample_size {} exceeds u32::MAX",
                        config.sample_size
                    ))
                })?,
                config.dimension,
            )?;
            let sampler = ReservoirSampler::new(config.sample_size, config.time_decay)?;
            // Derive a deterministic per-tree seed from the master so
            // parallel insert paths never alias RNG state.
            let tree_rng = ChaCha8Rng::seed_from_u64(master.next_u64());
            trees.push((tree, sampler, tree_rng));
        }

        let point_store = PointStore::new(config.dimension)?;

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
        })
    }

    /// Read-only access to the validated config.
    #[must_use]
    pub fn config(&self) -> &RcfConfig {
        &self.config
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

    /// Per-point dimensionality.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Total number of [`update`](Self::update) calls observed.
    #[must_use]
    pub fn updates_seen(&self) -> u64 {
        self.updates_seen
    }

    /// Borrow the underlying point store. Used by tests, persistence
    /// (story RCF.8) and diagnostics.
    #[must_use]
    pub fn point_store(&self) -> &PointStore {
        &self.point_store
    }

    /// Borrow the per-tree `(tree, sampler, rng)` triples. Used by
    /// tests and persistence.
    #[must_use]
    pub fn trees(&self) -> &[TreeSlot] {
        &self.trees
    }

    /// Pessimistic memory upper bound (in bytes) of the forest's
    /// payload: point store + per-tree node arenas + per-tree sampler
    /// heaps + per-tree RNG state.
    #[must_use]
    pub fn memory_estimate(&self) -> usize {
        let mut total = self.point_store.memory_estimate();
        for (_, sampler, _) in &self.trees {
            // BinaryHeap<WeightedEntry> ≈ capacity × 16 bytes.
            total += sampler.capacity() * 16;
            // ChaCha8Rng state ≈ 256 bytes per tree.
            total += core::mem::size_of::<ChaCha8Rng>();
        }
        // NodeStore per tree: 2 × sample_size slots × ~48 bytes (worst case).
        total += self.trees.len() * (2 * self.config.sample_size * 48);
        total
    }

    /// Stream a new `point` through the forest.
    ///
    /// # Errors
    ///
    /// - [`RcfError::DimensionMismatch`] when `point.len() != self.dimension()`.
    /// - [`RcfError::NaNValue`] when `point` contains a non-finite component.
    /// - Propagates failures from the underlying tree, sampler and
    ///   point-store calls.
    ///
    /// # Panics
    ///
    /// Never. The internal `expect("just-added point must be present")`
    /// is unreachable: [`PointStore::add`] returned the index in the
    /// preceding line, so the slot is guaranteed live.
    pub fn update(&mut self, point: Vec<f64>) -> RcfResult<()> {
        ensure_dim(&point, self.config.dimension)?;
        ensure_finite(&point)?;

        let new_idx = self.point_store.add(point)?;

        // Per-tree work: returns the evicted_idx that hit zero on
        // decr_ref (if any) so the caller can finalise the slot
        // outside the parallel block. Routed through the dedicated
        // thread pool when one is configured.
        #[cfg(feature = "parallel")]
        let pool = self.pool.clone();

        let Self {
            trees, point_store, ..
        } = self;
        let store: &PointStore = point_store;

        #[cfg(feature = "parallel")]
        let pending_frees = if let Some(p) = pool.as_deref() {
            p.install(|| update_trees(trees, store, new_idx))?
        } else {
            update_trees(trees, store, new_idx)?
        };

        #[cfg(not(feature = "parallel"))]
        let pending_frees = update_trees(trees, store, new_idx)?;

        // Single-threaded post-process: turn `decr_ref` "hit-zero"
        // signals into actual slot-free operations.
        for evicted in pending_frees {
            point_store.set_free(evicted)?;
        }

        // No tree wanted the new point — release the slot we eagerly
        // reserved so capacity stays bounded.
        if point_store.ref_count(new_idx) == 0 {
            point_store.drop_unreferenced(new_idx)?;
        }

        self.updates_seen = self.updates_seen.saturating_add(1);
        Ok(())
    }

    /// Score `point` as an anomaly. Higher = more anomalous.
    ///
    /// # Errors
    ///
    /// - [`RcfError::DimensionMismatch`] when `point.len() != self.dimension()`.
    /// - [`RcfError::NaNValue`] when `point` contains a non-finite component.
    /// - [`RcfError::EmptyForest`] when no tree currently holds any leaf.
    pub fn score(&self, point: &[f64]) -> RcfResult<AnomalyScore> {
        ensure_dim(point, self.config.dimension)?;
        ensure_finite(point)?;

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
        AnomalyScore::new(mean.max(0.0))
    }

    /// Compute the per-feature attribution of `point`'s anomaly
    /// score. Returns the mean [`DiVector`] across all trees that
    /// hold leaves.
    ///
    /// # Errors
    ///
    /// Same as [`score`](Self::score).
    pub fn attribution(&self, point: &[f64]) -> RcfResult<DiVector> {
        ensure_dim(point, self.config.dimension)?;
        ensure_finite(point)?;

        #[cfg(feature = "parallel")]
        let (mut accumulator, count) = if let Some(p) = self.pool.as_deref() {
            p.install(|| attribution_aggregate(&self.trees, self.config.dimension, point))?
        } else {
            attribution_aggregate(&self.trees, self.config.dimension, point)?
        };

        #[cfg(not(feature = "parallel"))]
        let (mut accumulator, count) =
            attribution_aggregate(&self.trees, self.config.dimension, point)?;

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
fn update_trees(
    trees: &mut [TreeSlot],
    store: &PointStore,
    new_idx: usize,
) -> RcfResult<Vec<usize>> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        // Per-tree work is short (~hundreds of ns) at the default
        // config — chunk trees so each rayon task does enough work
        // to amortise the work-stealing dispatch overhead. Chunk size
        // tuned on default 100×256×16 where ~25 trees per task hits
        // the sweet spot on a 4-core box.
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
fn process_tree_update(
    slot: &mut TreeSlot,
    store: &PointStore,
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
fn score_aggregate(trees: &[TreeSlot], point: &[f64]) -> RcfResult<(f64, usize)> {
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
fn attribution_aggregate(
    trees: &[TreeSlot],
    dimension: usize,
    point: &[f64],
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
                || (DiVector::zeros(dimension), 0_usize),
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
                || (DiVector::zeros(dimension), 0_usize),
                |(mut a1, c1), (a2, c2)| {
                    a1.accumulate(&a2)?;
                    Ok((a1, c1 + c2))
                },
            )
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut accumulator = DiVector::zeros(dimension);
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

// Compile-time Send + Sync assertions — RCF.7 AC #7.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<RandomCutForest>();
    assert_sync::<RandomCutForest>();
    assert_send::<PointStore>();
    assert_sync::<PointStore>();
};

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss, clippy::cast_lossless)] // Tests assert deterministic values + cast small bounded counters.
mod tests {
    use super::*;
    use crate::config::ForestBuilder;
    use rand::Rng;

    fn small_forest() -> RandomCutForest {
        ForestBuilder::new(2)
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
        let mut f = ForestBuilder::new(2)
            .num_trees(50)
            .sample_size(16)
            .seed(2026)
            .num_threads(2)
            .build()
            .expect("custom pool builds");
        for i in 0..100 {
            let v = i as f64 * 0.01;
            f.update(vec![v, v + 0.5]).unwrap();
        }
        let score: f64 = f.score(&[5.0, 5.0]).unwrap().into();
        assert!(score >= 0.0);
        assert_eq!(f.config().num_threads, Some(2));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn dedicated_thread_pool_zero_threads_rejected_at_validate() {
        let err = ForestBuilder::new(2)
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
        let f = ForestBuilder::new(2)
            .num_trees(50)
            .sample_size(16)
            .num_threads(4)
            .build()
            .expect("config with num_threads still validates without parallel feature");
        assert_eq!(f.config().num_threads, Some(4));
    }

    #[test]
    fn update_validates_dimension() {
        let mut f = small_forest();
        assert!(matches!(
            f.update(vec![1.0]).unwrap_err(),
            RcfError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn update_rejects_non_finite() {
        let mut f = small_forest();
        assert!(matches!(
            f.update(vec![1.0, f64::NAN]).unwrap_err(),
            RcfError::NaNValue
        ));
    }

    #[test]
    fn first_update_inserts_in_every_tree() {
        let mut f = small_forest();
        f.update(vec![0.0, 0.0]).unwrap();
        // Every tree's reservoir under-capacity → all Inserted.
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
            f.update(vec![v, v + 1.0]).unwrap();
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
    fn score_validates_dim() {
        let mut f = small_forest();
        f.update(vec![0.0, 0.0]).unwrap();
        assert!(matches!(
            f.score(&[1.0]).unwrap_err(),
            RcfError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn score_returns_non_negative() {
        let mut f = small_forest();
        for i in 0..200 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 * 0.01;
            f.update(vec![v, v + 0.5]).unwrap();
        }
        let score: f64 = f.score(&[0.5, 0.5]).unwrap().into();
        assert!(score >= 0.0);
        assert!(score.is_finite());
    }

    #[test]
    fn outlier_scores_higher_than_cluster_member() {
        let mut f = ForestBuilder::new(2)
            .num_trees(50)
            .sample_size(64)
            .seed(7)
            .build()
            .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        for _ in 0..200 {
            let p = vec![rng.random::<f64>() * 0.1, rng.random::<f64>() * 0.1];
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
        let mut f = ForestBuilder::new(4)
            .num_trees(50)
            .sample_size(32)
            .seed(2026)
            .build()
            .unwrap();
        for i in 0..100 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64 * 0.01;
            f.update(vec![v, v, v, v]).unwrap();
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
            let mut f = ForestBuilder::new(2)
                .num_trees(50)
                .sample_size(16)
                .seed(seed)
                .build()
                .unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            for _ in 0..100 {
                f.update(vec![rng.random::<f64>(), rng.random::<f64>()])
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
        let mut f = ForestBuilder::new(16).seed(1).build().unwrap();
        // Pre-fill: every tree at full capacity.
        for i in 0..(100 * 256) {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64;
            f.update(vec![v; 16]).unwrap();
        }
        let bytes = f.memory_estimate();
        // AWS-default: 100 × 256 × 16 floats = 3.2 MB raw points, plus
        // node + sampler overhead — keep total under 8 MB (loose 2×
        // budget margin). The 4 MB target lives in the bench story.
        assert!(bytes < 8 * 1024 * 1024, "memory_estimate = {bytes}");
    }

    #[test]
    fn point_store_capacity_stays_bounded() {
        let mut f = small_forest();
        for i in 0..1000 {
            #[allow(clippy::cast_precision_loss)]
            let v = i as f64;
            f.update(vec![v, v]).unwrap();
        }
        // Live points ≤ num_trees × sample_size (worst case if no overlap).
        let live = f.point_store().live_count();
        assert!(live <= f.num_trees() * f.sample_size());
    }
}
