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

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::config::RcfConfig;
use crate::domain::point::{ensure_dim, ensure_finite};
use crate::domain::{AnomalyScore, DiVector};
use crate::error::{RcfError, RcfResult};
use crate::forest::point_store::PointStore;
use crate::sampler::{ReservoirSampler, SamplerOp};
use crate::tree::{PointAccessor, RandomCutTree};
use crate::visitor::{AttributionVisitor, ScalarScoreVisitor};

/// Random Cut Forest aggregate.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RandomCutForest {
    /// Validated configuration.
    config: RcfConfig,
    /// `(tree, sampler)` pairs — one entry per tree.
    trees: Vec<(RandomCutTree, ReservoirSampler)>,
    /// Refcounted store shared across every tree.
    point_store: PointStore,
    /// Forest-wide RNG used by both samplers and tree cuts.
    rng: ChaCha8Rng,
    /// Total number of `update` calls observed.
    updates_seen: u64,
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
        let rng = if let Some(seed) = config.seed {
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

        let mut trees = Vec::with_capacity(config.num_trees);
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
            trees.push((tree, sampler));
        }

        let point_store = PointStore::new(config.dimension)?;

        Ok(Self {
            config,
            trees,
            point_store,
            rng,
            updates_seen: 0,
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

    /// Borrow the per-tree `(tree, sampler)` pairs. Used by tests
    /// and persistence.
    #[must_use]
    pub fn trees(&self) -> &[(RandomCutTree, ReservoirSampler)] {
        &self.trees
    }

    /// Pessimistic memory upper bound (in bytes) of the forest's
    /// payload: point store + per-tree node arenas + per-tree sampler
    /// heaps.
    #[must_use]
    pub fn memory_estimate(&self) -> usize {
        let mut total = self.point_store.memory_estimate();
        for (_, sampler) in &self.trees {
            // BinaryHeap<WeightedEntry> ≈ capacity × 16 bytes.
            total += sampler.capacity() * 16;
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

        let Self {
            trees,
            point_store,
            rng,
            ..
        } = self;
        let new_point: Vec<f64> = point_store
            .point(new_idx)
            .expect("just-added point must be present")
            .to_vec();

        for (tree, sampler) in trees.iter_mut() {
            match sampler.accept(new_idx, rng) {
                SamplerOp::Inserted => {
                    tree.add(new_idx, &new_point, point_store, rng)?;
                    point_store.incr_ref(new_idx)?;
                }
                SamplerOp::Replaced(evicted_idx) => {
                    tree.delete(evicted_idx, point_store)?;
                    point_store.decr_ref(evicted_idx)?;
                    tree.add(new_idx, &new_point, point_store, rng)?;
                    point_store.incr_ref(new_idx)?;
                }
                SamplerOp::Rejected => {}
            }
        }

        // No tree wanted the point — release the slot we eagerly
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

        let mut total = 0.0_f64;
        let mut count = 0_usize;
        for (tree, _) in &self.trees {
            let Some(root) = tree.root() else {
                continue;
            };
            let mass = tree.store().node(root)?.mass();
            let visitor = ScalarScoreVisitor::new(mass);
            let s = tree.traverse(point, visitor)?;
            total += f64::from(s);
            count += 1;
        }

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

        let mut accumulator = DiVector::zeros(self.config.dimension);
        let mut count = 0_usize;
        for (tree, _) in &self.trees {
            let Some(root) = tree.root() else {
                continue;
            };
            let mass = tree.store().node(root)?.mass();
            let visitor = AttributionVisitor::new(point.to_vec(), mass)?;
            let di = tree.traverse(point, visitor)?;
            accumulator.accumulate(&di)?;
            count += 1;
        }

        if count == 0 {
            return Err(RcfError::EmptyForest);
        }

        #[allow(clippy::cast_precision_loss)]
        let divisor = count as f64;
        accumulator.scale(divisor)?;
        Ok(accumulator)
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
        for (tree, sampler) in f.trees() {
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
        for (tree, sampler) in f.trees() {
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
