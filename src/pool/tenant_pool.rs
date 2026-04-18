//! Per-tenant pool of [`ThresholdedForest`] detectors.
//!
//! A single forest shared across every tenant pollutes baselines —
//! tenant A's quiet traffic pushes tenant B's threshold down, and
//! vice versa. [`TenantForestPool`] keeps one detector per tenant
//! key, instantiated on demand, with an LRU eviction policy so the
//! pool's footprint stays bounded even when the caller sees a long
//! tail of one-off tenants.
//!
//! # Life cycle
//!
//! - **Construction**: [`TenantForestPool::new`] takes a `capacity`
//!   (maximum simultaneous tenants) and a *factory* closure that
//!   knows how to build a fresh detector. The factory is invoked
//!   lazily — a tenant that never sends a point never allocates its
//!   detector.
//! - **Processing**: [`TenantForestPool::process`] looks up the
//!   tenant, creating the detector with the factory if absent,
//!   evicting the least-recently-used tenant when insertion would
//!   push the pool past `capacity`.
//! - **Inspection / migration**: [`TenantForestPool::iter`] and
//!   [`TenantForestPool::iter_mut`] walk every live tenant; combine
//!   with [`ThresholdedForest::to_path`] to checkpoint the whole
//!   pool to disk.
//!
//! # Thread safety
//!
//! The pool is `!Sync` and mutation takes `&mut self`. Wrap in a
//! [`std::sync::Mutex`] (or a sharded `RwLock` for read-heavy paths)
//! if multiple threads must share the same pool.

use core::hash::Hash;
use std::collections::HashMap;

use crate::bootstrap::BootstrapReport;
use crate::domain::DiVector;
use crate::error::{RcfError, RcfResult};
use crate::thresholded::{AnomalyGrade, ThresholdedForest};

/// Factory closure type carried by the pool.
type ForestFactory<const D: usize> = dyn Fn() -> RcfResult<ThresholdedForest<D>>;

/// Stored per-tenant entry: the detector + its LRU timestamp.
#[derive(Debug)]
struct TenantSlot<const D: usize> {
    /// Detector owned by this tenant. `Box` keeps the hashmap entry
    /// small (one pointer) so rehashes on eviction are cheap even
    /// with large `D`.
    forest: Box<ThresholdedForest<D>>,
    /// Monotonically increasing tick assigned by the pool on every
    /// access. The minimum tick identifies the LRU victim.
    last_access: u64,
}

/// Per-tenant pool of [`ThresholdedForest`] detectors.
///
/// `K` is the tenant key type. Typical choices: `String`,
/// `u64`, a small `enum`, or a newtype wrapping a UUID.
///
/// `D` is the per-point dimensionality, identical across every
/// tenant in the pool — mixed-dimension pools are not supported
/// because each tenant's `const D: usize` is baked into the forest's
/// type.
///
/// # Examples
///
/// ```
/// use rcf_rs::{TenantForestPool, ThresholdedForestBuilder};
///
/// let mut pool: TenantForestPool<String, 2> = TenantForestPool::new(
///     4,
///     || ThresholdedForestBuilder::<2>::new()
///         .num_trees(50)
///         .sample_size(16)
///         .min_observations(4)
///         .seed(42)
///         .build(),
/// ).unwrap();
///
/// let verdict_a = pool.process(&"tenant-a".to_string(), [0.1, 0.2]).unwrap();
/// let verdict_b = pool.process(&"tenant-b".to_string(), [5.0, 5.0]).unwrap();
/// assert_eq!(pool.len(), 2);
/// # let _ = (verdict_a, verdict_b);
/// ```
pub struct TenantForestPool<K, const D: usize>
where
    K: Hash + Eq + Clone,
{
    /// Live per-tenant detectors keyed by tenant id.
    forests: HashMap<K, TenantSlot<D>>,
    /// Maximum number of tenants the pool holds at once.
    capacity: usize,
    /// Monotonic access counter — bumped on every `process`,
    /// `get`, `get_mut`, or `score_only` call.
    access_counter: u64,
    /// Factory used to build a detector when a tenant is seen for
    /// the first time (or after its entry has been evicted).
    factory: Box<ForestFactory<D>>,
    /// Observability sink for pool-level events (LRU evictions,
    /// resident-count gauge). Per-tenant detectors have their own
    /// sinks that the caller can install inside the factory closure.
    metrics: std::sync::Arc<dyn crate::metrics::MetricsSink>,
}

impl<K, const D: usize> core::fmt::Debug for TenantForestPool<K, D>
where
    K: Hash + Eq + Clone + core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // `factory` is a trait object without a Debug bound — emit
        // its type-erased marker instead of the struct field.
        f.debug_struct("TenantForestPool")
            .field("capacity", &self.capacity)
            .field("len", &self.forests.len())
            .field("access_counter", &self.access_counter)
            .field("tenants", &self.forests.keys().collect::<Vec<_>>())
            .field("factory", &"<dyn Fn>")
            .field("metrics", &self.metrics)
            .finish()
    }
}

impl<K, const D: usize> TenantForestPool<K, D>
where
    K: Hash + Eq + Clone,
{
    /// Build a pool bounded at `capacity` tenants.
    ///
    /// The factory is stored and invoked on every first-seen tenant —
    /// it must be able to build a detector repeatedly, not just once.
    /// Seed the factory's builder deterministically (or from a
    /// per-tenant seed inside the closure) when reproducibility
    /// matters.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when `capacity == 0`.
    pub fn new<F>(capacity: usize, factory: F) -> RcfResult<Self>
    where
        F: Fn() -> RcfResult<ThresholdedForest<D>> + 'static,
    {
        if capacity == 0 {
            return Err(RcfError::InvalidConfig(
                "TenantForestPool capacity must be > 0".into(),
            ));
        }
        Ok(Self {
            forests: HashMap::with_capacity(capacity),
            capacity,
            access_counter: 0,
            factory: Box::new(factory),
            metrics: crate::metrics::default_sink(),
        })
    }

    /// Install a [`crate::MetricsSink`] for pool-level events.
    /// Emits `rcf_tenants_resident` gauge updates on every public
    /// mutation and `rcf_tenant_evictions_total` on LRU evictions.
    /// Per-tenant detector metrics are the factory's responsibility.
    #[must_use]
    pub fn with_metrics_sink(
        mut self,
        sink: std::sync::Arc<dyn crate::metrics::MetricsSink>,
    ) -> Self {
        #[allow(clippy::cast_precision_loss)]
        sink.set_gauge(
            crate::metrics::names::TENANTS_RESIDENT,
            self.forests.len() as f64,
        );
        self.metrics = sink;
        self
    }

    /// Read-only handle to the installed pool-level sink.
    #[must_use]
    pub fn metrics_sink(&self) -> &std::sync::Arc<dyn crate::metrics::MetricsSink> {
        &self.metrics
    }

    /// Refresh the `rcf_tenants_resident` gauge — called internally
    /// after every op that mutates the resident set.
    fn emit_resident_gauge(&self) {
        #[allow(clippy::cast_precision_loss)]
        self.metrics.set_gauge(
            crate::metrics::names::TENANTS_RESIDENT,
            self.forests.len() as f64,
        );
    }

    /// Maximum number of tenants the pool holds simultaneously.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current number of live tenants.
    #[must_use]
    pub fn len(&self) -> usize {
        self.forests.len()
    }

    /// Whether the pool is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.forests.is_empty()
    }

    /// Whether the tenant currently has a detector in the pool.
    #[must_use]
    pub fn contains(&self, key: &K) -> bool {
        self.forests.contains_key(key)
    }

    /// Read-only handle to a tenant's detector. Returns `None` when
    /// the tenant has never processed a point or has been evicted.
    /// Does **not** bump the LRU access counter so diagnostic tools
    /// can inspect state without disturbing eviction order.
    #[must_use]
    pub fn peek(&self, key: &K) -> Option<&ThresholdedForest<D>> {
        self.forests.get(key).map(|slot| slot.forest.as_ref())
    }

    /// Read-only handle with an LRU touch — the tenant is treated as
    /// freshly accessed.
    pub fn get(&mut self, key: &K) -> Option<&ThresholdedForest<D>> {
        let tick = self.bump_access();
        self.forests.get_mut(key).map(|slot| {
            slot.last_access = tick;
            &*slot.forest
        })
    }

    /// Mutable handle with an LRU touch. Prefer
    /// [`Self::process`] / [`Self::score_only`] unless you need
    /// direct access to [`ThresholdedForest`] methods not exposed by
    /// the pool.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut ThresholdedForest<D>> {
        let tick = self.bump_access();
        self.forests.get_mut(key).map(|slot| {
            slot.last_access = tick;
            slot.forest.as_mut()
        })
    }

    /// Score a point through the tenant's detector and graduate it
    /// through the adaptive threshold, creating the detector with
    /// the factory if this is the first point for the tenant.
    ///
    /// If inserting would push the pool past `capacity` the
    /// least-recently-used tenant is evicted first.
    ///
    /// # Errors
    ///
    /// Propagates factory errors (when a new detector must be built)
    /// and [`ThresholdedForest::process`] errors (malformed point,
    /// etc.).
    pub fn process(&mut self, key: &K, point: [f64; D]) -> RcfResult<AnomalyGrade> {
        self.touch_or_create(key)?.process(point)
    }

    /// Score a point against the tenant's detector without mutating
    /// the underlying forest or its statistics. Creates the detector
    /// on first use just like [`Self::process`] so the very first
    /// call for a tenant is not surprising — the detector exists
    /// but returns a warming-up verdict (no observations yet).
    ///
    /// # Errors
    ///
    /// Propagates factory errors and [`ThresholdedForest::score_only`]
    /// errors.
    pub fn score_only(&mut self, key: &K, point: &[f64; D]) -> RcfResult<AnomalyGrade> {
        self.touch_or_create(key)?.score_only(point)
    }

    /// Per-feature attribution for a tenant's view of a point.
    ///
    /// # Errors
    ///
    /// Propagates factory errors and
    /// [`ThresholdedForest::attribution`] errors.
    pub fn attribution(&mut self, key: &K, point: &[f64; D]) -> RcfResult<DiVector> {
        self.touch_or_create(key)?.attribution(point)
    }

    /// Retract a previously-observed point from a tenant's forest by
    /// its `point_idx`. Returns `Ok(false)` (and does not create the
    /// tenant) when the tenant is absent — SOC retraction paths must
    /// not silently spin up fresh detectors.
    ///
    /// # Errors
    ///
    /// Propagates [`ThresholdedForest::delete`] errors.
    pub fn delete(&mut self, key: &K, point_idx: usize) -> RcfResult<bool> {
        match self.get_mut(key) {
            Some(detector) => detector.delete(point_idx),
            None => Ok(false),
        }
    }

    /// Retract every point whose stored value bit-matches `point`
    /// for a given tenant. Returns `Ok(0)` (and does not create the
    /// tenant) when the tenant is absent.
    ///
    /// # Errors
    ///
    /// Propagates [`ThresholdedForest::delete_by_value`] errors.
    pub fn delete_by_value(&mut self, key: &K, point: &[f64; D]) -> RcfResult<usize> {
        match self.get_mut(key) {
            Some(detector) => detector.delete_by_value(point),
            None => Ok(0),
        }
    }

    /// Replay historical `points` into the tenant's detector before
    /// any live traffic. Lazily instantiates the tenant (like
    /// [`Self::process`]), then delegates to
    /// [`ThresholdedForest::bootstrap`]. Returns a report summarising
    /// ingestion.
    ///
    /// Use this when restarting a long-running agent: pull recent
    /// per-tenant history from the upstream TSDB (`Prometheus`,
    /// `Loki`, `InfluxDB`, parquet dump…), hand it to this method per
    /// tenant, and the pool is hot before the live streaming pipeline
    /// is switched back on — avoiding the per-tenant warmup coverage
    /// hole.
    ///
    /// # Errors
    ///
    /// Propagates factory and
    /// [`ThresholdedForest::bootstrap`] errors.
    pub fn bootstrap<I>(&mut self, key: &K, points: I) -> RcfResult<BootstrapReport>
    where
        I: IntoIterator<Item = [f64; D]>,
    {
        self.touch_or_create(key)?.bootstrap(points)
    }

    /// Install a pre-built detector for `key`, replacing any
    /// existing entry. Useful for warm reload — iterate a directory
    /// of per-tenant snapshots and pump them back into a fresh pool.
    ///
    /// Returns the displaced detector if one was already resident.
    /// When inserting a brand-new tenant would push the pool past
    /// `capacity`, the least-recently-used tenant is evicted first.
    pub fn insert(&mut self, key: K, forest: ThresholdedForest<D>) -> Option<ThresholdedForest<D>> {
        let tick = self.bump_access();
        if !self.forests.contains_key(&key) && self.forests.len() >= self.capacity {
            self.evict_lru();
        }
        let previous = self
            .forests
            .insert(
                key,
                TenantSlot {
                    forest: Box::new(forest),
                    last_access: tick,
                },
            )
            .map(|slot| *slot.forest);
        self.emit_resident_gauge();
        previous
    }

    /// Drop the tenant's detector. Returns the detector so callers
    /// can hand it back to the factory or persist it before release.
    pub fn remove(&mut self, key: &K) -> Option<ThresholdedForest<D>> {
        let out = self.forests.remove(key).map(|slot| *slot.forest);
        if out.is_some() {
            self.emit_resident_gauge();
        }
        out
    }

    /// Drop every tenant's detector.
    pub fn clear(&mut self) {
        self.forests.clear();
        self.emit_resident_gauge();
    }

    /// Iterate `(key, detector)` pairs in an unspecified order —
    /// use this for snapshot / migration.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &ThresholdedForest<D>)> + '_ {
        self.forests.iter().map(|(k, slot)| (k, &*slot.forest))
    }

    /// Mutable iteration over `(key, detector)` pairs. Does not bump
    /// any tenant's LRU tick — callers are assumed to be scanning
    /// for bulk operations (save to disk, migrate, reset stats).
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut ThresholdedForest<D>)> + '_ {
        self.forests
            .iter_mut()
            .map(|(k, slot)| (k, slot.forest.as_mut()))
    }

    /// Snapshot every tenant key currently held.
    #[must_use]
    pub fn tenants(&self) -> Vec<K> {
        self.forests.keys().cloned().collect()
    }

    /// Evict the least-recently-used tenant explicitly. Returns the
    /// evicted `(key, detector)` pair so callers can persist it
    /// before release.
    ///
    /// Public so custom eviction strategies (time-based, manual
    /// shedding, etc.) can drive the pool without going through the
    /// auto-eviction path.
    pub fn evict_lru(&mut self) -> Option<(K, ThresholdedForest<D>)> {
        let victim_key = self
            .forests
            .iter()
            .min_by_key(|(_, slot)| slot.last_access)
            .map(|(k, _)| k.clone())?;
        let slot = self.forests.remove(&victim_key)?;
        self.metrics
            .inc_counter(crate::metrics::names::TENANT_EVICTIONS_TOTAL, 1);
        self.emit_resident_gauge();
        Some((victim_key, *slot.forest))
    }

    /// Bump the monotonic access counter and return the new value.
    fn bump_access(&mut self) -> u64 {
        self.access_counter = self.access_counter.saturating_add(1);
        self.access_counter
    }

    /// Shared entry point for every access path: update the LRU
    /// tick, invoke the factory on first-seen tenants, and evict
    /// when the pool is full.
    fn touch_or_create(&mut self, key: &K) -> RcfResult<&mut ThresholdedForest<D>> {
        let tick = self.bump_access();
        if !self.forests.contains_key(key) {
            if self.forests.len() >= self.capacity {
                self.evict_lru();
            }
            let forest = (self.factory)()?;
            self.forests.insert(
                key.clone(),
                TenantSlot {
                    forest: Box::new(forest),
                    last_access: tick,
                },
            );
            self.emit_resident_gauge();
        }
        // At this point the entry exists; bump its access stamp and
        // return a mutable handle.
        let slot = self.forests.get_mut(key).expect("tenant was just inserted");
        slot.last_access = tick;
        Ok(slot.forest.as_mut())
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert bounds on closed-form quantities.
mod tests {
    use super::*;
    use crate::ThresholdedForestBuilder;

    fn factory_2d() -> impl Fn() -> RcfResult<ThresholdedForest<2>> {
        || {
            ThresholdedForestBuilder::<2>::new()
                .num_trees(50)
                .sample_size(16)
                .min_observations(4)
                .min_threshold(0.0)
                .seed(42)
                .build()
        }
    }

    #[test]
    fn new_rejects_zero_capacity() {
        let err = TenantForestPool::<String, 2>::new(0, factory_2d()).unwrap_err();
        assert!(matches!(err, RcfError::InvalidConfig(_)));
    }

    #[test]
    fn new_accepts_capacity_one() {
        let p = TenantForestPool::<String, 2>::new(1, factory_2d()).unwrap();
        assert_eq!(p.capacity(), 1);
        assert_eq!(p.len(), 0);
        assert!(p.is_empty());
    }

    #[test]
    fn process_auto_creates_tenant() {
        let mut p = TenantForestPool::<&'static str, 2>::new(4, factory_2d()).unwrap();
        assert!(!p.contains(&"a"));
        p.process(&"a", [0.0, 0.0]).unwrap();
        assert!(p.contains(&"a"));
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn process_evicts_lru_when_full() {
        let mut p = TenantForestPool::<&'static str, 2>::new(2, factory_2d()).unwrap();
        p.process(&"a", [0.0, 0.0]).unwrap();
        p.process(&"b", [1.0, 1.0]).unwrap();
        // Touch `a` so `b` becomes LRU.
        p.process(&"a", [0.1, 0.1]).unwrap();
        p.process(&"c", [2.0, 2.0]).unwrap();
        assert!(p.contains(&"a"));
        assert!(!p.contains(&"b"), "b should have been evicted");
        assert!(p.contains(&"c"));
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn peek_does_not_update_lru() {
        let mut p = TenantForestPool::<&'static str, 2>::new(2, factory_2d()).unwrap();
        p.process(&"old", [0.0, 0.0]).unwrap();
        p.process(&"new", [1.0, 1.0]).unwrap();
        // peek `old` — should NOT prevent its eviction.
        let _ = p.peek(&"old");
        p.process(&"newer", [2.0, 2.0]).unwrap();
        assert!(!p.contains(&"old"), "peek should not refresh LRU");
    }

    #[test]
    fn get_does_update_lru() {
        let mut p = TenantForestPool::<&'static str, 2>::new(2, factory_2d()).unwrap();
        p.process(&"old", [0.0, 0.0]).unwrap();
        p.process(&"new", [1.0, 1.0]).unwrap();
        let _ = p.get(&"old");
        p.process(&"newer", [2.0, 2.0]).unwrap();
        assert!(p.contains(&"old"), "get should refresh LRU");
        assert!(!p.contains(&"new"), "new should be evicted instead");
    }

    #[test]
    fn remove_returns_detector() {
        let mut p = TenantForestPool::<&'static str, 2>::new(2, factory_2d()).unwrap();
        p.process(&"a", [0.0, 0.0]).unwrap();
        let detector = p.remove(&"a").unwrap();
        assert_eq!(detector.forest().num_trees(), 50);
        assert!(!p.contains(&"a"));
    }

    #[test]
    fn remove_returns_none_for_missing_tenant() {
        let mut p = TenantForestPool::<&'static str, 2>::new(2, factory_2d()).unwrap();
        assert!(p.remove(&"nope").is_none());
    }

    #[test]
    fn insert_replaces_existing() {
        let mut p = TenantForestPool::<&'static str, 2>::new(2, factory_2d()).unwrap();
        p.process(&"a", [0.0, 0.0]).unwrap();
        let fresh = (factory_2d())().unwrap();
        let old = p.insert("a", fresh).unwrap();
        assert_eq!(old.forest().num_trees(), 50);
        assert!(p.contains(&"a"));
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn insert_evicts_when_full_and_key_new() {
        let mut p = TenantForestPool::<&'static str, 2>::new(2, factory_2d()).unwrap();
        p.process(&"a", [0.0, 0.0]).unwrap();
        p.process(&"b", [1.0, 1.0]).unwrap();
        let fresh = (factory_2d())().unwrap();
        p.insert("c", fresh);
        assert_eq!(p.len(), 2);
        assert!(p.contains(&"c"));
    }

    #[test]
    fn clear_drops_all_tenants() {
        let mut p = TenantForestPool::<&'static str, 2>::new(4, factory_2d()).unwrap();
        p.process(&"a", [0.0, 0.0]).unwrap();
        p.process(&"b", [1.0, 1.0]).unwrap();
        p.clear();
        assert!(p.is_empty());
    }

    #[test]
    fn iter_visits_every_tenant() {
        let mut p = TenantForestPool::<&'static str, 2>::new(4, factory_2d()).unwrap();
        p.process(&"a", [0.0, 0.0]).unwrap();
        p.process(&"b", [1.0, 1.0]).unwrap();
        let mut keys: Vec<&&str> = p.iter().map(|(k, _)| k).collect();
        keys.sort();
        assert_eq!(keys, vec![&"a", &"b"]);
    }

    #[test]
    fn evict_lru_returns_oldest_tenant() {
        let mut p = TenantForestPool::<&'static str, 2>::new(4, factory_2d()).unwrap();
        p.process(&"a", [0.0, 0.0]).unwrap();
        p.process(&"b", [1.0, 1.0]).unwrap();
        p.process(&"a", [0.1, 0.1]).unwrap();
        let (key, _) = p.evict_lru().unwrap();
        assert_eq!(key, "b");
    }

    #[test]
    fn evict_lru_on_empty_pool_returns_none() {
        let mut p = TenantForestPool::<&'static str, 2>::new(4, factory_2d()).unwrap();
        assert!(p.evict_lru().is_none());
    }

    #[test]
    fn factory_error_propagates() {
        let mut p = TenantForestPool::<&'static str, 2>::new(4, || {
            Err(RcfError::InvalidConfig("forced".into()))
        })
        .unwrap();
        let err = p.process(&"x", [0.0, 0.0]).unwrap_err();
        assert!(matches!(err, RcfError::InvalidConfig(_)));
        assert!(p.is_empty(), "failed factory should not leave an entry");
    }

    #[test]
    fn score_only_auto_creates_but_leaves_stats_empty() {
        let mut p = TenantForestPool::<&'static str, 2>::new(4, factory_2d()).unwrap();
        let verdict = p.score_only(&"a", &[0.0, 0.0]).unwrap();
        assert!(!verdict.ready(), "brand-new detector should warming-up");
        assert!(p.contains(&"a"));
    }

    #[test]
    fn tenants_returns_live_keys() {
        let mut p = TenantForestPool::<&'static str, 2>::new(4, factory_2d()).unwrap();
        p.process(&"a", [0.0, 0.0]).unwrap();
        p.process(&"b", [1.0, 1.0]).unwrap();
        let mut ts = p.tenants();
        ts.sort_unstable();
        assert_eq!(ts, vec!["a", "b"]);
    }

    #[test]
    fn isolation_between_tenants() {
        // A big outlier for tenant A must not raise tenant B's
        // threshold. Each tenant owns its own stats.
        let mut p = TenantForestPool::<&'static str, 2>::new(4, factory_2d()).unwrap();
        for i in 0_u32..32 {
            let v = f64::from(i) * 0.01;
            p.process(&"a", [v, v]).unwrap();
            p.process(&"b", [v, v]).unwrap();
        }
        // Shock tenant A with an outlier many times.
        for _ in 0..10 {
            p.process(&"a", [100.0, 100.0]).unwrap();
        }
        let a_threshold = p.peek(&"a").unwrap().current_threshold();
        let b_threshold = p.peek(&"b").unwrap().current_threshold();
        assert!(
            a_threshold > b_threshold,
            "tenant A threshold {a_threshold} should be > tenant B threshold {b_threshold}",
        );
    }
}
