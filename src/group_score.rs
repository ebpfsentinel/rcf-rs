//! Score decomposition by named feature groups.
//!
//! The forest's per-dimension attribution ([`DiVector`]) already
//! answers "which dim pushes the score up" at the component level.
//! Real-world feature vectors, though, tend to be semantically
//! grouped — a 14-D traffic vector might be 2 rate features, 4
//! protocol ratios, 3 entropy / cardinality fields, 2 payload stats,
//! 3 cardinality HLL estimates. An analyst triaging an alert is
//! far better served by *"score is driven by cardinality +
//! entropy, not by rate"* than by a ranked list of 14 dims.
//!
//! [`FeatureGroups`] lets callers declare those semantic groups
//! once, at detector-build time, and query a decomposed
//! [`GroupScores`] on every `group_scores` call — available on
//! [`crate::RandomCutForest`], [`crate::ThresholdedForest`], and
//! [`crate::TenantForestPool`]. The decomposition is a pure sum of
//! per-dim contributions, so
//! `sum(group_scores) = DiVector::total()` when the groups
//! partition every dimension (no overlap, no gap).
//!
//! # Example
//!
//! ```
//! use rcf_rs::{FeatureGroups, ForestBuilder};
//!
//! let mut forest = ForestBuilder::<4>::new().seed(7).build().unwrap();
//! let groups = FeatureGroups::builder()
//!     .add("rate", [0, 1])
//!     .add("payload", [2, 3])
//!     .build()
//!     .unwrap();
//! # for i in 0_u32..32 { let v = f64::from(i) * 0.01; forest.update([v, v, v, v]).unwrap(); }
//! let decomposition = forest.group_scores(&[0.3, 0.3, 0.3, 0.3], &groups).unwrap();
//! assert_eq!(decomposition.len(), 2);
//! ```
//!
//! # Overlap and gaps
//!
//! Groups may overlap (the same dim can appear in two groups) —
//! useful when a dim is meaningful to both a "traffic intensity"
//! and a "burstiness" grouping, for example. A gap in the group
//! coverage is legal too — the summed group contributions will
//! then be less than [`DiVector::total`]. [`GroupScores::coverage`]
//! exposes the ratio so callers can spot either case.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::domain::DiVector;
use crate::error::{RcfError, RcfResult};

/// A single named group of dimension indices.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FeatureGroup {
    /// Caller-facing name shown back in [`GroupScores`].
    name: String,
    /// Dimension indices that belong to this group.
    indices: Vec<usize>,
}

impl FeatureGroup {
    /// Build a group explicitly. Prefer the fluent
    /// [`FeatureGroups::builder`] path when defining multiple groups
    /// — it validates index bounds and name uniqueness.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::InvalidConfig`] when the name is empty or
    /// the indices slice is empty.
    pub fn new(
        name: impl Into<String>,
        indices: impl IntoIterator<Item = usize>,
    ) -> RcfResult<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(RcfError::InvalidConfig(
                "FeatureGroup name must not be empty".into(),
            ));
        }
        let indices: Vec<usize> = indices.into_iter().collect();
        if indices.is_empty() {
            return Err(RcfError::InvalidConfig(format!(
                "FeatureGroup \"{name}\" must declare at least one index"
            )));
        }
        Ok(Self { name, indices })
    }

    /// Group name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Dimension indices covered by this group.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
}

/// Validated ordered set of feature groups.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FeatureGroups {
    /// Ordered list of groups. Output of [`RandomCutForest::group_scores`]
    /// preserves this order so a caller-facing UI can pin columns.
    groups: Vec<FeatureGroup>,
    /// Largest index referenced across every group — used to check
    /// `max_index < D` at `group_scores` time.
    max_index: usize,
}

impl FeatureGroups {
    /// Start a fluent builder.
    #[must_use]
    pub fn builder() -> FeatureGroupsBuilder {
        FeatureGroupsBuilder::default()
    }

    /// Number of groups.
    #[must_use]
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Whether any group has been declared.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Read-only access to the declared groups, preserving
    /// insertion order.
    #[must_use]
    pub fn groups(&self) -> &[FeatureGroup] {
        &self.groups
    }

    /// Largest dimension index referenced across every group. `0`
    /// when the set is empty — callers that rely on `max_index < D`
    /// must also check emptiness.
    #[must_use]
    pub fn max_index(&self) -> usize {
        self.max_index
    }

    /// Validate the declared group indices against a per-point
    /// dimension `d`.
    ///
    /// # Errors
    ///
    /// Returns [`RcfError::OutOfBounds`] when any group references
    /// an index `>= d`.
    pub fn validate_for_dimension(&self, d: usize) -> RcfResult<()> {
        if self.is_empty() {
            return Ok(());
        }
        if self.max_index >= d {
            return Err(RcfError::OutOfBounds {
                index: self.max_index,
                len: d,
            });
        }
        Ok(())
    }
}

/// Fluent builder for [`FeatureGroups`].
#[derive(Debug, Default, Clone)]
pub struct FeatureGroupsBuilder {
    /// Groups accumulated under construction.
    groups: Vec<FeatureGroup>,
}

impl FeatureGroupsBuilder {
    /// Declare a new group. Builder calls are cheap; every failure
    /// is surfaced later by [`Self::build`].
    #[must_use]
    pub fn add(
        mut self,
        name: impl Into<String>,
        indices: impl IntoIterator<Item = usize>,
    ) -> Self {
        let name = name.into();
        let indices: Vec<usize> = indices.into_iter().collect();
        self.groups.push(FeatureGroup { name, indices });
        self
    }

    /// Finalise the builder, validating every group.
    ///
    /// # Errors
    ///
    /// - [`RcfError::InvalidConfig`] on an empty group name, an empty
    ///   index slice, or duplicate group names.
    pub fn build(self) -> RcfResult<FeatureGroups> {
        let mut max_index: usize = 0;
        for (i, g) in self.groups.iter().enumerate() {
            if g.name.is_empty() {
                return Err(RcfError::InvalidConfig(format!(
                    "FeatureGroup at position {i} has an empty name"
                )));
            }
            if g.indices.is_empty() {
                return Err(RcfError::InvalidConfig(format!(
                    "FeatureGroup \"{}\" must declare at least one index",
                    g.name
                )));
            }
            for &idx in &g.indices {
                if idx > max_index {
                    max_index = idx;
                }
            }
        }
        // Uniqueness check on names.
        for i in 0..self.groups.len() {
            for j in (i + 1)..self.groups.len() {
                if self.groups[i].name == self.groups[j].name {
                    return Err(RcfError::InvalidConfig(format!(
                        "duplicate FeatureGroup name \"{}\"",
                        self.groups[i].name
                    )));
                }
            }
        }
        Ok(FeatureGroups {
            groups: self.groups,
            max_index,
        })
    }
}

/// Decomposed anomaly score, one entry per [`FeatureGroup`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GroupScores {
    /// `(group_name, contribution)` pairs in the order declared on
    /// the source [`FeatureGroups`].
    scores: Vec<(String, f64)>,
    /// Raw total from [`DiVector::total`] — sum of **every**
    /// per-dim contribution, regardless of group coverage.
    total: f64,
}

impl GroupScores {
    /// Build from components. Normally produced by
    /// [`RandomCutForest::group_scores`] / [`ThresholdedForest::group_scores`];
    /// the constructor is public so tests and alternative backends
    /// can synthesise instances too.
    #[must_use]
    pub fn new(scores: Vec<(String, f64)>, total: f64) -> Self {
        Self { scores, total }
    }

    /// `(group_name, contribution)` pairs in declaration order.
    #[must_use]
    pub fn scores(&self) -> &[(String, f64)] {
        &self.scores
    }

    /// Raw sum across **all** per-dim contributions (equal to
    /// [`DiVector::total`] on the queried point, independent of
    /// group coverage).
    #[must_use]
    pub fn total(&self) -> f64 {
        self.total
    }

    /// Number of groups.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Whether any group is present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Sum of per-group contributions. When groups partition every
    /// dimension with no overlap, this equals [`Self::total`]. With
    /// gaps, `explained` < `total`; with overlap, `explained` >
    /// `total`.
    #[must_use]
    pub fn explained(&self) -> f64 {
        self.scores.iter().map(|(_, s)| *s).sum()
    }

    /// Ratio of [`Self::explained`] over [`Self::total`], capturing
    /// how much of the raw score is accounted for by the declared
    /// groups. `1.0` for a partitioning coverage, `< 1.0` for gaps,
    /// `> 1.0` for overlaps. Returns `0.0` when total is zero.
    #[must_use]
    pub fn coverage(&self) -> f64 {
        if self.total == 0.0 || !self.total.is_finite() {
            return 0.0;
        }
        self.explained() / self.total
    }

    /// `(name, contribution)` of the group with the highest
    /// contribution. Returns `None` on an empty decomposition.
    #[must_use]
    pub fn top_group(&self) -> Option<(&str, f64)> {
        self.scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(n, s)| (n.as_str(), *s))
    }
}

/// Build a [`GroupScores`] from a [`DiVector`] and a validated
/// [`FeatureGroups`]. Shared between the forest-, thresholded-, and
/// pool-level entry points.
#[must_use]
pub fn decompose(di: &DiVector, groups: &FeatureGroups) -> GroupScores {
    let mut scores = Vec::with_capacity(groups.len());
    for group in groups.groups() {
        let contribution: f64 = group.indices.iter().map(|&i| di.per_dim_total(i)).sum();
        scores.push((group.name.clone(), contribution));
    }
    GroupScores::new(scores, di.total())
}

impl<const D: usize> crate::forest::RandomCutForest<D> {
    /// Decompose the anomaly attribution of `point` over `groups`.
    ///
    /// Internally runs [`Self::attribution`] and aggregates the
    /// per-dim contributions through the declared groups.
    ///
    /// # Errors
    ///
    /// - [`RcfError::OutOfBounds`] when any group references an
    ///   index `>= D`.
    /// - Any error bubbled up from [`Self::attribution`].
    pub fn group_scores(&self, point: &[f64; D], groups: &FeatureGroups) -> RcfResult<GroupScores> {
        groups.validate_for_dimension(D)?;
        let di = self.attribution(point)?;
        Ok(decompose(&di, groups))
    }
}

impl<const D: usize> crate::thresholded::ThresholdedForest<D> {
    /// Decompose the anomaly attribution of `point` over `groups`.
    /// Delegates to the underlying forest's attribution — the
    /// adaptive threshold layer does not influence the
    /// decomposition.
    ///
    /// # Errors
    ///
    /// Same as [`crate::RandomCutForest::group_scores`].
    pub fn group_scores(&self, point: &[f64; D], groups: &FeatureGroups) -> RcfResult<GroupScores> {
        self.forest().group_scores(point, groups)
    }
}

#[cfg(feature = "std")]
impl<K, const D: usize> crate::pool::TenantForestPool<K, D>
where
    K: core::hash::Hash + Eq + Clone,
{
    /// Per-tenant decomposition. Lazily instantiates the tenant's
    /// detector (like [`Self::process`]) then delegates to
    /// [`crate::ThresholdedForest::group_scores`].
    ///
    /// # Errors
    ///
    /// Same as [`crate::ThresholdedForest::group_scores`] plus
    /// factory errors.
    ///
    /// # Panics
    ///
    /// Never under normal use. The fall-through branch uses
    /// [`Self::score_only`] to force an entry for the tenant, then
    /// asserts the tenant is resident — the assertion is only
    /// defensive and cannot fire unless a concurrent mutation
    /// evicts the tenant between the two calls, which cannot happen
    /// through `&mut self`.
    pub fn group_scores(
        &mut self,
        key: &K,
        point: &[f64; D],
        groups: &FeatureGroups,
    ) -> RcfResult<GroupScores> {
        // Force an entry for the tenant if it's absent — score_only
        // auto-creates on first use with warming-up semantics, so
        // the subsequent get_mut is guaranteed to return a slot.
        if !self.contains(key) {
            self.score_only(key, point)?;
        }
        let detector = self
            .get_mut(key)
            .expect("tenant was just forced into the pool");
        detector.group_scores(point, groups)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert exact equality on closed-form sums.
mod tests {
    use super::*;

    fn four_d() -> FeatureGroups {
        FeatureGroups::builder()
            .add("rate", [0, 1])
            .add("payload", [2, 3])
            .build()
            .unwrap()
    }

    #[test]
    fn feature_group_rejects_empty_name() {
        assert!(FeatureGroup::new("", [0]).is_err());
    }

    #[test]
    fn feature_group_rejects_empty_indices() {
        assert!(FeatureGroup::new("rate", std::iter::empty::<usize>()).is_err());
    }

    #[test]
    fn builder_rejects_empty_name() {
        let err = FeatureGroups::builder().add("", [0]).build().unwrap_err();
        assert!(matches!(err, RcfError::InvalidConfig(_)));
    }

    #[test]
    fn builder_rejects_empty_indices() {
        let err = FeatureGroups::builder()
            .add("rate", std::iter::empty::<usize>())
            .build()
            .unwrap_err();
        assert!(matches!(err, RcfError::InvalidConfig(_)));
    }

    #[test]
    fn builder_rejects_duplicate_names() {
        let err = FeatureGroups::builder()
            .add("rate", [0])
            .add("rate", [1])
            .build()
            .unwrap_err();
        assert!(matches!(err, RcfError::InvalidConfig(_)));
    }

    #[test]
    fn builder_tracks_max_index() {
        let g = FeatureGroups::builder()
            .add("a", [0, 3])
            .add("b", [7])
            .build()
            .unwrap();
        assert_eq!(g.max_index(), 7);
    }

    #[test]
    fn validate_for_dimension_passes_on_fit() {
        let g = four_d();
        g.validate_for_dimension(4).unwrap();
    }

    #[test]
    fn validate_for_dimension_rejects_out_of_bounds() {
        let g = four_d();
        let err = g.validate_for_dimension(3).unwrap_err();
        assert!(matches!(err, RcfError::OutOfBounds { .. }));
    }

    #[test]
    fn validate_for_dimension_accepts_empty_set() {
        let g = FeatureGroups::default();
        g.validate_for_dimension(0).unwrap();
        g.validate_for_dimension(100).unwrap();
    }

    #[test]
    fn decompose_partitioning_matches_total() {
        let mut di = DiVector::zeros(4);
        di.add_high(0, 1.0).unwrap();
        di.add_high(1, 2.0).unwrap();
        di.add_low(2, 3.0).unwrap();
        di.add_low(3, 4.0).unwrap();
        let scores = decompose(&di, &four_d());
        assert_eq!(scores.scores()[0], ("rate".to_string(), 3.0));
        assert_eq!(scores.scores()[1], ("payload".to_string(), 7.0));
        assert_eq!(scores.total(), 10.0);
        assert_eq!(scores.explained(), 10.0);
        assert_eq!(scores.coverage(), 1.0);
    }

    #[test]
    fn decompose_with_gap_has_coverage_below_one() {
        let mut di = DiVector::zeros(4);
        di.add_high(0, 1.0).unwrap();
        di.add_high(1, 1.0).unwrap();
        // Dim 2 and 3 contribute but are not in any declared group.
        di.add_high(2, 2.0).unwrap();
        di.add_high(3, 2.0).unwrap();
        let only_rate = FeatureGroups::builder()
            .add("rate", [0, 1])
            .build()
            .unwrap();
        let scores = decompose(&di, &only_rate);
        assert_eq!(scores.scores()[0], ("rate".to_string(), 2.0));
        assert_eq!(scores.total(), 6.0);
        assert!((scores.coverage() - 2.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn decompose_with_overlap_has_coverage_above_one() {
        let mut di = DiVector::zeros(2);
        di.add_high(0, 1.0).unwrap();
        di.add_high(1, 1.0).unwrap();
        let overlap = FeatureGroups::builder()
            .add("a", [0])
            .add("b", [0, 1])
            .build()
            .unwrap();
        let scores = decompose(&di, &overlap);
        assert_eq!(scores.scores()[0], ("a".to_string(), 1.0));
        assert_eq!(scores.scores()[1], ("b".to_string(), 2.0));
        assert_eq!(scores.total(), 2.0);
        assert!((scores.coverage() - 3.0 / 2.0).abs() < 1e-12);
    }

    #[test]
    fn top_group_picks_max_contribution() {
        let mut di = DiVector::zeros(4);
        di.add_high(0, 1.0).unwrap();
        di.add_high(1, 1.0).unwrap();
        di.add_low(2, 5.0).unwrap();
        di.add_low(3, 5.0).unwrap();
        let scores = decompose(&di, &four_d());
        let (name, value) = scores.top_group().unwrap();
        assert_eq!(name, "payload");
        assert_eq!(value, 10.0);
    }

    #[test]
    fn empty_group_scores_top_is_none() {
        let scores = GroupScores::new(vec![], 0.0);
        assert!(scores.top_group().is_none());
        assert!(scores.is_empty());
        assert_eq!(scores.coverage(), 0.0);
    }

    #[test]
    fn coverage_handles_zero_total() {
        let scores = GroupScores::new(vec![("a".into(), 0.0)], 0.0);
        assert_eq!(scores.coverage(), 0.0);
    }
}
