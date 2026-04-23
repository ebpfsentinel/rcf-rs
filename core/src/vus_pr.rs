//! VUS-PR — Volume Under the Surface, Precision-Recall variant.
//! Threshold-free, length-aware quality metric for time-series
//! anomaly detection (Paparrizos et al. VLDB 2022).
//!
//! Standard AUC-PR treats every timestamp independently and
//! requires picking a threshold; on time-series anomaly detection
//! this penalises detectors whose scores peak *near* the true
//! anomaly instead of on it. VUS-PR fixes both issues:
//!
//! - **Range-awareness** — each buffer size `l` inflates the
//!   anomaly mask by `l` positions on each side for the precision
//!   numerator, and dilates the prediction set by `l` on each side
//!   for the recall denominator. A prediction that lands within
//!   `l` of a true anomaly still counts as a hit.
//! - **Threshold-free** — sweeps every possible threshold (by
//!   walking the score-sorted prefix) and trapezoidally integrates
//!   the resulting (R, P) curve → `RangeAUCPR(l)`.
//! - **Length-agnostic** — integrates `RangeAUCPR(l)` over a range
//!   of buffer sizes `l ∈ [0, L]` → the "volume under the surface".
//!
//! Useful as the bench target for benchmarks such as TSB-AD-M
//! (Liu & Paparrizos, `NeurIPS` 2024) — see
//! [`crate::tsb_ad_m`] for the matching CSV loader and
//! `examples/tsb_ad_m_eval.rs` for an end-to-end runner.
//!
//! Gated behind `std` — needs sorting + allocation.
//!
//! # Reference
//!
//! J. Paparrizos, P. Boniol, T. Palpanas, R. S. Tsay, A. Elmore,
//! M. J. Franklin, "Volume Under the Surface: A New Accuracy
//! Evaluation Measure for Time-Series Anomaly Detection", VLDB
//! 2022 (PVLDB 15.11).

use alloc::vec;
use alloc::vec::Vec;

use crate::error::{RcfError, RcfResult};

/// Default maximum buffer — 100 positions. Corresponds to the
/// typical average anomaly length on TSB-AD-M. Override via
/// [`vus_pr_with_buffer`] when the expected anomaly length is
/// known.
pub const DEFAULT_MAX_BUFFER: usize = 100;

/// `RangeAUCPR(l)` for a single buffer size `l`. Exposed as a
/// standalone helper for callers that want the per-`l` slice of
/// the VUS surface (e.g. plotting the surface).
///
/// # Errors
///
/// Returns [`RcfError::InvalidConfig`] when `scores.len() !=
/// labels.len()`, the input is empty, or a score is non-finite.
pub fn range_auc_pr(scores: &[f64], labels: &[bool], buffer: usize) -> RcfResult<f64> {
    validate(scores, labels)?;
    Ok(range_auc_pr_inner(scores, labels, buffer))
}

/// VUS-PR integrated over `l ∈ [0, DEFAULT_MAX_BUFFER]`.
///
/// # Errors
///
/// Returns [`RcfError::InvalidConfig`] when inputs are invalid —
/// see [`range_auc_pr`].
pub fn vus_pr(scores: &[f64], labels: &[bool]) -> RcfResult<f64> {
    vus_pr_with_buffer(scores, labels, DEFAULT_MAX_BUFFER)
}

/// VUS-PR integrated over `l ∈ [0, max_buffer]`. Larger `max_buffer`
/// = more tolerant to off-by-N score peaks but costs proportionally
/// more compute (`O(n · max_buffer²)`).
///
/// # Errors
///
/// Returns [`RcfError::InvalidConfig`] when inputs are invalid —
/// see [`range_auc_pr`].
pub fn vus_pr_with_buffer(scores: &[f64], labels: &[bool], max_buffer: usize) -> RcfResult<f64> {
    validate(scores, labels)?;
    if max_buffer == 0 {
        return Ok(range_auc_pr_inner(scores, labels, 0));
    }
    let mut per_l = Vec::with_capacity(max_buffer + 1);
    for l in 0..=max_buffer {
        per_l.push(range_auc_pr_inner(scores, labels, l));
    }
    // Trapezoidal integral over l, normalised to the interval width.
    let mut acc = 0.0_f64;
    for pair in per_l.windows(2) {
        acc += (pair[0] + pair[1]) * 0.5;
    }
    #[allow(clippy::cast_precision_loss)]
    let width = (per_l.len() - 1) as f64;
    Ok(acc / width)
}

/// Shared precondition check for the public entry points.
fn validate(scores: &[f64], labels: &[bool]) -> RcfResult<()> {
    if scores.len() != labels.len() {
        return Err(RcfError::InvalidConfig(alloc::format!(
            "vus_pr: length mismatch — scores {} vs labels {}",
            scores.len(),
            labels.len()
        )));
    }
    if scores.is_empty() {
        return Err(RcfError::InvalidConfig(alloc::string::ToString::to_string(
            "vus_pr: empty input",
        )));
    }
    if scores.iter().any(|v| !v.is_finite()) {
        return Err(RcfError::InvalidConfig(alloc::string::ToString::to_string(
            "vus_pr: scores contain non-finite values",
        )));
    }
    Ok(())
}

/// Validated, allocating core of [`range_auc_pr`] — assumes inputs
/// already passed [`validate`].
#[allow(clippy::cast_precision_loss)]
fn range_auc_pr_inner(scores: &[f64], labels: &[bool], buffer: usize) -> f64 {
    let n = scores.len();
    let positive_count = labels.iter().filter(|&&b| b).count();
    if positive_count == 0 {
        return 0.0;
    }

    // Inflate the label mask by `buffer` on each side — precision
    // numerator uses this as its "true-positive" definition.
    let y_inflated = dilate(labels, buffer);

    // Indices sorted by descending score.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));

    // Tracks whether each true-anomaly position has been "covered"
    // by a prediction within `buffer` distance. Recall = fraction
    // of true-anomaly positions with `covered = true`.
    let mut covered = vec![false; n];
    let mut recall_hits = 0_usize;
    let mut precision_tp = 0_usize;
    let mut emitted = 0_usize;

    // PR curve starts at (R=0, P=1) — the convention in
    // scikit-learn / VUS reference code.
    let mut prev_recall = 0.0_f64;
    let mut prev_precision = 1.0_f64;
    let mut auc = 0.0_f64;

    for &p in &order {
        emitted += 1;
        if y_inflated[p] {
            precision_tp += 1;
        }
        // Expand the `covered` set by the `[p-buffer, p+buffer]`
        // neighbourhood. Only `labels[q] == true` positions matter;
        // each transitions from `false` → `true` at most once, so
        // the amortised cost is `O(n · buffer)` per `buffer` value.
        let lo = p.saturating_sub(buffer);
        let hi = (p + buffer).min(n - 1);
        for q in lo..=hi {
            if labels[q] && !covered[q] {
                covered[q] = true;
                recall_hits += 1;
            }
        }
        let recall = recall_hits as f64 / positive_count as f64;
        let precision = precision_tp as f64 / emitted as f64;
        // Trapezoidal segment. Only non-zero when recall advances.
        let dr = recall - prev_recall;
        if dr > 0.0 {
            auc += (prev_precision + precision) * 0.5 * dr;
        }
        prev_recall = recall;
        prev_precision = precision;
    }
    auc
}

/// Binary dilation — `out[i] = true` iff any `labels[j]` with
/// `|i - j| ≤ buffer` is `true`. Computed in `O(n)` via a cumulative
/// sum over the indicator array.
fn dilate(labels: &[bool], buffer: usize) -> Vec<bool> {
    let n = labels.len();
    if buffer == 0 {
        return labels.to_vec();
    }
    let mut cumsum = vec![0_u32; n + 1];
    for (i, &b) in labels.iter().enumerate() {
        cumsum[i + 1] = cumsum[i] + u32::from(b);
    }
    let mut out = vec![false; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let lo = i.saturating_sub(buffer);
        let hi = (i + buffer).min(n - 1);
        if cumsum[hi + 1] > cumsum[lo] {
            *slot = true;
        }
    }
    out
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::needless_range_loop
)]
mod tests {
    use super::*;

    #[test]
    fn rejects_length_mismatch() {
        let scores = [0.1_f64, 0.2];
        let labels = [false];
        assert!(vus_pr(&scores, &labels).is_err());
    }

    #[test]
    fn rejects_empty_input() {
        assert!(vus_pr(&[], &[]).is_err());
    }

    #[test]
    fn rejects_non_finite_scores() {
        let scores = [0.1_f64, f64::NAN];
        let labels = [false, true];
        assert!(vus_pr(&scores, &labels).is_err());
    }

    #[test]
    fn zero_true_positives_yields_zero() {
        let scores = [0.9_f64, 0.5, 0.1];
        let labels = [false, false, false];
        let v = vus_pr_with_buffer(&scores, &labels, 1).unwrap();
        assert_eq!(v, 0.0);
    }

    #[test]
    fn perfect_detector_scores_one() {
        // Every anomaly ranked above every normal → AUC-PR = 1
        // at every buffer.
        let scores = [0.9_f64, 0.8, 0.1, 0.05];
        let labels = [true, true, false, false];
        let v = vus_pr_with_buffer(&scores, &labels, 0).unwrap();
        assert!((v - 1.0).abs() < 1e-9, "v = {v}");
        let v2 = vus_pr_with_buffer(&scores, &labels, 2).unwrap();
        assert!((v2 - 1.0).abs() < 1e-9, "v2 = {v2}");
    }

    #[test]
    fn worst_detector_approaches_positive_rate_at_buffer_zero() {
        // Inverted ranking → precision ≈ baseline (positive rate)
        let scores = [0.1_f64, 0.2, 0.3, 0.4, 0.9, 0.95];
        let labels = [true, true, false, false, false, false];
        let v = vus_pr_with_buffer(&scores, &labels, 0).unwrap();
        assert!(v < 0.5, "v = {v}");
    }

    #[test]
    fn buffer_rewards_near_miss() {
        // Anomaly at index 5, top score at index 6 (off by 1).
        // At buffer = 0 the prediction misses entirely (range-AUC-PR
        // near the baseline ~1/n); at buffer ≥ 1 the prediction
        // sits inside the tolerance window and scores perfectly.
        let n = 20;
        let mut scores = vec![0.0_f64; n];
        scores[6] = 1.0;
        let mut labels = vec![false; n];
        labels[5] = true;
        let r0 = range_auc_pr(&scores, &labels, 0).unwrap();
        let r1 = range_auc_pr(&scores, &labels, 1).unwrap();
        let r2 = range_auc_pr(&scores, &labels, 2).unwrap();
        assert!(r0 < 0.2, "r0 = {r0}");
        assert!((r1 - 1.0).abs() < 1e-9, "r1 = {r1}");
        assert!((r2 - 1.0).abs() < 1e-9, "r2 = {r2}");
    }

    #[test]
    fn monotone_in_score_quality() {
        // A "bad" + "good" detector — good should score higher.
        let n = 64;
        let mut labels = vec![false; n];
        for i in 30..34 {
            labels[i] = true;
        }
        let good: Vec<f64> = (0..n)
            .map(|i| if (30..34).contains(&i) { 1.0 } else { 0.1 })
            .collect();
        let bad: Vec<f64> = (0..n)
            .map(|i| if (30..34).contains(&i) { 0.1 } else { 1.0 })
            .collect();
        let vg = vus_pr_with_buffer(&good, &labels, 5).unwrap();
        let vb = vus_pr_with_buffer(&bad, &labels, 5).unwrap();
        assert!(vg > vb);
        assert!(vg > 0.9);
    }

    #[test]
    fn bounded_unit_interval() {
        let mut labels = vec![false; 200];
        for i in 100..110 {
            labels[i] = true;
        }
        let scores: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let v = vus_pr_with_buffer(&scores, &labels, 20).unwrap();
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn range_auc_pr_matches_vus_pr_at_single_buffer() {
        // VUS with `max_buffer = 0` is exactly `range_auc_pr(l=0)`.
        let scores = [0.9_f64, 0.2, 0.7, 0.1];
        let labels = [true, false, true, false];
        let single = range_auc_pr(&scores, &labels, 0).unwrap();
        let vus = vus_pr_with_buffer(&scores, &labels, 0).unwrap();
        assert!((single - vus).abs() < 1e-12);
    }

    #[test]
    fn dilate_matches_manual_reference() {
        let labels = [false, true, false, false, false, true, false];
        let d0 = dilate(&labels, 0);
        assert_eq!(d0, labels);
        let d1 = dilate(&labels, 1);
        assert_eq!(d1, [true, true, true, false, true, true, true]);
        let d2 = dilate(&labels, 2);
        assert_eq!(d2, [true, true, true, true, true, true, true]);
    }
}
