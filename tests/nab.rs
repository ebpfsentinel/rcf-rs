#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::redundant_closure_for_method_calls
)]
//! Numenta Anomaly Benchmark (NAB) — detection-quality test on
//! real-world labeled streams.
//!
//! `#[ignore]` by default: NAB is a ~50 MB Apache-2.0 dataset and
//! the test needs it fetched to disk. Run manually:
//!
//! ```bash
//! git clone --depth 1 https://github.com/numenta/NAB.git /opt/nab
//! RCF_NAB_PATH=/opt/nab \
//!     cargo test --test nab --all-features -- --ignored --nocapture
//! ```
//!
//! See `scripts/nab/README.md` for scoring protocol + caveats.

#![cfg(all(feature = "serde_json", feature = "parallel"))]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use rcf_rs::{AnomalyScore, ForestBuilder};
use serde_json::Value;

// Pipeline tuned via `examples/nab_ablation.rs`:
//
// 1. 32-lag temporal embedding (~160 min of context on 5-min NAB
//    series) — longer context absorbs more of the contextual-shift
//    anomaly structure.
// 2. Z-score normalise each embedding dim against the warm-phase
//    mean/stddev — NAB series have wildly different scales
//    (CPU %, taxi counts, temperatures) and RCF's cut sampling
//    is range-weighted.
// 3. EMA-smooth the raw score stream (`alpha = 0.02`, half-life
//    ~35 steps / ~3 h). Cuts per-point noise without losing the
//    wide-window anomaly shape.
//
// Aggregate weighted AUC measured on `realKnownCause`: 0.719
// (baseline `lag=8`: 0.615). See `docs/performance.md` for the
// full ablation table.
const D: usize = 32;
const WARM_FRACTION: f64 = 0.15;
const SMOOTH_ALPHA: f64 = 0.02;

/// One CSV row after parsing: `(timestamp_str, value)`.
type Row = (String, f64);

fn nab_root() -> Option<PathBuf> {
    std::env::var("RCF_NAB_PATH").ok().map(PathBuf::from)
}

fn load_csv(path: &Path) -> Vec<Row> {
    let text = fs::read_to_string(path).expect("read CSV");
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 && line.starts_with("timestamp") {
            continue;
        }
        let mut parts = line.splitn(2, ',');
        let ts = parts.next().expect("timestamp col").to_string();
        let v: f64 = parts.next().expect("value col").parse().expect("f64");
        rows.push((ts, v));
    }
    rows
}

/// Parse NAB's `combined_windows.json` into
/// `{ file_path: [(start_ts, end_ts), ...] }`. Timestamps are
/// normalised to 19-char ISO-ish so they string-compare against
/// CSV rows directly (NAB CSV is `YYYY-MM-DD HH:MM:SS`, labels add
/// `.000000`).
fn load_windows(path: &Path) -> HashMap<String, Vec<(String, String)>> {
    let text = fs::read_to_string(path).expect("read windows");
    let json: Value = serde_json::from_str(&text).expect("valid JSON");
    let mut out = HashMap::new();
    if let Value::Object(map) = json {
        for (k, v) in map {
            let mut windows = Vec::new();
            if let Value::Array(arr) = v {
                for pair in arr {
                    if let Value::Array(pair) = pair
                        && pair.len() == 2
                    {
                        let s = pair[0].as_str().map(trunc19).expect("string ts");
                        let e = pair[1].as_str().map(trunc19).expect("string ts");
                        windows.push((s, e));
                    }
                }
            }
            out.insert(k, windows);
        }
    }
    out
}

fn trunc19(s: &str) -> String {
    // `YYYY-MM-DD HH:MM:SS` = 19 chars.
    s.chars().take(19).collect()
}

fn in_any_window(ts: &str, windows: &[(String, String)]) -> bool {
    windows
        .iter()
        .any(|(s, e)| ts >= s.as_str() && ts <= e.as_str())
}

/// Trapezoidal ROC-AUC on descending scores. Identical to the
/// helper in `tests/detection_quality.rs` — duplicated here to
/// keep the test files self-contained.
fn auc(scores: &[f64], labels: &[u8]) -> f64 {
    assert_eq!(scores.len(), labels.len());
    let mut pairs: Vec<(f64, u8)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
    let total_pos: u64 = labels.iter().map(|&l| u64::from(l)).sum();
    #[allow(clippy::cast_possible_truncation)]
    let total_neg: u64 = labels.len() as u64 - total_pos;
    if total_pos == 0 || total_neg == 0 {
        return 0.5;
    }
    let mut auc_val = 0.0_f64;
    let mut tp = 0_u64;
    let mut fp = 0_u64;
    let mut prev_tpr = 0.0_f64;
    let mut prev_fpr = 0.0_f64;
    for (_, label) in &pairs {
        if *label == 1 {
            tp += 1;
        } else {
            fp += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        let tpr = tp as f64 / total_pos as f64;
        #[allow(clippy::cast_precision_loss)]
        let fpr = fp as f64 / total_neg as f64;
        auc_val += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
        prev_tpr = tpr;
        prev_fpr = fpr;
    }
    auc_val
}

/// Which scoring API to exercise per probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scorer {
    /// Fast isolation-depth `score()` — non-mutating, rayon-
    /// parallel, eBPF-hot-path friendly.
    IsolationDepth,
    /// Probe-based codisp via `score_codisp` — AWS Java / rrcf
    /// semantic, mutates the forest per probe (known drift on long
    /// streams, see `score_codisp_stateless` for the fix).
    Codisp,
    /// Stateless codisp via `score_codisp_stateless` — walks root
    /// → leaf along stored cuts, no reservoir mutation, preserves
    /// the frozen-baseline promise and parallelises across trees.
    CodispStateless,
}

/// Score one NAB file: `D`-lag embedding → warm-phase z-score
/// normalisation → frozen-baseline scoring → EMA-smoothed score
/// stream. `scorer` selects between isolation-depth and
/// probe-based codisp.
fn score_file(
    rows: &[Row],
    windows: &[(String, String)],
    scorer: Scorer,
) -> (Vec<f64>, Vec<u8>) {
    if rows.len() < 2 * D {
        return (Vec::new(), Vec::new());
    }
    // Raw lag embedding.
    let embed_len = rows.len() - (D - 1);
    let mut embeddings: Vec<[f64; D]> = Vec::with_capacity(embed_len);
    for i in (D - 1)..rows.len() {
        let mut e = [0.0_f64; D];
        for (k, slot) in e.iter_mut().enumerate() {
            *slot = rows[i + k + 1 - D].1;
        }
        embeddings.push(e);
    }

    #[allow(clippy::cast_precision_loss)]
    let warm_end = ((embed_len as f64) * WARM_FRACTION) as usize;

    // Per-dim z-score using warm-phase mean / stddev.
    if warm_end > 1 {
        let mut means = [0.0_f64; D];
        let mut m2 = [0.0_f64; D];
        #[allow(clippy::cast_precision_loss)]
        let n_f = warm_end as f64;
        for emb in &embeddings[..warm_end] {
            for d in 0..D {
                means[d] += emb[d];
            }
        }
        for mean in &mut means {
            *mean /= n_f;
        }
        for emb in &embeddings[..warm_end] {
            for d in 0..D {
                let delta = emb[d] - means[d];
                m2[d] += delta * delta;
            }
        }
        let mut stddevs = [1.0_f64; D];
        for d in 0..D {
            stddevs[d] = (m2[d] / n_f).sqrt().max(1.0e-9);
        }
        for emb in &mut embeddings {
            for d in 0..D {
                emb[d] = (emb[d] - means[d]) / stddevs[d];
            }
        }
    }

    let mut forest = ForestBuilder::<D>::new()
        .num_trees(100)
        .sample_size(256)
        .seed(2026)
        .build()
        .unwrap();

    // Phase 1 — warm on normalised embeddings, no score collection.
    for p in &embeddings[..warm_end] {
        forest.update(*p).ok();
    }

    // Phase 2 — score against the frozen warm-phase forest. We do
    // NOT call `update` on the eval set: NAB anomaly windows are
    // wide (days), and folding anomaly points back into the
    // reservoir drags the baseline toward them and drops recall.
    let eval = &embeddings[warm_end..];
    let mut raw_scores: Vec<f64> = Vec::with_capacity(eval.len());
    for p in eval {
        let s: AnomalyScore = match scorer {
            Scorer::IsolationDepth => forest
                .score(p)
                .unwrap_or_else(|_| AnomalyScore::new(0.0).expect("zero valid")),
            Scorer::Codisp => forest
                .score_codisp(p)
                .unwrap_or_else(|_| AnomalyScore::new(0.0).expect("zero valid")),
            Scorer::CodispStateless => forest
                .score_codisp_stateless(p)
                .unwrap_or_else(|_| AnomalyScore::new(0.0).expect("zero valid")),
        };
        raw_scores.push(f64::from(s));
    }

    let mut labels = Vec::with_capacity(raw_scores.len());
    for idx in 0..raw_scores.len() {
        let row_idx = warm_end + idx + (D - 1);
        let ts = &rows[row_idx].0;
        labels.push(u8::from(in_any_window(ts, windows)));
    }

    // Phase 3 — EMA-smooth the score stream.
    let mut scores = Vec::with_capacity(raw_scores.len());
    if let Some(&first) = raw_scores.first() {
        let mut acc = first;
        for &s in &raw_scores {
            acc = SMOOTH_ALPHA * s + (1.0 - SMOOTH_ALPHA) * acc;
            scores.push(acc);
        }
    }
    (scores, labels)
}

/// Shared corpus runner — iterates `data/realKnownCause` in
/// parallel (one rayon worker per file), prints per-file AUC and
/// the weighted aggregate, returns the aggregate for floor-based
/// regression guards.
fn run_corpus(scorer: Scorer, label: &str) -> f64 {
    let Some(root) = nab_root() else {
        panic!(
            "RCF_NAB_PATH not set — clone https://github.com/numenta/NAB \
             and export the path before running this ignored test"
        );
    };
    let data_dir = root.join("data/realKnownCause");
    let windows = load_windows(&root.join("labels/combined_windows.json"));

    let mut entries: Vec<_> = fs::read_dir(&data_dir)
        .expect("read realKnownCause/")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| x == "csv")
        })
        .collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    assert!(
        !entries.is_empty(),
        "no CSV files under {}",
        data_dir.display()
    );

    // Parallel across files — each file owns its own forest, so
    // per-file pipelines are independent. `par_iter` fans out one
    // CSV per rayon worker.
    let paths: Vec<_> = entries.iter().map(std::fs::DirEntry::path).collect();
    let per_file: Vec<(String, f64, u64)> = paths
        .par_iter()
        .map(|csv_path| {
            let file_name = csv_path.file_name().unwrap().to_string_lossy().into_owned();
            let window_key = format!("realKnownCause/{file_name}");
            let empty = Vec::new();
            let w = windows.get(&window_key).unwrap_or(&empty);
            let rows = load_csv(csv_path);
            let (scores, labels) = score_file(&rows, w, scorer);
            let pos: u64 = labels.iter().map(|&l| u64::from(l)).sum();
            let a = auc(&scores, &labels);
            (file_name, a, pos)
        })
        .collect();

    let mut weighted_sum = 0.0_f64;
    let mut total_anoms = 0_u64;
    println!("\nNAB realKnownCause [{label}] per-file AUC:");
    for (name, a, pos) in &per_file {
        println!("  {a:.3}  pos={pos:<6}  {name}");
        #[allow(clippy::cast_precision_loss)]
        {
            weighted_sum += a * (*pos as f64);
        }
        total_anoms += *pos;
    }
    #[allow(clippy::cast_precision_loss)]
    let weighted_auc = if total_anoms == 0 {
        0.0
    } else {
        weighted_sum / total_anoms as f64
    };
    println!("[{label}] aggregate weighted AUC: {weighted_auc:.3}");
    weighted_auc
}

#[test]
#[ignore = "requires RCF_NAB_PATH pointing at a cloned NAB repo"]
fn realknowncause_aggregate_auc_above_floor() {
    let weighted_auc = run_corpus(Scorer::IsolationDepth, "score()");
    // Floor is a regression guard, not a quality claim. Adjust
    // downward *only* with a commit message explaining which
    // detector change moved the needle, and by how much.
    assert!(
        weighted_auc > 0.70,
        "aggregate weighted AUC = {weighted_auc:.3} below floor 0.70 — \
         detector regression or dataset change?"
    );
}

#[test]
#[ignore = "requires RCF_NAB_PATH; codisp path is ~30× slower than score()"]
fn realknowncause_codisp_aggregate_auc_above_floor() {
    let weighted_auc = run_corpus(Scorer::Codisp, "score_codisp()");
    // Codisp floor is independent: probe-based scoring hits a
    // higher AUC on NAB but the magnitude depends on the batched
    // shared-walk behaviour, so pin a conservative guard.
    assert!(
        weighted_auc > 0.70,
        "codisp aggregate weighted AUC = {weighted_auc:.3} below floor 0.70"
    );
}

#[test]
#[ignore = "requires RCF_NAB_PATH; stateless codisp is non-mutating and parallel"]
fn realknowncause_codisp_stateless_aggregate_auc_above_floor() {
    let weighted_auc = run_corpus(Scorer::CodispStateless, "score_codisp_stateless()");
    assert!(
        weighted_auc > 0.70,
        "stateless codisp aggregate weighted AUC = {weighted_auc:.3} below floor 0.70"
    );
}
