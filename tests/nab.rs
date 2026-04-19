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

#![cfg(feature = "serde_json")]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use rcf_rs::{AnomalyScore, ForestBuilder};
use serde_json::Value;

// 32-lag temporal embedding (~160 min of context on 5-min NAB
// series). Ablation in `examples/nab_ablation.rs` measured lag=8
// → 0.615, lag=16 → 0.650, lag=32 → 0.665 weighted aggregate AUC
// — longer context absorbs more of the contextual-shift anomaly
// structure NAB ships with.
const D: usize = 32;
const WARM_FRACTION: f64 = 0.15;

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

/// Score one NAB file: feature-engineer the 4-lag embedding,
/// warm on the first `WARM_FRACTION` of the series, stream-score
/// the rest, return `(scores_per_row, labels_per_row)` for the
/// scored suffix.
fn score_file(rows: &[Row], windows: &[(String, String)]) -> (Vec<f64>, Vec<u8>) {
    if rows.len() < 16 {
        return (Vec::new(), Vec::new());
    }
    // Build lag embeddings starting at index 3.
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
    let mut forest = ForestBuilder::<D>::new()
        .num_trees(100)
        .sample_size(256)
        .seed(2026)
        .build()
        .unwrap();

    // Phase 1 — warm, no score collection.
    for p in &embeddings[..warm_end] {
        forest.update(*p).ok();
    }

    // Phase 2 — score against the frozen warm-phase forest. We do
    // NOT call `update` on the eval set: NAB anomaly windows are
    // wide (days), and folding anomaly points back into the
    // reservoir drags the baseline toward them and drops recall.
    // Matches realistic deployments where an initial "known clean"
    // window trains the detector and production traffic is then
    // scored against that frozen model.
    let mut scores = Vec::with_capacity(embed_len.saturating_sub(warm_end));
    let mut labels = Vec::with_capacity(embed_len.saturating_sub(warm_end));
    for (idx, p) in embeddings[warm_end..].iter().enumerate() {
        let s: AnomalyScore = forest
            .score(p)
            .unwrap_or_else(|_| AnomalyScore::new(0.0).expect("zero score is valid"));
        scores.push(f64::from(s));
        let row_idx = warm_end + idx + (D - 1);
        let ts = &rows[row_idx].0;
        labels.push(u8::from(in_any_window(ts, windows)));
    }
    (scores, labels)
}

#[test]
#[ignore = "requires RCF_NAB_PATH pointing at a cloned NAB repo"]
fn realknowncause_aggregate_auc_above_floor() {
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

    let mut weighted_sum = 0.0_f64;
    let mut total_anoms = 0_u64;
    let mut per_file = Vec::new();

    for entry in &entries {
        let csv_path = entry.path();
        let file_name = csv_path.file_name().unwrap().to_string_lossy().into_owned();
        let window_key = format!("realKnownCause/{file_name}");
        let empty = Vec::new();
        let w = windows.get(&window_key).unwrap_or(&empty);
        let rows = load_csv(&csv_path);
        let (scores, labels) = score_file(&rows, w);
        let pos: u64 = labels.iter().map(|&l| u64::from(l)).sum();
        let a = auc(&scores, &labels);
        per_file.push((file_name.clone(), a, pos));
        #[allow(clippy::cast_precision_loss)]
        {
            weighted_sum += a * pos as f64;
        }
        total_anoms += pos;
    }

    println!("\nNAB realKnownCause per-file AUC:");
    for (name, a, pos) in &per_file {
        println!("  {a:.3}  pos={pos:<6}  {name}");
    }
    #[allow(clippy::cast_precision_loss)]
    let weighted_auc = if total_anoms == 0 {
        0.0
    } else {
        weighted_sum / total_anoms as f64
    };
    println!("aggregate weighted AUC (by positive count): {weighted_auc:.3}");

    // Floor is a regression guard, not a quality claim. Adjust
    // downward *only* with a commit message explaining which
    // detector change moved the needle, and by how much.
    assert!(
        weighted_auc > 0.60,
        "aggregate weighted AUC = {weighted_auc:.3} below floor 0.60 — \
         detector regression or dataset change?"
    );
}
