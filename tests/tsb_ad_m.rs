#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::redundant_closure_for_method_calls,
    dead_code
)]
//! TSB-AD multivariate track — detection-quality regression guard on
//! a modern, per-point-labeled time-series corpus.
//!
//! TSB-AD-M covers 200 multivariate series across 16 source datasets
//! (MSL, SMAP, SMD, MITDB, SVDB, PSM, GHL, Exathlon, …). Each CSV
//! exposes the feature columns followed by a `Label` column (0/1).
//! The filename encodes the train-split boundary (`tr_<N>`) and the
//! index of the first anomaly (`1st_<N>`), so no separate windows
//! JSON is needed.
//!
//! Unlike NAB, TSB-AD-M exercises the **native multivariate** RCF
//! path — no lag embedding, each feature column maps to one RCF dim.
//!
//! `#[ignore]` by default: dataset is ~515 MB. Run manually:
//!
//! ```bash
//! scripts/tsb_ad/fetch.sh /tmp/tsb-ad
//! RCF_TSB_AD_M_PATH=/tmp/tsb-ad/TSB-AD-M \
//!     cargo test --test tsb_ad_m --all-features -- --ignored --nocapture
//! ```
//!
//! Coverage note: the dispatch whitelist is
//! `{2, 3, 7, 8, 9, 12, 16, 17, 18, 19, 25, 29, 31, 38, 51, 55, 66}`
//! — 192 / 200 files (96 %). The eight D=248 files are skipped so
//! monomorphisation cost stays bounded; eBPFsentinel's production
//! feature-vector dim is typically ≤ 64 anyway.

#![cfg(all(feature = "serde_json", feature = "parallel"))]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use rcf_rs::{AnomalyScore, ForestBuilder};

/// Fraction of the `tr_<N>` warm split reserved to compute the
/// per-dim mean / stddev used for z-score normalisation. The full
/// warm split feeds the forest reservoir.
const NUM_TREES: usize = 100;
/// Reservoir size per tree — matches the NAB pipeline so the two
/// benchmarks are directly comparable at equal detector capacity.
const SAMPLE_SIZE: usize = 256;
/// EMA alpha applied to the raw score stream — same 0.02 value
/// tuned via the NAB ablation (`docs/performance.md`).
const SMOOTH_ALPHA: f64 = 0.02;
/// Seed pinned for reproducibility across runs.
const SEED: u64 = 2026;
/// Minimum positives required in the eval window for AUC to be
/// meaningful — below this the file is reported but excluded from
/// the aggregate.
const MIN_POSITIVES: u64 = 5;
/// Stride-subsample cap for the codisp variant — matches the AWS
/// Java bench's max-eval so the two scorers land on identical row
/// coverage. `score_codisp()` is ~30× slower than `score()` and
/// mutates the forest per probe, so scanning 100 % of eval rows on
/// a ~1 M-point corpus is session-infeasible. Stride-subsampling is
/// uniform across time.
const CODISP_MAX_EVAL: usize = 50_000;

/// Parsed filename metadata: source dataset name, dim, train-split
/// end index (`tr_<N>`), index of the first anomaly (`1st_<N>`).
#[derive(Debug, Clone)]
struct FileMeta {
    /// Full file name (for reporting).
    file: String,
    /// Source dataset label (`MSL`, `SMAP`, `SMD`, …). Used as the
    /// per-dataset aggregation key.
    dataset: String,
    /// Feature-vector dimensionality — equals `header_cols - 1`.
    dim: usize,
    /// Index of the last warm-split row (`tr_<N>` field).
    train_end: usize,
}

/// Per-file result used for reporting.
#[derive(Debug, Clone)]
struct FileResult {
    /// File name for the report.
    file: String,
    /// Dataset label.
    dataset: String,
    /// Dim of the series.
    dim: usize,
    /// Point-wise AUC over the eval split.
    auc: f64,
    /// Number of positive labels in the eval split.
    positives: u64,
    /// Number of eval-split points.
    eval_len: usize,
}

fn root() -> Option<PathBuf> {
    std::env::var("RCF_TSB_AD_M_PATH").ok().map(PathBuf::from)
}

/// Parse the filename into a [`FileMeta`]. Returns `None` on any
/// shape mismatch — unrecognised files are skipped rather than
/// crashing the whole test.
fn parse_meta(path: &Path, dim: usize) -> Option<FileMeta> {
    let file = path.file_name()?.to_str()?.to_string();
    let stem = file.strip_suffix(".csv")?;
    // Layout: NNN_DATASET_id_X_Category_tr_<train>_1st_<first>
    let parts: Vec<&str> = stem.split('_').collect();
    let dataset = parts.get(1)?.to_string();
    let tr_pos = parts.iter().position(|&p| p == "tr")?;
    let train_end: usize = parts.get(tr_pos + 1)?.parse().ok()?;
    Some(FileMeta {
        file,
        dataset,
        dim,
        train_end,
    })
}

/// Load a TSB-AD-M CSV. Returns `(features[], labels[])` — the
/// feature matrix is row-major (`row * dim + d`) to avoid the
/// const-generic tax of carrying `[f64; D]` through parsing; the
/// const-generic forest pipeline copies into fixed-size arrays at
/// score time.
fn load_csv(path: &Path) -> (Vec<f64>, Vec<u8>, usize) {
    let text = fs::read_to_string(path).expect("read CSV");
    let mut lines = text.lines();
    let header = lines.next().expect("header row");
    let cols = header.split(',').count();
    assert!(cols >= 2, "TSB-AD-M CSV has at least one feature + Label");
    let dim = cols - 1;
    let mut features: Vec<f64> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let mut tail = line;
        for _ in 0..dim {
            let (val_str, rest) = tail.split_once(',').expect("dim cols");
            features.push(val_str.parse().expect("f64"));
            tail = rest;
        }
        // Labels are stored as floats on some exports ("0.0") — parse
        // tolerantly and round to the nearest integer.
        let label_f: f64 = tail.parse().expect("label f64");
        labels.push(u8::from(label_f >= 0.5));
    }
    (features, labels, dim)
}

/// Trapezoidal ROC-AUC — shared with `tests/nab.rs`, duplicated to
/// keep the file self-contained.
fn auc(scores: &[f64], labels: &[u8]) -> f64 {
    assert_eq!(scores.len(), labels.len());
    let mut pairs: Vec<(f64, u8)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
    let total_pos: u64 = labels.iter().map(|&l| u64::from(l)).sum();
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
        let tpr = tp as f64 / total_pos as f64;
        let fpr = fp as f64 / total_neg as f64;
        auc_val += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
        prev_tpr = tpr;
        prev_fpr = fpr;
    }
    auc_val
}

/// Which scoring API to exercise per-point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scorer {
    /// Isolation-depth `score()` — non-mutating, rayon-parallel,
    /// full eval scan. Production hot-path API.
    IsolationDepth,
    /// Probe-based `score_codisp()` — inserts + walks leaf→root +
    /// deletes per call. ~30× slower than `score()`, matches the
    /// AWS Java `getAnomalyScore()` / rrcf `codisp()` semantic.
    /// Stride-subsampled eval to `CODISP_MAX_EVAL` rows.
    Codisp,
    /// Stateless codisp — no reservoir mutation, preserves
    /// frozen-baseline semantic across the full eval stream.
    CodispStateless,
}

/// Core per-file scoring pipeline, specialised on `D`. Reads the
/// row-major `features` matrix, normalises per-dim against the
/// train-split mean / stddev, warms the forest on the train split,
/// then scores the eval split (frozen baseline). Raw scores are
/// EMA-smoothed before AUC is computed. `scorer` selects between
/// isolation depth (full scan) and codisp (stride-subsampled).
fn score_file<const D: usize>(
    features: &[f64],
    labels: &[u8],
    dim: usize,
    train_end: usize,
    scorer: Scorer,
) -> (f64, u64, usize) {
    assert_eq!(dim, D, "score_file<D>: D mismatch");
    let n = labels.len();
    if n <= train_end + 1 {
        return (0.5, 0, 0);
    }

    // Per-dim mean / stddev on the train split.
    let mut means = [0.0_f64; D];
    let mut m2 = [0.0_f64; D];
    let n_train_f = train_end as f64;
    for r in 0..train_end {
        for (d, mean) in means.iter_mut().enumerate() {
            *mean += features[r * D + d];
        }
    }
    for mean in &mut means {
        *mean /= n_train_f.max(1.0);
    }
    for r in 0..train_end {
        for (d, m2_d) in m2.iter_mut().enumerate() {
            let delta = features[r * D + d] - means[d];
            *m2_d += delta * delta;
        }
    }
    let mut stddevs = [1.0_f64; D];
    for d in 0..D {
        stddevs[d] = (m2[d] / n_train_f.max(1.0)).sqrt().max(1.0e-9);
    }

    // z-score into fixed arrays for forest ingress.
    let read_point = |r: usize| -> [f64; D] {
        let mut p = [0.0_f64; D];
        for (d, slot) in p.iter_mut().enumerate() {
            *slot = (features[r * D + d] - means[d]) / stddevs[d];
        }
        p
    };

    let mut forest = ForestBuilder::<D>::new()
        .num_trees(NUM_TREES)
        .sample_size(SAMPLE_SIZE)
        .seed(SEED)
        .build()
        .expect("forest build");

    // Warm on the train split.
    for r in 0..train_end {
        let p = read_point(r);
        let _ = forest.update(p);
    }

    // Build eval index set: mutating codisp is stride-subsampled
    // because it churns the reservoir per probe; isolation depth
    // and stateless codisp scan the full eval tail.
    let eval_indices: Vec<usize> = match scorer {
        Scorer::IsolationDepth | Scorer::CodispStateless => (train_end..n).collect(),
        Scorer::Codisp => {
            let eval_n = n - train_end;
            let stride = eval_n.div_ceil(CODISP_MAX_EVAL).max(1);
            (train_end..n).step_by(stride).collect()
        }
    };

    let mut raw_scores = Vec::with_capacity(eval_indices.len());
    let mut eval_labels = Vec::with_capacity(eval_indices.len());
    for &r in &eval_indices {
        let p = read_point(r);
        let s: AnomalyScore = match scorer {
            Scorer::IsolationDepth => forest
                .score(&p)
                .unwrap_or_else(|_| AnomalyScore::new(0.0).expect("zero valid")),
            Scorer::Codisp => forest
                .score_codisp(&p)
                .unwrap_or_else(|_| AnomalyScore::new(0.0).expect("zero valid")),
            Scorer::CodispStateless => forest
                .score_codisp_stateless(&p)
                .unwrap_or_else(|_| AnomalyScore::new(0.0).expect("zero valid")),
        };
        raw_scores.push(f64::from(s));
        eval_labels.push(labels[r]);
    }

    // EMA-smoothed stream.
    let mut smoothed = Vec::with_capacity(raw_scores.len());
    if let Some(&first) = raw_scores.first() {
        let mut acc = first;
        for &s in &raw_scores {
            acc = SMOOTH_ALPHA * s + (1.0 - SMOOTH_ALPHA) * acc;
            smoothed.push(acc);
        }
    }

    let positives: u64 = eval_labels.iter().map(|&l| u64::from(l)).sum();
    let auc_val = auc(&smoothed, &eval_labels);
    (auc_val, positives, smoothed.len())
}

/// Central match-dispatch — every whitelisted `D` is monomorphised
/// here. Unlisted dims fall through to `None` and are skipped.
fn dispatch(
    features: &[f64],
    labels: &[u8],
    dim: usize,
    train_end: usize,
    scorer: Scorer,
) -> Option<(f64, u64, usize)> {
    macro_rules! arm {
        ($($d:literal),* $(,)?) => {
            match dim {
                $($d => Some(score_file::<$d>(features, labels, dim, train_end, scorer)),)*
                _ => None,
            }
        };
    }
    arm!(2, 3, 7, 8, 9, 12, 16, 17, 18, 19, 25, 29, 31, 38, 51, 55, 66)
}

/// Shared test body — iterate the corpus with the selected scorer,
/// print the per-dataset AUC breakdown, return the aggregate
/// positive-weighted AUC. The two `#[test]` entry points differ
/// only in the scorer variant and the floor they assert.
fn run_corpus(scorer: Scorer, label: &str) -> f64 {
    let Some(root_path) = root() else {
        panic!(
            "RCF_TSB_AD_M_PATH not set — run scripts/tsb_ad/fetch.sh and \
             export RCF_TSB_AD_M_PATH=<dir>/TSB-AD-M before running this ignored test"
        );
    };

    let mut entries: Vec<_> = fs::read_dir(&root_path)
        .expect("read TSB-AD-M dir")
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
        root_path.display()
    );

    // Parallelise at the file level — each file owns an independent
    // `RandomCutForest`, so rayon can fan out across cores. The
    // per-file pipeline is still sequential (const-generic D dispatch,
    // `score_codisp` mutates the forest serially per probe), but 14
    // files running concurrently on a 14C / 20T host dominates the
    // outer loop cost.
    let paths: Vec<_> = entries.iter().map(std::fs::DirEntry::path).collect();
    let outputs: Vec<Result<FileResult, usize>> = paths
        .par_iter()
        .filter_map(|path| {
            let (features, labels, dim) = load_csv(path);
            let meta = parse_meta(path, dim)?;
            let res = dispatch(&features, &labels, dim, meta.train_end, scorer);
            Some(match res {
                Some((auc_val, positives, eval_len)) => Ok(FileResult {
                    file: meta.file,
                    dataset: meta.dataset,
                    dim,
                    auc: auc_val,
                    positives,
                    eval_len,
                }),
                None => Err(dim),
            })
        })
        .collect();

    let mut results: Vec<FileResult> = Vec::new();
    let mut skipped_dim: BTreeMap<usize, usize> = BTreeMap::new();
    for out in outputs {
        match out {
            Ok(r) => results.push(r),
            Err(d) => *skipped_dim.entry(d).or_default() += 1,
        }
    }

    // Per-dataset aggregate (weighted by positive count).
    let mut per_dataset: BTreeMap<String, (f64, u64, usize)> = BTreeMap::new();
    for r in &results {
        let entry = per_dataset
            .entry(r.dataset.clone())
            .or_insert((0.0, 0, 0));
        if r.positives >= MIN_POSITIVES {
            entry.0 += r.auc * r.positives as f64;
            entry.1 += r.positives;
            entry.2 += 1;
        }
    }

    println!("\nTSB-AD-M [{label}] per-dataset AUC (weighted by positives, files counted):");
    let mut overall_weighted = 0.0_f64;
    let mut overall_positives = 0_u64;
    let mut overall_files = 0_usize;
    for (ds, (sum, pos, files)) in &per_dataset {
        if *pos == 0 {
            continue;
        }
        let a = sum / *pos as f64;
        overall_weighted += sum;
        overall_positives += *pos;
        overall_files += *files;
        println!("  {a:.3}  files={files:<3}  pos={pos:<7}  {ds}");
    }
    let overall_auc = if overall_positives == 0 {
        0.0
    } else {
        overall_weighted / overall_positives as f64
    };
    println!(
        "\n[{label}] aggregate weighted AUC: {overall_auc:.3} \
         across {overall_files} files / {overall_positives} positives"
    );

    if !skipped_dim.is_empty() {
        print!("\nskipped D (not in whitelist): ");
        for (d, n) in &skipped_dim {
            print!("{d}:{n} ");
        }
        println!();
    }
    overall_auc
}

#[test]
#[ignore = "requires RCF_TSB_AD_M_PATH pointing at the extracted TSB-AD-M corpus"]
fn tsb_ad_m_aggregate_auc_above_floor() {
    let overall_auc = run_corpus(Scorer::IsolationDepth, "score()");
    // Floor is a regression guard on the current pipeline. Adjust
    // only alongside a commit message explaining the detector change.
    assert!(
        overall_auc > 0.55,
        "aggregate weighted AUC = {overall_auc:.3} below floor 0.55"
    );
}

#[test]
#[ignore = "requires RCF_TSB_AD_M_PATH; codisp path is ~30× slower, runs ~1h at CODISP_MAX_EVAL=50k stride"]
fn tsb_ad_m_codisp_aggregate_auc_above_floor() {
    let overall_auc = run_corpus(Scorer::Codisp, "score_codisp()");
    // Codisp floor is independent of the isolation-depth floor —
    // mutation-per-probe plus stride subsampling drifts the number.
    assert!(
        overall_auc > 0.55,
        "codisp aggregate weighted AUC = {overall_auc:.3} below floor 0.55"
    );
}

#[test]
#[ignore = "requires RCF_TSB_AD_M_PATH; stateless codisp scans full eval, no reservoir mutation"]
fn tsb_ad_m_codisp_stateless_aggregate_auc_above_floor() {
    let overall_auc = run_corpus(Scorer::CodispStateless, "score_codisp_stateless()");
    assert!(
        overall_auc > 0.55,
        "stateless codisp aggregate weighted AUC = {overall_auc:.3} below floor 0.55"
    );
}
