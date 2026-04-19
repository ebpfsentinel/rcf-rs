#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::redundant_closure_for_method_calls
)]
//! NAB ablation harness — try multiple hyperparameter / scoring
//! configurations on the `realKnownCause` subset, print weighted
//! aggregate AUC per config. Used to gauge whether NAB results
//! can be improved beyond the 0.615 baseline measured by
//! `tests/nab.rs`.
//!
//! Run with `cargo run --release --example nab_ablation -- /opt/nab`.

#![cfg(feature = "serde_json")]

use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

use rcf_rs::{AnomalyScore, ForestBuilder, ThresholdedForestBuilder};
use serde_json::Value;

#[derive(Clone, Copy)]
enum Mode {
    /// Bare RCF, warm-then-frozen, `score()` per probe.
    ForestFrozen,
    /// Bare RCF, probe-based hack (`update_indexed → score → delete`).
    ForestProbe,
    /// Bare RCF, proper codisp via `score_codisp` — insert leaf,
    /// walk ancestors summing `max(sibling.mass / subtree.mass)`,
    /// remove leaf. The rrcf / AWS-Java scoring semantic.
    ForestCodisp,
    /// TRCF with online updates — EMA-adaptive threshold evolves
    /// with the stream, `time_decay > 0` lets the baseline age out.
    /// `score_only` is called first (frozen view) and then the
    /// point is also folded back into the forest via `process` so
    /// the threshold keeps adapting.
    TrcfOnline,
}

#[derive(Clone, Copy)]
struct Config {
    lag: usize,
    trees: usize,
    sample: usize,
    warm_frac: f64,
    iaf: f64,
    mode: Mode,
    /// Differencing embedding: `[v_t - v_{t-1}, v_{t-1} - v_{t-2}, …]`
    /// instead of raw lag values. Stationarises drifting series.
    diff_embedding: bool,
    /// Z-score normalise the embedding by warm-phase stddev.
    zscore: bool,
    /// EMA-smooth the raw scores with factor `alpha` before AUC
    /// (`0.0` = no smoothing, `0.3` = moderate).
    smooth_alpha: f64,
    label: &'static str,
}

fn main() {
    let Some(nab_root) = env::args().nth(1) else {
        eprintln!("usage: nab_ablation <path/to/NAB>");
        std::process::exit(2);
    };
    let root = Path::new(&nab_root);
    let mk = |label, lag, mode, diff, zscore, smooth| Config {
        lag,
        trees: 100,
        sample: 256,
        warm_frac: 0.15,
        iaf: 1.0,
        mode,
        diff_embedding: diff,
        zscore,
        smooth_alpha: smooth,
        label,
    };
    let configs = vec![
        // baseline sweep (from earlier)
        mk("baseline D=8", 8, Mode::ForestFrozen, false, false, 0.0),
        mk("lag=32", 32, Mode::ForestFrozen, false, false, 0.0),
        // new axes on top of lag=32
        mk("lag=32 + diff", 32, Mode::ForestFrozen, true, false, 0.0),
        mk("lag=32 + zscore", 32, Mode::ForestFrozen, false, true, 0.0),
        mk(
            "lag=32 + diff + zscore",
            32,
            Mode::ForestFrozen,
            true,
            true,
            0.0,
        ),
        mk(
            "lag=32 + smooth(0.3)",
            32,
            Mode::ForestFrozen,
            false,
            false,
            0.3,
        ),
        mk(
            "lag=32 + smooth(0.1)",
            32,
            Mode::ForestFrozen,
            false,
            false,
            0.1,
        ),
        // TRCF online — real design for contextual anomalies
        mk("trcf-online D=8", 8, Mode::TrcfOnline, false, false, 0.0),
        mk("trcf-online D=32", 32, Mode::TrcfOnline, false, false, 0.0),
        mk(
            "trcf-online D=32 + diff",
            32,
            Mode::TrcfOnline,
            true,
            false,
            0.0,
        ),
        mk(
            "trcf-online D=32 + diff + zscore",
            32,
            Mode::TrcfOnline,
            true,
            true,
            0.0,
        ),
        mk(
            "trcf-online D=32 + diff + smooth(0.3)",
            32,
            Mode::TrcfOnline,
            true,
            false,
            0.3,
        ),
        // stacked combinations
        mk(
            "lag=32 + zscore + smooth(0.1)",
            32,
            Mode::ForestFrozen,
            false,
            true,
            0.1,
        ),
        mk(
            "lag=32 + zscore + smooth(0.3)",
            32,
            Mode::ForestFrozen,
            false,
            true,
            0.3,
        ),
        mk(
            "lag=32 + zscore + smooth(0.05)",
            32,
            Mode::ForestFrozen,
            false,
            true,
            0.05,
        ),
        mk(
            "lag=32 + zscore + smooth(0.02)",
            32,
            Mode::ForestFrozen,
            false,
            true,
            0.02,
        ),
        mk(
            "lag=32 + zscore + smooth(0.01)",
            32,
            Mode::ForestFrozen,
            false,
            true,
            0.01,
        ),
        mk(
            "lag=64 + zscore + smooth(0.05)",
            64,
            Mode::ForestFrozen,
            false,
            true,
            0.05,
        ),
        // probe variants kept for reference
        mk("probe-score D=8", 8, Mode::ForestProbe, false, false, 0.0),
        // proper codisp (probe-based, walk leaf→root)
        mk("codisp D=8", 8, Mode::ForestCodisp, false, false, 0.0),
        mk("codisp D=32", 32, Mode::ForestCodisp, false, false, 0.0),
        mk(
            "codisp D=32 + zscore",
            32,
            Mode::ForestCodisp,
            false,
            true,
            0.0,
        ),
        mk(
            "codisp D=32 + zscore + smooth(0.02)",
            32,
            Mode::ForestCodisp,
            false,
            true,
            0.02,
        ),
    ];

    let windows = load_windows(&root.join("labels/combined_windows.json"));
    let mut files: Vec<_> = fs::read_dir(root.join("data/realKnownCause"))
        .expect("realKnownCause dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "csv"))
        .collect();
    files.sort_by_key(std::fs::DirEntry::file_name);

    println!("{:18}  {:>6}  {:>8}", "config", "AUC", "time(s)");
    for cfg in &configs {
        let t0 = Instant::now();
        let mut sum = 0.0_f64;
        let mut total = 0_u64;
        for entry in &files {
            let csv = entry.path();
            let name = csv.file_name().unwrap().to_string_lossy().into_owned();
            let key = format!("realKnownCause/{name}");
            let empty = Vec::new();
            let w = windows.get(&key).unwrap_or(&empty);
            let rows = load_csv(&csv);
            let (scores, labels) = match cfg.lag {
                8 => run::<8>(cfg, &rows, w),
                16 => run::<16>(cfg, &rows, w),
                32 => run::<32>(cfg, &rows, w),
                64 => run::<64>(cfg, &rows, w),
                _ => panic!("unhandled lag"),
            };
            let pos = labels.iter().map(|&l| u64::from(l)).sum::<u64>();
            let a = auc(&scores, &labels);
            #[allow(clippy::cast_precision_loss)]
            {
                sum += a * pos as f64;
            }
            total += pos;
        }
        #[allow(clippy::cast_precision_loss)]
        let agg = if total == 0 { 0.0 } else { sum / total as f64 };
        let elapsed = t0.elapsed().as_secs_f64();
        println!("{:18}  {:>6.3}  {:>8.1}", cfg.label, agg, elapsed);
    }
}

fn run<const D: usize>(
    cfg: &Config,
    rows: &[(String, f64)],
    windows: &[(String, String)],
) -> (Vec<f64>, Vec<u8>) {
    if rows.len() < 2 * D + 1 {
        return (Vec::new(), Vec::new());
    }

    // Build embeddings — either raw lag or first-difference.
    let (embeddings, ts_offset) = if cfg.diff_embedding {
        // [Δ_t, Δ_{t-1}, …, Δ_{t-(D-1)}] where Δ_t = v_t - v_{t-1}
        // Need D+1 history to build the first diff embedding →
        // start at index D.
        let mut emb = Vec::with_capacity(rows.len().saturating_sub(D));
        for i in D..rows.len() {
            let mut e = [0.0_f64; D];
            for (k, slot) in e.iter_mut().enumerate() {
                let j = i - k;
                *slot = rows[j].1 - rows[j - 1].1;
            }
            emb.push(e);
        }
        (emb, D)
    } else {
        let mut emb = Vec::with_capacity(rows.len() - (D - 1));
        for i in (D - 1)..rows.len() {
            let mut e = [0.0_f64; D];
            for (k, slot) in e.iter_mut().enumerate() {
                *slot = rows[i + k + 1 - D].1;
            }
            emb.push(e);
        }
        (emb, D - 1)
    };
    let mut embeddings = embeddings;
    let embed_len = embeddings.len();
    #[allow(clippy::cast_precision_loss)]
    let warm_end = ((embed_len as f64) * cfg.warm_frac) as usize;

    // Z-score normalise using warm-phase per-dim stddev.
    if cfg.zscore && warm_end > 1 {
        let mut means = [0.0_f64; D];
        let mut m2 = [0.0_f64; D];
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
        for (d, s) in stddevs.iter_mut().enumerate() {
            *s = (m2[d] / n_f).sqrt().max(1.0e-9);
        }
        for emb in &mut embeddings {
            for d in 0..D {
                emb[d] = (emb[d] - means[d]) / stddevs[d];
            }
        }
    }

    let mut raw_scores = Vec::with_capacity(embed_len.saturating_sub(warm_end));
    let mut labels = Vec::with_capacity(embed_len.saturating_sub(warm_end));

    match cfg.mode {
        Mode::ForestFrozen | Mode::ForestProbe | Mode::ForestCodisp => {
            let mut forest = ForestBuilder::<D>::new()
                .num_trees(cfg.trees)
                .sample_size(cfg.sample)
                .initial_accept_fraction(cfg.iaf)
                .seed(2026)
                .build()
                .unwrap();
            for p in &embeddings[..warm_end] {
                forest.update(*p).ok();
            }
            for (i, p) in embeddings[warm_end..].iter().enumerate() {
                let s = match cfg.mode {
                    Mode::ForestCodisp => forest
                        .score_codisp(p)
                        .unwrap_or_else(|_| AnomalyScore::new(0.0).unwrap()),
                    Mode::ForestProbe => match forest.update_indexed(*p) {
                        Ok(idx) => {
                            let s = forest
                                .score(p)
                                .unwrap_or_else(|_| AnomalyScore::new(0.0).unwrap());
                            forest.delete(idx).ok();
                            s
                        }
                        Err(_) => AnomalyScore::new(0.0).unwrap(),
                    },
                    _ => forest
                        .score(p)
                        .unwrap_or_else(|_| AnomalyScore::new(0.0).unwrap()),
                };
                raw_scores.push(f64::from(s));
                let row_idx = warm_end + i + ts_offset;
                labels.push(u8::from(in_any_window(&rows[row_idx].0, windows)));
            }
        }
        Mode::TrcfOnline => {
            let mut trcf = ThresholdedForestBuilder::<D>::new()
                .num_trees(cfg.trees)
                .sample_size(cfg.sample)
                .min_observations(u64::try_from(warm_end.max(1)).unwrap_or(1))
                .min_threshold(0.0)
                .seed(2026)
                .build()
                .unwrap();
            // Warm phase — process to feed both the forest and the
            // score-stream EMA so the threshold stabilises.
            for p in &embeddings[..warm_end] {
                trcf.process(*p).ok();
            }
            // Eval phase — continue processing, collect the graded
            // verdict. `grade()` is already normalised against the
            // adaptive threshold, which is the point.
            for (i, p) in embeddings[warm_end..].iter().enumerate() {
                let grade = trcf.process(*p).map_or(0.0, |g| g.grade());
                raw_scores.push(grade);
                let row_idx = warm_end + i + ts_offset;
                labels.push(u8::from(in_any_window(&rows[row_idx].0, windows)));
            }
        }
    }

    // Optional EMA smoothing of raw scores.
    let scores = if cfg.smooth_alpha > 0.0 && !raw_scores.is_empty() {
        let mut smoothed = Vec::with_capacity(raw_scores.len());
        let mut acc = raw_scores[0];
        for &s in &raw_scores {
            acc = cfg.smooth_alpha * s + (1.0 - cfg.smooth_alpha) * acc;
            smoothed.push(acc);
        }
        smoothed
    } else {
        raw_scores
    };
    (scores, labels)
}

fn load_csv(path: &Path) -> Vec<(String, f64)> {
    let text = fs::read_to_string(path).expect("read csv");
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 && line.starts_with("timestamp") {
            continue;
        }
        let mut parts = line.splitn(2, ',');
        let ts = parts.next().unwrap().to_string();
        let v: f64 = parts.next().unwrap().parse().unwrap();
        out.push((ts, v));
    }
    out
}

fn load_windows(path: &Path) -> std::collections::HashMap<String, Vec<(String, String)>> {
    let text = fs::read_to_string(path).expect("read windows");
    let json: Value = serde_json::from_str(&text).unwrap();
    let mut out = std::collections::HashMap::new();
    if let Value::Object(map) = json {
        for (k, v) in map {
            let mut windows = Vec::new();
            if let Value::Array(arr) = v {
                for pair in arr {
                    if let Value::Array(pair) = pair
                        && pair.len() == 2
                    {
                        let s = pair[0].as_str().map(trunc19).unwrap();
                        let e = pair[1].as_str().map(trunc19).unwrap();
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
    s.chars().take(19).collect()
}

fn in_any_window(ts: &str, w: &[(String, String)]) -> bool {
    w.iter().any(|(s, e)| ts >= s.as_str() && ts <= e.as_str())
}

fn auc(scores: &[f64], labels: &[u8]) -> f64 {
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
    let mut auc = 0.0_f64;
    let mut tp = 0_u64;
    let mut fp = 0_u64;
    let mut prev_tpr = 0.0_f64;
    let mut prev_fpr = 0.0_f64;
    for (_, l) in &pairs {
        if *l == 1 {
            tp += 1;
        } else {
            fp += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        let tpr = tp as f64 / total_pos as f64;
        #[allow(clippy::cast_precision_loss)]
        let fpr = fp as f64 / total_neg as f64;
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
        prev_tpr = tpr;
        prev_fpr = fpr;
    }
    auc
}
