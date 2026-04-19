#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
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

use rcf_rs::{AnomalyScore, ForestBuilder};
use serde_json::Value;

#[derive(Clone, Copy)]
struct Config {
    lag: usize,
    trees: usize,
    sample: usize,
    warm_frac: f64,
    iaf: f64,
    probe: bool, // probe-based scoring (update+score+delete per probe)
    label: &'static str,
}

fn main() {
    let Some(nab_root) = env::args().nth(1) else {
        eprintln!("usage: nab_ablation <path/to/NAB>");
        std::process::exit(2);
    };
    let root = Path::new(&nab_root);
    let configs = vec![
        Config {
            lag: 8,
            trees: 100,
            sample: 256,
            warm_frac: 0.15,
            iaf: 1.0,
            probe: false,
            label: "baseline",
        },
        Config {
            lag: 16,
            trees: 100,
            sample: 256,
            warm_frac: 0.15,
            iaf: 1.0,
            probe: false,
            label: "lag=16",
        },
        Config {
            lag: 32,
            trees: 100,
            sample: 256,
            warm_frac: 0.15,
            iaf: 1.0,
            probe: false,
            label: "lag=32",
        },
        Config {
            lag: 8,
            trees: 200,
            sample: 256,
            warm_frac: 0.15,
            iaf: 1.0,
            probe: false,
            label: "trees=200",
        },
        Config {
            lag: 8,
            trees: 100,
            sample: 512,
            warm_frac: 0.15,
            iaf: 1.0,
            probe: false,
            label: "sample=512",
        },
        Config {
            lag: 8,
            trees: 100,
            sample: 256,
            warm_frac: 0.15,
            iaf: 0.125,
            probe: false,
            label: "iaf=0.125",
        },
        Config {
            lag: 8,
            trees: 100,
            sample: 256,
            warm_frac: 0.30,
            iaf: 1.0,
            probe: false,
            label: "warm=0.30",
        },
        Config {
            lag: 8,
            trees: 100,
            sample: 256,
            warm_frac: 0.15,
            iaf: 1.0,
            probe: true,
            label: "probe-score",
        },
        Config {
            lag: 16,
            trees: 100,
            sample: 256,
            warm_frac: 0.15,
            iaf: 1.0,
            probe: true,
            label: "lag=16+probe",
        },
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
    if rows.len() < 2 * D {
        return (Vec::new(), Vec::new());
    }
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
    let warm_end = ((embed_len as f64) * cfg.warm_frac) as usize;

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

    let mut scores = Vec::with_capacity(embed_len.saturating_sub(warm_end));
    let mut labels = Vec::with_capacity(embed_len.saturating_sub(warm_end));
    for (i, p) in embeddings[warm_end..].iter().enumerate() {
        let s: AnomalyScore = if cfg.probe {
            // Probe-based: insert, score, delete. Approximates the
            // rrcf codisp / AWS getAnomalyScore semantic — the
            // probe temporarily joins the tree and is scored
            // against itself-included.
            match forest.update_indexed(*p) {
                Ok(idx) => {
                    let s = forest
                        .score(p)
                        .unwrap_or_else(|_| AnomalyScore::new(0.0).unwrap());
                    forest.delete(idx).ok();
                    s
                }
                Err(_) => AnomalyScore::new(0.0).unwrap(),
            }
        } else {
            forest
                .score(p)
                .unwrap_or_else(|_| AnomalyScore::new(0.0).unwrap())
        };
        scores.push(f64::from(s));
        let row_idx = warm_end + i + (D - 1);
        let ts = &rows[row_idx].0;
        labels.push(u8::from(in_any_window(ts, windows)));
    }
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
