#![allow(clippy::unwrap_used, clippy::panic, clippy::similar_names)]
//! rcf-rs side of the external-bench comparison — reads the CSV
//! emitted by `scripts/external-bench/gen_points.py`, warms a
//! forest, then reports inserts/s, scores/s, and `AUC` against
//! the first-column label. Matches the metric shape of
//! `bench_rrcf.py` / `bench_sklearn_iforest.py` so the numbers sit
//! side by side in `docs/performance.md`.
//!
//! Deliberately `D = 16` (AWS-default) — regenerate the CSV with
//! a matching width if you want a different dimensionality.
//!
//! Run with:
//!     `cargo run --release --example external_bench_driver -- data.csv 100 256`

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use rcf_rs::{AnomalyScore, ForestBuilder, RcfError};

const D: usize = 16;

fn main() -> Result<(), RcfError> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: external_bench_driver <csv> <num_trees> <sample_size>");
        std::process::exit(2);
    }
    let path = &args[1];
    let num_trees: usize = args[2].parse().expect("num_trees integer");
    let sample_size: usize = args[3].parse().expect("sample_size integer");

    let (points, labels) = load_csv(path);
    let n = points.len();
    // Split 30 / 70 — warm on the first slice, score the rest
    // against the frozen-ish baseline. Matches realistic agent
    // deployments and gives a non-degenerate AUC signal on
    // synthetic outlier corpora.
    let split = n * 3 / 10;
    println!("points={n} dim={D} trees={num_trees} sample={sample_size} warm={split}");

    let mut forest = ForestBuilder::<D>::new()
        .num_trees(num_trees)
        .sample_size(sample_size)
        .seed(2026)
        .build()?;

    // Warm phase — insert the first slice, no scoring yet.
    let t_insert = Instant::now();
    for p in &points[..split] {
        forest.update(*p)?;
    }
    let insert_ns = t_insert.elapsed().as_nanos();
    #[allow(clippy::cast_precision_loss)]
    let insert_per_s = split as f64 * 1.0e9 / insert_ns as f64;

    // Eval phase — score the remaining points against the trained
    // baseline via the parallel `score_many` path. rayon fans out
    // across points on top of the per-tree parallelism, matching
    // sklearn's `n_jobs=-1` — apples to apples throughput.
    let eval = &points[split..];
    let eval_labels = &labels[split..];
    let t_score = Instant::now();
    let scored: Vec<AnomalyScore> = forest.score_many(eval)?;
    let scores: Vec<f64> = scored.iter().map(|s| f64::from(*s)).collect();
    let score_ns = t_score.elapsed().as_nanos();
    #[allow(clippy::cast_precision_loss)]
    let score_per_s = eval.len() as f64 * 1.0e9 / score_ns as f64;

    let a = auc(&scores, eval_labels);

    #[allow(clippy::cast_precision_loss)]
    {
        println!(
            "  inserts        = {split}, total {:.2} ms",
            insert_ns as f64 / 1.0e6
        );
        println!(
            "  scores         = {}, total {:.2} ms",
            eval.len(),
            score_ns as f64 / 1.0e6
        );
        println!(
            "  per-op insert  = {:.0} ns",
            insert_ns as f64 / split as f64
        );
        println!(
            "  per-op score   = {:.0} ns",
            score_ns as f64 / eval.len() as f64
        );
    }
    println!("  updates_per_s  = {insert_per_s:.0}");
    println!("  scores_per_s   = {score_per_s:.0}");
    println!("  auc            = {a:.3}");
    Ok(())
}

fn load_csv(path: &str) -> (Vec<[f64; D]>, Vec<u8>) {
    let file = File::open(path).expect("open input CSV");
    let reader = BufReader::new(file);
    let mut points: Vec<[f64; D]> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.expect("read line");
        if i == 0 && line.starts_with("label") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        assert!(
            parts.len() == D + 1,
            "expected {} columns, got {} on line {i}",
            D + 1,
            parts.len()
        );
        let label: u8 = parts[0].parse().expect("label u8");
        labels.push(label);
        let mut point = [0.0_f64; D];
        for (d, cell) in parts[1..].iter().enumerate() {
            point[d] = cell.parse().expect("cell f64");
        }
        points.push(point);
    }
    (points, labels)
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
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
        prev_tpr = tpr;
        prev_fpr = fpr;
    }
    auc
}
