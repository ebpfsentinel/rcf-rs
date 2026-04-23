#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! End-to-end TSB-AD-M evaluation harness.
//!
//! Loads one TSB-AD-M CSV, runs [`DynamicForest`] over its feature
//! columns with a 50 % calibration / 50 % scoring split, and
//! reports the VUS-PR of the resulting score trace against the
//! dataset's ground-truth labels.
//!
//! To evaluate a full TSB-AD-M folder, wrap this runner in a shell
//! loop over the `.csv` files — the library intentionally does not
//! bundle the dataset itself.
//!
//! ```bash
//! # Download TSB-AD-M first (≈ 1 GiB):
//! # https://huggingface.co/datasets/yzhao062/TSB-AD-M
//!
//! cargo run --release --example tsb_ad_m_eval -- \
//!     /path/to/TSB-AD-M/MSL_1_001.csv
//! ```
//!
//! Output (one line):
//!
//! ```text
//! MSL_1_001.csv  n=2000  dim=55  pos=123  VUS-PR=0.4312  elapsed=218ms
//! ```

use std::env;
use std::process;
use std::time::Instant;

use anomstream_core::{DynamicForest, ForestBuilder, RcfError, TsbAdMDataset, vus_pr_with_buffer};

const MAX_D: usize = 128;
const CALIBRATION_FRACTION: f64 = 0.5;
const NUM_TREES: usize = 64;
const SAMPLE_SIZE: usize = 256;
const SEED: u64 = 2026;
const VUS_BUFFER: usize = 100;

fn main() -> Result<(), RcfError> {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!(
            "usage: tsb_ad_m_eval <path/to/dataset.csv>\n\n\
             downloads: https://huggingface.co/datasets/yzhao062/TSB-AD-M"
        );
        process::exit(2);
    };

    let started = Instant::now();
    let dataset = TsbAdMDataset::load_csv(&path)?;
    let n = dataset.len();
    let dim = dataset.feature_dim();
    if dim > MAX_D {
        eprintln!(
            "tsb_ad_m_eval: feature dim {dim} > MAX_D {MAX_D} — \
             increase `MAX_D` in examples/tsb_ad_m_eval.rs"
        );
        process::exit(3);
    }

    let builder = ForestBuilder::<MAX_D>::new()
        .num_trees(NUM_TREES)
        .sample_size(SAMPLE_SIZE)
        .seed(SEED);
    let mut forest: DynamicForest<MAX_D> = DynamicForest::new(builder, dim)?;

    // Calibration split — warm the forest on the first fraction of
    // the series without emitting scores for it.
    let calib = (n as f64 * CALIBRATION_FRACTION) as usize;
    for row in dataset.features.iter().take(calib) {
        forest.update(row)?;
    }

    // Scoring pass — emit one score per remaining timestamp. The
    // calibration prefix gets a neutral score (forest is not trained
    // on itself in-order).
    let mut scores = vec![0.0_f64; n];
    for (i, row) in dataset.features.iter().enumerate().skip(calib) {
        let s = forest.score(row)?;
        scores[i] = f64::from(s);
        forest.update(row)?;
    }

    let vus = vus_pr_with_buffer(&scores, &dataset.labels, VUS_BUFFER)
        .expect("vus_pr on non-empty trace");
    let elapsed = started.elapsed();

    let filename = std::path::Path::new(&path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(&path);

    println!(
        "{filename}  n={n}  dim={dim}  pos={pos}  VUS-PR={vus:.4}  elapsed={elapsed_ms}ms",
        pos = dataset.positive_count(),
        elapsed_ms = elapsed.as_millis()
    );
    Ok(())
}
