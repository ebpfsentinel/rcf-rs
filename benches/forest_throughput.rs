//! Criterion benchmarks for [`rcf_rs::RandomCutForest`] insert and
//! score throughput across a representative `(num_trees, sample_size,
//! dim)` matrix. Targets per RCF.9 AC #6: ≥ 100k inserts/sec and
//! ≥ 50k scores/sec at the AWS-default `(100, 256, 16)` configuration
//! on a typical `x86_64` dev box.
//!
//! Run with `cargo bench`. Compile-only check via `cargo bench --no-run`.

#![allow(clippy::cast_precision_loss)]

use criterion::{Criterion, criterion_group, criterion_main};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rcf_rs::{ForestBuilder, RandomCutForest};
use std::hint::black_box;

const CONFIG_MATRIX: &[(usize, usize, usize)] = &[
    (50, 128, 16),
    (100, 256, 4),
    (100, 256, 16),
    (100, 256, 64),
    (200, 512, 16),
];

fn build_warm_forest(
    num_trees: usize,
    sample_size: usize,
    dim: usize,
    seed: u64,
) -> RandomCutForest {
    let mut forest = ForestBuilder::new(dim)
        .num_trees(num_trees)
        .sample_size(sample_size)
        .seed(seed)
        .build()
        .expect("AWS-conformant config");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Pre-fill so each tree's reservoir is at capacity — the
    // post-warmup hot path mixes Inserted/Replaced/Rejected outcomes,
    // which is what production workloads see.
    for _ in 0..(sample_size * 4) {
        let p: Vec<f64> = (0..dim)
            .map(|_| <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng))
            .collect();
        forest.update(p).expect("update succeeds");
    }
    forest
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_update");
    for &(num_trees, sample_size, dim) in CONFIG_MATRIX {
        let id = format!("{num_trees}t_{sample_size}s_{dim}d");
        group.bench_function(&id, |b| {
            let mut forest = build_warm_forest(num_trees, sample_size, dim, 2026);
            let mut rng = ChaCha8Rng::seed_from_u64(7);
            b.iter(|| {
                let p: Vec<f64> = (0..dim)
                    .map(|_| <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng))
                    .collect();
                forest.update(black_box(p)).expect("update succeeds");
            });
        });
    }
    group.finish();
}

fn bench_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_score");
    for &(num_trees, sample_size, dim) in CONFIG_MATRIX {
        let id = format!("{num_trees}t_{sample_size}s_{dim}d");
        group.bench_function(&id, |b| {
            let forest = build_warm_forest(num_trees, sample_size, dim, 2026);
            let mut rng = ChaCha8Rng::seed_from_u64(11);
            b.iter(|| {
                let p: Vec<f64> = (0..dim)
                    .map(|_| <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng))
                    .collect();
                let s = forest.score(black_box(&p)).expect("score succeeds");
                black_box(s);
            });
        });
    }
    group.finish();
}

fn bench_attribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_attribution");
    // Attribution is heavier than score — bench only the AWS default
    // configuration plus the two extremes for a sense of scale.
    for &(num_trees, sample_size, dim) in &[(100, 256, 4), (100, 256, 16), (100, 256, 64)] {
        let id = format!("{num_trees}t_{sample_size}s_{dim}d");
        group.bench_function(&id, |b| {
            let forest = build_warm_forest(num_trees, sample_size, dim, 2026);
            let mut rng = ChaCha8Rng::seed_from_u64(13);
            b.iter(|| {
                let p: Vec<f64> = (0..dim)
                    .map(|_| <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng))
                    .collect();
                let di = forest
                    .attribution(black_box(&p))
                    .expect("attribution succeeds");
                black_box(di);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_insert, bench_score, bench_attribution);
criterion_main!(benches);
