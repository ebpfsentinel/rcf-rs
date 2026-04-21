//! Criterion benchmarks for [`rcf_rs::RandomCutForest`] insert and
//! score throughput across a representative `(num_trees, sample_size,
//! dim)` matrix. Targets ≥ 100k inserts/sec and ≥ 50k scores/sec at
//! the AWS-default `(100, 256, 16)` configuration on a typical
//! `x86_64` dev box.
//!
//! Forests are now const-generic over `D`, so each `dim` is its own
//! monomorphisation. The matrix entries are expanded inline through
//! the [`bench_dim`] helper, parameterised by the target dimension.
//!
//! The bench process pins `mimalloc` as the global allocator —
//! reduces small-allocation overhead vs the system allocator, which
//! is the same trick callers can apply at their `main.rs` to free
//! a few percent on every `update`/`score` call.
//!
//! Run with `cargo bench`. Compile-only check via `cargo bench --no-run`.

#![allow(clippy::cast_precision_loss)]

use criterion::{
    BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use mimalloc::MiMalloc;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rcf_rs::{ForestBuilder, RandomCutForest};
use std::hint::black_box;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn build_warm_forest<const D: usize>(
    num_trees: usize,
    sample_size: usize,
    seed: u64,
) -> RandomCutForest<D> {
    let mut forest = ForestBuilder::<D>::new()
        .num_trees(num_trees)
        .sample_size(sample_size)
        .seed(seed)
        .build()
        .expect("AWS-conformant config");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for _ in 0..(sample_size * 4) {
        let mut p = [0.0_f64; D];
        for slot in &mut p {
            *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng);
        }
        forest.update(p).expect("update succeeds");
    }
    forest
}

fn bench_update_for<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    num_trees: usize,
    sample_size: usize,
) {
    let id = format!("{num_trees}t_{sample_size}s_{D}d");
    group.bench_function(&id, |b| {
        let mut forest = build_warm_forest::<D>(num_trees, sample_size, 2026);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        b.iter(|| {
            let mut p = [0.0_f64; D];
            for slot in &mut p {
                *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng);
            }
            forest.update(black_box(p)).expect("update succeeds");
        });
    });
}

fn bench_score_for<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    num_trees: usize,
    sample_size: usize,
) {
    let id = format!("{num_trees}t_{sample_size}s_{D}d");
    group.bench_function(&id, |b| {
        let forest = build_warm_forest::<D>(num_trees, sample_size, 2026);
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        b.iter(|| {
            let mut p = [0.0_f64; D];
            for slot in &mut p {
                *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng);
            }
            let s = forest.score(black_box(&p)).expect("score succeeds");
            black_box(s);
        });
    });
}

fn bench_attribution_for<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    num_trees: usize,
    sample_size: usize,
) {
    let id = format!("{num_trees}t_{sample_size}s_{D}d");
    group.bench_function(&id, |b| {
        let forest = build_warm_forest::<D>(num_trees, sample_size, 2026);
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        b.iter(|| {
            let mut p = [0.0_f64; D];
            for slot in &mut p {
                *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng);
            }
            let di = forest
                .attribution(black_box(&p))
                .expect("attribution succeeds");
            black_box(di);
        });
    });
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_update");
    bench_update_for::<16>(&mut group, 50, 128);
    bench_update_for::<4>(&mut group, 100, 256);
    bench_update_for::<16>(&mut group, 100, 256);
    bench_update_for::<64>(&mut group, 100, 256);
    bench_update_for::<16>(&mut group, 200, 512);
    group.finish();
}

fn bench_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_score");
    bench_score_for::<16>(&mut group, 50, 128);
    bench_score_for::<4>(&mut group, 100, 256);
    bench_score_for::<16>(&mut group, 100, 256);
    bench_score_for::<64>(&mut group, 100, 256);
    bench_score_for::<16>(&mut group, 200, 512);
    group.finish();
}

fn bench_attribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_attribution");
    bench_attribution_for::<4>(&mut group, 100, 256);
    bench_attribution_for::<16>(&mut group, 100, 256);
    bench_attribution_for::<64>(&mut group, 100, 256);
    group.finish();
}

fn bench_score_and_attribution_for<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    num_trees: usize,
    sample_size: usize,
) {
    let id = format!("{num_trees}t_{sample_size}s_{D}d");
    group.bench_function(&id, |b| {
        let forest = build_warm_forest::<D>(num_trees, sample_size, 2026);
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        b.iter(|| {
            let mut p = [0.0_f64; D];
            for slot in &mut p {
                *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng);
            }
            let out = forest
                .score_and_attribution(black_box(&p))
                .expect("score_and_attribution succeeds");
            black_box(out);
        });
    });
}

fn bench_split_score_then_attribution_for<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    num_trees: usize,
    sample_size: usize,
) {
    let id = format!("{num_trees}t_{sample_size}s_{D}d");
    group.bench_function(&id, |b| {
        let forest = build_warm_forest::<D>(num_trees, sample_size, 2026);
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        b.iter(|| {
            let mut p = [0.0_f64; D];
            for slot in &mut p {
                *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng);
            }
            let s = forest.score(black_box(&p)).expect("score succeeds");
            let di = forest
                .attribution(black_box(&p))
                .expect("attribution succeeds");
            black_box((s, di));
        });
    });
}

fn bench_combined(c: &mut Criterion) {
    let mut merged = c.benchmark_group("forest_score_and_attribution");
    bench_score_and_attribution_for::<16>(&mut merged, 100, 256);
    merged.finish();
    let mut split = c.benchmark_group("forest_split_score_then_attribution");
    bench_split_score_then_attribution_for::<16>(&mut split, 100, 256);
    split.finish();
}

fn bench_codisp_many_for<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    num_trees: usize,
    sample_size: usize,
    batch: usize,
) {
    let id = format!("{num_trees}t_{sample_size}s_{D}d_k{batch}");
    group.bench_function(&id, |b| {
        let mut rng = ChaCha8Rng::seed_from_u64(19);
        let probes: Vec<[f64; D]> = (0..batch)
            .map(|_| {
                let mut p = [0.0_f64; D];
                for slot in &mut p {
                    *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng);
                }
                p
            })
            .collect();
        let mut forest = build_warm_forest::<D>(num_trees, sample_size, 2026);
        b.iter(|| {
            let out = forest
                .score_codisp_many(black_box(&probes))
                .expect("score_codisp_many succeeds");
            black_box(out);
        });
    });
}

fn bench_codisp_loop_for<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    num_trees: usize,
    sample_size: usize,
    batch: usize,
) {
    let id = format!("{num_trees}t_{sample_size}s_{D}d_k{batch}");
    group.bench_function(&id, |b| {
        let mut rng = ChaCha8Rng::seed_from_u64(19);
        let probes: Vec<[f64; D]> = (0..batch)
            .map(|_| {
                let mut p = [0.0_f64; D];
                for slot in &mut p {
                    *slot = <ChaCha8Rng as rand::Rng>::random::<f64>(&mut rng);
                }
                p
            })
            .collect();
        let mut forest = build_warm_forest::<D>(num_trees, sample_size, 2026);
        b.iter(|| {
            let mut out = Vec::with_capacity(probes.len());
            for p in &probes {
                out.push(
                    forest
                        .score_codisp(black_box(p))
                        .expect("score_codisp succeeds"),
                );
            }
            black_box(out);
        });
    });
}

fn bench_codisp(c: &mut Criterion) {
    let mut batch = c.benchmark_group("forest_codisp_many");
    bench_codisp_many_for::<16>(&mut batch, 100, 256, 16);
    bench_codisp_many_for::<16>(&mut batch, 100, 256, 64);
    batch.finish();
    let mut loopg = c.benchmark_group("forest_codisp_loop");
    bench_codisp_loop_for::<16>(&mut loopg, 100, 256, 16);
    bench_codisp_loop_for::<16>(&mut loopg, 100, 256, 64);
    loopg.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_score,
    bench_attribution,
    bench_combined,
    bench_codisp
);
criterion_main!(benches);
