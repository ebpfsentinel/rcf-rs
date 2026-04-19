//! Extended criterion bench suite covering the value-add APIs
//! beyond `forest_throughput.rs`: bulk scoring, early-termination,
//! forensic baseline, and tenant similarity / cross-tenant.
//!
//! Measures wall-clock on `x86_64` with `mimalloc` pinned
//! globally. Run with `cargo bench --bench extended` or
//! `cargo bench --bench extended -- bulk_scoring/` for a group.

#![allow(clippy::cast_precision_loss)]

use criterion::{Criterion, criterion_group, criterion_main};
use mimalloc::MiMalloc;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{
    EarlyTermConfig, ForestBuilder, RandomCutForest, TenantForestPool, ThresholdedForestBuilder,
};
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
            *slot = rng.random::<f64>();
        }
        forest.update(p).expect("update succeeds");
    }
    forest
}

fn make_batch<const D: usize>(seed: u64, count: usize) -> Vec<[f64; D]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let mut p = [0.0_f64; D];
            for slot in &mut p {
                *slot = rng.random::<f64>();
            }
            p
        })
        .collect()
}

/// Bulk batch scoring vs. serial for-loop. Tests three batch
/// sizes so callers can pick the configuration matching their
/// expected backfill / replay workload.
fn bench_bulk_scoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_scoring");
    let forest = build_warm_forest::<16>(100, 256, 2026);

    for &batch in &[64_usize, 512, 4096] {
        let probes: Vec<[f64; 16]> = make_batch::<16>(7, batch);

        group.bench_function(format!("score_many_{batch}"), |b| {
            b.iter(|| {
                let out = forest
                    .score_many(black_box(&probes))
                    .expect("score_many succeeds");
                black_box(out);
            });
        });

        group.bench_function(format!("serial_score_{batch}"), |b| {
            b.iter(|| {
                let mut sum = 0.0_f64;
                for p in &probes {
                    sum += f64::from(forest.score(black_box(p)).expect("score"));
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

/// Early-termination scoring vs. full-ensemble `score()`. Two
/// thresholds: tight (rarely stops early) and loose (stops often).
fn bench_early_term(c: &mut Criterion) {
    let mut group = c.benchmark_group("early_term");
    let forest = build_warm_forest::<16>(100, 256, 2026);
    let probe: [f64; 16] = make_batch::<16>(7, 1)[0];

    let tight = EarlyTermConfig {
        min_trees: 16,
        confidence_threshold: 0.02,
    };
    let loose = EarlyTermConfig {
        min_trees: 16,
        confidence_threshold: 0.20,
    };

    group.bench_function("full_score", |b| {
        b.iter(|| {
            let s = forest.score(black_box(&probe)).expect("score");
            black_box(s);
        });
    });
    group.bench_function("early_term_tight_0.02", |b| {
        b.iter(|| {
            let r = forest
                .score_early_term(black_box(&probe), tight)
                .expect("early");
            black_box(r);
        });
    });
    group.bench_function("early_term_loose_0.20", |b| {
        b.iter(|| {
            let r = forest
                .score_early_term(black_box(&probe), loose)
                .expect("early");
            black_box(r);
        });
    });
    group.finish();
}

/// Forensic baseline — O(`live_points` × D). Sweep
/// `(sample_size, D)` to show how the aggregate scales.
fn bench_forensic(c: &mut Criterion) {
    let mut group = c.benchmark_group("forensic_baseline");

    let f16_256 = build_warm_forest::<16>(100, 256, 2026);
    let probe_16: [f64; 16] = make_batch::<16>(7, 1)[0];
    group.bench_function("100t_256s_16d", |b| {
        b.iter(|| {
            let r = f16_256
                .forensic_baseline(black_box(&probe_16))
                .expect("forensic");
            black_box(r);
        });
    });

    let f4_256 = build_warm_forest::<4>(100, 256, 2026);
    let probe_4: [f64; 4] = make_batch::<4>(7, 1)[0];
    group.bench_function("100t_256s_4d", |b| {
        b.iter(|| {
            let r = f4_256
                .forensic_baseline(black_box(&probe_4))
                .expect("forensic");
            black_box(r);
        });
    });

    let f16_1024 = build_warm_forest::<16>(100, 1024, 2026);
    group.bench_function("100t_1024s_16d", |b| {
        b.iter(|| {
            let r = f16_1024
                .forensic_baseline(black_box(&probe_16))
                .expect("forensic");
            black_box(r);
        });
    });

    group.finish();
}

fn build_warm_pool(num_tenants: usize, samples_per_tenant: usize) -> TenantForestPool<u32, 4> {
    let mut pool: TenantForestPool<u32, 4> = TenantForestPool::new(num_tenants * 2, || {
        ThresholdedForestBuilder::<4>::new()
            .num_trees(50)
            .sample_size(64)
            .min_observations(16)
            .min_threshold(0.1)
            .seed(2026)
            .build()
    })
    .expect("pool build");
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let tenant_count = u32::try_from(num_tenants).expect("bench num_tenants fits u32");
    for t in 0..tenant_count {
        for _ in 0..samples_per_tenant {
            let p = [
                rng.random::<f64>(),
                rng.random::<f64>(),
                rng.random::<f64>(),
                rng.random::<f64>(),
            ];
            pool.process(&t, p).expect("process");
        }
    }
    pool
}

/// Tenant similarity matrix + cross-tenant scoring. Show the
/// `N²` growth of `similarity_matrix` and the `O(N)` cost of
/// `score_across_tenants`.
fn bench_tenant(c: &mut Criterion) {
    let mut group = c.benchmark_group("tenant_pool");

    for &n in &[32_usize, 128, 512] {
        let pool = build_warm_pool(n, 128);
        let probe = [0.5_f64, 0.5, 0.5, 0.5];

        group.bench_function(format!("similarity_matrix_{n}t"), |b| {
            b.iter(|| {
                let m = pool.similarity_matrix(black_box(16));
                black_box(m);
            });
        });
        group.bench_function(format!("score_across_{n}t"), |b| {
            b.iter(|| {
                let r = pool.score_across_tenants(black_box(&probe)).expect("cross");
                black_box(r);
            });
        });
        group.bench_function(format!("most_similar_top5_{n}t"), |b| {
            b.iter(|| {
                let r = pool.most_similar(black_box(&0_u32), 5, 16);
                black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bulk_scoring,
    bench_early_term,
    bench_forensic,
    bench_tenant
);
criterion_main!(benches);
