//! Criterion bench suite for `anomstream-hotpath`: eBPF-style
//! ingress primitives (`UpdateSampler`, `PrefixRateCap`, bounded
//! MPSC `channel`). Run with
//! `cargo bench -p anomstream-hotpath --bench modules`.

#![allow(clippy::cast_precision_loss)]

use anomstream_hotpath::{PrefixRateCap, UpdateSampler, update_channel};
use criterion::{Criterion, criterion_group, criterion_main};
use mimalloc::MiMalloc;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// `UpdateSampler` `accept_stride` + `accept_hash` + keyed
/// `accept_hash` (murmur-mix secret). Target: per-packet overhead
/// on the classifier hot path.
fn bench_hot_path_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path_sampler");

    group.bench_function("accept_stride_keep_8", |b| {
        let s = UpdateSampler::new(8);
        b.iter(|| {
            let v = s.accept_stride();
            black_box(v);
        });
    });

    group.bench_function("accept_hash_keep_8", |b| {
        let s = UpdateSampler::new(8);
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            let h: u64 = rng.random();
            let v = s.accept_hash(black_box(h));
            black_box(v);
        });
    });

    group.bench_function("accept_hash_keyed_keep_8", |b| {
        let s = UpdateSampler::new_keyed(8).expect("getrandom");
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            let h: u64 = rng.random();
            let v = s.accept_hash(black_box(h));
            black_box(v);
        });
    });

    group.finish();
}

/// `PrefixRateCap::check_and_record` — 256-bucket atomic counter
/// sketch + window roll. Lock-free, `O(1)`.
fn bench_hot_path_prefix_cap(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path_prefix_cap");

    group.bench_function("check_and_record_100cap_1s", |b| {
        let cap = PrefixRateCap::new(100, 1_000);
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        let mut now_ms = 0_u64;
        b.iter(|| {
            let h: u64 = rng.random();
            now_ms = now_ms.wrapping_add(1);
            let v = cap.check_and_record(black_box(h), now_ms);
            black_box(v);
        });
    });

    group.finish();
}

/// Bounded MPSC `update_channel::try_enqueue` with background
/// drain thread keeping the queue non-full.
fn bench_hot_path_channel(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path_channel");

    group.bench_function("try_enqueue_4096cap", |b| {
        let (producer, consumer) = update_channel::<16>(4096);
        let drain = std::thread::spawn(move || while consumer.recv().is_some() {});
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng.random();
            }
            let ok = producer.try_enqueue(black_box(p));
            black_box(ok);
        });
        drop(producer);
        let _ = drain.join();
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hot_path_sampler,
    bench_hot_path_prefix_cap,
    bench_hot_path_channel
);
criterion_main!(benches);
