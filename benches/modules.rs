//! Criterion bench suite for the post-core modules added on top of
//! the bare forest: hot-path ingress primitives, TRCF shingled
//! wrapper, streaming quantiles (t-digest + histogram), feature /
//! meta drift detectors, ADWIN, LSH clustering, SPOT/DSPOT, Platt
//! calibration, Fisher p-value combination, dynamic-dim forest,
//! drift-aware shadow swap, and SAGE Shapley explanations.
//!
//! Each group targets one module's hot-path entry point so any
//! perf regression from a refactor surfaces as a single flipped
//! number rather than a disappearing line of code.
//!
//! Run with `cargo bench --bench modules` or
//! `cargo bench --bench modules -- hot_path/` for one group.

#![allow(clippy::cast_precision_loss)]

use criterion::{Criterion, criterion_group, criterion_main};
use mimalloc::MiMalloc;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rcf_rs::{
    AdwinDetector, CusumConfig, DiVector, DriftAwareForest, DriftRecoveryConfig, DynamicForest,
    FeatureDriftDetector, ForestBuilder, LshAlertClusterer, MetaDriftDetector, OnlineStats,
    PlattCalibrator, PlattFitConfig, PotDetector, SageEstimator, ScoreHistogram,
    ShingledForestBuilder, TDigest, ensemble::fisher_combine, hot_path,
};
use std::hint::black_box;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// `hot_path::UpdateSampler` `accept_stride` + `accept_hash` +
/// keyed `accept_hash` (murmur-mix secret). Target: per-packet
/// overhead on the classifier hot path.
fn bench_hot_path_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path_sampler");

    group.bench_function("accept_stride_keep_8", |b| {
        let s = hot_path::UpdateSampler::new(8);
        b.iter(|| {
            let v = s.accept_stride();
            black_box(v);
        });
    });

    group.bench_function("accept_hash_keep_8", |b| {
        let s = hot_path::UpdateSampler::new(8);
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            let h: u64 = rng.random();
            let v = s.accept_hash(black_box(h));
            black_box(v);
        });
    });

    group.bench_function("accept_hash_keyed_keep_8", |b| {
        let s = hot_path::UpdateSampler::new_keyed(8).expect("getrandom");
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            let h: u64 = rng.random();
            let v = s.accept_hash(black_box(h));
            black_box(v);
        });
    });

    group.finish();
}

/// `hot_path::PrefixRateCap::check_and_record` — 256-bucket
/// atomic counter sketch + window roll. Lock-free, O(1).
fn bench_hot_path_prefix_cap(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path_prefix_cap");

    group.bench_function("check_and_record_100cap_1s", |b| {
        let cap = hot_path::PrefixRateCap::new(100, 1_000);
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

/// `hot_path::channel` — bounded MPSC `try_enqueue` throughput
/// with a background drain thread keeping the queue non-full.
fn bench_hot_path_channel(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path_channel");

    group.bench_function("try_enqueue_4096cap", |b| {
        let (producer, consumer) = hot_path::channel::<16>(4096);
        // Background drain thread — keeps the queue from saturating.
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

/// `ShingledForest::update_scalar` + `score_scalar` — ring-buffer
/// push then forest score on the embedded shingle.
fn bench_shingled(c: &mut Criterion) {
    let mut group = c.benchmark_group("shingled");

    group.bench_function("update_then_score_dim16", |b| {
        let mut sf = ShingledForestBuilder::<16>::new()
            .num_trees(100)
            .sample_size(256)
            .seed(2026)
            .build()
            .expect("shingled build");
        // Warm the shingle + reservoir.
        let mut rng_warm = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..1024 {
            let _ = sf.update_scalar(rng_warm.random::<f64>());
        }
        let mut rng_live = ChaCha8Rng::seed_from_u64(7);
        b.iter(|| {
            let v: f64 = rng_live.random();
            let _ = sf.update_scalar(black_box(v));
            let s = sf.score_scalar(black_box(v)).expect("score_scalar");
            black_box(s);
        });
    });

    group.finish();
}

/// `TDigest::record` + `quantile` — streaming quantile primitive.
fn bench_tdigest(c: &mut Criterion) {
    let mut group = c.benchmark_group("tdigest");

    group.bench_function("record", |b| {
        let mut d = TDigest::with_default_compression();
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            let v: f64 = rng.random();
            d.record(black_box(v));
        });
    });

    group.bench_function("quantile_p99_after_100k", |b| {
        let mut d = TDigest::with_default_compression();
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..100_000 {
            d.record(rng.random::<f64>());
        }
        b.iter(|| {
            let mut scratch = d.clone();
            let q = scratch.quantile(black_box(0.99));
            black_box(q);
        });
    });

    group.finish();
}

/// `ScoreHistogram::record` — fixed-bin streaming sketch. Cheap
/// per-record, reads are trivial.
fn bench_histogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("score_histogram");

    group.bench_function("record_default_range", |b| {
        let mut h = ScoreHistogram::with_range(0.0, 5.0).expect("hist build");
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            let v: f64 = rng.random();
            h.record(black_box(v * 5.0));
        });
    });

    group.finish();
}

/// `FeatureDriftDetector::observe` + `psi` — per-dim PSI scan.
fn bench_feature_drift(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_drift");

    group.bench_function("observe_d16_10bin", |b| {
        let mut det: FeatureDriftDetector<16> =
            FeatureDriftDetector::<16>::new(10).expect("fdd build");
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        // Warm + freeze so observe is the post-freeze histogram
        // update path.
        for _ in 0..512 {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng.random();
            }
            let _ = det.observe(&p);
        }
        det.freeze_baseline().expect("freeze");
        b.iter(|| {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng.random();
            }
            let _ = det.observe(black_box(&p));
        });
    });

    group.bench_function("psi_d16_10bin", |b| {
        let mut det: FeatureDriftDetector<16> =
            FeatureDriftDetector::<16>::new(10).expect("fdd build");
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..2048 {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng.random();
            }
            let _ = det.observe(&p);
        }
        det.freeze_baseline().expect("freeze");
        // Accumulate a production window so psi does real work.
        let mut rng_prod = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..2048 {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng_prod.random();
            }
            let _ = det.observe(&p);
        }
        b.iter(|| {
            let v = det.psi().expect("psi");
            black_box(v);
        });
    });

    group.finish();
}

/// `MetaDriftDetector::observe` — two-sided CUSUM update.
fn bench_meta_drift(c: &mut Criterion) {
    let mut group = c.benchmark_group("meta_drift");

    group.bench_function("observe_cusum", |b| {
        let mut det = MetaDriftDetector::new(CusumConfig::default()).expect("cusum");
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            let v: f64 = rng.random();
            let d = det.observe(black_box(v));
            black_box(d);
        });
    });

    group.finish();
}

/// `AdwinDetector::update` — bounded ring + Hoeffding split scan
/// (O(N) per update).
fn bench_adwin(c: &mut Criterion) {
    let mut group = c.benchmark_group("adwin");

    group.bench_function("update_cap_4096", |b| {
        let mut det = AdwinDetector::default_bounded();
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        // Warm the ring so every update pays the full scan cost.
        for _ in 0..4096 {
            let _ = det.update(rng.random());
        }
        b.iter(|| {
            let v: f64 = rng.random();
            let d = det.update(black_box(v));
            black_box(d);
        });
    });

    group.finish();
}

/// `LshAlertClusterer::hash_divector` + `observe` — quantise the
/// `DiVector` into per-dim hex symbols, look up the bucket.
fn bench_lsh_cluster(c: &mut Criterion) {
    use rcf_rs::{AlertRecord, AnomalyScore, ForensicBaseline};
    let mut group = c.benchmark_group("lsh_cluster");

    // Build a record template we can reuse — DiVector of dim 16
    // with a single strong dim.
    let mut di = DiVector::zeros(16);
    di.add_high(3, 2.5).expect("add_high");
    di.add_low(7, 1.5).expect("add_low");
    let rec: AlertRecord<u32, 16> = AlertRecord {
        version: rcf_rs::ALERT_RECORD_VERSION,
        tenant: Some(1_u32),
        timestamp_ms: 0,
        point: [0.0; 16],
        score: AnomalyScore::new(1.0).expect("score"),
        grade: None,
        severity: None,
        attribution: di.clone(),
        baseline: ForensicBaseline::<16> {
            observed: [0.0; 16],
            expected: [0.0; 16],
            stddev: [0.0; 16],
            delta: [0.0; 16],
            zscore: [0.0; 16],
            live_points: 0,
        },
    };

    group.bench_function("hash_divector_d16", |b| {
        let clusterer = LshAlertClusterer::default_lsh();
        b.iter(|| {
            let h = clusterer.hash_divector(black_box(&di));
            black_box(h);
        });
    });

    group.bench_function("observe_d16", |b| {
        let mut clusterer = LshAlertClusterer::default_lsh();
        b.iter(|| {
            let (h, d) = clusterer.observe(black_box(&rec));
            black_box((h, d));
        });
    });

    group.finish();
}

/// `PotDetector::record` + `p_value` — SPOT/DSPOT streaming.
fn bench_univariate_spot(c: &mut Criterion) {
    let mut group = c.benchmark_group("univariate_spot");

    group.bench_function("record_post_freeze", |b| {
        let mut det = PotDetector::default_spot();
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..2048 {
            det.record(rng.random::<f64>());
        }
        det.freeze_baseline().expect("freeze");
        b.iter(|| {
            let v: f64 = rng.random();
            det.record(black_box(v));
        });
    });

    group.bench_function("p_value_post_freeze", |b| {
        let mut det = PotDetector::default_spot();
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..4096 {
            det.record(rng.random::<f64>());
        }
        det.freeze_baseline().expect("freeze");
        // Feed peaks so the GPD fit is non-degenerate.
        for _ in 0..256 {
            det.record(0.98 + rng.random::<f64>() * 0.05);
        }
        b.iter(|| {
            let v: f64 = 0.99 + rng.random::<f64>() * 0.05;
            let p = det.p_value(black_box(v));
            black_box(p);
        });
    });

    group.finish();
}

/// `PlattCalibrator::fit` (Newton-Raphson) + `calibrate` (σ).
fn bench_calibrator(c: &mut Criterion) {
    let mut group = c.benchmark_group("calibrator");

    group.bench_function("fit_2048_samples", |b| {
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        let data: Vec<(f64, bool)> = (0..2048)
            .map(|_| {
                let s: f64 = rng.random::<f64>() * 5.0;
                (s, s > 3.0)
            })
            .collect();
        b.iter(|| {
            let cal =
                PlattCalibrator::fit(black_box(&data), PlattFitConfig::default()).expect("fit");
            black_box(cal);
        });
    });

    group.bench_function("calibrate_single", |b| {
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        let data: Vec<(f64, bool)> = (0..1024)
            .map(|_| {
                let s: f64 = rng.random::<f64>() * 5.0;
                (s, s > 3.0)
            })
            .collect();
        let cal = PlattCalibrator::fit(&data, PlattFitConfig::default()).expect("fit");
        let mut rng_live = ChaCha8Rng::seed_from_u64(7);
        b.iter(|| {
            let s: f64 = rng_live.random::<f64>() * 5.0;
            let p = cal.calibrate(black_box(s));
            black_box(p);
        });
    });

    group.finish();
}

/// `ensemble::fisher_combine` — Kahan-compensated χ² sum.
fn bench_fisher(c: &mut Criterion) {
    let mut group = c.benchmark_group("fisher_combine");

    for &k in &[8_usize, 32, 128] {
        group.bench_function(format!("k{k}"), |b| {
            let mut rng = ChaCha8Rng::seed_from_u64(2026);
            let p_values: Vec<f64> = (0..k).map(|_| rng.random::<f64>().max(1.0e-12)).collect();
            b.iter(|| {
                let p = fisher_combine(black_box(&p_values));
                black_box(p);
            });
        });
    }

    group.finish();
}

/// `DynamicForest::update` — runtime-dim wrapper that zero-pads
/// into a `MAX_D` const-generic forest. Overhead vs the native
/// const-generic path is the headline number.
fn bench_dynamic_forest(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_forest");

    group.bench_function("update_active8_max16", |b| {
        let builder = ForestBuilder::<16>::new()
            .num_trees(100)
            .sample_size(256)
            .seed(2026);
        let mut dyn_forest: DynamicForest<16> = DynamicForest::new(builder, 8).expect("dyn build");
        let mut rng_warm = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..1024 {
            let mut p = [0.0_f64; 8];
            for slot in &mut p {
                *slot = rng_warm.random();
            }
            dyn_forest.update(&p).expect("warm update");
        }
        let mut rng_live = ChaCha8Rng::seed_from_u64(7);
        b.iter(|| {
            let mut p = [0.0_f64; 8];
            for slot in &mut p {
                *slot = rng_live.random();
            }
            dyn_forest.update(black_box(&p)).expect("update");
        });
    });

    group.finish();
}

/// `DriftAwareForest::update` — no shadow (baseline) vs. with
/// active shadow (measures the shadow overhead).
fn bench_drift_aware(c: &mut Criterion) {
    let mut group = c.benchmark_group("drift_aware");

    group.bench_function("update_no_shadow", |b| {
        let builder = ForestBuilder::<16>::new()
            .num_trees(100)
            .sample_size(256)
            .seed(2026);
        let mut d: DriftAwareForest<16> =
            DriftAwareForest::new(builder, DriftRecoveryConfig::default()).expect("build");
        let mut rng_warm = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..1024 {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng_warm.random();
            }
            d.update(p).expect("warm");
        }
        let mut rng_live = ChaCha8Rng::seed_from_u64(7);
        b.iter(|| {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng_live.random();
            }
            d.update(black_box(p)).expect("update");
        });
    });

    group.bench_function("update_with_shadow", |b| {
        let builder = ForestBuilder::<16>::new()
            .num_trees(100)
            .sample_size(256)
            .seed(2026);
        let mut d: DriftAwareForest<16> = DriftAwareForest::new(
            builder,
            DriftRecoveryConfig {
                shadow_warmup: 1_000_000,
                min_primary_age: 100,
            },
        )
        .expect("build");
        let mut rng_warm = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..1024 {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng_warm.random();
            }
            d.update(p).expect("warm");
        }
        d.on_drift().expect("spawn shadow");
        let mut rng_live = ChaCha8Rng::seed_from_u64(7);
        b.iter(|| {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng_live.random();
            }
            d.update(black_box(p)).expect("update");
        });
    });

    group.finish();
}

/// `SageEstimator::explain` — permutation Shapley with default
/// 64 permutations. Expensive: `K · D` forest scores per call.
fn bench_sage(c: &mut Criterion) {
    let mut group = c.benchmark_group("sage");
    group.sample_size(20);

    group.bench_function("explain_d16_k64", |b| {
        let mut forest = ForestBuilder::<16>::new()
            .num_trees(50)
            .sample_size(128)
            .seed(2026)
            .build()
            .expect("forest");
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..512 {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng.random();
            }
            forest.update(p).expect("update");
        }
        let sage: SageEstimator<16> = SageEstimator::default_anchor([0.0_f64; 16]).expect("sage");
        let probe: [f64; 16] = core::array::from_fn(|_| rng.random::<f64>());
        b.iter(|| {
            let e = sage.explain(&forest, black_box(&probe)).expect("explain");
            black_box(e);
        });
    });

    group.finish();
}

/// `OnlineStats` — Welford streaming mean + variance. Target:
/// per-sample `update()` cost and `variance()` read cost on a
/// warmed accumulator.
fn bench_online_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("online_stats");

    group.bench_function("update_cold", |b| {
        b.iter(|| {
            let mut s = OnlineStats::default();
            for i in 0..32_u64 {
                s.update(black_box(f64::from(u32::try_from(i).unwrap_or(0))));
            }
            black_box(s);
        });
    });

    group.bench_function("update_hot", |b| {
        let mut s = OnlineStats::default();
        for i in 0..1_000_u64 {
            s.update(f64::from(u32::try_from(i).unwrap_or(0)));
        }
        let mut v = 0.5_f64;
        b.iter(|| {
            v = v.mul_add(1.000_000_1, 1.0);
            s.update(black_box(v));
        });
    });

    group.bench_function("variance_read", |b| {
        let mut s = OnlineStats::default();
        for i in 0..1_000_u64 {
            s.update(f64::from(u32::try_from(i).unwrap_or(0)));
        }
        b.iter(|| {
            let v = s.variance();
            black_box(v);
        });
    });

    group.bench_function("std_dev_read", |b| {
        let mut s = OnlineStats::default();
        for i in 0..1_000_u64 {
            s.update(f64::from(u32::try_from(i).unwrap_or(0)));
        }
        b.iter(|| {
            let v = s.std_dev();
            black_box(v);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hot_path_sampler,
    bench_hot_path_prefix_cap,
    bench_hot_path_channel,
    bench_shingled,
    bench_tdigest,
    bench_histogram,
    bench_feature_drift,
    bench_meta_drift,
    bench_adwin,
    bench_lsh_cluster,
    bench_univariate_spot,
    bench_calibrator,
    bench_fisher,
    bench_dynamic_forest,
    bench_drift_aware,
    bench_sage,
    bench_online_stats
);
criterion_main!(benches);
