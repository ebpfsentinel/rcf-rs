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
//! `cargo bench --bench modules -- shingled/` for one group.

#![allow(clippy::cast_precision_loss)]

use anomstream_core::{
    AdwinDetector, CountMinSketch, CusumConfig, DriftAwareForest, DriftRecoveryConfig,
    DynamicForest, FeatureDriftDetector, FeatureGroups, ForestBuilder, HyperLogLog,
    MetaDriftDetector, NormStrategy, Normalizer, OnlineStats, PerFeatureCusum,
    PerFeatureCusumConfig, PerFeatureEwma, PerFeatureEwmaConfig, PotDetector, RandomCutForest,
    ScoreHistogram, ShingledForestBuilder, SpaceSaving, TDigest, ensemble::fisher_combine,
};
use criterion::{Criterion, criterion_group, criterion_main};
use mimalloc::MiMalloc;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

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

/// `PerFeatureCusum<D>` — parallel two-sided CUSUM change-point
/// detector. Target: per-observation cost at `D=16` in three
/// regimes — below threshold (common), trip path (rare), and
/// stable after reference seeded.
fn bench_per_feature_cusum(c: &mut Criterion) {
    let mut group = c.benchmark_group("per_feature_cusum");
    let cfg = PerFeatureCusumConfig {
        slack: 0.5,
        threshold: 5.0,
    };

    group.bench_function("observe_below_threshold_d16", |b| {
        let mut det: PerFeatureCusum<16> = PerFeatureCusum::new(cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        det.observe(&[10.0_f64; 16]); // seed refs
        b.iter(|| {
            let mut p = [10.0_f64; 16];
            for slot in &mut p {
                *slot += rng.random::<f64>() * 0.1; // tiny wiggle, below slack
            }
            let r = det.observe(black_box(&p));
            black_box(r);
        });
    });

    group.bench_function("observe_alert_trip_d16", |b| {
        let mut det: PerFeatureCusum<16> = PerFeatureCusum::new(cfg);
        det.observe(&[10.0_f64; 16]);
        // Warm the charts so every tick trips an alert.
        for _ in 0..20 {
            det.observe(&[15.0_f64; 16]);
        }
        b.iter(|| {
            let r = det.observe(black_box(&[15.0_f64; 16]));
            black_box(r);
        });
    });

    group.bench_function("observe_stable_d16", |b| {
        let mut det: PerFeatureCusum<16> = PerFeatureCusum::new(cfg);
        det.observe(&[10.0_f64; 16]);
        b.iter(|| {
            let r = det.observe(black_box(&[10.0_f64; 16]));
            black_box(r);
        });
    });

    group.finish();
}

/// `PerFeatureEwma<D>` — parallel univariate EWMA z-score. Target:
/// per-observation cost at `D=16` in two regimes — hot (warmed,
/// returning z-scores) and cold (warming, no z-score math).
fn bench_per_feature_ewma(c: &mut Criterion) {
    let mut group = c.benchmark_group("per_feature_ewma");
    let cfg = PerFeatureEwmaConfig {
        alpha: 0.1,
        warmup_samples: 32,
    };

    group.bench_function("observe_warmed_d16", |b| {
        let mut ewma: PerFeatureEwma<16> = PerFeatureEwma::new(cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..256 {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng.random::<f64>() * 10.0;
            }
            ewma.observe(&p);
        }
        b.iter(|| {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng.random::<f64>() * 10.0;
            }
            let r = ewma.observe(black_box(&p));
            black_box(r);
        });
    });

    group.bench_function("observe_cold_d16", |b| {
        b.iter(|| {
            let mut ewma: PerFeatureEwma<16> = PerFeatureEwma::new(cfg);
            let mut rng = ChaCha8Rng::seed_from_u64(7);
            for _ in 0..16 {
                let mut p = [0.0_f64; 16];
                for slot in &mut p {
                    *slot = rng.random::<f64>() * 10.0;
                }
                ewma.observe(black_box(&p));
            }
            black_box(&ewma);
        });
    });

    group.bench_function("observe_spike_d16", |b| {
        let mut ewma: PerFeatureEwma<16> = PerFeatureEwma::new(cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..256 {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng.random::<f64>() * 10.0;
            }
            ewma.observe(&p);
        }
        let mut spike = [10.0_f64; 16];
        spike[3] = 1_000.0;
        b.iter(|| {
            let r = ewma.observe(black_box(&spike));
            black_box(r);
        });
    });

    group.finish();
}

/// `Normalizer<D>` — per-feature min-max / z-score transform.
/// Target: per-point transform cost at `D=16` (AWS-typical
/// dim), plus the `fit` 2-pass cost on a 1024-sample batch.
fn bench_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize");

    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let samples: Vec<[f64; 16]> = (0..1024)
        .map(|_| {
            let mut p = [0.0_f64; 16];
            for slot in &mut p {
                *slot = rng.random::<f64>() * 100.0;
            }
            p
        })
        .collect();
    let n_minmax = Normalizer::<16>::fit(NormStrategy::MinMax, &samples);
    let n_zscore = Normalizer::<16>::fit(NormStrategy::ZScore, &samples);
    let n_none = Normalizer::<16>::identity(NormStrategy::None);
    let probe: [f64; 16] = core::array::from_fn(|_| rng.random::<f64>() * 100.0);

    group.bench_function("transform_minmax_d16", |b| {
        b.iter(|| {
            let out = n_minmax.transform(black_box(&probe));
            black_box(out);
        });
    });
    group.bench_function("transform_zscore_d16", |b| {
        b.iter(|| {
            let out = n_zscore.transform(black_box(&probe));
            black_box(out);
        });
    });
    group.bench_function("transform_none_d16", |b| {
        b.iter(|| {
            let out = n_none.transform(black_box(&probe));
            black_box(out);
        });
    });
    group.bench_function("fit_minmax_1024x16", |b| {
        b.iter(|| {
            let n = Normalizer::<16>::fit(NormStrategy::MinMax, black_box(&samples));
            black_box(n);
        });
    });

    group.finish();
}

/// `CountMinSketch` — probabilistic frequency sketch. Target:
/// per-call `increment` + `estimate` cost at AWS-typical size
/// (`w=2048`, `d=4`), plus a saturation check so the hash-free
/// accounting path stays in profile.
fn bench_count_min_sketch(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_min_sketch");

    group.bench_function("increment_2048x4", |b| {
        let mut cms = CountMinSketch::new(2048, 4);
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            let key: u64 = rng.random();
            cms.increment(black_box(&key.to_le_bytes()), 1);
        });
    });

    group.bench_function("estimate_2048x4", |b| {
        let mut cms = CountMinSketch::new(2048, 4);
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..10_000_u64 {
            let key: u64 = rng.random();
            cms.increment(&key.to_le_bytes(), 1);
        }
        let probe: u64 = rng.random();
        let probe_bytes = probe.to_le_bytes();
        b.iter(|| {
            let v = cms.estimate(black_box(&probe_bytes));
            black_box(v);
        });
    });

    group.bench_function("reset_2048x4", |b| {
        let mut cms = CountMinSketch::new(2048, 4);
        for i in 0..1_000_u64 {
            cms.increment(&i.to_le_bytes(), 1);
        }
        b.iter(|| {
            cms.reset();
            black_box(&cms);
        });
    });

    group.finish();
}

/// `FeatureGroups::group_scores` — per-group sum reduction over
/// the per-dim `DiVector` with coverage calc. `D=16` split into
/// three named groups mirroring the enterprise ML detection layer
/// (rate / payload / cardinality).
fn bench_group_scores(c: &mut Criterion) {
    let mut group = c.benchmark_group("group_scores");
    let mut forest: RandomCutForest<16> = ForestBuilder::<16>::new()
        .num_trees(100)
        .sample_size(256)
        .seed(2026)
        .build()
        .expect("forest");
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    for _ in 0..1024 {
        let mut p = [0.0_f64; 16];
        for slot in &mut p {
            *slot = rng.random::<f64>();
        }
        forest.update(p).expect("update");
    }
    let groups = FeatureGroups::builder()
        .add("rate", [0_usize, 1, 2])
        .add("payload", [3_usize, 4, 5, 6, 7])
        .add("cardinality", [8_usize, 9, 10, 11, 12, 13, 14, 15])
        .build()
        .expect("groups");
    let probe: [f64; 16] = core::array::from_fn(|_| rng.random::<f64>());

    group.bench_function("from_forest_d16_3groups", |b| {
        b.iter(|| {
            let gs = forest.group_scores(black_box(&probe), &groups).expect("gs");
            black_box(gs);
        });
    });

    group.finish();
}

/// `AttributionStability::from_forest` — inter-tree dispersion
/// scan. `O(num_trees · D)` per probe. Target: cost relative to
/// plain `attribution()`.
fn bench_attribution_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("attribution_stability");
    let mut forest: RandomCutForest<16> = ForestBuilder::<16>::new()
        .num_trees(100)
        .sample_size(256)
        .seed(2026)
        .build()
        .expect("forest");
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    for _ in 0..1024 {
        let mut p = [0.0_f64; 16];
        for slot in &mut p {
            *slot = rng.random::<f64>();
        }
        forest.update(p).expect("update");
    }
    let probe: [f64; 16] = core::array::from_fn(|_| rng.random::<f64>());

    group.bench_function("from_forest_d16", |b| {
        b.iter(|| {
            let s = forest
                .attribution_stability(black_box(&probe))
                .expect("stab");
            black_box(s);
        });
    });

    group.finish();
}

/// `RandomCutForest::score_with_confidence` — mean + stderr over
/// per-tree score dispersion. Always walks every tree (no early
/// termination); dispersion is the whole point.
fn bench_score_with_confidence(c: &mut Criterion) {
    let mut group = c.benchmark_group("score_ci");
    let mut forest: RandomCutForest<16> = ForestBuilder::<16>::new()
        .num_trees(100)
        .sample_size(256)
        .seed(2026)
        .build()
        .expect("forest");
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    for _ in 0..1024 {
        let mut p = [0.0_f64; 16];
        for slot in &mut p {
            *slot = rng.random::<f64>();
        }
        forest.update(p).expect("update");
    }
    let probe: [f64; 16] = core::array::from_fn(|_| rng.random::<f64>());

    group.bench_function("single_probe_d16", |b| {
        b.iter(|| {
            let s = forest
                .score_with_confidence(black_box(&probe))
                .expect("score_ci");
            black_box(s);
        });
    });

    group.finish();
}

/// `RandomCutForest::bootstrap` — historical replay throughput.
/// Measures the ingest rate when the forest is warm-filled from a
/// pre-made batch (typical cold-start restart path).
fn bench_bootstrap(c: &mut Criterion) {
    let mut group = c.benchmark_group("bootstrap");

    group.bench_function("replay_4096_d16", |b| {
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        let points: Vec<[f64; 16]> = (0..4096)
            .map(|_| {
                let mut p = [0.0_f64; 16];
                for slot in &mut p {
                    *slot = rng.random::<f64>();
                }
                p
            })
            .collect();
        b.iter(|| {
            let mut forest: RandomCutForest<16> = ForestBuilder::<16>::new()
                .num_trees(50)
                .sample_size(128)
                .seed(2026)
                .build()
                .expect("forest");
            let r = forest.bootstrap(points.iter().copied()).expect("bootstrap");
            black_box(r);
        });
    });

    group.finish();
}

/// Persistence — `to_bytes` / `from_bytes` roundtrip throughput
/// via postcard. Measures serialization cost on a warm forest
/// (AWS default config) independent of I/O.
#[cfg(all(feature = "postcard", feature = "serde"))]
fn bench_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistence");
    let mut forest: RandomCutForest<16> = ForestBuilder::<16>::new()
        .num_trees(100)
        .sample_size(256)
        .seed(2026)
        .build()
        .expect("forest");
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    for _ in 0..1024 {
        let mut p = [0.0_f64; 16];
        for slot in &mut p {
            *slot = rng.random::<f64>();
        }
        forest.update(p).expect("update");
    }

    group.bench_function("to_bytes_100t_256s_d16", |b| {
        b.iter(|| {
            let bytes = forest.to_bytes().expect("to_bytes");
            black_box(bytes);
        });
    });

    group.bench_function("from_bytes_100t_256s_d16", |b| {
        let bytes = forest.to_bytes().expect("to_bytes");
        b.iter(|| {
            let f: RandomCutForest<16> =
                RandomCutForest::from_bytes(black_box(&bytes)).expect("from_bytes");
            black_box(f);
        });
    });

    group.finish();
}

#[cfg(not(all(feature = "postcard", feature = "serde")))]
fn bench_persistence(_: &mut Criterion) {}

/// `HyperLogLog` — cardinality sketch. Target: per-sample `add`
/// cost at `p=12` (4 096 registers, ~1.6 % std error), plus
/// `estimate` read cost after warming and a shard-merge pass.
fn bench_hyperloglog(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperloglog");

    group.bench_function("add_bytes_p12", |b| {
        let mut hll = HyperLogLog::with_default_precision();
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            let v: u64 = rng.random();
            hll.add_bytes(black_box(&v.to_le_bytes()));
        });
    });

    group.bench_function("add_hash_p12", |b| {
        let mut hll = HyperLogLog::with_default_precision();
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            let h: u64 = rng.random();
            hll.add_hash(black_box(h));
        });
    });

    group.bench_function("estimate_after_100k_p12", |b| {
        let mut hll = HyperLogLog::with_default_precision();
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..100_000 {
            let v: u64 = rng.random();
            hll.add_bytes(&v.to_le_bytes());
        }
        b.iter(|| {
            let n = hll.estimate();
            black_box(n);
        });
    });

    group.bench_function("merge_p12", |b| {
        let mut a = HyperLogLog::with_default_precision();
        let mut b_hll = HyperLogLog::with_default_precision();
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..50_000 {
            a.add_bytes(&rng.random::<u64>().to_le_bytes());
            b_hll.add_bytes(&rng.random::<u64>().to_le_bytes());
        }
        b.iter(|| {
            let mut scratch = a.clone();
            scratch.merge(black_box(&b_hll)).expect("merge");
            black_box(scratch);
        });
    });

    group.finish();
}

/// `SpaceSaving` — deterministic top-K heavy hitters. Covers the
/// hot (tracked-key) path, the cold/evict path (linear `O(K)`
/// scan), and the `top_k` ranking step after saturating the table
/// with distinct noise keys.
fn bench_space_saving(c: &mut Criterion) {
    let mut group = c.benchmark_group("space_saving");

    group.bench_function("observe_hot_k128", |b| {
        let mut ss: SpaceSaving<u32> = SpaceSaving::with_default_capacity().expect("ss build");
        // Warm the table with 128 tracked keys.
        for k in 0..128_u32 {
            ss.observe(k);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            // Keys in [0, 128) already tracked → hot path, no evict.
            let k: u32 = rng.random_range(0..128);
            ss.observe(black_box(k));
        });
    });

    group.bench_function("observe_cold_k128", |b| {
        let mut ss: SpaceSaving<u64> = SpaceSaving::with_default_capacity().expect("ss build");
        // Saturate with 128 distinct keys — every subsequent
        // observation hits the evict path.
        for k in 0..128_u64 {
            ss.observe(k);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        b.iter(|| {
            // Fresh keys → never tracked → linear min-scan + swap.
            let k: u64 = rng.random();
            ss.observe(black_box(k));
        });
    });

    group.bench_function("top_k_from_1024_distinct", |b| {
        let mut ss: SpaceSaving<u32> = SpaceSaving::new(1024).expect("ss build");
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        for _ in 0..100_000 {
            let k: u32 = rng.random();
            ss.observe(k);
        }
        b.iter(|| {
            let top = ss.top_k(black_box(10));
            black_box(top);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_shingled,
    bench_tdigest,
    bench_histogram,
    bench_feature_drift,
    bench_meta_drift,
    bench_adwin,
    bench_univariate_spot,
    bench_fisher,
    bench_dynamic_forest,
    bench_drift_aware,
    bench_online_stats,
    bench_count_min_sketch,
    bench_normalize,
    bench_per_feature_ewma,
    bench_per_feature_cusum,
    bench_group_scores,
    bench_attribution_stability,
    bench_score_with_confidence,
    bench_bootstrap,
    bench_persistence,
    bench_hyperloglog,
    bench_space_saving
);
criterion_main!(benches);
