# Performance

Criterion (`cargo bench`) — wall-clock mean point estimate,
`mimalloc` pinned globally. Two bench files:

- `benches/forest_throughput.rs` — core ops (insert, score,
  attribution).
- `benches/extended.rs` — bulk, early-term, forensic, tenant.

```bash
cargo bench                                            # full
cargo bench -- --sample-size 10 --measurement-time 2   # quick
```

## Reference hardware

|           |                                                    |
| --------- | -------------------------------------------------- |
| CPU       | Intel Core i7-1370P (13th gen), 14C/20T, L3 24 MiB |
| Memory    | 32 GB DDR5                                         |
| Kernel    | Linux 6.17                                         |
| Allocator | mimalloc 0.1 (pinned in bench harness)             |
| Compiler  | rustc 1.95 stable                                  |

Absolute values scale with CPU / memory bandwidth; _ratios_
(parallel speedup, early-term savings, tenant scaling) are the
portable signal.

## Caveats

- **Cross-group variance**: `b.iter()` mutates a persistent
  forest and criterion chooses batch sizes per-op, so reservoir
  state + per-iter overhead drift across bench groups. Trust
  ratios within a group, not absolute numbers across groups.
- **Parallel ceiling**: `score_many` plateaus at ~6× speedup on
  a 14-core host — memory-bandwidth-bound past L3 working set.

## Core ops

`(trees=100, sample=256, D=16)` on a cool-CPU single-seed run
(powersave governor, no competing load):

| Workload                              | Time   | Throughput |
| ------------------------------------- | ------ | ---------- |
| `forest_update`                       | ~25 µs | ~40 k/s    |
| `forest_score`                        | ~24 µs | ~42 k/s    |
| `forest_attribution`                  | ~45 µs | ~22 k/s    |
| `forest_score_and_attribution`        | ~46 µs | ~22 k/s    |
| `forest_split_score_then_attribution` | ~76 µs | ~13 k/s    |

The fused `score_and_attribution` walk is **~40 % faster** than
calling `score` + `attribution` back-to-back (single traversal
instead of two). The fused bbox SIMD kernel
(`total_probability_of_cut`) saves one pass over `min`/`max` loads
per internal node. Post split-typed-arena refactor (persistence
v4) leaf arena memory is −90 % (~320 B → ~40 B per slot).

Other `(trees, samples, D)` tuples below:

| Config           | `forest_update` | `forest_score` | `forest_attribution` |
| ---------------- | --------------- | -------------- | -------------------- |
| `(50, 128, 16)`  | ~20 µs          | ~18 µs         | —                    |
| `(100, 256, 4)`  | ~20 µs          | ~21 µs         | ~36 µs               |
| `(100, 256, 16)` | ~25 µs          | ~24 µs         | ~45 µs               |
| `(100, 256, 64)` | ~66 µs          | ~28 µs         | ~91 µs               |
| `(200, 512, 16)` | ~34 µs          | ~51 µs         | —                    |

Criterion HTML report lives at `target/criterion/`.

## Bulk batch scoring

`D=16`, forest `(100, 256)`:

| Batch size | `score_many` (parallel) | Serial loop | Speedup |
| ---------- | ----------------------- | ----------- | ------- |
| 64         | 511 µs                  | 2.16 ms     | 4.2×    |
| 512        | 3.74 ms                 | 16.9 ms     | 4.5×    |
| 4096       | 27.8 ms                 | 135 ms      | 4.9×    |

### Codisp batched scoring

Probe-based codisp batched API (`score_codisp_many`) pre-inserts
probes into every tree, shared-walk amortises per-tree leaf →
root descent, rayon across trees:

| Batch K | `score_codisp_many` | `score_codisp` loop | Speedup |
| ------- | ------------------- | ------------------- | ------- |
| 16      | 1.78 ms             | 2.36 ms             | 1.3×    |
| 64      | 6.65 ms             | 9.48 ms             | 1.4×    |

Gain caps at ~1.5× because insert/delete mutation phases still
scale with `K × num_trees`; only the walk phase benefits from
shared-walk. For **frozen-baseline batched codisp** on any batch
size, prefer `score_codisp_stateless_many` (single-probe stateless
codisp mapped in parallel — no reservoir mutation, no `O(K)`
saturation limit).

### Memory-bandwidth plateau @ ~6× / 14 C

`score_many` tops out at ~6× rayon speedup on the 14-core / 20-thread
reference host — once the tree arena for 100 trees × 256 samples (~6 MB
at `D = 16`) fits in L3 but the per-probe working set thrashes L1/L2,
all workers compete for the same LLC → DRAM channel and further
threads contend rather than scale. Two avenues have been explored:

- **Cache-aware probe reordering** via [`locality_bucket`] +
  [`score_many_locality_sorted`] — shipped, opt-in, bench-driven.
  Sorting probes by leading-dim quantised key groups similar tree
  descents so each rayon worker re-uses warm arena cache lines.
  At `k = 1024`, `D = 16`, correlated cluster: plain `score_many`
  5.10 ms, sorted variant 5.69 ms — the `O(N log N)` sort + double
  gather outweighs the cache gain on uniformly-random batches.
  Callers with strongly-correlated batches (SOC alert replay of a
  single flow, periodic tenant-scan) can bench their own workload
  and swap in; do not swap blindly.
- **Packed Cut (`dim: u8` + `value: f32`, 8 B vs 16 B)** — **not
  shipped**. Halves the per-internal-node cut footprint, which
  would improve L1 fit at `D = 16`, but `value: f32` changes the
  isolation-depth boundary on points near a cut — bit-level
  divergence from the f64-cut scorer. Correctness audit + full
  NAB/TSB-AD AUC regression required before wiring. Documented
  here so the decision is explicit, not lost to future archaeology.

[`locality_bucket`]: ../src/forest/random_cut_forest.rs
[`score_many_locality_sorted`]: ../src/forest/random_cut_forest.rs

## Early-termination

`D=16`, forest `(100, 256)`, single probe:

| Path                                                       | Time   |
| ---------------------------------------------------------- | ------ |
| `score` (parallel ensemble)                                | 32 µs  |
| `score_early_term` threshold=0.02 (tight)                  | 32 µs  |
| `score_early_term` threshold=0.20 (loose, stops ~20 trees) | 4.7 µs |

Loose threshold → 6.9× speedup on baseline-dominated traffic;
tight threshold matches parallel `score` (sequential walk
rarely short-circuits).

## Forensic baseline

`forensic_baseline`:

| `(trees, samples, D)` | Time  |
| --------------------- | ----- |
| `(100, 256, 4)`       | 13 µs |
| `(100, 256, 16)`      | 16 µs |
| `(100, 1024, 16)`     | 64 µs |

Cost ≈ `O(live_points × D)` Welford sweep — `sample_size` ×4
→ ×4 time (close to linear on the current run), dim cost
grows ~1.3× over 4 → 16 (rayon hides per-dim cost until L1
pressure sets in).

## Tenant pool at scale

`tenant_pool`, each tenant `D=4` / `(50, 64)`, warmed 128 samples:

| N   | `similarity_matrix` | `score_across_tenants` | `most_similar_top5` |
| --- | ------------------- | ---------------------- | ------------------- |
| 32  | 28 µs               | 85 µs                  | 0.33 µs             |
| 128 | 72 µs               | 311 µs                 | 1.10 µs             |
| 512 | 550 µs              | 2.67 ms                | 4.59 µs             |

Scaling `N=32→512` (16× tenants):

- `similarity_matrix` O(N²) parallelised: 20× (rayon fan-out
  hides quadratic until core saturation).
- `score_across_tenants` O(N) parallelised: 31×.
- `most_similar_top5` O(N·log k) bounded heap: 14×.

## External baselines (synthetic)

Input: 10k points, `D=16`, 1 % outliers, 30 % warm / 70 % eval,
frozen baseline. Each impl on its idiomatic fast path
(rcf-rs rayon / rrcf single-process / sklearn NumPy-Cython SIMD
/ AWS Java cold JVM). **5-seed variance** (seeds 2026–2030),
mean ± stddev, coefficient of variation in parens.

| Impl                                   | Backend              | Updates/s                   | Scores/s                  | AUC       |
| -------------------------------------- | -------------------- | --------------------------- | ------------------------- | --------- |
| `rcf-rs` 0.0.0-dev, `score()`          | Rust, rayon-parallel | **31 500** (single seed)    | **197 900** (single seed) | 1.000     |
| `rcf-rs` 0.0.0-dev, `score_codisp()`   | Rust, parallel walk  | — (per-probe insert/delete) | 8 150 (single seed)       | 1.000     |
| `rcf-rs` 0.0.0-dev, `score()` (5-seed) | Rust, rayon-parallel | 17 500 ± 1 240 (7 %)        | 125 900 ± 1 840 (1.5 %)   | 1.000 ± 0 |
| `randomcutforest-java` 4.4.0           | JVM 26, cold         | 2 090 ± 134 (6 %)           | 8 870 ± 415 (5 %)         | 1.000 ± 0 |
| `rrcf` 0.4.4                           | Python + NumPy       | 73 ± 3 (4 %)                | 94 150 ± 4 840 (5 %)      | 0.992 ± 0 |
| `sklearn.IsolationForest`              | NumPy + Cython       | batch-only                  | 136 300 ± 2 450 (2 %)     | 1.000 ± 0 |

Ratios (mean/mean):

- **Updates**: rcf-rs is ~8.4× faster than AWS Java, ~240× faster
  than rrcf. CVs around 5-7 % on all impls; the ratios sit well
  outside the noise floor.
- **Scores (fast path)**: sklearn edges rcf-rs `score()` by 8 %
  (136k vs 126k) — real but small (stddevs combined ≈ 3k, so the
  10k delta is ~3σ significant). rrcf trails rcf-rs by ~25 %;
  AWS Java trails by ~14×.
- **Scores (codisp path)**: rcf-rs `score_codisp()` mutates the
  forest per probe (insert → walk leaf→root → delete). Post the
  rayon-per-tree parallel refactor (walk + delete now fan out
  across trees) it hits ~8 k probes/s at `(100, 256, D=16)` —
  ~25× slower than the isolation-depth `score()` fast path.
  Matches AWS Java `getAnomalyScore` / rrcf `codisp()` semantic;
  use it for SOC triage / forensic replay,
  not the eBPF hot path.
- **AUC**: identical within measurement precision across every
  seed (0.992 for rrcf, 1.000 for the other three).

Noise sources documented: machine thermal state varies across
runs — single-seed numbers from an earlier cool-CPU session
landed at ~32k/203k for rcf-rs, dropping to ~17k/126k on this
run. The **ratios are portable, the absolute numbers aren't**.

Reproduce the sweep:

```bash
scripts/synthetic/variance_sweep.sh /tmp/aws-rcf/randomcutforest-core-4.4.0.jar
```

## Detection quality — NAB `realKnownCause`

Three scoring APIs, three trade-offs:

- **`RandomCutForest::score()`** — isolation-depth, non-mutating,
  rayon-parallel, eBPF-hot-path friendly.
  On NAB: **0.719** aggregate AUC after the lag=32 + zscore +
  smooth(0.02) pipeline.
- **`RandomCutForest::score_codisp()`** — probe-based (insert,
  walk leaf→root accumulating `max(sibling.mass /
subtree.mass)`, remove). Matches rrcf / AWS Java scoring
  semantic. ~30× slower than `score()` and **mutates the
  reservoir** — insertion evicts baseline points the following
  `delete` cannot restore, so long eval streams drift away from
  the frozen warm-phase baseline. On NAB: **0.776** aggregate
  AUC, beats rrcf (0.748) and AWS Java (0.757).
- **`RandomCutForest::score_codisp_stateless()`** — root → leaf
  walk along stored cuts, accumulates `max(sibling_mass /
subtree_mass)` per level without inserting the probe. Preserves
  the frozen-baseline promise exactly, takes `&self`, rayon-
  parallel across trees. On NAB: **0.763** aggregate AUC
  (**~0.013 shy of mutating codisp, ~0.044 above `score()`**).
  Runtime for the 7-file corpus: **1.09 s parallel** — 12× faster
  than the mutating variant.

Same embedding pipeline (32-lag → warm-phase z-score → EMA
α = 0.02), 15 % warm, 100 trees × 256 sample. `tests/nab.rs`
runs the 7-file corpus in parallel via rayon `par_iter` over
files — each file owns an independent forest. Full run
(both variants, parallel file iter) completes in ~12 s.

| File                                   | `score()` | `score_codisp()` | `score_codisp_stateless()` | rrcf  | AWS Java  |
| -------------------------------------- | --------- | ---------------- | -------------------------- | ----- | --------- |
| `ambient_temperature_system_failure`   | **0.813** | **0.813**        | 0.793                      | 0.734 | 0.786     |
| `cpu_utilization_asg_misconfiguration` | 0.953     | **0.969**        | 0.963                      | 0.849 | 0.906     |
| `ec2_request_latency_system_failure`   | **0.709** | 0.706            | 0.621                      | 0.481 | 0.482     |
| `machine_temperature_system_failure`   | 0.578     | **0.817**        | 0.815                      | 0.880 | 0.883     |
| `nyc_taxi`                             | **0.698** | 0.636            | 0.623                      | 0.571 | 0.540     |
| `rogue_agent_key_hold`                 | 0.145     | 0.198            | 0.181                      | 0.535 | **0.633** |
| `rogue_agent_key_updown`               | **0.633** | 0.579            | 0.563                      | 0.657 | 0.542     |
| **weighted aggregate**                 | 0.719     | **0.776**        | 0.763                      | 0.748 | 0.757     |

### Hyperparameter ablation

`examples/nab_ablation.rs` on the same corpus:

| Config                                  | Aggregate AUC |
| --------------------------------------- | ------------- |
| baseline (lag=8, raw score)             | 0.615         |
| lag=32                                  | 0.665         |
| lag=32 + diff                           | 0.640         |
| lag=32 + zscore                         | 0.683         |
| lag=32 + smooth(0.1)                    | 0.687         |
| lag=32 + zscore + smooth(0.05)          | 0.718         |
| **lag=32 + zscore + smooth(0.02)**      | **0.719**     |
| lag=64 + zscore + smooth(0.05)          | 0.672         |
| trcf-online D=32                        | 0.320         |
| probe-score D=8 (naive hack)            | 0.330         |
| **codisp D=32 + zscore + smooth(0.02)** | **0.776**     |

- **Longer embedding** (lag=32) captures wider context — +0.050.
- **Z-score per warm-phase dim stddev** compensates NAB's scale
  variance (CPU % vs taxi counts vs temperatures) — RCF cuts are
  range-weighted, so un-normalised inputs let one dim dominate.
  +0.018 on top of lag=32.
- **EMA smoothing of raw scores** (α ≈ 0.02, half-life ~35 steps)
  cuts per-point noise without losing the wide-window shape.
  +0.036 on top of lag=32 + zscore.
- **Differencing** (first-diff of lag values) regresses — NAB's
  contextual signal lives in absolute values, not rate-of-change.
- **TRCF online** regresses catastrophically (0.72 → 0.32): the
  EMA threshold adapts UP during the multi-day anomaly windows
  and stops flagging them. Frozen baseline is the right paradigm
  for NAB's wide-window labels.
- **Probe-based naive hack** (`update_indexed → score → delete`)
  tanks AUC (0.330) — post-insert `score` ranks the freshly-
  inserted probe as seen. Proper codisp is `score_codisp()`:
  walks from inserted leaf → root accumulating
  `max(sibling.mass / subtree.mass)`, then deletes the probe.

### Two-API split

`score()` (isolation-depth, non-mutating, parallel) and
`score_codisp()` (probe-based, mutating, sequential per tree)
serve different use cases. `score()` is the eBPF hot-path
default; `score_codisp()` is for SOC triage / forensic replay
where the extra ~30× latency is acceptable for the +0.057 AUC
gain. On NAB, `score_codisp()` (0.776) leads both rrcf (0.748)
and AWS Java (0.757).

- `tests/detection_quality.rs` pins synthetic-corpus regression
  guards: AUC > 0.95 on separable clusters, > 0.90 on transition.
- `tests/nab.rs` pins NAB aggregate floor at 0.70.

Reproduce:

```bash
git clone --depth 1 https://github.com/numenta/NAB.git /opt/nab
RCF_NAB_PATH=/opt/nab cargo test --test nab --all-features -- --ignored --nocapture
python3 scripts/nab/bench_rrcf_nab.py --nab /opt/nab
java -cp "scripts/nab:/tmp/aws-rcf/randomcutforest-core-4.4.0.jar" RcfBenchNab /opt/nab
```

## Detection quality — TSB-AD-M (multivariate)

TSB-AD-M (TheDatumOrg, 2024): 200 multivariate series across 16
source datasets, per-point binary labels, native multivariate
(no lag embedding). Pipeline: per-dim z-score on the upstream
`tr_<N>` train split, frozen-baseline scoring, EMA-smooth
α = 0.02. Forest `(100, 256)`, seed `2026`. Const-generic
whitelist `{2, 3, 7, 8, 9, 12, 16, 17, 18, 19, 25, 29, 31, 38, 51,
55, 66}` covers **192 / 200 files (96 %)**; the eight D=248 files
are skipped. `tests/tsb_ad_m.rs` runs the corpus in parallel via
rayon `par_iter` over files. Runtime on reference hardware:
~3 min `score()`, ~6 min `score_codisp()` (stride-subsampled to
50 k eval rows / file), ~3 min `score_codisp_stateless()` on the
**full** eval stream.

Per-dataset ROC-AUC (weighted by positive count) against
`randomcutforest-java` 4.4.0 on the same corpus. rrcf 0.4.4 was
benched with the same protocol
(`scripts/tsb_ad/bench_rrcf_tsb_ad.py`, parallel across files)
but wall-time is prohibitive on the full corpus — ~3–4 h at
14 workers / `--max-eval 1500`. Numbers are left for the reader
to reproduce; the script is provided for reproducibility.

| Source dataset         | Files             | `score()` | `score_codisp()` | `score_codisp_stateless()` | AWS Java  |
| ---------------------- | ----------------- | --------- | ---------------- | -------------------------- | --------- |
| Genesis                | 1                 | 0.968     | **0.991**        | **0.994**                  | 0.982     |
| SMAP                   | 27                | 0.803     | **0.823**        | 0.716                      | 0.805     |
| SMD                    | 22                | 0.618     | **0.760**        | 0.752                      | 0.806     |
| MSL                    | 16                | **0.705** | 0.746            | 0.599                      | 0.762     |
| SVDB                   | 31                | 0.692     | 0.737            | **0.779**                  | 0.757     |
| LTDB                   | 5                 | 0.601     | 0.755            | **0.758**                  | 0.755     |
| Exathlon               | 27                | 0.491     | 0.894            | **0.996**                  | 0.865     |
| MITDB                  | 13                | 0.597     | **0.678**        | 0.603                      | 0.660     |
| PSM                    | 1                 | 0.608     | 0.595            | **0.613**                  | 0.611     |
| CATSv2                 | 6                 | **0.580** | 0.547            | 0.496                      | 0.547     |
| CreditCard             | 1                 | 0.589     | 0.679            | 0.658                      | **0.693** |
| Daphnet                | 1                 | 0.309     | 0.885            | **0.926**                  | 0.944     |
| GECCO                  | 1                 | 0.412     | 0.523            | **0.753**                  | 0.594     |
| GHL                    | 25                | 0.454     | 0.461            | **0.570**                  | 0.419     |
| OPPORTUNITY            | 8 (skipped D=248) | —         | —                | —                          | 0.298     |
| SWaT                   | 2                 | 0.282     | **0.825**        | 0.715                      | 0.825     |
| TAO                    | 13                | 0.451     | 0.453            | **0.487**                  | 0.471     |
| **aggregate weighted** | **192 / 200**     | 0.583     | **0.768**        | 0.751                      | 0.753     |

- **rcf-rs `score()`** — isolation depth, rayon-parallel, full
  eval scan. Same fast API eBPFsentinel ships on the hot path.
  `tests/tsb_ad_m.rs::tsb_ad_m_aggregate_auc_above_floor` pins
  the aggregate floor at 0.55 — regression guard, not a quality
  claim.
- **rcf-rs `score_codisp()`** — probe-based codisp walk (leaf → root,
  `max(sibling.mass / subtree.mass)`), sequential per tree,
  mutates the reservoir per probe. Stride-subsampled to 50 000
  eval rows per file (const `CODISP_MAX_EVAL`). Directly
  comparable to the AWS Java / rrcf semantic; leads aggregate
  **0.768** vs AWS Java 0.753.
  `tests/tsb_ad_m.rs::tsb_ad_m_codisp_aggregate_auc_above_floor`.
- **rcf-rs `score_codisp_stateless()`** — root → leaf walk along
  stored cuts, max `sibling_mass / subtree_mass` per level, no
  reservoir mutation. Takes `&self` → rayon-parallel across
  trees. Covers the **full** eval stream (no stride) and
  preserves the frozen-baseline semantic across long runs.
  Aggregate **0.751** — ~0.017 below the drift-affected mutating
  codisp but within measurement noise of AWS Java (0.753) and
  the only variant that scales past the `CODISP_MAX_EVAL` cap.
  `tests/tsb_ad_m.rs::tsb_ad_m_codisp_stateless_aggregate_auc_above_floor`.
- **AWS Java `getAnomalyScore()`** — codisp-like, stride-
  subsampled to 50 000 eval rows per file (essentially full-scan
  for 95 % of the corpus). Covers all 200 files including the
  eight D=248 OPPORTUNITY series the const-generic whitelist
  skips.

Caveats:

- **Plain point-wise ROC-AUC**; the official TSB-AD leaderboard
  ranks on **VUS-PR** (Paparrizos et al. 2022) which integrates
  range-based precision / recall across a sliding window.
- **RCF is classical by design** — transformer-based SOTA
  (TimesNet, Anomaly Transformer) outscores every impl here on
  heavy-physics datasets (SWaT, Daphnet, GECCO) where the anomaly
  signature lives in higher-order cross-channel structure. RCF
  stays competitive on Genesis / SMAP / MSL / SVDB where per-dim
  statistical drift dominates — closer to eBPFsentinel's
  production feature mix (rate, ratio, entropy, cardinality).

Reproduce:

```bash
scripts/tsb_ad/fetch.sh /tmp/tsb-ad
RCF_TSB_AD_M_PATH=/tmp/tsb-ad/TSB-AD-M \
    cargo test --release --test tsb_ad_m --all-features -- --ignored --nocapture
python3 scripts/tsb_ad/bench_rrcf_tsb_ad.py \
    --dir /tmp/tsb-ad/TSB-AD-M --max-eval 1500 --workers "$(nproc)"
javac -cp /tmp/aws-rcf-central/randomcutforest-core-4.4.0.jar \
    scripts/tsb_ad/RcfBenchTsbAd.java
java -cp scripts/tsb_ad:/tmp/aws-rcf-central/randomcutforest-core-4.4.0.jar \
    RcfBenchTsbAd /tmp/tsb-ad/TSB-AD-M 50000
```
